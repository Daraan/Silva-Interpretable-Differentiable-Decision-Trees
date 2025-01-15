from __future__ import annotations

import argparse
import logging
import os
from functools import partial
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

import gymnasium as gym
import ray
from packaging.version import parse as parse_version
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.experimental import tqdm_ray
from ray.tune import CLIReporter

# NOTE: JSON, CSV, and Tensorboard loggers are created automatically by Tune
from ray.tune.logger import (  # noqa: F401
    CSVLoggerCallback,
)

from interpretable_ddts.callbacks.tuner.adv_comet_callback import AdvCometLoggerCallback
from interpretable_ddts.callbacks.tuner.adv_json_logger_callback import AdvJsonLoggerCallback
from interpretable_ddts.callbacks.tuner.adv_tbx_logger_callback import AdvTBXLoggerCallback
from interpretable_ddts.runfiles._ddt_trainable import build_and_train, create_ddt_config
from interpretable_ddts.runfiles.constants import DISC_EVAL_METRIC_RETURN_MEAN
from interpretable_ddts.tools import comet_upload_offline_experiments
from interpretable_ddts.tools import trial_name_creator

os.environ["RAY_COLOR_PREFIX"] = "1"

RAY_VERSION = parse_version(ray.__version__)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ray.tune.logger import LoggerCallback

if __name__ == "__main__":
    # full parser see: https://github.com/ray-project/ray/blob/master/rllib/utils/test_utils.py#L61
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default="ddt")
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=1000)
    parser.add_argument("-l", "--num_leaves", help="number of leaves for DDT/DRL ", type=int, default=8)
    parser.add_argument(
        "-L", "--legacy", help="Use original code without an algorithm", default=False, action="store_true"
    )
    parser.add_argument("-n", "--num_hidden", help="number of hidden layers for MLP ", type=int, default=0)
    parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default="cart")
    parser.add_argument("-gpu", "--gpu", help="run on GPU?", action="store_true")
    parser.add_argument("-r", "--rule_list", help="Use rule list setup", action="store_true", default=False)
    parser.add_argument("-s", "--seed", help="Seed", default=None, type=int)
    parser.add_argument("-J", "--num_jobs", help="Amount of jobs the Tuner does start", default=5, type=int)
    parser.add_argument(
        "-np",
        "--not_parallel",
        help="Do not run multiple models in parallel, i.e. the Tuner will execute one job only. "
        "This is equivalent to num_jobs=1",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-mp",
        "--parallel",
        help="Use multiple CPUs per worker",
        action="store_true",
        default=False,
    )
    parser.add_argument("-nd", "--not_discrete", help="Disable non-discrete eval", action="store_true", default=False)
    parser.add_argument("-p", "--process_number", help="Process number", type=int, default=None)
    parser.add_argument(
        "--silent",
        help="supress prints",
        action="store_true",
        default=False,
    )
    parser.add_argument("--test", "--dry-run", help="Do not save any models", action="store_true", default=False)
    parser.add_argument("--wandb", "-wb", help="Log to WandB", action="store_true", default=False)
    parser.add_argument(
        "--comet",
        nargs="?",
        help="Log to Comet",
        const="1",
        default="off",
        choices=["offline", "offline+upload", "0", "1", "False", "off", "on"],
        type=str,
    )
    parser.add_argument(
        "--render_mode",
        "--render",
        nargs="?",
        const="rgb_array",
        help="Render mode",
        type=str,
        default=None,
        choices=["human", "rgb_array", "ansi"],
    )
    parser.add_argument("--extra", help="extra arguments", nargs="+", choices=["silva_loss"])
    parser.add_argument("--use_silva_loss", "--silva_loss", help="Use Silva loss", action="store_true", default=False)
    parser.add_argument("--comment", "-c", help="Add comment to this run", type=str, default="")

    args = parser.parse_args()
    if args.agent_type == "ddt" and args.num_hidden:
        raise ValueError("Do not use --num_hidden with DDT")
    use_comet_offline = args.comet.lower().startswith("offline")
    if args.comet.lower() in ("0", "false", "off"):
        args.comet = False
    if not args.test and not args.comet:
        logger.warning("Not in test mode and comet disabled. Will not log to Comet")
        import time

        time.sleep(4)  # give user time to cancel

    if args.seed == -1:
        args.seed = None
    if args.env_type == "lunar":
        init_env = gym.make("LunarLander-v2")
    elif args.env_type == "cart":
        init_env = gym.make("CartPole-v1")
    else:
        # Allow different environments
        init_env = gym.make(args.env_type)
    env_name = init_env.unwrapped.spec.id  # pyright: ignore[reportOptionalMemberAccess]
    args.env_type = env_name

    # note config will be passed as first positional argument
    if args.legacy:
        # Do not use an algorithm but the gym_runner.py code
        config, module_spec = create_ddt_config(args)
        from interpretable_ddts.runfiles import gym_runner

        trainable = partial(
            gym_runner.start_process,
            args=argparse.Namespace(
                agent_type=module_spec,
                env_type=config.env,
                seed=args.seed,
                gpu=args.gpu,
                rule_list=args.rule_list,
                num_leaves=args.num_leaves,
                test=args.test,
                num_hidden=args.num_hidden,
                use_pbar=tqdm_ray.tqdm,
                episodes=args.episodes,
                # Note: cast to int as it might be an np.int type
                dim_in=int(module_spec.observation_space.shape[0]),  # pyright: ignore[reportOptionalSubscript, reportOptionalMemberAccess]
                dim_out=int(module_spec.action_space.n),  # type: ignore[attr-defined],
                render_mode=None,
                comment=args.comment,
            ),
            use_rllib_output=True,
        )
    else:
        config, module_spec = create_ddt_config(args)
        trainable = partial(build_and_train, use_pbar=True)

    # If videos are logged use custom callbacks for correct logging
    callbacks: list[LoggerCallback] = (
        [
            AdvJsonLoggerCallback(),
            AdvTBXLoggerCallback(),
        ]
        if args.render_mode
        else []
    )
    tags = ["dev", env_name, args.agent_type]
    if args.test:
        tags.append("test")
    if args.legacy:
        tags.append("legacy")
    if args.use_silva_loss:
        tags.append("silva_loss")
    if args.gpu:
        tags.append("gpu")
    if args.rule_list:
        tags.append("RuleList")
    if args.wandb:
        callbacks.append(
            WandbLoggerCallback(
                project="SilvaWandB-Test",
                group="test_experiment",  # if not set Tuner name is used
                excludes=["system/*"],
                upload_checkpoints=False,
                save_code=False,  # Code diff
                # For more keywords see: https://docs.wandb.ai/ref/python/init/
                # Log gym
                # https://docs.wandb.ai/guides/integrations/openai-gym/
                monitor_gym=False,
                # Special comment
                notes=args.comment if args.comment else None,
                tags=tags,
            ),
        )
    else:
        # could use wandb offline
        logger.info("Not logging to WandB")
    if args.comet or args.test:
        # API KEY
        from dotenv import load_dotenv

        load_dotenv(Path("~/.comet_api_key.env").expanduser())

        comet_callback = AdvCometLoggerCallback(
            disabled=not args.comet and args.test,
            online=not use_comet_offline,  # do not upload
            project_name="test-project",  # "general" for Uncategorized Experiments
            workspace="dev-workspace" if args.test else None,
            save_checkpoints=False,
            tags=tags,
            # Other keywords see: https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/
            auto_metric_step_rate=10,  # How often batch metrics are logged. Default 10
            auto_histogram_epoch_rate=1,  # How often histograms are logged. Default 1
            parse_args=False,
            log_git_metadata=(
                not args.test and (args.num_jobs <= 10 or use_comet_offline)
            ),  # disabled by rllib; might cause throttling
            log_git_patch=False,
            log_graph=False,  # computation graph, Default True
            log_code=not args.test,  # Default True
            log_env_details=True,
            # Subkeys of env details:
            log_env_network=False,
            log_env_disk=False,
            log_env_gpu=args.num_jobs <= 5 and args.gpu,
            log_env_host=False,
            log_env_cpu=args.num_jobs <= 5,
            # ---
            auto_log_co2=False,  # needs codecarbon
            auto_histogram_weight_logging=False,  # Default False
            auto_histogram_gradient_logging=False,  # Default False
            auto_histogram_activation_logging=False,  # Default False
            # Custom keywords of Adv Callback
            exclude_metrics=(
                "time_since_restore",
                "iterations_since_restore",
                "timestamp",
                # "training_iteration", #  needed for the callback
            ),
            log_to_other=("comment", "cli_args/comment", "cli_args"),
            log_cli_args=True,
        )
        # Metrics to exclude
        # keep only time_this_iter_s
        callbacks.append(comet_callback)
    # Will use these resources per job
    # NOTE: Even if not used will allocate these resources per run
    # trainable_with_resources = tune.with_resources(trainable, tune.PlacementGroupFactory(
    #    [{'CPU': 1.0}] + [{'CPU': 1.0}] * (4 if args.parallel else 0),
    # ))
    # Use tune.with_parameters to pass large objects to the trainable
    # Create a dict to upload as hyperparameters

    # -- Preprocess Parameters --

    upload_args = vars(args).copy()
    upload_args["extra"] = repr(args.extra)
    for key in ("test", "wandb", "comet", "comment", "not_parallel", "num_jobs", "silent"):
        del upload_args[key]
    if upload_args["process_number"] is None:
        del upload_args["process_number"]

    param_space: dict[str, Any] = {
        "env": config.env if isinstance(config.env, str) else config.env.unwrapped.spec.id,
        "algo": config.algo_class.__name__,
        "module": config.rl_module_spec.module_class.__name__,
        "model_config": config.rl_module_spec.model_config,
    }
    # WandB might not log them if they are not selected as choice
    param_space = {k: tune.choice([v]) for k, v in param_space.items()}
    param_space["cli_args"] = upload_args

    if args.test and args.not_parallel:
        # will spew some warnings about train.report
        print("-- TEST MODE --")
        if args.legacy:
            trainable({})
        else:
            result = build_and_train(param_space, disable_report=True)
        sys.exit()

    # -- Tune --
    tuner = tune.Tuner(
        trainable,  # Note: possibly can also be a list
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=1 if args.not_parallel else args.num_jobs,
            # metric=
            #    (EVALUATION_RESULTS + "/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
            #     if config.evaluation_interval else ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN),
            mode="max",
            trial_name_creator=trial_name_creator,
        ),
        run_config=train.RunConfig(
            # Trial artifacts are uploaded periodically to this directory
            storage_path=Path("../outputs").resolve(),  # type: ignore[argument]
            name="test_experiment",
            log_to_file=False,  # True for hydra like logging to files; or (stoud, stderr.log) files
            progress_reporter=CLIReporter(mode="max", max_report_frequency=45),
            # JSON, CSV, and Tensorboard loggers are created automatically by Tune
            # to disable set TUNE_DISABLE_AUTO_CALLBACK_LOGGERS environment variable to "1"
            callbacks=callbacks,
            # Use fail_fast for during debugging/testing to stop all experiments
            failure_config=train.FailureConfig(fail_fast=True),
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=4,
                checkpoint_score_order="max",
                checkpoint_score_attribute=DISC_EVAL_METRIC_RETURN_MEAN,
            ),
        ),
    )
    results = tuner.fit()
    if args.comet == "offline+upload":
        comet_upload_offline_experiments()
