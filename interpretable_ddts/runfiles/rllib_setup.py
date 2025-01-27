from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import ray
from packaging.version import parse as parse_version
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.tune import CLIReporter

from ray_utilities.comet import comet_upload_offline_experiments, get_default_workspace  # isort: skip # comet should be imported before torch

from interpretable_ddts.runfiles._ddt_trainable import build_and_train
from ray_utilities.constants import DISC_EVAL_METRIC_RETURN_MEAN
from ray_utilities import trial_name_creator
from ray_utilities.callbacks.tuner import (
    AdvCometLoggerCallback,
    create_tuner_callbacks,
)

from interpretable_ddts.runfiles.ddt_setup import DDTSetup

os.environ["RAY_COLOR_PREFIX"] = "1"

RAY_VERSION = parse_version(ray.__version__)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ray.tune.callback import Callback


def main():
    # full parser example see: https://github.com/ray-project/ray/blob/master/rllib/utils/test_utils.py#L61

    setup = DDTSetup()
    args = setup.get_args()
    env_name = args.env_type
    use_comet_offline: bool = args.comet and args.comet.lower().startswith("offline")
    trainable = setup.trainable

    # Will use these resources per job
    # NOTE: Even if not used will allocate these resources per run
    # trainable_with_resources = tune.with_resources(trainable, tune.PlacementGroupFactory(
    #    [{'CPU': 1.0}] + [{'CPU': 1.0}] * (4 if args.parallel else 0),
    # ))
    # Use tune.with_parameters to pass large objects to the trainable

    # Callbacks

    # If videos are logged use custom callbacks for correct logging
    # NOTE: JSON, CSV, and Tensorboard loggers are created automatically by Tune if not disabled
    tags = setup.create_tags()
    callbacks: list[Callback] = create_tuner_callbacks(render=bool(args.render_mode))

    # WandB
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

    # Comet
    if args.comet or args.test:
        # API KEY
        from dotenv import load_dotenv

        load_dotenv(Path("~/.comet_api_key.env").expanduser())

        workspace_name = (
            "dev-workspace"
            if args.test
            else (
                get_default_workspace()
                if args.comet and not use_comet_offline
                else None  # if disabled no need to query
            )
        )
        project_name = "_".join([args.agent_type, env_name, "-dev", ("-test" if args.test else "")])

        comet_callback = AdvCometLoggerCallback(
            disabled=not args.comet and args.test,
            online=not use_comet_offline,  # do not upload
            project_name=project_name,  # "general" for Uncategorized Experiments
            workspace=workspace_name,
            save_checkpoints=False,
            tags=tags,
            # Other keywords see: https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/
            auto_metric_step_rate=10,  # How often batch metrics are logged. Default 10
            auto_histogram_epoch_rate=1,  # How often histograms are logged. Default 1
            parse_args=False,
            log_git_metadata=not args.test,  # disabled by rllib; might cause throttling -> needed for Reproduce button
            log_git_patch=False,
            log_graph=False,  # computation graph, Default True
            log_code=False,  # Default True; use if not using git_metadata
            log_env_details=not args.test,
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
            log_to_other=("comment", "cli_args/comment", "cli_args/test", "cli_args/num_jobs"),
            log_cli_args=True,
            log_pip_packages=True,  # only relevant if log_env_details=False
        )
        # Metrics to exclude
        # keep only time_this_iter_s
        callbacks.append(comet_callback)
    else:
        logger.info("Not logging to Comet")

    # -- Test --
    if args.test and args.not_parallel:
        # will spew some warnings about train.report
        print("-- TEST MODE --")
        if args.legacy:
            trainable({})
        else:
            results = build_and_train(setup.param_space, disable_report=True)
        sys.exit()

    # -- Tune --
    tuner = tune.Tuner(
        trainable,  # Note: possibly can also be a list
        param_space=setup.param_space,
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
    return results


if __name__ == "__main__":
    results = main()
