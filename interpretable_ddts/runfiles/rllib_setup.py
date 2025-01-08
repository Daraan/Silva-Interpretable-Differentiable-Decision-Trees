from __future__ import annotations

import argparse
import logging
import math
import os
from functools import partial
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Optional, TypeVar

import gymnasium as gym
import ray
from packaging.version import parse as parse_version
from ray import train, tune
from ray.air.integrations.comet import CometLoggerCallback
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.experimental import tqdm_ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
)
from ray.tune import CLIReporter

# NOTE: JSON, CSV, and Tensorboard loggers are created automatically by Tune
from ray.tune.logger import (  # noqa: F401
    CSVLoggerCallback,
    JsonLoggerCallback,
    TBXLoggerCallback,
)

from interpretable_ddts.agents.ddt_catalog import DDTCatalog
from interpretable_ddts.agents.ddt_ppo_module import DDTModule
from interpretable_ddts.agents.ppo_learner import SilvaLearner
from interpretable_ddts.agents.rllib_port.discrete_evaluation import eval_with_discrete
from interpretable_ddts.tools import is_pbar


os.environ["RAY_COLOR_PREFIX"] = "1"

RAY_VERSION = parse_version(ray.__version__)

# Keys
TRAIN_METRIC_RETURN_MEAN = ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
EVAL_METRIC_RETURN_MEAN = EVALUATION_RESULTS + "/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
DISC_EVAL_METRIC_RETURN_MEAN = EVALUATION_RESULTS + "/discrete/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN

logger = logging.getLogger(__name__)

_ConfigType = TypeVar("_ConfigType", bound=PPOConfig)


def create_ddt_config(
    args: argparse.Namespace,
    env: str | gym.Env,
    config_class: type[_ConfigType] = PPOConfig,
) -> tuple[_ConfigType, RLModuleSpec]:
    config = config_class()
    config.environment(env)
    config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    config.resources(
        # num_gpus=1 if args.gpu else 0,4
        # process that runs Algorithm.training_step() during Tune
        num_cpus_for_main_process=1,
        # num_learner_workers=4 if args.parallel else 1,
        # num_cpus_per_learner_worker=1,
        # num_cpus_per_worker=1,
    )
    config.env_runners(
        num_env_runners=2 if args.parallel else 0,
        num_cpus_per_env_runner=1,  # num_cpus_per_worker
        # explore=False,
        # How long an rollout episode lasts, for "auto" calculated from batch_size
        # total_train_batch_size / (num_envs_per_env_runner * num_env_runners)
        # rollout_fragment_length=1,  # Default: "auto"
        num_envs_per_env_runner=1,
        validate_env_runners_after_construction=args.test,
        # 1) "truncate_episodes": Each call to `EnvRunner.sample()` returns a
        #    batch of at most `rollout_fragment_length * num_envs_per_env_runner` in
        #    size. The batch is exactly `rollout_fragment_length * num_envs`
        #    in size if postprocessing does not change batch sizes.
        # Use if not using GAE
        # 2) "complete_episodes": Each call to `EnvRunner.sample()` returns a
        #    batch of at least `rollout_fragment_length * num_envs_per_env_runner` in
        #    size. Episodes aren't truncated, but multiple episodes
        #    may be packed within one batch to meet the (minimum) batch size.
        batch_mode="truncate_episodes",
    )
    config.learners(
        # for fractional GPUs, you should always set num_learners to 0 or 1
        num_learners=0 if args.parallel else 0,
        num_cpus_per_learner=1,
        num_gpus_per_learner=1 if args.gpu else 0,
    )

    USE_SILVA_LOSS = True
    config.framework("torch")
    config.training(
        learner_class=SilvaLearner,
        learner_config_dict={"use_silva_loss": USE_SILVA_LOSS},
        gamma=0.99,
        use_critic=True,
        # with a growing number of Learners and to increase the learning rate as follows:
        # lr = [original_lr] * ([num_learners] ** 0.5)
        lr=(
            1e-3
            if True
            # Shedule LR
            else [
                [0, 8e-3],  # <- initial value at timestep 0
                [100, 4e-3],
                [400, 1e-3],
                [800, 1e-4],
            ]
        ),
        clip_param=0.2,
        grad_clip=0.5,
        # grad_clip_by="norm",
        entropy_coeff=0.01,
        # vf_clip_param=10,
        train_batch_size_per_learner=36,
        # The total effective batch size is then
        # `num_learners` x `train_batch_size_per_learner` and you can
        # access it with the property `AlgorithmConfig.total_train_batch_size`.
        minibatch_size=8,
        num_epochs=20,
        use_kl_loss=False,
        use_gae=True,  # Must be true to use "truncate_episodes"
    )
    # Create a single agent RL module spec.
    module_spec = RLModuleSpec(
        module_class=DDTModule,
        observation_space=init_env.observation_space,
        action_space=init_env.action_space,
        model_config={
            "bot_name": args.agent_type + args.env_type,
            "rule_list": args.rule_list,
            "num_rules": args.num_leaves,
            "save_output": not args.test,
            "use_gpu": args.gpu,
            "vf_double_output": USE_SILVA_LOSS,
            "action_use_softmax": USE_SILVA_LOSS,
            "use_silva_loss": USE_SILVA_LOSS,  # unused by model config
        },
        catalog_class=DDTCatalog,
    )
    # module = module_spec.build()

    config.rl_module(
        rl_module_spec=module_spec,
    )
    # https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.evaluation.html
    config.evaluation(
        custom_evaluation_function=eval_with_discrete,
        evaluation_interval=10,
        evaluation_duration=5,
        evaluation_duration_unit="episodes",
        evaluation_num_env_runners=2 if args.parallel else 0,
    )

    config.reporting(
        keep_per_episode_custom_metrics=True,  # If True calculate max min mean
        log_gradients=False,  # Default is True
        # Will smooth metrics in the reports, e.g. tensorboard
        metrics_num_episodes_for_smoothing=1,  # Default is 100
    )
    config.debugging(
        # https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.debugging.html#ray-rllib-algorithms-algorithm-config-algorithmconfig-debugging
        # seed=args.seed,
    )
    # Checks
    config.validate_train_batch_size_vs_rollout_fragment_length()
    assert (
        config.rl_module_spec.model_config["vf_double_output"]  # type: ignore
        == config.learner_config_dict["use_silva_loss"]
    )
    return config, module_spec


if __name__ == "__main__":
    # full parser see: https://github.com/ray-project/ray/blob/master/rllib/utils/test_utils.py#L61
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default="ddt")
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=1000)
    parser.add_argument("-l", "--num_leaves", help="number of leaves for DDT/DRL ", type=int, default=8)
    parser.add_argument("-n", "--num_hidden", help="number of hidden layers for MLP ", type=int, default=0)
    parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default="cart")
    parser.add_argument("-gpu", "--gpu", help="run on GPU?", action="store_true")
    parser.add_argument("-r", "--rule_list", help="Use rule list setup", action="store_true", default=False)
    parser.add_argument("-s", "--seed", help="Seed", default=-1, type=int)
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
    parser.add_argument("-p", "--process_number", help="Process number", type=int, default=0)
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
        choices=["offline", "0", "1", "False", "off"],
    )
    parser.add_argument(
        "-rl",
        "--rllib",
        help="Use rllib",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    if args.comet.lower() in ("0", "false", "off"):
        args.comet = False

    if args.seed == -1:
        args.seed = None
    if args.env_type == "lunar":
        init_env = gym.make("LunarLander-v2")
    elif args.env_type == "cart":
        init_env = gym.make("CartPole-v1")
    else:
        raise ValueError(f"No valid environment {args.env_type}")
    env_name = init_env.unwrapped.spec.id  # pyright: ignore[reportOptionalMemberAccess]

    config, module_spec = create_ddt_config(args, init_env)

    def build_and_train(index: Optional[int | dict[str, Any]] = None, *, use_pbar=True, disable_report=False):
        """
        Args:
            index: Is a `dict` / `param_spec` if this is used by Tune.

        Warning:
            Best practice is to not refer to any objects from outer scope in the training_function
        """
        config, _ = create_ddt_config(args, env_name)
        algo = config.build()

        if use_pbar:
            pbar = tqdm_ray.tqdm(range(args.episodes), position=index if isinstance(index, int) else None)
        else:
            pbar = range(args.episodes)
        running_eval_rewards = []
        running_disc_eval_rewards = []
        running_rewards = []
        for _episode in pbar:
            result = algo.train()
            # Results
            # Training:
            train_reward = result[ENV_RUNNER_RESULTS].get(EPISODE_RETURN_MEAN, float("nan"))
            if not math.isnan(train_reward):
                running_rewards.append(train_reward)
            running_reward = sum(running_rewards[-100:]) / (
                min(100, len(running_rewards)) or float("nan")  # nan for 0
            )
            # Evaluation:
            eval_results = result.get(EVALUATION_RESULTS, {})
            eval_env_runner_results = eval_results.get(ENV_RUNNER_RESULTS, {})
            eval_mean = eval_env_runner_results.get(EPISODE_RETURN_MEAN, float("nan"))
            if not math.isnan(eval_mean):
                running_eval_rewards.append(eval_mean)
            running_eval_reward = sum(running_eval_rewards[-100:]) / (
                min(100, len(running_eval_rewards)) or float("nan")  # nan for 0
            )
            # Discrete rewards:
            discrete_evaluation = eval_results.get("discrete", {})
            disc_eval_env_runner_results = discrete_evaluation.get(ENV_RUNNER_RESULTS, {})
            disc_eval_mean = disc_eval_env_runner_results.get(EPISODE_RETURN_MEAN, float("nan"))
            if not math.isnan(disc_eval_mean):
                running_disc_eval_rewards.append(disc_eval_mean)
            disc_running_eval_reward = sum(running_disc_eval_rewards[-100:]) / (
                min(100, len(running_disc_eval_rewards)) or float("nan")  # nan for 0
            )

            metrics = {
                TRAIN_METRIC_RETURN_MEAN: result[ENV_RUNNER_RESULTS].get(
                    EPISODE_RETURN_MEAN,
                    float("nan"),
                ),
                EVAL_METRIC_RETURN_MEAN: eval_mean,
                DISC_EVAL_METRIC_RETURN_MEAN: disc_eval_mean,
            }
            # Report metrics
            if not disable_report:
                train.report(metrics)

            # Update progress bar
            if not is_pbar(pbar):
                continue
            try:
                pbar.set_description(
                    f"R mean: {metrics[TRAIN_METRIC_RETURN_MEAN]:>6.1f} |"
                    f"R max: {result['env_runners']['episode_return_max']:>4.0f} |"
                    f"R roll: {running_reward:>6.1f} |"
                    f"Eval Rew: {eval_mean:>6.1f} |"
                    f"Roll Eval Rew: {running_eval_reward:>6.1f} |"
                    f"Disc Eval Rew: {disc_eval_mean:>6.1f} |"
                    f"Roll Disc Rew: {disc_running_eval_reward:>6.1f} |"
                )
            except KeyError as e:
                print("Error with Key", e)
                pbar.set_description("")
        eval_result = algo.evaluate()
        eval_result["done"] = True
        return eval_result

    # note config will be passed as first positional argument
    if False:
        from interpretable_ddts.runfiles import gym_runner
        from interpretable_ddts.agents.ddt_ppo_module import DDTModuleGymRunner

        module_spec.module_class = DDTModuleGymRunner
        module_spec.model_config.update(
            {
                "save_output": False,
            }
        )
        module_spec.model_config.update(
            {
                "save_output": not args.test,
                "use_gpu": args.gpu,
                "vf_double_output": True,
                "action_use_softmax": True,
                "use_silva_loss": True,
            },
        )
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
                dim_in=int(init_env.observation_space.shape[0]),  # pyright: ignore[reportOptionalSubscript],
                dim_out=int(init_env.action_space.n),  # type: ignore[attr-defined],
                render_mode=None,
            ),
        )
    else:
        trainable = partial(build_and_train, use_pbar=True)
    param_space = {
        "env": str(config.env),
        "algo": config.algo_class.__name__,
        "module": config.rl_module_spec.module_class.__name__,
        "model_config": config.rl_module_spec.model_config,
    }
    param_space = {k: tune.choice([v]) for k, v in param_space.items()}

    callbacks = []
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
                notes="test save code",
            ),
        )
    else:
        # could use wandb offline
        logger.info("Not logging to WandB")
    if args.comet:
        os.environ["COMET_API_KEY"] = "XXXX"  # or use keyword api_key
        callbacks.append(
            CometLoggerCallback(
                disabled=args.comet == "offline",  # do not upload
                save_checkpoints=False,
                tags=["test", "dev"],
                # Other keywords see: https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/
                auto_metric_step_rate=10,  # How often batch metrics are logged
                log_git_metadata=True,  # disabled by rllib
                log_graph=True,  # Default True
                # api_key=,
                log_env_details=True,
                auto_log_co2=False,  # needs codecarbon
                auto_histogram_weight_logging=True,  # Default False
                auto_histogram_gradient_logging=True,  # Default False
                auto_histogram_activation_logging=True,  # Default False
            ),
        )
    # Will use these resources per job
    # NOTE: Even if not used will allocate these resources per run
    # trainable_with_resources = tune.with_resources(trainable, tune.PlacementGroupFactory(
    #    [{'CPU': 1.0}] + [{'CPU': 1.0}] * (4 if args.parallel else 0),
    # ))
    # Use tune.with_parameters to pass large objects to the trainable
    if args.test and args.not_parallel:
        # will spew some warnings about train.report
        result = build_and_train(disable_report=True)
        sys.exit()

    tuner = tune.Tuner(
        trainable,  # Note: possibly can also be a list
        # "PPO",
        # run_config=air.RunConfig(stop={"training_iteration": 1}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=1 if args.not_parallel else args.num_jobs,
            # metric=
            #    (EVALUATION_RESULTS + "/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
            #     if config.evaluation_interval else ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN),
            mode="max",
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
        ),
    )
    results = tuner.fit()
