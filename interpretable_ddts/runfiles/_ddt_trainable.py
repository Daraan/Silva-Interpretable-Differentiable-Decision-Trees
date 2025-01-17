from __future__ import annotations

import argparse
import logging
import math
import os
import tempfile
from typing import TYPE_CHECKING, Any, Optional, TypeVar

import gymnasium as gym
import ray
import torch
from ray import train
from ray.air.integrations.comet import CometLoggerCallback
from ray.experimental import tqdm_ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks, make_multi_callbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    EVALUATION_RESULTS,
)
from ray.tune import logger as tune_logger

from interpretable_ddts.agents.ddt_catalog import DDTCatalog
from interpretable_ddts.agents.ddt_ppo_module import DDTModule
from interpretable_ddts.agents.ppo_learner import SilvaLearner
from interpretable_ddts.runfiles._pbar_updates import update_pbar
from interpretable_ddts.runfiles.constants import (
    DISC_EVAL_METRIC_RETURN_MEAN,
    DISCRETE_EVALUATION_BEST_VIDEO,
    DISCRETE_EVALUATION_WORST_VIDEO,
    EVAL_METRIC_RETURN_MEAN,
    EVALUATION_BEST_VIDEO,
    EVALUATION_WORST_VIDEO,
    TRAIN_METRIC_RETURN_MEAN,
)
from ray_utilities import is_pbar
from ray_utilities.callbacks.algorithm.discrete_eval_callback import DiscreteEvalCallback
from ray_utilities.callbacks.algorithm.env_render_callback import make_render_callback

logger = logging.getLogger(__name__)

_ConfigType = TypeVar("_ConfigType", bound=PPOConfig)


def create_ddt_config(
    args: dict[str, Any] | argparse.Namespace,
    env_type: Optional[str | type[gym.Env]] = None,
    *,
    config_class: type[_ConfigType] = PPOConfig,
) -> tuple[_ConfigType, RLModuleSpec]:
    """
    Args:
        legacy: Use the legacy code based on `gym_runner.py` and not an algorithm class.
    """
    if isinstance(args, argparse.Namespace):
        args = vars(args).copy()
    if not env_type and not args["env_type"]:
        raise ValueError("No environment specified")
    env_type = env_type or args["env_type"]
    assert env_type, "No environment specified"
    config = config_class()
    if args["render_mode"]:
        env_config = {"render_mode": args["render_mode"]}
        config.environment(env_type, env_config=env_config)
    else:
        config.environment(env_type)
    config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    config.resources(
        # num_gpus=1 if args["gpu"] else 0,4
        # process that runs Algorithm.training_step() during Tune
        num_cpus_for_main_process=1,
        # num_learner_workers=4 if args["parallel"] else 1,
        # num_cpus_per_learner_worker=1,
        # num_cpus_per_worker=1,
    )
    config.env_runners(
        num_env_runners=2 if args["parallel"] else 0,
        num_cpus_per_env_runner=1,  # num_cpus_per_worker
        # How long an rollout episode lasts, for "auto" calculated from batch_size
        # total_train_batch_size / (num_envs_per_env_runner * num_env_runners)
        # rollout_fragment_length=1,  # Default: "auto"
        num_envs_per_env_runner=1,
        # validate_env_runners_after_construction=args["test"],
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
        num_learners=0 if args["parallel"] else 0,
        num_cpus_per_learner=1,
        num_gpus_per_learner=1 if args["gpu"] else 0,
    )

    config.framework("torch")
    config.training(
        learner_class=SilvaLearner,
        learner_config_dict={"use_silva_loss": args["use_silva_loss"]},
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
    if isinstance(config.env, str):
        init_env = gym.make(config.env)
    else:
        assert not TYPE_CHECKING or config.env
        init_env = gym.make(config.env.unwrapped.spec.id)  # pyright: ignore[reportOptionalMemberAccess]
    module_spec = RLModuleSpec(
        module_class=DDTModule,
        observation_space=init_env.observation_space,
        action_space=init_env.action_space,
        model_config={
            "bot_name": args["agent_type"] + args["env_type"],
            "rule_list": args["rule_list"],
            "num_rules": args["num_leaves"],
            "use_gpu": args["gpu"],
            "vf_double_output": args["use_silva_loss"],
            "action_use_softmax": args["use_silva_loss"],
            "use_silva_loss": args["use_silva_loss"],  # unused by model config
        },
        catalog_class=DDTCatalog,
    )
    # module = module_spec.build()
    config.rl_module(
        rl_module_spec=module_spec,
    )
    # https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.evaluation.html
    config.evaluation(
        evaluation_interval=10,
        evaluation_duration=5,
        evaluation_duration_unit="episodes",
        evaluation_num_env_runners=2 if args["parallel"] else 0,
        # NOTE: Policy gradient algorithms are able to find the optimal
        # policy, even if this is a stochastic one. Setting "explore=False" here
        # results in the evaluation workers not using this optimal policy!
        evaluation_config=PPOConfig.overrides(
            explore=False,
        ),
    )
    callbacks: list[type[DefaultCallbacks]] = [DiscreteEvalCallback]
    if args["render_mode"]:
        callbacks.append(make_render_callback())

    if callbacks:
        if len(callbacks) == 1:
            callback = callbacks[0]
        else:
            callback = make_multi_callbacks(callbacks)
            # Necessary patch for new_api, cannot use this callback with new API
            callback.on_episode_created = DefaultCallbacks.on_episode_created
        config.callbacks(callbacks_class=callback)

    config.reporting(
        keep_per_episode_custom_metrics=True,  # If True calculate max min mean
        log_gradients=False,  # Default is True
        # Will smooth metrics in the reports, e.g. tensorboard
        metrics_num_episodes_for_smoothing=1,  # Default is 100
    )
    config.debugging(
        # https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.debugging.html#ray-rllib-algorithms-algorithm-config-algorithmconfig-debugging
        seed=args["seed"],
        log_sys_usage=False,
        # These loggers will log more metrics which are stored less-accessible in the ~/ray_results/logdir
        # Using these could be useful if no Tuner is used
        logger_config={"type": tune_logger.NoopLogger},
    )
    # Checks
    config.validate_train_batch_size_vs_rollout_fragment_length()
    assert (
        config.rl_module_spec.model_config["vf_double_output"]  # type: ignore
        == config.learner_config_dict["use_silva_loss"]
    )
    if args["legacy"]:
        from interpretable_ddts.agents.ddt_ppo_module import LegacyDDTModule

        module_spec.module_class = LegacyDDTModule
        module_spec.model_config.update(  # type: ignore
            {
                "save_output": False,  # TODO: Add checkpoint for legacy
                "use_gpu": args["gpu"],
                "vf_double_output": True,
                "action_use_softmax": True,
                "use_silva_loss": True,
            },
        )
        config.evaluation(custom_evaluation_function=None)

    return config, module_spec


def build_and_train(hparams: dict[str, Any], *, use_pbar=True, disable_report=False):
    """
    Args:
        hparams: The hyperparameters selected for the trial from the search space from ray tune.
            Should include an `args` key with the parsed arguments.

    Attention:
        Best practice is to not refer to any objects from outer scope in the training_function
    """
    args: dict = hparams["cli_args"]
    # TODO: this should use the parameters from the search space
    config, _ = create_ddt_config(args)
    algo = config.build()

    if use_pbar:
        pbar = tqdm_ray.tqdm(range(args["episodes"]), position=hparams.get("process_number", None))
    else:
        pbar = range(args["episodes"])
    running_eval_rewards = []
    running_disc_eval_rewards = []
    running_rewards = []
    result = {}
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
        elif running_disc_eval_rewards:  # TEMP: As long as metrics logger for discrete evaluation is not shared
            disc_eval_mean = running_disc_eval_rewards[-1]
        disc_running_eval_reward = sum(running_disc_eval_rewards[-100:]) / (
            min(100, len(running_disc_eval_rewards)) or float("nan")  # nan for 0
        )

        # NOTE: The csv logger will only log keys that are present in the first result,
        #       i.e. the videos will not be logged if they are added later; but everytime otherwise!
        metrics = {
            TRAIN_METRIC_RETURN_MEAN: result[ENV_RUNNER_RESULTS].get(
                EPISODE_RETURN_MEAN,
                float("nan"),
            ),
            EVAL_METRIC_RETURN_MEAN: eval_mean,
            DISC_EVAL_METRIC_RETURN_MEAN: disc_eval_mean,
        }
        if EVALUATION_RESULTS in result:
            # Store videos
            if evaluation_videos_best := result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get(
                "episode_videos_best",
            ):
                metrics[EVALUATION_BEST_VIDEO] = {
                    "video": evaluation_videos_best,
                    "reward": result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MAX],
                }
            if evaluation_videos_worst := result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get(
                "episode_videos_worst",
            ):
                metrics[EVALUATION_WORST_VIDEO] = {
                    "video": evaluation_videos_worst,
                    "reward": result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MIN],
                }
            if discrete_evaluation_videos_best := result[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS].get(
                "episode_videos_best"
            ):
                metrics[DISCRETE_EVALUATION_BEST_VIDEO] = {
                    "video": discrete_evaluation_videos_best,
                    "reward": result[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS][EPISODE_RETURN_MAX],
                }
            if discrete_evaluation_videos_worst := result[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS].get(
                "episode_videos_worst"
            ):
                metrics[DISCRETE_EVALUATION_WORST_VIDEO] = {
                    "video": discrete_evaluation_videos_worst,
                    "reward": result[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS][EPISODE_RETURN_MIN],
                }
            # Check for NaN values, if they are not the evaluation metrics warn.
            if any(isinstance(value, float) and math.isnan(value) for value in metrics.values()):
                logger.warning("NaN values in metrics: %s", metrics)

        # Checkpoint & metrics
        if False and not disable_report and ray.train.get_context().get_world_rank() == 0:
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": _episode, "model_state": algo.get_module().state_dict()},
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                train.report(metrics=metrics, checkpoint=train.Checkpoint.from_directory(tempdir))
        # Report metrics
        elif not disable_report:
            train.report(metrics, checkpoint=None)

        # Update progress bar
        if not is_pbar(pbar):
            continue
        update_pbar(
            pbar,
            train_results={
                "mean": metrics[TRAIN_METRIC_RETURN_MEAN],
                "max": result["env_runners"].get("episode_return_max", float("nan")),
                "roll": running_reward,
            },
            eval_results={
                "mean": eval_mean,
                "roll": running_eval_reward,
            },
            discrete_eval_results={
                "mean": disc_eval_mean,
                "roll": disc_running_eval_reward,
            },
        )
    if EVALUATION_RESULTS not in result:
        result[EVALUATION_RESULTS] = algo.evaluate()
    result["done"] = True
    if args.get("comment"):
        result["comment"] = args["comment"]
    # Postprocess results and return
    try:
        reduced_results = reduce_results(
            result,
            extra_keys_to_keep=[
                # Should log as video! not array
                # (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, "episode_videos_best"),
                # (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, "episode_videos_worst"),
                # (EVALUATION_RESULTS, "discrete", ENV_RUNNER_RESULTS, "episode_videos_best"),
                # (EVALUATION_RESULTS, "discrete", ENV_RUNNER_RESULTS, "episode_videos_worst"),
            ],
        )  # if not args["test"] else [(LEARNER_RESULTS,)])
    except Exception:
        logger.exception("Failed to reduce results")
        return result
    else:
        return reduced_results


# NOTE: This should not overlap!
RESULTS_TO_KEEP = {
    (ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN),
    # (ENV_RUNNER_RESULTS, NUM_EPISODES),
    (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN),
    (EVALUATION_RESULTS, "discrete", ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN),
    ("comment",),
}
RESULTS_TO_KEEP.update((key,) for key in CometLoggerCallback._other_results)
RESULTS_TO_KEEP.update((key,) for key in CometLoggerCallback._system_results)
RESULTS_TO_KEEP.update((key,) for key in CometLoggerCallback._exclude_results)
assert all(isinstance(key, (tuple, list)) for key in RESULTS_TO_KEEP)

_MISSING = object()


def _find_item(obj: dict[str, Any], keys: list[str]) -> Any:
    if len(keys) == 1:
        return obj.get(keys[0], _MISSING)
    value = obj.get(keys[0], _MISSING)
    if isinstance(value, dict):
        return _find_item(value, keys[1:])
    if value is not _MISSING and len(keys) > 0:
        raise TypeError(f"Expected dict at {keys[0]} but got {value}")
    return value


def reduce_results(
    results: dict[str, Any], extra_keys_to_keep: Optional[list[tuple[str, ...]]] = None
) -> dict[str, Any]:
    # from omegaconf import OmegaConf
    # res = OmegaConf.create(results, flags={"allow_objects": True})
    # return OmegaConf.to_container(OmegaConf.merge((OmegaConf.select(res, key) for key in RESULTS_TO_KEEP)))

    reduced: dict[str, Any] = {}
    _count = 0
    if extra_keys_to_keep:
        keys_to_keep = RESULTS_TO_KEEP.copy()
        keys_to_keep.update(extra_keys_to_keep)
    else:
        keys_to_keep = RESULTS_TO_KEEP

    for keys in keys_to_keep:
        value = _find_item(results, keys if not isinstance(keys, str) else [keys])
        if value is not _MISSING:
            sub_dir = reduced
            for key in keys[:-1]:
                sub_dir = sub_dir.setdefault(key, {})
            if keys[-1] in sub_dir:
                raise ValueError(f"Key {keys[-1]} already exists in {sub_dir}")
            sub_dir[keys[-1]] = value
            _count += 1
    if _count != len(RESULTS_TO_KEEP):
        logger.warning("Reduced results do not match the expected amount of keys: %s", reduced)
    return reduced
