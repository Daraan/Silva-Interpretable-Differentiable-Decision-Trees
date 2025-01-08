from __future__ import annotations

import os
import gymnasium as gym
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import torch

import tempfile
from interpretable_ddts.agents.ddt_catalog import DDTCatalog
from interpretable_ddts.agents.ddt_ppo_module import DDTModule
from interpretable_ddts.agents.ppo_learner import SilvaLearner
from interpretable_ddts.agents.rllib_port.discrete_evaluation import eval_with_discrete
from interpretable_ddts.tools import is_pbar

import ray
from ray import train
from ray.experimental import tqdm_ray
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN, EVALUATION_RESULTS


import math
from typing import Any, Optional, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

_ConfigType = TypeVar("_ConfigType", bound=PPOConfig)

# Keys
TRAIN_METRIC_RETURN_MEAN = ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
DISC_EVAL_METRIC_RETURN_MEAN = EVALUATION_RESULTS + "/discrete/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
EVAL_METRIC_RETURN_MEAN = EVALUATION_RESULTS + "/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN


def create_ddt_config(
    args: argparse.Namespace,
    env_type: Optional[str | gym.Env] = None,
    *,
    config_class: type[_ConfigType] = PPOConfig,
) -> tuple[_ConfigType, RLModuleSpec]:
    """
    Args:
        legacy: Use the legacy code based on `gym_runner.py` and not an algorithm class.
    """
    if not env_type and not args.env_type:
        raise ValueError("No environment specified")
    env_type = env_type or args.env_type
    config = config_class()
    config.environment(env_type)
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
    init_env = gym.make(config.env)  # type: ignore[arg-type]
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
        # NOTE: Policy gradient algorithms are able to find the optimal
        # policy, even if this is a stochastic one. Setting "explore=False" here
        # results in the evaluation workers not using this optimal policy!
        evaluation_config=AlgorithmConfig.overrides(
            explore=False,
        ),
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
    if args.legacy:
        from interpretable_ddts.agents.ddt_ppo_module import LegacyDDTModule

        module_spec.module_class = LegacyDDTModule
        module_spec.model_config.update(  # type: ignore
            {
                "save_output": not args.test,
                "use_gpu": args.gpu,
                "vf_double_output": True,
                "action_use_softmax": True,
                "use_silva_loss": True,
            },
        )
        config.evaluation(custom_evaluation_function=None)
        assert config.rl_module_spec.module_class == LegacyDDTModule  # type: ignore
        assert config.rl_module_spec.model_config["vf_double_output"]  # type: ignore
        assert config.rl_module_spec.model_config["action_use_softmax"]  # type: ignore
        assert config.rl_module_spec.model_config["use_silva_loss"]  # type: ignore
        assert config.custom_evaluation_function is None

    return config, module_spec


def build_and_train(hparams: dict[str, Any], *, use_pbar=True, disable_report=False):
    """
    Args:
        hparams: The hyperparameters selected for the trial from the search space from ray tune.
            Should include an `args` key with the parsed arguments.

    Attention:
        Best practice is to not refer to any objects from outer scope in the training_function
    """
    args = hparams["args"]
    config, _ = create_ddt_config(args)
    algo = config.build()

    if use_pbar:
        pbar = tqdm_ray.tqdm(range(args.episodes), position=hparams.get("process_number", None))
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
