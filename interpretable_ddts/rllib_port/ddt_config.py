from __future__ import annotations
# pyright: enableExperimentalFeatures=true

from interpretable_ddts.rllib_port.ddt_catalog import DDTCatalog
from interpretable_ddts.rllib_port.ddt_ppo_module import DDTModule, ModelConfigDict
from interpretable_ddts.rllib_port.ppo_learner import SilvaLearner
from ray_utilities.callbacks.algorithm.discrete_eval_callback import DiscreteEvalCallback
from ray_utilities.callbacks.algorithm.env_render_callback import make_render_callback
from ray_utilities.constants import RAY_NEW_API_STACK_ENABLED


import gymnasium as gym
from ray.rllib.algorithms.callbacks import DefaultCallbacks, make_multi_callbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune import logger as tune_logger

from typing import TYPE_CHECKING, Any, Final, Optional, TypeVar, cast

if TYPE_CHECKING:
    from interpretable_ddts.ddt_setup import DDTArgumentParser
    from ray_utilities.config.experiment_base import NamespaceType


_ConfigType = TypeVar("_ConfigType", bound=PPOConfig)


def create_ddt_config(
    args: dict[str, Any] | NamespaceType[DDTArgumentParser],
    env_type: Optional[str | gym.Env] = None,
    env_seed: Optional[int] = None,
    *,
    config_class: type[_ConfigType] = PPOConfig,
) -> tuple[_ConfigType, RLModuleSpec]:
    """
    Args:
        legacy: Use the legacy code based on `gym_runner.py` and not an algorithm class.
    """
    if not isinstance(args, dict):
        if hasattr(args, "as_dict"):  # Tap
            args = cast("dict[str, Any]", args.as_dict())
        else:
            args = vars(args).copy()
    if not env_type and not args["env_type"]:
        raise ValueError("No environment specified")
    env_spec: Final = env_type or args["env_type"]
    del env_type
    assert env_spec, "No environment specified"
    config = config_class()

    env_config = {}
    if args["render_mode"]:
        env_config["render_mode"] = args["render_mode"]
    if env_seed is not None:
        env_config.update({"seed": env_seed, "env_type": env_spec})
        config.environment("seeded_env", env_config=env_config)
    elif env_config:
        config.environment(env_spec, env_config=env_config)
    else:
        config.environment(env_spec)
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
        num_cpus_per_learner=0 if args["test"] else 1,
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
    # NOTE: This might needs adjustment when using VectorEnv
    if isinstance(config.env, str) and config.env != "seeded_env":
        init_env = gym.make(config.env)
    elif config.env == "seeded_env":
        if isinstance(env_spec, str):
            init_env = gym.make(env_spec)
        else:
            init_env = env_spec
    else:
        assert not TYPE_CHECKING or config.env
        init_env = gym.make(config.env.unwrapped.spec.id)  # pyright: ignore[reportOptionalMemberAccess]
    # Note: legacy keys are updated below
    model_config: ModelConfigDict = {
        "rule_list": args["rule_list"],
        "num_rules": args["num_leaves"],
        "use_gpu": args["gpu"],
        "vf_double_output": args["use_silva_loss"],
        "action_use_softmax": args["use_silva_loss"],
    }
    module_spec = RLModuleSpec(
        module_class=DDTModule,
        observation_space=init_env.observation_space,
        action_space=init_env.action_space,
        model_config=cast("dict[str, Any]", model_config),
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
    if not RAY_NEW_API_STACK_ENABLED:
        # by default enabled from ray 2.40.0; should be called after .exploration
        config.api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
    # Checks
    config.validate_train_batch_size_vs_rollout_fragment_length()
    assert (
        config.rl_module_spec.model_config["vf_double_output"]  # type: ignore
        == config.learner_config_dict["use_silva_loss"]
    )
    if args["legacy"]:
        from interpretable_ddts.rllib_port.ddt_ppo_module import LegacyDDTModule  # noqa: PLC0415

        module_spec.module_class = LegacyDDTModule
        model_config: ModelConfigDict = module_spec.model_config  # type: ignore[assignment]
        model_config.update(  # type: ignore
            {
                "bot_name": args["agent_type"] + args["env_type"],
                "save_output": False,  # TODO: Add checkpoint for legacy
                "use_gpu": args["gpu"],
                "vf_double_output": True,
                "action_use_softmax": True,
                # "use_silva_loss": True,
            },
        )
        config.evaluation(custom_evaluation_function=None)

    return config, module_spec
