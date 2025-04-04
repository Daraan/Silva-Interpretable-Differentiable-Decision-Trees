from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast

from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig

# pyright: enableExperimentalFeatures=true
from interpretable_ddts.rllib_port.ddt_catalog import DDTPPOCatalog
from interpretable_ddts.rllib_port.ddt_ppo_module import DDTModelConfigDict, DDTModule
from interpretable_ddts.rllib_port.ppo_learner import SilvaLearner
from ray_utilities.config.create_algorithm import create_algorithm_config

if TYPE_CHECKING:
    import gymnasium as gym
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    from interpretable_ddts.ddt_setup import DDTArgumentParser
    from ray_utilities.config.experiment_base import NamespaceType


_ConfigType = TypeVar("_ConfigType", bound=AlgorithmConfig)


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
        # Add dataclass
        else:
            args = vars(args).copy()
    model_config: DDTModelConfigDict = {
        "rule_list": args["rule_list"],
        "num_rules": args["num_leaves"],
        "use_gpu": args["gpu"],
        "vf_double_output": args["use_silva_loss"],
        "action_use_softmax": args["use_silva_loss"],
    }
    config, module_spec = create_algorithm_config(
        args,
        env_type,
        env_seed,
        config_class=config_class,
        model_config=model_config,
        module_class=DDTModule,
        catalog_class=DDTPPOCatalog,
        framework="torch",
    )
    config.training(
        learner_class=SilvaLearner,
        learner_config_dict={"use_silva_loss": args["use_silva_loss"]},
    )
    assert (
        config.rl_module_spec.model_config["vf_double_output"]  # type: ignore
        == config.learner_config_dict["use_silva_loss"]
    )
    if args["legacy"]:
        from interpretable_ddts.rllib_port.ddt_ppo_module import LegacyDDTModule  # noqa: PLC0415

        module_spec.module_class = LegacyDDTModule
        model_config: DDTModelConfigDict = module_spec.model_config  # type: ignore[assignment]
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
        config.evaluation(
            custom_evaluation_function=None,
            evaluation_num_env_runners=1 if args["parallel"] else 0,  # NOTE: Parallel evaluation not implemented
        )
    return config, module_spec
