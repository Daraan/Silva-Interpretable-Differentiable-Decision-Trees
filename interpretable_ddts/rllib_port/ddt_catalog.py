from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, cast

from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.torch.base import TorchModel

from interpretable_ddts.agents.ddt import DDT
from interpretable_ddts.agents.ddt_agent import init_rule_list
from ray_utilities.dummy_encoder import DummyActorCriticEncoder, DummyActorCriticEncoderConfig
from ray_utilities.typing.discrete_module import DiscreteModelBase

if TYPE_CHECKING:
    import gymnasium as gym
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig  # noqa: F401

    from interpretable_ddts.agents.ddt_agent import LeafInfo
    from interpretable_ddts.rllib_port.ddt_ppo_module import ModelConfigDict


class DDTCatalog(Catalog):
    """Catalog class to create custom model and not a predefined one with unnecessary modules."""

    def _determine_components_hook(self) -> None:
        """Hook to determine the components of the model."""
        # We do not need an encoder; no not set _encoder_config, hence do not call super()

        assert not hasattr(self, "_encoder_config")

        # Create a function that can be called when framework is known to retrieve the
        # class type for action distributions
        self._action_dist_class_fn = functools.partial(
            self._get_dist_cls_from_action_space,
            action_space=self.action_space,
        )


class DDTModel(DDT, DiscreteModelBase, TorchModel):
    """Compatible with rays Model interface. Expect not having a config"""

    def __init__(
        self,
        *,
        input_dim: int,
        weights,
        comparators,
        leaves: int | list[LeafInfo],
        output_dim: int | None = None,
        alpha=1.0,
        is_value=False,
        use_gpu=False,
    ):
        # TorchModel.__init__ expects a config whch we do not have
        DDT.__init__(
            self,
            input_dim=input_dim,
            output_dim=output_dim,
            weights=weights,
            comparators=comparators,
            leaves=leaves,
            alpha=alpha,
            is_value=is_value,
            use_gpu=use_gpu,
        )


if TYPE_CHECKING:
    # Test ABC class
    DDTModel(input_dim=-1, weights=None, comparators=None, leaves=-1)

_NOT_SET: Any = object()


class DDTPPOCatalog(PPOCatalog):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: ModelConfigDict,
    ):
        """Initializes the PPOCatalog.

        Args:
            observation_space: The observation space of the Encoder.
            action_space: The action space for the Pi Head.
            model_config_dict: The model config to use.
        """
        # Skip PPOCatalog init
        super(PPOCatalog, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=cast("dict[str, Any]", model_config_dict),
        )
        """This is a dict union of these two"""

        # PPOCatalog code
        self.actor_critic_encoder_config = DummyActorCriticEncoderConfig()
        # TODO: Infer from num_rules
        self.pi_and_vf_head_hiddens = self._model_config_dict["head_fcnet_hiddens"]
        self.pi_and_vf_head_activation = self._model_config_dict["head_fcnet_activation"]

        self.pi_head_config = None
        self.vf_head_config = None

        # self._model_config_dict is merged with DefaultModelConfig
        # if TYPE_CHECKING:
        #    self._model_config_dict = model_config_dict

        self._init_weights = _NOT_SET
        self._init_comparators = _NOT_SET
        self._init_leaves = _NOT_SET

    def _init_shared(self):
        if self._init_weights is not _NOT_SET:
            return
        rule_list: bool = self._model_config_dict["rule_list"]
        input_dim = self.observation_space.shape[0]  # type: ignore
        output_dim = int(self.action_space.n)  # type: ignore
        num_rules = self._model_config_dict["num_rules"]
        if rule_list:
            self._init_weight, self._init_comparators, self._init_leaves = init_rule_list(
                num_rules,
                input_dim,
                output_dim,
            )
        else:
            self._init_weights = None
            self._init_comparators = None
            self._init_leaves = num_rules

    def build_actor_critic_encoder(self, framework: str) -> DummyActorCriticEncoder:
        return self.actor_critic_encoder_config.build(framework)

    def build_pi_head(self, framework: str):
        assert framework == "torch"
        self._init_shared()
        return DDTModel(
            input_dim=self.observation_space.shape[0],  # type: ignore
            output_dim=int(self.action_space.n),  # type: ignore
            weights=self._init_weights,
            comparators=self._init_comparators,
            leaves=self._init_leaves,
            alpha=1,
            # For rllib should return logits
            # for Silva should return probs
            is_value=not self._model_config_dict.get("action_use_softmax", False),
            use_gpu=self._model_config_dict["use_gpu"],
        )

    def build_vf_head(self, framework: str):
        assert framework == "torch"
        self._init_shared()
        return DDTModel(
            input_dim=self.observation_space.shape[0],  # type: ignore
            output_dim=1 if not self._model_config_dict["vf_double_output"] else 2,
            weights=self._init_weights,
            comparators=self._init_comparators,
            leaves=self._model_config_dict["num_rules"],
            alpha=1,
            is_value=True,
            use_gpu=self._model_config_dict["use_gpu"],
        )
