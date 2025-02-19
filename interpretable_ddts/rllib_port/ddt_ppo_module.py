from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

import numpy as np
import ray.train

# from ray.rllib import SampleBatch  # input for model
try:
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
except ImportError:
    # Older API
    from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule as DefaultPPOTorchRLModule  # pyright: ignore[reportPrivateImportUsage]
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.deprecation import logger as _deprecation_logger
from typing_extensions import NotRequired, Self

from interpretable_ddts.agents._agent_interface import AgentBase
from interpretable_ddts.agents.ddt import DDT
from interpretable_ddts.opt_helpers import ppo_update
from interpretable_ddts.opt_helpers.replay_buffer import (
    ReplayBufferSingleAgent as SilvaReplayBuffer,
)
from ray_utilities.constants import (
    DISC_EVAL_METRIC_RETURN_MEAN,
    EVAL_METRIC_RETURN_MEAN,
)
from ray_utilities.typing.discrete_module import DiscreteModule

# This suppresses a deprecation warning from RLModuleConfig
__old_level = _deprecation_logger.getEffectiveLevel()
_deprecation_logger.setLevel(logging.ERROR)
RLModuleConfig()
_deprecation_logger.setLevel(__old_level)


if TYPE_CHECKING:
    import gymnasium as gym

    from interpretable_ddts.agents.ddt import LeafInfo

logger = logging.getLogger(__name__)


def init_rule_list(num_rules, dim_in, dim_out):
    weights = np.random.rand(num_rules, dim_in)
    leaves: list[LeafInfo] = []
    comparators = np.random.rand(num_rules, 1)
    leaves = [
        ([leaf_index], np.arange(0, leaf_index).tolist(), np.random.rand(dim_out).tolist())
        for leaf_index in range(num_rules)
    ]

    leaves.append(([], np.arange(0, num_rules).tolist(), np.random.rand(dim_out).tolist()))
    return weights, comparators, leaves


class ModelConfigDict(TypedDict):
    bot_name: NotRequired[str]
    num_rules: int
    rule_list: bool
    save_output: NotRequired[bool]
    use_gpu: bool
    action_use_softmax: bool
    vf_double_output: bool


class DDTModule(DiscreteModule, DefaultPPOTorchRLModule):
    observation_space: gym.Space
    action_space: gym.Space
    config: RLModuleConfig
    model_config: Optional[ModelConfigDict]

    CAN_USE_DISCRETE_EVAL = True

    bot_name: Optional[str] = None

    def __init__(
        self,
        config: RLModuleConfig = DEPRECATED_VALUE,  # type: ignore[arg-type]  # use -1 here to avoid errors
        *,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        inference_only: Optional[bool] = None,
        learner_only: bool = False,
        model_config: Optional[dict[str, Any] | ModelConfigDict] = None,
        catalog_class=None,
    ) -> None:
        if config and config != DEPRECATED_VALUE:  # type: ignore[comparison-overlap]
            super().__init__(
                config,
                observation_space=observation_space,
                action_space=action_space,
                inference_only=inference_only,
                learner_only=learner_only,
                model_config=model_config,
                catalog_class=catalog_class,
            )
        else:
            super().__init__(
                observation_space=observation_space,
                action_space=action_space,
                inference_only=inference_only,
                learner_only=learner_only,
                model_config=model_config,
                catalog_class=catalog_class,
            )

    def setup(self) -> None:
        # super().setup() # Might create more modules, e.g. encoder
        assert isinstance(self.model_config, dict)

        if "bot_name" in self.model_config:
            self.bot_name = self.model_config["bot_name"] + "_"
        num_rules: int = self.model_config["num_rules"]
        rule_list: bool = self.model_config["rule_list"]
        input_dim = self.observation_space.shape[0]  # type: ignore
        output_dim = int(self.action_space.n)  # type: ignore
        if rule_list:
            init_weights, init_comparators, init_leaves = init_rule_list(
                num_rules,
                input_dim,
                output_dim,
            )
        else:
            init_weights = None
            init_comparators = None
            init_leaves = num_rules

        # Use is_value=True to NOT apply the softmax and return logits
        self.pi = DDT(
            input_dim=input_dim,
            output_dim=output_dim,
            weights=init_weights,
            comparators=init_comparators,
            leaves=init_leaves,
            alpha=1,
            # For rllib should return logits
            # for Silva should return probs
            is_value=not self.model_config.get("action_use_softmax", False),
            use_gpu=self.model_config["use_gpu"],
        )
        if not self.inference_only:
            self.vf = DDT(
                input_dim=input_dim,
                output_dim=1 if not self.model_config["vf_double_output"] else 2,
                weights=init_weights,
                comparators=init_comparators,
                leaves=self.model_config["num_rules"],
                alpha=1,
                is_value=True,
                use_gpu=self.model_config["use_gpu"],
            )

        self.is_discrete = False
        self._max_inputs = 10
        self._setup_name()

    def _setup_name(self):
        assert isinstance(self.model_config, dict)

        if "bot_name" not in self.model_config:
            return
        self.bot_name = self.model_config["bot_name"] + "_"
        num_rules: int = self.model_config["num_rules"]
        rule_list: bool = self.model_config["rule_list"]
        if rule_list and (str(num_rules) + "_rules" not in self.bot_name):
            self.bot_name += str(num_rules) + "_rules"
        elif not rule_list and (str(num_rules) + "_leaves" not in self.bot_name):
            self.bot_name += str(num_rules) + "_leaves"

    def encoder(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        No encoder is used return inputs in an ActorCriticEncoder output form
        to be passed to the action(pi) and value(vf) networks
        """
        return {
            ENCODER_OUT: {
                ACTOR: inputs,
                # Add critic from value network
                **({} if self.config.inference_only else {CRITIC: inputs}),
            },
        }

    def switch_mode(self, *, discrete: bool):
        if discrete and not self.is_discrete:
            self.pi = self.pi.create_discrete_copy()
            self.pi.eval()
            if not self.inference_only:  # vf is not used in inference and missing in later ray versions
                self.vf = self.vf.create_discrete_copy()
                self.vf.eval()
            self.is_discrete = True
        elif not discrete and self.is_discrete:
            self.pi = self.pi
            if not self.inference_only:
                self.vf = self.vf
            self.is_discrete = False


class LegacyDDTModule(DDTModule, AgentBase):
    """
    Version of the DDTModule that is used by gym_runner.py
    it is compatible with the AgentBase interface
    """

    bot_name: str  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def action_network(self):  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.pi

    @property
    def value_network(self):  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.vf

    def save_reward(self, reward):
        self.replay_buffer.insert(
            obs=[self.last_state],
            action_log_probs=self.last_action_probs,
            value_preds=self.last_value_pred[self.last_action.item()],
            last_action=self.last_action.item(),
            full_probs_vector=self.full_probs,
            rewards=reward,
        )
        return True

    def setup(self, *, duplicate=False) -> None:
        super().setup()
        assert self.bot_name

        self.rewards_file = None
        self._version = None
        self._duplicate = duplicate
        self.save_output = getattr(self, "model_config", {}).get("save_output", False)
        self.replay_buffer = SilvaReplayBuffer()
        self.reward_history = []
        self._check_version()

        self.ppo = ppo_update.PPO([self.action_network, self.value_network], two_nets=True, use_gpu=False)
        self.num_steps = 0

    def _write_hparams(self):
        pass

    def save(self, path):
        pass

    @classmethod
    def duplicate_agent(cls, agent: Self, *, discrete: bool = False) -> Self:
        """
        Creates a **shallow** duplicate of the agent with identical(!) networks, however
        with a new replay buffer.

        Args:
            discrete: If True, the new agent will have a discrete copy of the networks.
        """
        # from copy import deepcopy
        # new_agent = deepcopy(self)
        new_agent = cls(
            observation_space=agent.observation_space,
            action_space=agent.action_space,
            inference_only=agent.inference_only,  # could possibly set this to False, value missing then?
            learner_only=False,
            model_config=agent.model_config,
            catalog_class=agent.catalog.__class__,
        )
        new_agent.setup(duplicate=True)  # this creates pi, vf and ppo; adjust afterwards!
        # NOTE: Networks are shared!
        if discrete:
            new_agent.pi = agent.pi.create_discrete_copy()
            new_agent.vf = (
                agent.vf.create_discrete_copy()
            )  # TODO: this should be based on pi!; check if really necessary
            new_agent.ppo = ppo_update.PPO([new_agent.pi, new_agent.vf], two_nets=True, use_gpu=False)
            new_agent.is_discrete = True
        else:
            new_agent.pi = agent.pi
            new_agent.vf = agent.vf
            new_agent.ppo = agent.ppo
        return new_agent

    def duplicate(self, *, discrete=False):
        return self.duplicate_agent(self, discrete=discrete)

    def end_episode(self, reward, *, discrete_reward: Optional[float] = None):
        if self._duplicate:
            logger.warning("Calling end_episode on a duplicate agent")
        loss = AgentBase.end_episode(self, reward)
        # This should only be used in gym_runner which does not use ray train; only one report per episode
        metrics = {
            EVAL_METRIC_RETURN_MEAN: reward,
        }
        if discrete_reward is not None:
            metrics[DISC_EVAL_METRIC_RETURN_MEAN] = discrete_reward
        ray.train.report(
            metrics,
            checkpoint=None,
        )
        return loss


if TYPE_CHECKING:  # Create ABC errors
    DDTModule()
    LegacyDDTModule()
