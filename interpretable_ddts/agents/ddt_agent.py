# Created by Andrew Silva on 8/28/19
from __future__ import annotations

import gymnasium
from ray.rllib.algorithms.ppo import PPO, PPOConfig, PPOTorchPolicy
from ray.rllib.models.modelv2 import ModelConfigDict
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.replay_buffers import ReplayBuffer, StorageUnit
import torch
from torch import nn
from ._agent_interface import AgentBase
from interpretable_ddts.agents.ddt import DDT
from interpretable_ddts.opt_helpers import replay_buffer, ppo_update
import os
import numpy as np
from typing import Literal, Optional, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import gymnasium.envs.registration

def save_ddt(fn, model):
    checkpoint = dict()
    mdl_data = dict()
    mdl_data['weights'] = model.layers
    mdl_data['comparators'] = model.comparators
    mdl_data['leaf_init_information'] = model.leaf_init_information
    mdl_data['action_probs'] = model.action_probs
    mdl_data['alpha'] = model.alpha
    mdl_data['input_dim'] = model.input_dim
    mdl_data['is_value'] = model.is_value
    checkpoint['model_data'] = mdl_data
    torch.save(checkpoint, fn)


def load_ddt(fn):
    model_checkpoint = torch.load(fn, map_location='cpu')
    model_data = model_checkpoint['model_data']
    init_weights = [weight.detach().clone().data.cpu().numpy() for weight in model_data['weights']]
    init_comparators = [comp.detach().clone().data.cpu().numpy() for comp in model_data['comparators']]

    new_model = DDT(input_dim=model_data['input_dim'],
                    weights=init_weights,
                    comparators=init_comparators,
                    leaves=model_data['leaf_init_information'],
                    alpha=model_data['alpha'].item(),
                    is_value=model_data['is_value'])
    # NOTE: what about output dim?
    new_model.action_probs = model_data['action_probs']
    return new_model


def init_rule_list(num_rules, dim_in, dim_out):
    weights = np.random.rand(num_rules, dim_in)
    leaves = []
    comparators = np.random.rand(num_rules, 1)
    for leaf_index in range(num_rules):
        leaves.append([[leaf_index], np.arange(0, leaf_index).tolist(), np.random.rand(dim_out)])
    leaves.append([[], np.arange(0, num_rules).tolist(), np.random.rand(dim_out)])
    return weights, comparators, leaves


class DDTAgent(AgentBase):
    def __init__(
        self,
        bot_name="DDT",
        input_dim=4,
        output_dim=2,
        rule_list=False,
        num_rules=4,
        version: Optional[int] = None,
        *,
        use_gpu=False,
        save_output=True,
        _duplicate=False,
    ):
        # bot_name before calling super
        self.bot_name = bot_name + '_'
        self.rule_list = rule_list
        self.num_rules = num_rules
        if rule_list:
            if str(num_rules) + '_rules' not in self.bot_name:
                self.bot_name += str(num_rules)+'_rules'
            init_weights, init_comparators, init_leaves = init_rule_list(num_rules, input_dim, output_dim)
        else:
            init_weights = None
            init_comparators = None
            init_leaves = num_rules
            if str(num_rules) + '_leaves' not in self.bot_name:
                self.bot_name += str(num_rules) + '_leaves'
        AgentBase.__init__(
            self,
            input_dim,
            output_dim,
            version=version,
            save_output=save_output,
            _duplicate=_duplicate,
        )

        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.action_network = DDT(
            input_dim=input_dim,
            output_dim=output_dim,
            weights=init_weights,
            comparators=init_comparators,
            leaves=init_leaves,
            alpha=1,
            is_value=False,
            use_gpu=use_gpu,
        )
        self.value_network = DDT(
            input_dim=input_dim,
            output_dim=output_dim,
            weights=init_weights,
            comparators=init_comparators,
            leaves=init_leaves,
            alpha=1,
            is_value=True,
            use_gpu=use_gpu,
        )

        self.ppo = ppo_update.PPO([self.action_network, self.value_network], two_nets=True, use_gpu=False)

        self.last_state = [0, 0, 0, 0]
        self.last_action: torch.IntTensor = torch.IntTensor([0])
        self.last_action_probs = torch.Tensor([0])
        self.last_value_pred = torch.Tensor([[0, 0]])
        self.last_deep_action_probs = None
        self.last_deep_value_pred = [None]*output_dim
        self.full_probs = None
        self.reward_history = []
        self.num_steps = 0
        self.deeper_full_probs = None

    def get_action(self, observation, max_inputs=10):
        return super().get_action(observation, max_inputs)

    def save_reward(self, reward):
        self.replay_buffer.insert(
            obs=[self.last_state],
            action_log_probs=self.last_action_probs,
            value_preds=self.last_value_pred[self.last_action.item()],
            deeper_action_log_probs=self.last_deep_action_probs,
            deeper_value_pred=self.last_deep_value_pred[self.last_action.item()],
            last_action=self.last_action.item(),
            full_probs_vector=self.full_probs,
            deeper_full_probs_vector=self.deeper_full_probs,
            rewards=reward,
        )
        return True

    def save(self, fn: Union[Path, str]='last', *, force_save: bool=False):
        """
        force_save: Still saves the output even in `save_output` is False
        """        
        assert self.version is not None
        if not (self.save_output or force_save):
            return
        act_fn = str(fn) + self.bot_name + '_actor' + f'_v{self.version}.pth.tar'
        val_fn = str(fn) + self.bot_name + "_critic" + f"_v{self.version}.pth.tar"

        save_ddt(act_fn, self.action_network)
        save_ddt(val_fn, self.value_network)

    def load(self, fn='last', version=None):
        assert version
        act_fn = str(fn) + self.bot_name + '_actor' + f'_v{version}.pth.tar'
        val_fn = str(fn) + self.bot_name + '_critic' + f'_v{version}.pth.tar'

        if os.path.exists(act_fn):
            self.action_network = load_ddt(act_fn)
            self.value_network = load_ddt(val_fn)
        else:
            msg = f"No such file or directory:' {act_fn}'"
            raise FileNotFoundError(msg)

    def __getstate__X(self):
        return {
            'action_network': self.action_network,
            'value_network': self.value_network,
            'ppo': self.ppo,
            'bot_name': self.bot_name,
            'rule_list': self.rule_list,
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'num_rules': self.num_rules
        }

    def __setstate__X(self, state):
        for key in state:
            setattr(self, key, state[key])

    @AgentBase.skip_if_no_output
    def _write_hparams(self):
        if self.save_output:
            self.rewards_file.open("w").write(", ".join([  # type: ignore[attribute]
                f"name: {self.bot_name}",
                "method: ddt",
                f"version: {self.version}",
                f"input_dim: {self.input_dim}",
                f"output_dim: {self.output_dim}",
                f"num_rules: {self.num_rules}",
                f"rule_list: {self.rule_list}",
            ]) + "\n")

    def duplicate(self):
        new_agent = DDTAgent(bot_name=self.bot_name.rstrip('_'),
                             input_dim=self.input_dim,
                             output_dim=self.output_dim,
                             rule_list=self.rule_list,
                             num_rules=self.num_rules,
                             version=self.version,
                             save_output=self.save_output,
                             _duplicate=True,
                             use_gpu=False  # <-----
                             )
        new_agent.__setstate__X(self.__getstate__X())
        return new_agent


class PPOConfigDDT(PPOConfig):
    custom_model: str = "DDT"


    def __post_init__(self):
        super().__post_init__()
        self["model"] = {"custom_model": self.custom_model}


class RLlibDDT(DDTAgent, TorchModelV2, nn.Module):
    # Note that this class by itself is not a valid model unless you inherit from nn.Module and implement forward() in a subclass.

    # ModelV2
    obs_space: gymnasium.spaces.Space
    action_space: gymnasium.spaces.Space
    num_outputs: int
    model_config: ModelConfigDict
    name: str | Literal["default_model"]

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
    ):
        nn.Module.__init__(self)
        DDTAgent.__init__(self, **model_config)
        delattr(self, "replay_buffer")
        self.replay_buffer = ReplayBuffer(capacity=1, storage_unit=StorageUnit.EPISODES)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

    def forward(
        self,
        input_dict: dict[str, TensorType],
        state: list,
        seq_lens: TensorType,
    ) -> tuple[TensorType, list[TensorType]]:
        probs = self.action_network(input_dict["obs"])
        value_pred = self.value_network(input_dict["obs"])
        # check https://docs.ray.io/en/releases-2.10.0/rllib/rllib-models.html#custom-pytorch-models
        #probs = probs.view(-1).cpu()
        #probs = probs.squeeze(0).cpu()
        #self.last_value_pred = value_pred.squeeze(0)[input_dict["actions"]]
        # Note: value pred is softmax over actions
        self.last_value_pred = value_pred.sum(axis=1)
        self.full_probs = probs
        if self.action_network.input_dim > 30:
            top_probs, inds = torch.topk(probs, 3)
            return top_probs, []
        return probs, []
        #return super().forward(input_dict, state, seq_lens)

    def value_function(self):
        """ "
        Returns the value function output for the most recent forward pass.

        Returns:
            Value estimate tensor of shape [BATCH].
        """
        # Note: afterwards the batch, dim is removed.
        #self.model.value_function()[0].item()
        #print(self.last_value_pred)
        return self.last_value_pred

class SilvaPPO(PPO):
    def get_default_policy_class(self, config):
        return SilvaPPOPolicy

class SilvaPPOPolicy(PPOTorchPolicy):
    @staticmethod
    def policy_compute_actions(policy,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        with torch.no_grad():
            obs = torch.Tensor(observation)
            obs = obs.view(1, -1)
            policy.last_state = obs

            probs = policy.action_network(obs)
            value_pred = policy.value_network(obs)
            probs_v = probs.view(-1).cpu()  # not equivalent to squeeze if multiple ops
            probs_s = probs.squeeze(0).cpu()  # this flattens the array
            assert probs_v.shape == probs_s.shape
            probs = probs_s
            
            policy.full_probs = probs
            if policy.action_network.input_dim > max_inputs:
                probs, inds = torch.topk(probs, 3)
            m = Categorical(probs)
            action = m.sample()
            log_probs = m.log_prob(action)
            policy.last_action_probs = log_probs.cpu()
            policy.last_value_pred = value_pred.view(-1).cpu()

            if policy.action_network.input_dim > max_inputs:
                policy.last_action = inds[action].cpu()
            else:
                policy.last_action = action.cpu()
        if policy.action_network.input_dim > max_inputs:
            action = inds[action].item()
        else:
            action = action.item()
        return action

    @staticmethod
    def policy_action_sampler_fn(
                policy: PPOTorchPolicy,
                model:RLlibDDT,
                obs_batch: dict[str, TensorType],
                state_batches: Optional[list[TensorType]],
                explore: Optional[bool],
                timestep: Optional[int],
        
            ):
        with torch.no_grad():
            ... # TODO # XXX
        
        return actions, logp, dist_inputs, state_out