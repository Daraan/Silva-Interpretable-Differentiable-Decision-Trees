# Created by Andrew Silva on 8/28/19
import torch
from torch.distributions import Categorical
from ._agent_interface import AgentBase
from interpretable_ddts.agents.ddt import DDT
from interpretable_ddts.opt_helpers import replay_buffer, ppo_update
import os
import numpy as np
from typing import Optional, Union
from pathlib import Path


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
        *, _duplicate=False,
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
        super().__init__(input_dim, output_dim, _duplicate=_duplicate)
        
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.action_network = DDT(input_dim=input_dim,
                                  output_dim=output_dim,
                                  weights=init_weights,
                                  comparators=init_comparators,
                                  leaves=init_leaves,
                                  alpha=1,
                                  is_value=False,
                                  use_gpu=False)
        self.value_network = DDT(input_dim=input_dim,
                                 output_dim=output_dim,
                                 weights=init_weights,
                                 comparators=init_comparators,
                                 leaves=init_leaves,
                                 alpha=1,
                                 is_value=True,
                                 use_gpu=False)

        self.ppo = ppo_update.PPO([self.action_network, self.value_network], two_nets=True, use_gpu=False)

        self.last_state = [0, 0, 0, 0]
        self.last_action = 0
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
        self.replay_buffer.insert(obs=[self.last_state],
                                  action_log_probs=self.last_action_probs,
                                  value_preds=self.last_value_pred[self.last_action.item()],
                                  deeper_action_log_probs=self.last_deep_action_probs,
                                  deeper_value_pred=self.last_deep_value_pred[self.last_action.item()],
                                  last_action=self.last_action,
                                  full_probs_vector=self.full_probs,
                                  deeper_full_probs_vector=self.deeper_full_probs,
                                  rewards=reward)
        return True

    def save(self, fn: Union[Path, str]='last'):
        assert self.version is not None
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

    def __getstate__(self):
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

    def __setstate__(self, state):
        for key in state:
            setattr(self, key, state[key])

    def _write_hparams(self):
        self.rewards_file.open("w").write(", ".join([
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
                             _duplicate=True
                             )
        new_agent.__setstate__(self.__getstate__())
        return new_agent
