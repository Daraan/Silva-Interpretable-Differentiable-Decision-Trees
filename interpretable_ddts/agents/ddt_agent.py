# Created by Andrew Silva on 8/28/19
from __future__ import annotations

import torch
from ._agent_interface import AgentBase
from interpretable_ddts.agents.ddt import DDT
from interpretable_ddts.opt_helpers import replay_buffer, ppo_update
import os
import numpy as np
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from pathlib import Path
    LeafInfo = tuple[list[int], list[int], list[float] | float]


def save_ddt(fn, model):
    checkpoint = {}
    mdl_data = {
        'weights': model.layers,
        'comparators': model.comparators,
        'leaf_init_information': model.leaf_init_information,
        'action_probs': model.action_probs,
        'alpha': model.alpha,
        'input_dim': model.input_dim,
        'is_value': model.is_value,
    }
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


def init_rule_list(num_rules: int, dim_in: int, dim_out: int):
    weights = np.random.rand(num_rules, dim_in)
    comparators = np.random.rand(num_rules, 1)
    leaves: list[LeafInfo] = [
        # for dim_out > 1 use tolist
        ([leaf_index], np.arange(0, leaf_index).tolist(), np.random.rand(dim_out).tolist())
        for leaf_index in range(num_rules)
    ]
    leaves.append(([], np.arange(0, num_rules).tolist(), np.random.rand(dim_out).tolist()))
    return weights, comparators, leaves


class DDTAgent(AgentBase):

    action_network: DDT
    value_network: DDT

    def __init__(
        self,
        bot_name="DDT",
        input_dim=4,
        output_dim=2,
        *,
        rule_list=False,
        num_rules=4,
        version: Optional[int] = None,
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
            value_preds=self.last_value_pred[
                self.last_action.item()
            ],  # pyright: ignore[reportArgumentType]  # item() -> Number and not int
            deeper_action_log_probs=self.last_deep_action_probs,
            deeper_value_pred=self.last_deep_value_pred[  # pyright: ignore[reportArgumentType, reportCallIssue]
                self.last_action.item()
            ],
            last_action=self.last_action.item(),
            full_probs_vector=self.full_probs,
            deeper_full_probs_vector=self.deeper_full_probs,
            rewards=reward,
        )
        return True

    def save(self, path: Union[Path, str]='last', *, force_save: bool=False):
        """
        The two outputs are saved as two separate files named

        ``str(path) + self.bot_name + '_actor|_critic' + f'_v{self.version}.pth.tar'``

        Args:
            force_save: Still saves the output even in `save_output` is False
        """
        assert self.version is not None
        if not (self.save_output or force_save):
            return
        act_fn = str(path) + self.bot_name + '_actor' + f'_v{self.version}.pth.tar'
        val_fn = str(path) + self.bot_name + "_critic" + f"_v{self.version}.pth.tar"

        save_ddt(act_fn, self.action_network)
        save_ddt(val_fn, self.value_network)

    def load(self, fn: str | Path ='last', *, version=None, auto_naming=True):
        """
        Replaced the action and value network of the agent

        Args:
            auto_naming: Will use the botn_ame and version to construct actor and critic network.
            Otherwise the fn must contain _actor in its name, which is replaced by _critic to load
            the critic network
        """
        if auto_naming:
            assert version is not None
            act_fn = str(fn) + self.bot_name + '_actor' + f'_v{version}.pth.tar'
            val_fn = str(fn) + self.bot_name + '_critic' + f'_v{version}.pth.tar'
        else:
            act_fn = str(fn)
            assert "_actor" in act_fn
            val_fn = act_fn.replace("_actor", "_critic")
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
            'num_rules': self.num_rules,
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
                             use_gpu=False,  # <-----
                             )
        new_agent.__setstate__X(self.__getstate__X())
        return new_agent
