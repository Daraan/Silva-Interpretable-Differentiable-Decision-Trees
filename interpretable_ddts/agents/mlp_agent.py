from pathlib import Path
import re
from ._agent_interface import AgentBase
import torch
import torch.nn as nn
from torch.distributions import Categorical
from interpretable_ddts.opt_helpers import replay_buffer, ppo_update
import copy
from typing import Union, Optional


class BaselineFCNet(nn.Module):
    def __init__(self, input_dim, is_value=False, output_dim=2, hidden_layers=1):
        super(BaselineFCNet, self).__init__()
        self.lin1 = nn.Linear(input_dim, input_dim)
        self.lin2 = None
        self.lin3 = nn.Linear(input_dim, output_dim)
        self.sig = nn.ReLU()
        self.input_dim = input_dim
        modules = []
        for h in range(hidden_layers):
            modules.append(nn.Linear(input_dim, input_dim))
        if len(modules) > 0:
            self.lin2 = nn.Sequential(*modules)
        self.softmax = nn.Softmax(dim=1)
        self.is_value = is_value

    def forward(self, input_data):
        if self.lin2 is not None:
            act_out = self.lin3(self.sig(self.lin2(self.sig(self.lin1(input_data)))))
        else:
            act_out = self.lin3(self.sig(self.lin1(input_data)))
        if self.is_value:
            return act_out
        else:
            return self.softmax(act_out)


class MLPAgent(AgentBase):
    def __init__(
        self,
        bot_name="MLPAgent",
        input_dim=4,
        output_dim=2,
        num_hidden=1,
        version: Optional[int] = None,
        _duplicate=False,
    ):
        self.bot_name = bot_name + "_" + str(num_hidden) + '_hid'
        self._duplicate = _duplicate   
        self.rewards_file = None   
        self._version = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        if version is None:
            self._check_version()
        else:
            self.version = version
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.action_network = BaselineFCNet(input_dim=input_dim,
                                            output_dim=output_dim,
                                            is_value=False,
                                            hidden_layers=num_hidden)
        self.value_network = BaselineFCNet(input_dim=input_dim,
                                           output_dim=output_dim,
                                           is_value=True,
                                           hidden_layers=num_hidden)

        self.ppo = ppo_update.PPO([self.action_network, self.value_network], two_nets=True)
        self.actor_opt = torch.optim.RMSprop(self.action_network.parameters(), lr=5e-3)
        self.value_opt = torch.optim.RMSprop(self.value_network.parameters(), lr=5e-3)

        self.last_state = [0, 0, 0, 0]
        self.last_action = 0
        self.last_action_probs = torch.Tensor([0])
        self.last_value_pred = torch.Tensor([[0, 0]])
        self.last_deep_action_probs = torch.Tensor([0])
        self.last_deep_value_pred = torch.Tensor([[0, 0]])
        self.full_probs = None
        self.reward_history = []
        self.num_steps = 0

    def get_action(self, observation):
        with torch.no_grad():
            obs = torch.Tensor(observation)
            obs = obs.view(1, -1)
            self.last_state = obs
            probs = self.action_network(obs)
            value_pred = self.value_network(obs)
            probs = probs.view(-1)
            self.full_probs = probs
            if self.action_network.input_dim > 30:
                probs, inds = torch.topk(probs, 3)
            m = Categorical(probs)
            action = m.sample()

            log_probs = m.log_prob(action)
            self.last_action_probs = log_probs
            self.last_value_pred = value_pred.view(-1).cpu()

            if self.action_network.input_dim > 30:
                self.last_action = inds[action]
            else:
                self.last_action = action
        if self.action_network.input_dim > 30:
            action = inds[action].item()
        else:
            action = action.item()
        return action

    def save_reward(self, reward):
        self.replay_buffer.insert(obs=[self.last_state],
                                  action_log_probs=self.last_action_probs,
                                  value_preds=self.last_value_pred[self.last_action.item()],
                                  last_action=self.last_action,
                                  full_probs_vector=self.full_probs,
                                  rewards=reward)
        return True

    def end_episode(self, timesteps):
        self.reward_history.append(timesteps)
        value_loss, action_loss = self.ppo.batch_updates(self.replay_buffer, self)
        if self.rewards_file is None:
            txts_path = Path('../txts')
            txts_path.mkdir(parents=True, exist_ok=True)
            if self.version is None:
                rewards_file = txts_path / (
                    self.bot_name + "_0_rewards.txt"
                )
                if not rewards_file.exists():
                    self.version = 0
                else:
                    stem = rewards_file.with_name(rewards_file.stem) # remove the _rewards.txt
                    match = re.match(
                        r"(?P<name>.+)_v(?P<number>\d+)'_rewards.txt'$", str(stem)
                    )
                    if match:
                        d = int(match.groupdict()["number"]) + 1
                    else:
                        d = 1
                    self.version = d
            rewards_file = txts_path / (self.bot_name + f"_v{self.version}" + "_rewards.txt")
            self.rewards_file = rewards_file
        self.rewards_file.open("a").write(str(timesteps) + "\n")
        self.num_steps += 1

    def reset(self):
        self.replay_buffer.clear()


    def save(self, fn: Union[Path, str]='last'):
        assert self.version is not None
        checkpoint = dict()
        checkpoint['actor'] = self.action_network.state_dict()
        checkpoint['value'] = self.value_network.state_dict()
        save_path = Path(str(fn))
        save_path = save_path.with_name(save_path.name + self.bot_name + f"_v{self.version}" + ".pth.tar")
        torch.save(checkpoint, save_path)

    def load(self, fn='last'):
        # fn = fn + self.bot_name + '.pth.tar'
        model_checkpoint = torch.load(fn, map_location='cpu')
        actor_data = model_checkpoint['actor']
        value_data = model_checkpoint['value']
        self.action_network.load_state_dict(actor_data)
        self.value_network.load_state_dict(value_data)

    def __getstate__(self):
        return {
            # 'replay_buffer': self.replay_buffer,
            'action_network': self.action_network,
            'value_network': self.value_network,
            'ppo': self.ppo,
            'actor_opt': self.actor_opt,
            'value_opt': self.value_opt,
            'num_hidden': self.num_hidden
        }

    def __setstate__(self, state):
        self.action_network = copy.deepcopy(state['action_network'])
        self.value_network = copy.deepcopy(state['value_network'])
        self.ppo = copy.deepcopy(state['ppo'])
        self.actor_opt = copy.deepcopy(state['actor_opt'])
        self.value_opt = copy.deepcopy(state['value_opt'])
        self.num_hidden = copy.deepcopy(state['num_hidden'])
        
    def _write_hparams(self):
        self.rewards_file.open("w").write(
            ", ".join(
                [
                    f"name: {self.bot_name}",
                    "method: mlp",
                    f"version: {self.version}",
                    f"input_dim: {self.input_dim}",
                    f"output_dim: {self.output_dim}",
                    f"num_hidden: {self.num_hidden}",
                    "rule_list: False",
                ]
            )
            + "\n"
        )

    def duplicate(self):
        new_agent = MLPAgent(
            bot_name=self.bot_name.rstrip('_'),
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_hidden=self.num_hidden,
            version=self.version,
            _duplicate=True
            )
        new_agent.__setstate__(self.__getstate__())
        return new_agent
