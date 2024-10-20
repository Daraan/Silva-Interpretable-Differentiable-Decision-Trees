from typing import Optional
from pathlib import Path

import torch
from torch.distributions import Categorical


class AgentBase:
    
    bot_name : str
    _duplicate : bool
    
    def __init__(self, input_dim=4, output_dim=2, *, version, _duplicate):
        assert self.bot_name, "self.bot_name should be set before calling super()"
        self.output_dim = output_dim
        self.input_dim = input_dim
        self._duplicate = _duplicate

        # check for next availiable version
        self.rewards_file = None
        self._version = None
        if version is None:
            self._check_version()
        else:
            self.version = version

    def _check_version(self) -> int:
        rewards_path = Path("../txts")
        rewards_path.mkdir(parents=True, exist_ok=True)
        rewards_file = rewards_path / (self.bot_name + "_v0_rewards.txt")
        if rewards_file.exists():
            files = list(rewards_path.glob(f"{self.bot_name}_v*"))
            latest = sorted(int(str(f).split("_v")[-1].split("_")[0]) for f in files)[
                -1
            ]
            self.version = latest + 1
        else:
            self.version = 0
        return self.version

    def _write_hparams(self):
        raise NotImplementedError

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value: Optional[int]):
        if value is not None and self._version is not None:
            raise ValueError("Version already set")
        self._version = value
        if value is None or self._duplicate or "crispytester" in self.bot_name:
            return
        if self.rewards_file:
            return
        txts_path = Path("../txts")
        txts_path.mkdir(parents=True, exist_ok=True)
        rewards_file = txts_path / (
            self.bot_name + f"_v{self.version}" + "_rewards.txt"
        )
        self.rewards_file = rewards_file
        self._write_hparams()
        self.rewards_file.open("a")

    #

    def get_action(self, observation, max_inputs):
        with torch.no_grad():
            obs = torch.Tensor(observation)
            obs = obs.view(1, -1)
            self.last_state = obs

            probs = self.action_network(obs)
            value_pred = self.value_network(obs)
            probs = probs.view(-1).cpu()
            self.full_probs = probs
            if self.action_network.input_dim > max_inputs:
                probs, inds = torch.topk(probs, 3)
            m = Categorical(probs)
            action = m.sample()
            log_probs = m.log_prob(action)
            self.last_action_probs = log_probs.cpu()
            self.last_value_pred = value_pred.view(-1).cpu()

            if self.action_network.input_dim > max_inputs:
                self.last_action = inds[action].cpu()
            else:
                self.last_action = action.cpu()
        if self.action_network.input_dim > max_inputs:
            action = inds[action].item()
        else:
            action = action.item()
        return action

    
    def end_episode(self, timesteps):
        assert self.version is not None and self.rewards_file
        self.reward_history.append(timesteps)
        value_loss, action_loss = self.ppo.batch_updates(self.replay_buffer, self)
        self.rewards_file.open("a").write(str(timesteps) + "\n")
        self.num_steps += 1

    def reset(self):
        self.replay_buffer.clear()

    def duplicate(self):
        raise NotImplementedError