from typing import Optional
from pathlib import Path

class AgentBase:
    
    bot_name : str
    _duplicate : bool
    
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
        rewards_file.open("w")
        self.rewards_file = rewards_file
    
    def duplicate(self):
        raise NotImplementedError