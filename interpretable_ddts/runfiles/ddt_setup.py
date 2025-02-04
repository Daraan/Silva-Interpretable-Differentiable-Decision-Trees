from __future__ import annotations

import logging
from argparse import Namespace
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from ray.rllib.algorithms.ppo.ppo import PPOConfig

from interpretable_ddts.runfiles._ddt_trainable import build_and_train, create_ddt_config
from ray_utilities.config.experiment_base import (
    DefaultArgumentParser,
    ExperimentSetupBase,
)
from ray_utilities.environment import create_env

if TYPE_CHECKING:
    from ray_utilities import AlgorithmReturnData

logger = logging.getLogger(__name__)

__all__ = ["DDTArgumentParser", "DDTSetup"]


class DDTArgumentParser(DefaultArgumentParser):
    agent_type: str = "ddt"

    num_leaves: int = 8
    """Number of leaves for DDT/DRL. Must be a square of 2."""

    rule_list: bool = False
    """Use rule list setup"""

    use_silva_loss: bool = False
    """Use Silva's loss implementation"""

    # MLP
    num_hidden: int = 0
    """Number of hidden layers when using MLP"""

    legacy: bool = False
    """Use original code without an algorithm"""

    def configure(self):
        super().configure()
        self.add_argument("-l", "--num_leaves")
        self.add_argument("-rl", "--rule_list")


class DDTSetup(ExperimentSetupBase[PPOConfig, DDTArgumentParser]):
    # region Argument Parsing

    default_extra_tags: ClassVar[list[str]] = [
        *ExperimentSetupBase.default_extra_tags,
        # base tags are: "dev", "<test>", "<gpu>", "<env_type>", "<agent_type>"
        "<legacy>",
        "<use_silva_loss>",
        "<rule_list>",
    ]

    PROJECT: str = "DDT-Silva"

    @property
    def project_name(self) -> str:
        """Name for the output folder, wandb project, and comet workspace."""
        return "dev-workspace" if self.args.test else self.PROJECT

    @project_name.setter
    def project_name(self, value: str):
        logger.warning("Setting project name to %s. Prefer creation of a new class", value)
        self.PROJECT = value

    @property
    def group_name(self) -> str:
        return "_".join([self.args.agent_type, self.args.env_type, ("-test" if self.args.test else "")])

    def create_parser(self) -> DDTArgumentParser:
        return DDTArgumentParser()

    def _create_config(self):
        config, _module_spec = create_ddt_config(self.args)
        return config

    def postprocess_args(self, args):
        args = super().postprocess_args(args)
        # Set env name
        init_env = create_env(args.env_type)
        env_name = init_env.unwrapped.spec.id  # pyright: ignore[reportOptionalMemberAccess]
        args.env_type = env_name
        # Assertions
        assert args.agent_type == "ddt", f"Only DDT is supported, got {args.agent_type}"
        if args.agent_type == "ddt" and args.num_hidden:
            raise ValueError("Do not use --num_hidden with DDT")
        if args.agent_type == "mlp" and args.num_hidden:
            raise ValueError("Must specify --num_hidden with MLP")
        if not args.test and not args.comet:
            logger.warning("Not in test mode and comet disabled. Will not log to Comet")
            import time

            time.sleep(4)  # give user time to cancel

        if args.seed == -1:
            args.seed = None
        return args

    # endregion

    def clean_args_to_hparams(self, args: Namespace | DDTArgumentParser | None = None):
        upload_args = super().clean_args_to_hparams(args)
        del args  # no not confuse variables
        upload_args["extra"] = None if not self.args.extra else repr([repr(e) for e in self.args.extra])
        if self.args.agent_type == "ddt":
            del upload_args["num_hidden"]
        return upload_args

    # region Config and Trainable

    def create_trainable(self) -> Callable[[dict[str, Any]], AlgorithmReturnData]:
        if self.args.legacy:
            # Do not use an algorithm but the gym_runner.py code
            from ray.experimental import tqdm_ray

            from interpretable_ddts.runfiles import gym_runner

            module_spec = self.config.get_rl_module_spec()
            trainable = partial(
                gym_runner.start_process,
                args=Namespace(
                    agent_type=module_spec,
                    env_type=self.config.env,
                    seed=self.args.seed,
                    gpu=self.args.gpu,
                    rule_list=self.args.rule_list,
                    num_leaves=self.args.num_leaves,
                    test=self.args.test,
                    num_hidden=self.args.num_hidden,
                    use_pbar=tqdm_ray.tqdm,
                    episodes=self.args.episodes,
                    # Note: cast to int as it might be an np.int type
                    dim_in=int(module_spec.observation_space.shape[0]),  # noqa: E501 # pyright: ignore[reportOptionalSubscript, reportOptionalMemberAccess]
                    dim_out=int(module_spec.action_space.n),  # type: ignore[attr-defined],
                    render_mode=None,
                    comment=self.args.comment,
                ),
                use_rllib_output=True,
            )
            return trainable

        trainable = partial(build_and_train, use_pbar=True)
        return trainable

    # endregion
