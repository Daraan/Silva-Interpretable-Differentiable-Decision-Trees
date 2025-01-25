from __future__ import annotations

import logging
from argparse import Namespace
from functools import partial
from typing import Any, Callable

from ray.rllib.algorithms.ppo.ppo import PPOConfig

from interpretable_ddts.runfiles._ddt_trainable import build_and_train, create_ddt_config
from ray_utilities.config.experiment_base import (
    DefaultArgumentParser,
    ExperimentSetupBase,
    TrainableReturnData,
)
from ray_utilities.environment import parse_env_name

logger = logging.getLogger(__name__)


class DDTArgumentParser(DefaultArgumentParser):
    agent_type: str = "ddt"

    num_leaves: int = 8
    """Number of leaves for DDT/DRL. Must be a square of 2."""

    rule_list: bool = False
    """Use rule list setup"""

    use_silva_loss: bool = False
    """Use Silva's loss implementation"""

    # MLP
    num_hidden: int | None = None
    """Number of hidden layers when using MLP"""

    legacy: bool = False
    """Use original code without an algorithm"""

    def configure(self):
        super().configure()
        self.add_argument("-l", "--num_leaves")
        self.add_argument("-rl", "--rule_list")


class DDTSetup(ExperimentSetupBase[PPOConfig, DDTArgumentParser]):
    @classmethod
    def create_parser(cls) -> DDTArgumentParser:
        return DDTArgumentParser()

    @classmethod
    def create_config(cls, args):
        config, _module_spec = create_ddt_config(args)
        return config

    @classmethod
    def postprocess_args(cls, args):
        args = super().postprocess_args(args)
        args.env_type = parse_env_name(args.env_type)
        assert args.agent_type == "ddt", f"Only DDT is supported, got {args.agent_type}"
        if args.agent_type == "ddt" and args.num_hidden:
            raise ValueError("Do not use --num_hidden with DDT")
        if args.agent_type == "mlp" and args.num_hidden is None:
            raise ValueError("Must specify --num_hidden with MLP")
        if not args.test and not args.comet:
            logger.warning("Not in test mode and comet disabled. Will not log to Comet")
            import time

            time.sleep(4)  # give user time to cancel

        if args.seed == -1:
            args.seed = None
        return args

    @classmethod
    def trainable_from_config(cls, *, args, config) -> Callable[[dict[str, Any]], TrainableReturnData]:
        if args.legacy:
            # Do not use an algorithm but the gym_runner.py code
            config, module_spec = create_ddt_config(vars(args).copy())
            from ray.experimental import tqdm_ray

            from interpretable_ddts.runfiles import gym_runner

            trainable = partial(
                gym_runner.start_process,
                args=Namespace(
                    agent_type=module_spec,
                    env_type=config.env,
                    seed=args.seed,
                    gpu=args.gpu,
                    rule_list=args.rule_list,
                    num_leaves=args.num_leaves,
                    test=args.test,
                    num_hidden=args.num_hidden,
                    use_pbar=tqdm_ray.tqdm,
                    episodes=args.episodes,
                    # Note: cast to int as it might be an np.int type
                    dim_in=int(
                        module_spec.observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript, reportOptionalMemberAccess]
                    ),
                    dim_out=int(module_spec.action_space.n),  # type: ignore[attr-defined],
                    render_mode=None,
                    comment=args.comment,
                ),
                use_rllib_output=True,
            )
            return trainable

        trainable = partial(build_and_train, use_pbar=True)
        return trainable
