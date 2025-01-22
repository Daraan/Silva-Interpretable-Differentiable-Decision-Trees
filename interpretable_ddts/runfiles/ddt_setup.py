from __future__ import annotations
from argparse import Namespace
from functools import partial

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray_utilities.config.experiment_base import ExperimentSetupBase, DefaultArgumentParser

from interpretable_ddts.runfiles._ddt_trainable import build_and_train, create_ddt_config
from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import RLModuleSpec
    from ray.rllib.algorithms import AlgorithmConfig


class DDTArgumentParser(DefaultArgumentParser):
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


class DDTSetup(ExperimentSetupBase):
    def create_parser(self) -> DDTArgumentParser:
        return DDTArgumentParser()

    @overload
    def create_config(self, args: Namespace, *, return_module_spec: Literal[False]) -> PPOConfig: ...

    @overload
    def create_config(
        self, args: Namespace, *, return_module_spec: Literal[True] = True
    ) -> tuple[PPOConfig, RLModuleSpec]: ...

    def create_config(self, args: Namespace, *, return_module_spec: bool = True):
        config, module_spec = create_ddt_config(args)
        if return_module_spec:
            return config, module_spec
        return config

    def create_trainable(self):
        return partial(build_and_train, use_pbar=True)

    def trainable_from_config(self, *, args: Namespace | DDTArgumentParser, config: AlgorithmConfig):  # pyright: ignore[reportIncompatibleMethodOverride]
        if args.legacy:
            # Do not use an algorithm but the gym_runner.py code
            config, module_spec = create_ddt_config(vars(args).copy())
            from interpretable_ddts.runfiles import gym_runner
            from ray.experimental import tqdm_ray

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

        trainable = partial(build_and_train, use_pbar=True)
        return trainable
