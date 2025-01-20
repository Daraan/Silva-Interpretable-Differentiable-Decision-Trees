# Created by Andrew Silva on 8/28/19
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, TypeGuard, Union, cast

import gymnasium as gym
import numpy as np
import torch.multiprocessing as mp
from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo  # pyright: ignore[reportPrivateImportUsage]
from joblib import Parallel, delayed
from tqdm import tqdm

from interpretable_ddts.agents.ddt_agent import DDTAgent
from interpretable_ddts.agents.mlp_agent import MLPAgent
from interpretable_ddts.opt_helpers.replay_buffer import discount_reward
from interpretable_ddts.runfiles._pbar_updates import update_pbar
from interpretable_ddts.runfiles.constants import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN, EVALUATION_RESULTS
from ray_utilities import GYM_V_0_26, seed_everything

if TYPE_CHECKING:
    from multiprocessing.synchronize import Lock

    from gymnasium.core import ActType, ObsType
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec  # for performance import only if used  # noqa: TC004

    from interpretable_ddts.agents._agent_interface import AgentBase


def run_episode(
    q,
    env: gym.Env[ObsType, ActType],
    agent_in: AgentBase,
    *,
    render_mode=None,
    **kwargs,
) -> tuple[float, dict[str, Any]]:
    agent = agent_in.duplicate(**kwargs)  # NOTE: Weights are not copied

    # Reset without resetting the RNG generator to get an initial observation
    if GYM_V_0_26:
        state, _ = env.reset()
    else:
        state: ObsType = env.reset()  # type: ignore[assignment]

    done = False
    while not done:
        action: ActType = agent.get_action(state)  # pyright: ignore[reportAssignmentType]
        # Step through environment using chosen action
        if GYM_V_0_26:
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        else:
            state, reward, done, _ = env.step(action)  # pyright: ignore[reportAssignmentType]
        # env.render()
        # Save reward
        agent.save_reward(reward)
        if done:
            break
    reward_sum = np.sum(agent.replay_buffer.rewards_list)
    rewards_list, advantage_list, deeper_advantage_list = discount_reward(
        reward=agent.replay_buffer.rewards_list,
        value=agent.replay_buffer.value_list,
        deeper_value=agent.replay_buffer.deeper_value_list,
    )
    agent.replay_buffer.rewards_list = rewards_list
    agent.replay_buffer.advantage_list = advantage_list
    agent.replay_buffer.deeper_advantage_list = deeper_advantage_list

    to_return = (reward_sum, copy.deepcopy(agent.replay_buffer.__getstate__()))
    if q is not None:
        try:
            q.put(to_return)
        except RuntimeError as e:
            print(e)
            return to_return
    return to_return


def main(
    episodes,
    agent: Union[DDTAgent, MLPAgent, AgentBase],
    env: str | gym.Env,
    *,
    seed=None,
    pbar: bool | Iterable[int] = True,
    render_mode=None,
):
    if agent.save_output:
        assert agent.rewards_file
        models_path = Path("../models") / (agent.bot_name + f"_v{agent.version}")
        rewards_path = Path("../txts")
        models_path.mkdir(parents=True, exist_ok=True)
        rewards_path.mkdir(parents=True, exist_ok=True)
        # Create a link to the rewards file
        (models_path / agent.rewards_file.name).symlink_to(agent.rewards_file.resolve())
    elif TYPE_CHECKING:
        models_path = Path("Is not used")

    if isinstance(env, str):
        if env == "lunar":
            env = gym.make("LunarLander-v2", render_mode=render_mode)
        elif env == "cart":
            env = gym.make("CartPole-v1", render_mode=render_mode)
        else:
            env = gym.make(env, render_mode=render_mode)
    else:
        assert env.render_mode == render_mode, "Render mode mismatch"
    if render_mode is not None:
        env = RecordVideo(
            env,
            "../outputs/video/",
            name_prefix="training",
            episode_trigger=lambda x: x % 250 == 0,
        )
        env = RecordEpisodeStatistics(env)
    next_seed, _ = seed_everything(env=env, seed=seed)
    if GYM_V_0_26 and next_seed is not None:
        # Note that with seed=None, the RNG for the environment will not be reset
        env.reset(seed=next_seed)

    if pbar is True:
        print("Running agent ", agent.bot_name, " version ", agent.version)
        pbar = tqdm(range(1, episodes + 1), miniters=10)
        use_pbar = True
    elif not pbar:
        pbar = range(1, episodes + 1)
        use_pbar = False
    else:
        use_pbar = True

    def is_pbar(pbar) -> TypeGuard[tqdm]:  # noqa: ARG001
        return use_pbar

    try:
        agent.duplicate(discrete=True)  # type: ignore
        can_duplicate_discrete = True
    except TypeError:
        can_duplicate_discrete = False

    reward = episode = running_reward = float("nan")
    running_reward_array = []
    discrete_running_reward_array = []
    for episode in pbar:
        returned_object = run_episode(
            None,
            env=env,
            agent_in=agent,
            render_mode=render_mode,
        )
        reward = returned_object[0]
        running_reward_array.append(reward)
        agent.replay_buffer.extend(returned_object[1])
        if (
            agent.save_output
            and (reward >= 499 or (env == "lunar" and reward >= 0))
            and episode % 500 != 0  # saved below
        ):
            agent.save(models_path / f"{episode}th")
        if can_duplicate_discrete:
            discrete_returned_object = run_episode(
                None,
                env=env,
                agent_in=agent,
                render_mode=render_mode,
                discrete=True,
            )
            discrete_running_reward_array.append(discrete_returned_object[0])
            agent.end_episode(reward, discrete_reward=discrete_returned_object[0])
        else:
            agent.end_episode(reward)

        running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))
        if is_pbar(pbar) and episode % 2 == 0:
            update_pbar(
                pbar,
                train_results=None,
                eval_results={
                    "mean": reward,
                    "roll": running_reward,
                },
                discrete_eval_results=(
                    {
                        "mean": discrete_running_reward_array[-1],
                        "roll": (
                            sum(discrete_running_reward_array[-100:])
                            / float(min(100.0, len(discrete_running_reward_array)))
                        ),
                    }
                    if can_duplicate_discrete
                    else None
                ),
            )
        if agent.save_output and episode % 500 == 0:
            agent.save(models_path / f"{episode}th")
    # Save final episode
    if is_pbar(pbar) and episode % 50 != 0:
        pbar.set_description(
            f"{agent.bot_name}_v{agent.version} "
            f"|Ep. {episode:<4} |Rwrd: {reward:>4.0f} "
            f"|Avg. Rwrd: {running_reward:>4.0f} "
            f"|Len {returned_object[1]['steps']:>3}",  # pyright: ignore[reportPossiblyUnboundVariable]
        )
    if agent.save_output and episode % 500 != 0:
        agent.save(models_path / f"{episode}th")

    return running_reward_array


def create_rlib_agent(args, init_env: gym.Env):
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec  # noqa: F811

    from interpretable_ddts.agents.ddt_catalog import DDTCatalog  # noqa: F811
    from interpretable_ddts.agents.ddt_ppo_module import LegacyDDTModule  # noqa: F811

    module_spec = RLModuleSpec(
        module_class=LegacyDDTModule,
        observation_space=init_env.observation_space,
        action_space=init_env.action_space,
        model_config={
            "bot_name": AGENT_TYPE + ENV_TYPE,
            "input_dim": dim_in,
            "output_dim": dim_out,
            "rule_list": args.rule_list,
            "num_rules": args.num_leaves,
            "save_output": not args.test,
            "use_gpu": USE_GPU,
            "vf_double_output": True,
            "action_use_softmax": True,
        },
        catalog_class=DDTCatalog,
    )
    policy_agent: LegacyDDTModule = cast(LegacyDDTModule, module_spec.build())
    policy_agent.setup()
    return policy_agent


def start_process(
    i,
    args: argparse.Namespace,
    init_env: Optional[gym.Env] = None,
    lock: Optional[Lock] = None,
    *,
    use_rllib_output: bool = False,
):
    """Wrapper of main that can be used in parallel."""
    agent_type: "str | RLModuleSpec" = args.agent_type
    env_type: str | gym.Env = args.env_type
    if not isinstance(env_type, str):
        env_type = env_type.unwrapped.spec.id
    seed: Optional[int] = args.seed
    # Initialize with different seed
    if isinstance(i, int):
        sub_seed = seed + i if seed is not None else None
        seed_everything(None, seed=sub_seed, torch_manual=False)
    seed2 = np.random.randint(0, 1000000)
    if agent_type.__class__.__name__ == "RLModuleSpec":  # avoid expensive import
        bot_name = "rllib" + env_type
    else:
        if TYPE_CHECKING:
            assert isinstance(agent_type, str)
        bot_name = agent_type + env_type
    if args.gpu:
        bot_name += "GPU"
    # Use a lock for file creation
    with lock or nullcontext():
        if agent_type == "ddt":
            policy_agent = DDTAgent(
                bot_name=bot_name,
                input_dim=args.dim_in,
                output_dim=args.dim_out,
                rule_list=args.rule_list,
                num_rules=args.num_leaves,
                save_output=not args.test,
            )
        elif agent_type == "mlp":
            policy_agent = MLPAgent(
                bot_name=bot_name,
                input_dim=args.dim_in,
                output_dim=args.dim_out,
                num_hidden=args.num_hidden,
                save_output=not args.test,
            )
        elif agent_type == "rllib":
            assert init_env
            policy_agent = create_rlib_agent(args, init_env)
        elif agent_type.__class__.__name__ == "RLModuleSpec":
            assert not TYPE_CHECKING or isinstance(agent_type, RLModuleSpec)
            policy_agent = agent_type.build()
            policy_agent.setup()
        else:
            raise ValueError(f"No valid network selected: {agent_type}")
    use_pbar: bool | type[tqdm] = getattr(args, "use_pbar", True)
    if use_pbar:
        if isinstance(use_pbar, type):
            pbar = use_pbar(range(1, args.episodes + 1))
        else:
            pbar = tqdm(
                range(1, args.episodes + 1),
                miniters=10,
                mininterval=0.2,
                maxinterval=1,
                position=i + (args.process_number % 5) * 5,
                postfix="Process " + str(i + (args.process_number * 5)),
            )
    else:
        pbar = False
    reward_array = main(
        args.episodes,
        policy_agent,
        args.env_type,
        seed=seed2,
        pbar=pbar,
        render_mode=args.render_mode,
    )
    if not use_rllib_output:
        results = {
            "running_reward_mean": np.mean(reward_array[-100:]),
            "num_episodes": len(reward_array),
            "perfect_episodes": sum([1 for r in reward_array if r >= 499]),
        }
    else:
        results = {
            EVALUATION_RESULTS: {
                ENV_RUNNER_RESULTS: {
                    EPISODE_RETURN_MEAN: max(reward_array[-5:]),
                },
            },
        }
    if "comment" in args:
        results["comment"] = args.comment
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default="ddt")
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=2000)
    parser.add_argument("-l", "--num_leaves", help="number of leaves for DDT/DRL ", type=int, default=8)
    parser.add_argument("-n", "--num_hidden", help="number of hidden layers for MLP ", type=int, default=0)
    parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default="cart")
    parser.add_argument("-gpu", "--gpu", help="run on GPU?", action="store_true")
    parser.add_argument("-r", "--rule_list", help="Use rule list setup", action="store_true", default=False)
    parser.add_argument("-s", "--seed", help="Seed", default=-1, type=int)
    parser.add_argument("-np", "--not_parallel", help="Do not run in parallel", action="store_true", default=False)
    parser.add_argument("-p", "--process_number", help="Process number", type=int, default=0)
    parser.add_argument("--silent", help="supress prints", action="store_true", default=False)
    parser.add_argument("--test", "--dry-run", help="Do not save any models", action="store_true", default=False)
    parser.add_argument(
        "--render_mode", "-rm", help="Render the environment", type=str, default=None, const="human", nargs="?"
    )

    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    if args.render_mode == "human" and not args.not_parallel:
        raise ValueError("Cannot use 'human' render in parallel.")
    SEED = args.seed
    AGENT_TYPE: str = args.agent_type  # 'ddt', 'mlp'
    NUM_EPS: int = args.episodes  # num episodes Default 1000
    ENV_TYPE: str = args.env_type  # 'cart' or 'lunar' Default 'cart'
    USE_GPU = args.gpu  # Applies for 'prolo' only. use gpu? Default false

    init_env: gym.Env
    if ENV_TYPE == "lunar":
        init_env = cast(LunarLander, gym.make("LunarLander-v2", render_mode=args.render_mode))
        dim_in = init_env.observation_space.shape[0]  # type: ignore
        dim_out = init_env.action_space.n  # type: ignore[attr-defined]
        env = "LunarLander-v2"
    elif ENV_TYPE == "cart":
        init_env = cast(CartPoleEnv, gym.make("CartPole-v1", render_mode=args.render_mode))
        dim_in = init_env.observation_space.shape[0]  # type: ignore
        dim_out = init_env.action_space.n  # type: ignore[attr-defined]
        env = "CartPole-v1"
    else:
        raise ValueError(f"No valid environment {ENV_TYPE}")
    args.dim_in = dim_in
    args.dim_out = dim_out

    if not args.silent:
        print(f"Agent {AGENT_TYPE} on {ENV_TYPE} seed {SEED}")
    # mp.set_start_method('spawn')
    # mp.set_sharing_strategy('file_system')
    # torch.backends.cudnn.deterministic = True

    if not args.not_parallel:
        lock = mp.Manager().Lock()
        data = Parallel(n_jobs=5, pre_dispatch="all")(
            delayed(start_process)(i, args, init_env, lock) for i in range(5)
        )  # fmt: skip
    else:
        data = [start_process(0, args, init_env) for _ in range(5)]
