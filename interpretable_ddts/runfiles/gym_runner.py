# Created by Andrew Silva on 8/28/19
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union, Any, cast
import gymnasium as gym
import numpy as np
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from interpretable_ddts.agents._agent_interface import AgentBase
from interpretable_ddts.agents.ddt import DDTCatalog
from interpretable_ddts.agents.ddt_agent import DDTAgent
from interpretable_ddts.agents.ddt_ppo_module import DDTModuleGymRunner
from interpretable_ddts.agents.mlp_agent import MLPAgent
from interpretable_ddts.opt_helpers.replay_buffer import discount_reward
from joblib import Parallel, delayed
import time
import torch.multiprocessing as mp
import argparse
import copy
from tqdm import tqdm

from interpretable_ddts.tools import seed_everything

from packaging.version import parse as parse_version, Version

from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

GYM_VERSION = parse_version(gym.__version__)
GYM_V_0_26 = GYM_VERSION >= Version("0.26")
"""First gymnasium version"""
GYM_V1 = GYM_VERSION >= Version("1.0.0")

def run_episode(q, env: gym.Env, agent_in: AgentBase, seed: Optional[int]=0, render_mode=None) -> tuple[float, dict[str, Any]]:
    agent = agent_in.duplicate()

    # docstring: returns an initial observation.
    # If the environment already has a random number generator and reset is called with seed=None, the RNG should not be reset.
    # Moreover, reset should (in the typical use case) be called with an integer seed right after initialization and then never again.
    # Reset environment and record the starting state
    if GYM_V_0_26:
        state, info = env.reset(seed=seed)
    else:
        state = env.reset()

    done = False
    while not done:
        action = agent.get_action(state)
        # Step through environment using chosen action
        if GYM_V_0_26:
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        else:
            state, reward, done, _ = env.step(action)
        # env.render()
        # Save reward
        agent.save_reward(reward)
        if done:
            break
    reward_sum = np.sum(agent.replay_buffer.rewards_list)
    rewards_list, advantage_list, deeper_advantage_list = discount_reward(agent.replay_buffer.rewards_list,
                                                                          agent.replay_buffer.value_list,
                                                                          agent.replay_buffer.deeper_value_list)
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
    seed=None,
    pbar: bool | Iterable[int]=True,
    render_mode=None,
):
    running_reward_array = []
    if agent.save_output:
        models_path = Path("../models") / (agent.bot_name + f"_v{agent.version}")
        rewards_path = Path('../txts')
        models_path.mkdir(parents=True, exist_ok=True)
        rewards_path.mkdir(parents=True, exist_ok=True)
        # Create a link to the rewards file
        (models_path / agent.rewards_file.name).symlink_to(agent.rewards_file.resolve())

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
    seed_everything(env, seed)
    if GYM_V_0_26:
        env.reset(seed=seed)

    if pbar is True:
        print("Running agent ", agent.bot_name, " version ", agent.version)
        pbar = tqdm(range(1, episodes + 1), miniters=10)
        use_pbar = True
    elif not pbar:
        pbar = range(1, episodes + 1)
        use_pbar = False
    else:
        use_pbar = True
    for episode in pbar:
        returned_object = run_episode(
            None,
            env=env,
            agent_in=agent,
            render_mode=render_mode,
            seed=None,
        )
        reward = returned_object[0]
        running_reward_array.append(returned_object[0])
        agent.replay_buffer.extend(returned_object[1])
        if (
            agent.save_output
            and (reward >= 499 or (env == "lunar" and reward >= 0))
            and episode % 500 != 0  # saved below
        ):
            agent.save(models_path / f"{episode}th")
        agent.end_episode(reward)

        running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))
        if use_pbar and episode % 2 == 0:
            pbar.set_description(
                f"{agent.bot_name}_v{agent.version} |Ep. {episode:<4} |Rwrd: {reward:>4.0f} |Avg. Rwrd: {running_reward:>4.0f} |Len {returned_object[1]['steps']:>3}"
            )
        if agent.save_output and episode % 500 == 0:
            agent.save(models_path / f"{episode}th")
    # Save final episode
    if use_pbar and episode % 50 != 0:
        pbar.set_description(
            f"{agent.bot_name}_v{agent.version} |Ep. {episode:<4} |Rwrd: {reward:>4.0f} |Avg. Rwrd: {running_reward:>4.0f} |Len {returned_object[1]['steps']:>3}"
        )
    if agent.save_output and episode % 500 != 0:
        agent.save(models_path / f"{episode}th")

    return running_reward_array


def start_process(i, args: argparse.Namespace, init_env=None):
    """
    Wrapper of main that can be used in parallel.
    """
    if isinstance(i, dict):
        # rllib config pass
        pass
    elif i > 0:  # delay the start for file existence checks
        time.sleep(i / 2)
    agent_type: str | RLModuleSpec = args.agent_type
    if agent_type == "rllib":
        print(init_env)
        assert init_env
    env_type: str = args.env_type
    seed: Optional[int] = args.seed
    # Initialize with different seed
    if isinstance(i, int) and False:
        sub_seed = seed + i if seed is not None else None
        seed_everything(None, sub_seed, torch_manual=False)
    seed2 = np.random.randint(0, 1000000)
    if isinstance(agent_type, RLModuleSpec):
        bot_name = "rllib" + env_type
    else:
        bot_name = agent_type + env_type
    if args.gpu:
        bot_name += "GPU"
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
        policy_agent = create_rlib_agent(args, init_env)
    elif isinstance(agent_type, RLModuleSpec):
        policy_agent = agent_type.build()
        policy_agent.setup()
    else:
        raise Exception("No valid network selected")
    use_pbar = getattr(args, "use_pbar", True)
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
    return {
        "running_reward_mean" : np.mean(reward_array[-100:]),
        "perfect_episodes" : sum([1 for r in reward_array if r >= 499])
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default='ddt')
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=2000)
    parser.add_argument("-l", "--num_leaves", help="number of leaves for DDT/DRL ", type=int, default=8)
    parser.add_argument("-n", "--num_hidden", help="number of hidden layers for MLP ", type=int, default=0)
    parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default='cart')
    parser.add_argument("-gpu", "--gpu", help="run on GPU?", action='store_true')
    parser.add_argument("-r", "--rule_list", help="Use rule list setup", action='store_true', default=False)
    parser.add_argument("-s", "--seed", help="Seed", default=-1, type=int)
    parser.add_argument("-np", "--not_parallel", help="Do not run in parallel", action='store_true', default=False)
    parser.add_argument("-p", "--process_number", help="Process number", type=int, default=0)
    parser.add_argument("--silent", help="supress prints", action="store_true", default=False,)
    parser.add_argument("--test", "--dry-run", help="Do not save any models", action="store_true", default=False)
    parser.add_argument("--render_mode", "-rm", help="Render the environment",
                        type=str, default=None, const="human", nargs='?')

    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    if args.render_mode == "human" and not args.not_parallel:
        raise Exception("Cannot use 'human' render in parallel.")
    SEED = args.seed
    AGENT_TYPE: str = args.agent_type  # 'ddt', 'mlp'
    NUM_EPS: int = args.episodes  # num episodes Default 1000
    ENV_TYPE: str = args.env_type  # 'cart' or 'lunar' Default 'cart'
    USE_GPU = args.gpu  # Applies for 'prolo' only. use gpu? Default false

    init_env: gym.Env
    if ENV_TYPE == 'lunar':
        init_env = cast(LunarLander, gym.make('LunarLander-v2', render_mode=args.render_mode))
        dim_in = init_env.observation_space.shape[0]
        dim_out = init_env.action_space.n
        env = "LunarLander-v2"
    elif ENV_TYPE == 'cart':
        init_env = cast(CartPoleEnv, gym.make("CartPole-v1", render_mode=args.render_mode))
        dim_in = init_env.observation_space.shape[0]
        dim_out = init_env.action_space.n
        env = "CartPole-v1"
    else:
        raise Exception('No valid environment selected')
    args.dim_in = dim_in
    args.dim_out = dim_out

    if not args.silent:
        print(f"Agent {AGENT_TYPE} on {ENV_TYPE} seed {SEED}")
    # mp.set_start_method('spawn')
    # mp.set_sharing_strategy('file_system')
    #torch.backends.cudnn.deterministic = True

    if not args.not_parallel:
        data = Parallel(n_jobs=5, pre_dispatch="all")(
            delayed(start_process)(i, args, init_env) for i in range(5)
        )
    else:
        data = [start_process(0, args, init_env) for _ in range(5)]
