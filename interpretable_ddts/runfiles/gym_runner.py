# Created by Andrew Silva on 8/28/19
from pathlib import Path
from typing import Optional, Union
import gym
import numpy as np
import torch
from interpretable_ddts.agents._agent_interface import AgentBase
from interpretable_ddts.agents.ddt_agent import DDTAgent
from interpretable_ddts.agents.mlp_agent import MLPAgent
from interpretable_ddts.opt_helpers.replay_buffer import discount_reward
from joblib import Parallel, delayed
import time
import torch.multiprocessing as mp
import argparse
import copy
from tqdm import tqdm

from interpretable_ddts.tools import seed_everything

def run_episode(q, agent_in: AgentBase, ENV_NAME, seed: Optional[int]=0):
    agent = agent_in.duplicate()
    if ENV_NAME == 'lunar':
        env = gym.make('LunarLander-v2')
    elif ENV_NAME == 'cart':
        env = gym.make('CartPole-v1')
    else:
        raise Exception('No valid environment selected')
    seed_everything(env, seed)
    # docstring: returns an initial observation.
    # If the environment already has a random number generator and reset is called with seed=None, the RNG should not be reset.
    # Moreover, reset should (in the typical use case) be called with an integer seed right after initialization and then never again.
    state = env.reset()  # Reset environment and record the starting state

    done = False
    while not done:
        action = agent.get_action(state)
        # Step through environment using chosen action
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

    to_return = [reward_sum, copy.deepcopy(agent.replay_buffer.__getstate__())]
    if q is not None:
        try:
            q.put(to_return)
        except RuntimeError as e:
            print(e)
            return to_return
    return to_return


def main(episodes, agent: Union[DDTAgent, MLPAgent], ENV_NAME, seed=None, pbar=None):
    running_reward_array = []
    models_path = Path('../models')
    rewards_path = Path('../txts')
    models_path.mkdir(parents=True, exist_ok=True)
    rewards_path.mkdir(parents=True, exist_ok=True)
    
    if pbar is None:
        print("Running agent ", agent.bot_name, " version ", agent.version)
        pbar = tqdm(range(1, episodes + 1), miniters=10, mininterval=0.2, maxinterval=1)
    for episode in pbar:
        reward = 0
        returned_object = run_episode(None, agent_in=agent, ENV_NAME=ENV_NAME)
        reward += returned_object[0]
        running_reward_array.append(returned_object[0])
        agent.replay_buffer.extend(returned_object[1])
        if reward >= 499:
            agent.save(models_path / f"{episode}th")
        agent.end_episode(reward)

        running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))
        if episode % 2 == 0:
            pbar.set_description(f"{agent.bot_name}_v{agent.version} | Episode {episode}  Last Reward: {reward:.2f}  Average Reward: {running_reward:.2f}")
        if episode % 500 == 0:
            agent.save(models_path / f"{episode}th")
    # Save final episode
    if episode % 50 != 0:
        pbar.set_description(f"Episode {episode}  Last Reward: {reward}  Average Reward: {running_reward}")
    if episode % 500 != 0:
        agent.save(models_path / f"{episode}th")

    return running_reward_array


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

    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    SEED = args.seed
    
    AGENT_TYPE = args.agent_type  # 'ddt', 'mlp'
    NUM_EPS = args.episodes  # num episodes Default 1000
    ENV_TYPE = args.env_type  # 'cart' or 'lunar' Default 'cart'
    USE_GPU = args.gpu  # Applies for 'prolo' only. use gpu? Default false
    if ENV_TYPE == 'lunar':
        init_env = gym.make('LunarLander-v2')
        dim_in = init_env.observation_space.shape[0]
        dim_out = init_env.action_space.n
    elif ENV_TYPE == 'cart':
        init_env = gym.make('CartPole-v1')
        dim_in = init_env.observation_space.shape[0]
        dim_out = init_env.action_space.n
    else:
        raise Exception('No valid environment selected')

    print(f"Agent {AGENT_TYPE} on {ENV_TYPE} seed {SEED}")
    # mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    def start_process(i):
        if i > 0:  # delay the start for file existence checks
            time.sleep(i/2)
        # Initialize with different seed
        seed_everything(None, SEED + i if SEED is not None else None)
        bot_name = AGENT_TYPE + ENV_TYPE
        if USE_GPU:
            bot_name += 'GPU'
        if AGENT_TYPE == 'ddt':
            policy_agent = DDTAgent(bot_name=bot_name,
                                    input_dim=dim_in,
                                    output_dim=dim_out,
                                    rule_list=args.rule_list,
                                    num_rules=args.num_leaves)
        elif AGENT_TYPE == 'mlp':
            policy_agent = MLPAgent(input_dim=dim_in,
                                    bot_name=bot_name,
                                    output_dim=dim_out,
                                    num_hidden=args.num_hidden)
        else:
            raise Exception('No valid network selected')
        pbar = tqdm(
            range(1, NUM_EPS + 1),
            miniters=10,
            mininterval=0.2,
            maxinterval=1,
            position=i + (args.process_number % 5) * 5,
            postfix="Process " + str(i + (args.process_number * 5)),
        )
        reward_array = main(NUM_EPS, policy_agent, ENV_TYPE, seed=SEED, pbar=pbar)
        return reward_array
    if not args.not_parallel:
        data = Parallel(n_jobs=5, pre_dispatch="all")(
            delayed(start_process)(i) for i in range(5)
        )
    else:
        data = [start_process(0) for _ in range(5)]