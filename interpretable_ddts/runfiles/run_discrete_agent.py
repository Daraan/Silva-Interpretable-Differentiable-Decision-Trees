# Created by Andrew Silva on 5/10/19
from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import NamedTuple, Optional, cast
from joblib import Parallel, delayed
import pandas as pd
import torch
import numpy as np
import os
from interpretable_ddts.opt_helpers.discretization import convert_to_discrete
from interpretable_ddts.agents.ddt_agent import load_ddt, DDTAgent
from interpretable_ddts.agents.ddt import DDT
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from interpretable_ddts.opt_helpers.sklearn_to_ddt import ddt_init_from_dt
import matplotlib.pyplot as plt
from interpretable_ddts.runfiles.sc2_minigame_runner import run_episode as micro_episode
from interpretable_ddts.runfiles.gym_runner import run_episode as gym_episode
from interpretable_ddts.tools import RE_PARSE_FILENAME_OLD, create_df_index, match_filename

RE_PARSE_FILENAME = re.compile(
    r"(?P<parent_dir>.+?/)?"
    r"(?P<episode>\d+)th"
    r"(?P<method>ddt|mlp)"
    r"(?P<env>[^_]+?)(?P<GPU>GPU)?"
    r"_(?P<features>(?P<num>\d+)_(?P<typ>[^_]+))"
    r"_actor"
    r"_v(?P<version>\d+)"
)

class Result(NamedTuple):
    fn : str
    fuzzy_reward : float      # np.mean(reward_after_five)
    fuzzy_reward_std : float  # np.std(reward_after_five), 
    discrete_reward : float     # np.mean(crispy_reward)
    discrete_reward_std: float  # np.std(crispy_reward)


def evaluate_model(fn: str | Path, env: str, verbose: bool | int=1, count: Optional[tuple[int, int]]=None) -> Result | None:
    num_runs = 15
    
    final_deep_actor_fn = os.path.join(MODEL_DIR, fn)
    try:
        fda = load_ddt(final_deep_actor_fn)
    except FileNotFoundError as e:
        print(e)
        logging.error(f"File not found: {final_deep_actor_fn}")
        return None

    policy_agent = DDTAgent(bot_name='crispytester',
                            input_dim=37,
                            output_dim=10)

    policy_agent.action_network = fda
    policy_agent.value_network = fda
    reward_after_five = []
    master_states = []
    for _ in range(15):
        if env == "FindAndDefeatZerglings":
            try:
                reward, replay_buffer = micro_episode(None, policy_agent, env)
            except Exception as e:
                print(e)
                continue
        elif env in ['cart', 'lunar']:
            reward, replay_buffer = gym_episode(None, policy_agent, env)
        master_states.extend(replay_buffer['states'])
        reward_after_five.append(reward)

    # discrete
    crispy_actor = convert_to_discrete(policy_agent.action_network, master_states)
    policy_agent.action_network = crispy_actor

    crispy_reward = []
    for _ in range(num_runs):
        if env == "FindAndDefeatZerglings":
            try:
                crispy_out, replay_buffer = micro_episode(None, policy_agent, env)
            except Exception as e:
                print(e)
                crispy_out = -3
                continue
        elif env in ['cart', 'lunar']:
            crispy_out, replay_buffer = gym_episode(None, policy_agent, env)

        crispy_reward.append(crispy_out)
    if verbose:
        print(f"FN = {fn}\n"
            f"Average reward after 5 runs is {np.mean(reward_after_five):.3f}\n"
            f"Average reward for the crispy network after {num_runs} runs is {np.mean(crispy_reward):.3f}\n"
        )
    elif count is not None:
        # not a precise but estimated progress count
        print(f"{'~'+str(count[0]):>7}/{count[1]}", end="\r", flush=True)
    else:
        print(".", end="", flush=True)
    return Result(str(fn), np.mean(reward_after_five), np.std(reward_after_five), np.mean(crispy_reward), np.std(crispy_reward))

def search_for_good_model(env, n_jobs=5, verbose=1):
    # Be sure to comment out gym_runner.gym_episode env.render
    max_reward = -float("inf")
    max_std = -float("inf")
    max_fuzzy_reward = -float('inf')
    max_fuzzy_std = -float("inf")
    best_fn = 'non'
    best_fuzzy_fn = 'non'

    model_path = Path(MODEL_DIR)
    files = [
        fn
        for fn in model_path.glob(f"**/*ddt*{env}*actor*")
        if not fn.parent.name.startswith("models")  # excluded subdirs
    ]
    total = len(files)
    if total == 0:
        print("No results found in", model_path, "subdirs excluded")
        import sys
        sys.exit(1)
        return
    if verbose:
        print(f"Found {total} models")
    if total >= 50 and verbose == "auto":
        print("Turning off full verbose output for more than 500 models")
        verbose = False
    elif verbose == "auto":
        verbose = True
    delayed_functions = [delayed(evaluate_model)(fn.relative_to(model_path), env, verbose, (i, total)) for i, fn in enumerate(files, 1)]
    filenames = [fn.relative_to(model_path).name for fn in files]

    all_results_ = cast("list[Result | None]", Parallel(n_jobs=25)(delayed_functions))
    if not all(all_results_):
        all_results: "list[Result]" = list(
            filter(None, all_results_)
        )  # filter potential FileNotFound
        filenames = [fn for fn, res in zip(filenames, all_results_) if res]
    else:
        all_results = cast("list[Result]", all_results_)
    if not verbose:
        print("\n")
    metadata = [
        (RE_PARSE_FILENAME.match(file) or RE_PARSE_FILENAME_OLD.match(file)).groupdict()
        for file in filenames
    ]
    results_df = pd.DataFrame(all_results)
    results_df.index = create_df_index(metadata)
    results_df.sort_values("discrete_reward", ascending=False, inplace=True)
    
    best_fuzzy_arg: int = results_df.fuzzy_reward.argmax()  # type: ignore
    best_arg: int = results_df.discrete_reward.argmax()  # type: ignore
    
    max_fuzzy_reward = all_results[best_fuzzy_arg].fuzzy_reward
    max_fuzzy_std = all_results[best_fuzzy_arg].fuzzy_reward_std
    best_fuzzy_fn = all_results[best_fuzzy_arg].fn
    max_reward = all_results[best_arg].discrete_reward
    max_std = all_results[best_arg].discrete_reward_std
    best_fn = all_results[best_arg].fn
    
    return (
        best_fuzzy_fn,
        best_fn,
        max_fuzzy_reward,
        max_fuzzy_std,
        max_reward,
        max_std,
        results_df,
    )
    
def best_model_from_data(results: pd.DataFrame):
    best_fuzzy_arg = results.fuzzy_reward.idxmax()  # type: ignore
    best_arg = results.discrete_reward.idxmax()  # type: ignore

    max_fuzzy_reward = results.loc[best_fuzzy_arg].fuzzy_reward
    max_fuzzy_std = results.loc[best_fuzzy_arg].fuzzy_reward_std
    best_fuzzy_fn = results.loc[best_fuzzy_arg].fn
    max_reward = results.loc[best_arg].discrete_reward
    max_std = results.loc[best_arg].discrete_reward_std
    best_fn = results.loc[best_arg].fn
    return (
        best_fuzzy_fn,
        best_fn,
        max_fuzzy_reward,
        max_fuzzy_std,
        max_reward,
        max_std,
    )

def run_a_model(fn: str, args_in: argparse.Namespace, seed: Optional[int]=0, verbose:int=1):
    num_runs = 15
    if 'cart' in fn:
        env = 'cart'
    elif 'lunar' in fn:
        env = 'lunar'
    elif 'FindAndDefeatZerglings' in fn:
        env = 'FindAndDefeatZerglings'
    final_deep_actor_fn = os.path.join(MODEL_DIR, fn) if not fn.startswith(MODEL_DIR) else fn
    final_deep_critic_fn = os.path.join(MODEL_DIR, fn) if not fn.startswith(MODEL_DIR) else fn

    fda = load_ddt(final_deep_actor_fn)
    fdc = load_ddt(final_deep_critic_fn)

    policy_agent = DDTAgent(bot_name='crispytester',
                            input_dim=37,
                            output_dim=10)

    # fda.comparators.data = fda.comparators.data.unsqueeze(-1)
    policy_agent.action_network = fda
    # fsc.comparators.data = fsc.comparators.data.unsqueeze(-1)
    policy_agent.value_network = fdc
    master_states = []
    master_actions = []
    reward_after_five = 0
    for _ in range(num_runs):
        if env == 'FindAndDefeatZerglings':
            reward, replay_buffer = micro_episode(None, policy_agent, game_mode=env)
        elif env in ['cart', 'lunar']:
            reward, replay_buffer = gym_episode(None, policy_agent, env)
        else:
            raise ValueError(f"Unknown environment {env}")
        master_states.extend(replay_buffer['states'])
        master_actions.extend(replay_buffer['actions_taken'])
        reward_after_five += reward
    if verbose:
        print(f"Average reward after {num_runs} runs is {reward_after_five/num_runs:.3f}")

    master_states = torch.cat([state[0] for state in master_states], dim=0)
    if args_in.discretize:
        crispy_actor = convert_to_discrete(policy_agent.action_network)  # Discretize DDT
    else:
        ###### test with a DT #######
        x_train = [state.cpu().numpy().reshape(-1) for state in master_states]
        y_train = [action.cpu().numpy().reshape(-1) for action in master_actions]
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(x_train, y_train)
        plt.figure(figsize=(20, 20))
        plot_tree(clf, filled=True)
        plt.savefig('tree.png')
        init_weights, init_comparators, init_leaves = ddt_init_from_dt(clf)
        crispy_actor = DDT(input_dim=len(x_train[0]),
                           output_dim=len(np.unique(y_train)),
                           weights=init_weights,
                           comparators=init_comparators,
                           leaves=init_leaves,
                           alpha=99999.,
                           is_value=False,
                           use_gpu=False)
    if verbose:
        print("-----------\nCrispy:\n")

    policy_agent.action_network = crispy_actor
    crispy_reward = []
    for i in range(num_runs):
        if env == 'FindAndDefeatZerglings':
            crispy_out, replay_buffer = micro_episode(None, policy_agent, game_mode=env)
        elif env in ['cart', 'lunar']:
            if seed is None:
                seed = np.random.RandomState().randint(100000)
            torch.random.manual_seed(seed + i)
            np.random.seed(seed + i)
            crispy_out, replay_buffer = gym_episode(None, policy_agent, env, seed=seed+i)
        crispy_reward.append(crispy_out)

    leaves = crispy_actor.leaf_init_information
    for leaf_ind in range(len(leaves)):
        leaves[leaf_ind][-1] = np.argmax(leaves[leaf_ind][-1])
    if verbose > 1:
        print(leaves)
        print(crispy_actor.comparators.detach().numpy().reshape(-1))
        ddt_weights = crispy_actor.layers.detach().numpy()
        print(np.argmax(np.abs(ddt_weights), axis=1))
    if verbose:
        print(f"Average reward after {num_runs} runs is {reward_after_five/num_runs:.3f}\n"
            f"Average reward for the crispy network after {num_runs} runs is {np.mean(crispy_reward)} with std {np.std(crispy_reward):.3f}"
        )
    return reward_after_five / num_runs, np.mean(crispy_reward), np.std(crispy_reward)


def fc_state_dict(fn=''):
    fc_model = torch.load(fn)
    print(fc_model['actor'])

def test_model(discrete_fn: Path | str, seed=None, verbose:int=True, count: Optional[tuple[int, int]]=None):
    """Allows parallel execution of run_a_model"""
    if verbose:
        print("\n------------------\nTesting", discrete_fn)
    elif count is not None:
        # not a precise but estimated progress count
        print(f"{'~'+str(count[0]):>7}/{count[1]}", end="\r", flush=True)
    else:
        print(".", end="", flush=True)
    filename = discrete_fn.name if isinstance(discrete_fn, Path) else discrete_fn
    # Run model
    (
        avg_reward_diff, 
        avg_reward_discrete, 
        std_reward_discrete
    ) = run_a_model(filename, args, seed=seed, verbose=verbose)
    # Gather results
    header = match_filename(filename).groupdict()
    version = header.get("version", 99)
    version = int(version) if version is not None else 99
    index = (
        header["env"],
        header["method"],
        header["typ"],
        int(header["num"]),
        bool(header["GPU"]),
        version,
        int(header["episode"]),
    )
    return (index, (avg_reward_diff, avg_reward_discrete, std_reward_discrete))
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--discretize", help="train sklearn tree or discretize ddt?", action='store_true')
    parser.add_argument("-env", "--env_type", help="FindAndDefeatZerglings, cart, or lunar", type=str, default="cart")
    parser.add_argument("-m", "--model_dir", help="where are models stored?", default="../models", type=str)
    parser.add_argument('-f', '--find_model', help="find the best models?", action="store_true")
    parser.add_argument('-r', '--run_model', help="run a model?", action="store_true")
    parser.add_argument('-n', '--model_fn', help="model filename for running", type=str, default="")
    parser.add_argument(
        "-s", "--seed", help="Seed; use -1 for None", type=int, default=12496
    )
    parser.add_argument(
        "-a", "--all", help="Test all models; not only the best", action="store_true", default=False
    )
    parser.add_argument(
        "-v", "--verbose", help="Verbose output. If more than 500 models are checked switches to False", action="store_true", default="auto"
    )
    parser.add_argument(
        "--silent", help="No output", action="store_true", default=False
    )
    parser.add_argument(
        "-N", "--n_jobs", help="Number of jobs for parallel execution", type=int, default=25
    )
    parser.add_argument(
        "--test", "--dry-run", help="Do not save any outputs", action="store_true", default=False
    )
    
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    SEED = args.seed
    N_JOBS = args.n_jobs

    envir = args.env_type
    MODEL_DIR = args.model_dir
    # args.run_model = True
    # args.discretize = True

    if args.find_model:
        print("\nFinding model...")
        *_, results_df = search_for_good_model(
            envir,
            n_jobs=N_JOBS,
            verbose=(not args.silent and (args.verbose or not args.all)),
        )
        Path("../outputs").mkdir(exist_ok=True)
        if not args.test:
            results_df.to_csv(f"../outputs/results_{envir}.csv")
    else:
        # reuse saved data
        results_df = (pd.read_csv(f"../outputs/results_{envir}.csv")
                      .set_index(["env", "method", "sub-method", "capacity", "GPU", "version", "episode"], drop=True))
    # Query df which is the best model
    best_disc_models = {}
    for sub_method in ["leaves", "rules"]:
        sub_df = results_df[
            results_df.index.get_level_values("sub-method") == sub_method
        ]
        if len(sub_df) == 0:
            continue
        (
            best_fuzzy_fn,
            best_fn,
            max_fuzzy_reward,
            max_fuzzy_std,
            disc_reward,
            disc_std,
        ) = best_model_from_data(sub_df)
        print(
            f"Best differentiable {sub_method} file: {best_fuzzy_fn} with {max_fuzzy_reward} reward and {max_fuzzy_std} std"
        )
        print(
            f"Best discrete {sub_method} file: {best_fn} with {disc_reward} reward and {disc_std} std"
        )
        best_disc_models[sub_method] = os.path.join(MODEL_DIR, best_fn)
    if args.run_model:
        print("\nRunning model...")
        # preselected or best
        if args.model_fn:
            models = [args.model_fn]
        elif args.all:
            models = results_df.fn.values
        else:  # only best
            models = best_disc_models.values()
        # cartpole random seeds include: [11421, 12494, 12495, 12496,
        # 30867, 30868, 30869, 30870, 30871, 30872, 34662, 38979, 38980, 45603, 45604, 45605, 45606, 46760, 46761,
        # 50266, 50267, 54857, 65926, 70614, 79986, 79987, 79988, 79989]
        best_results = []
        if args.all:  # execute parallel
            total = len(models)
            eval_functions = [
                delayed(test_model)(discrete_fn, SEED, False, (i, total)) for i, discrete_fn in enumerate(models, 1)
            ]
            results = Parallel(n_jobs=25)(eval_functions)
            print("\n")
        else:
            results = [test_model(discrete_fn) for discrete_fn in models]
        for (index, (avg_reward_diff, avg_reward_discrete, std_reward_discrete)) in results:
            results_df.loc[index, "test_diff_reward"] = round(avg_reward_diff, 3)
            results_df.loc[index, "test_disc_reward"] = round(avg_reward_discrete, 3)
            results_df.loc[index, "test_disc_std"] = round(std_reward_discrete, 3)
            best_results.append(index)
        if not args.test:
            results_df.to_csv(f"../outputs/results_{envir}.csv")
        if args.all:
            print("\nAll results:\n", results_df[["fn", "test_diff_reward", "test_disc_reward", "test_disc_std"]].sort_values("test_disc_reward", ascending=False))
        else:
            print("\nBest results:\n", results_df.loc[best_results, ["fn", "test_diff_reward", "test_disc_reward", "test_disc_std"]])
