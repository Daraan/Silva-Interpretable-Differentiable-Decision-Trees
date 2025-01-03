# Created by Andrew Silva on 5/10/19
from __future__ import annotations

from datetime import datetime
import re
import logging
from pathlib import Path
from typing import Optional, Sequence, TypedDict, cast, overload
from typing_extensions import Literal, NotRequired
from joblib import Parallel, delayed
import pandas as pd
import torch
import numpy as np
import os
import gymnasium as gym

from interpretable_ddts.opt_helpers.discretization import convert_to_discrete
from interpretable_ddts.agents.ddt_agent import DDTAgent
from interpretable_ddts.agents.ddt import DDT
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from interpretable_ddts.opt_helpers.sklearn_to_ddt import ddt_init_from_dt
import matplotlib.pyplot as plt

try:
    from interpretable_ddts.runfiles.sc2_minigame_runner import run_episode as sc_episode
except ModuleNotFoundError as e:
    logging.error("Cannot import Starcraft due to %s", e)
from interpretable_ddts.runfiles.gym_runner import run_episode as gym_episode
from interpretable_ddts.tools import RE_PARSE_FILENAME_OLD, create_df_index, match_filename, seed_everything

RE_PARSE_FILENAME = re.compile(
    r"(?P<parent_dir>.+?/)?"
    r"(?P<episode>\d+)th"
    r"(?P<method>ddt|mlp)"
    r"(?P<env>[^_]+?)(?P<GPU>GPU)?"
    r"_(?P<features>(?P<num>\d+)_(?P<typ>[^_]+))"
    r"_actor"
    r"_v(?P<version>\d+)",
)


class ResultDict(TypedDict):
    fn: str
    fuzzy_reward: float  # np.mean(reward_after_five)
    fuzzy_reward_std: float  # np.std(reward_after_five),
    discrete_reward: float  # np.mean(crispy_reward)
    discrete_reward_std: float  # np.std(crispy_reward)
    discrete_model: NotRequired[DDTAgent]


def create_gym_env(env: str | gym.Env, *, render_mode=None):
    if not isinstance(env, gym.Env):
        if env == "lunar":
            gym_env = gym.make("LunarLander-v2", render_mode=render_mode)
        elif env == "cart":
            gym_env = gym.make("CartPole-v1", render_mode=render_mode)
        elif env == "FindAndDefeatZerglings":
            return None
        else:
            gym_env = gym.make(env, render_mode=render_mode)
    else:
        gym_env = env
    return gym_env


def load_agent(fn: str | Path, bot_name="crispytester"):
    final_deep_actor_fn = os.path.join(MODEL_DIR, fn) if not str(fn).startswith(MODEL_DIR) else fn
    policy_agent = DDTAgent(
        bot_name=bot_name,
        # Dimensions do not matter as network is replaced
        input_dim=2,
        output_dim=2,
    )
    policy_agent.load(final_deep_actor_fn, auto_naming=False)
    return policy_agent


def search_for_good_model(env, n_jobs=5, verbose=1):
    # Be sure to comment out gym_runner.gym_episode env.render
    max_reward = -float("inf")
    max_std = -float("inf")
    max_fuzzy_reward = -float("inf")
    max_fuzzy_std = -float("inf")
    best_fn = "non"
    best_fuzzy_fn = "non"

    model_path = Path(MODEL_DIR)
    files = [
        fn
        for fn in model_path.glob(f"**/*ddt*{env}*actor*")
        if not fn.parent.name.startswith("models")  # excluded subdirs
    ]
    total = len(files)
    if total == 0:
        print("No results found in", model_path, "subdirs excluded. Exiting...")
        import sys

        sys.exit(1)
    if verbose:
        print(f"Found {total} models")
    _max_models_for_verbose = 50
    if total >= _max_models_for_verbose and verbose == "auto":
        print(f"Turning off full verbose output for more than {_max_models_for_verbose} models")
        verbose = False
    elif verbose == "auto":
        verbose = True
    delayed_functions = [
        delayed(evaluate_model)(
            fn.relative_to(model_path),
            env=env,
            verbose=verbose,
            parallel_count=(i, total),
            run_discrete=True,
        )
        for i, fn in enumerate(files, 1)
    ]
    filenames = [fn.relative_to(model_path).name for fn in files]
    if n_jobs > 1:
        all_results_ = cast("list[ResultDict | None]", Parallel(n_jobs=n_jobs)(delayed_functions))
    else:
        all_results_ = [foo[0](*foo[1], **foo[2]) for foo in delayed_functions]
    if not all(all_results_):
        all_results: "list[ResultDict]" = list(
            filter(None, all_results_),
        )  # filter potential FileNotFound
        filenames = [fn for fn, res in zip(filenames, all_results_) if res]
    else:
        all_results = cast("list[ResultDict]", all_results_)
    if not verbose:
        print("\n")
    parsed_filenames = [(RE_PARSE_FILENAME.match(file) or RE_PARSE_FILENAME_OLD.match(file)) for file in filenames]
    if not all(parsed_filenames):
        unparsable_names = [filename for filename, m in zip(filenames, parsed_filenames) if m is None]
        logging.error("Cannot parse filenames: %s", ", ".join(unparsable_names))
        all_results = [result for result, m in zip(all_results, parsed_filenames) if m]
    parsed_filenames = filter(None, parsed_filenames)
    metadata = [file.groupdict() for file in parsed_filenames]
    results_df_unsorted = pd.DataFrame(all_results)
    results_df_unsorted.index = create_df_index(metadata)  # do not sort before
    results_df = results_df_unsorted.sort_values("discrete_reward", ascending=False)

    best_fuzzy_arg: int = results_df.fuzzy_reward.argmax()  # type: ignore
    best_arg: int = results_df.discrete_reward.argmax()  # type: ignore

    max_fuzzy_reward = all_results[best_fuzzy_arg]["fuzzy_reward"]
    max_fuzzy_std = all_results[best_fuzzy_arg]["fuzzy_reward_std"]
    best_fuzzy_fn = all_results[best_fuzzy_arg]["fn"]
    max_reward = all_results[best_arg]["discrete_reward"]
    max_std = all_results[best_arg]["discrete_reward_std"]
    best_fn = all_results[best_arg]["fn"]
    best_discrete_model = all_results[best_arg].get("discrete_model")

    return (
        best_fuzzy_fn,
        best_fn,
        max_fuzzy_reward,
        max_fuzzy_std,
        max_reward,
        max_std,
        best_discrete_model,
        results_df,  # This should be last!
    )


def best_model_from_data(results: pd.DataFrame) -> tuple[str, str, float, float, float, float]:
    best_fuzzy_arg = results.fuzzy_reward.idxmax()
    best_arg = results.discrete_reward.idxmax()

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
    )  # pyright: ignore[reportReturnType]


@overload
def evaluate_model(
    fn: str,
    *,
    env: Optional[str | gym.Env] = None,
    seed: Optional[int] = 0,
    verbose: int = 1,
    render_mode=None,
    run_discrete: Literal[True] = True,
    classic_decision_tree=False,
    parallel_count: Optional[tuple[int, int]] = None,
) -> ResultDict: ...


@overload
def evaluate_model(
    fn: str,
    *,
    env: Optional[str | gym.Env] = None,
    seed: Optional[int] = 0,
    verbose: int = 1,
    render_mode=None,
    run_discrete: Literal[False],
    classic_decision_tree=False,
    parallel_count: Optional[tuple[int, int]] = None,
) -> float: ...


def evaluate_model(
    fn: str,
    *,
    env: Optional[str | gym.Env] = None,
    seed: Optional[int] = None,
    verbose: int = 1,
    render_mode=None,
    run_discrete=True,
    classic_decision_tree=False,
    parallel_count: Optional[tuple[int, int]] = None,
) -> ResultDict | float:
    num_runs = 15
    if env is None:
        if "cart" in fn:
            env = "cart"
        elif "lunar" in fn:
            env = "lunar"
        elif "FindAndDefeatZerglings" in fn:
            env = "FindAndDefeatZerglings"
        else:
            raise ValueError(f"Unknown environment used in {fn}")

    gym_env = create_gym_env(env, render_mode=render_mode)
    if gym_env and seed is not None:
        gym_env.reset(seed=seed)  # NOTE: The observation from this reset is not used.
        seed_everything(env=None, seed=seed, torch_manual=True)  # needed here to be reproducible

    policy_agent = load_agent(fn, bot_name="crispytester")
    policy_agent.value_network = policy_agent.action_network  # XXX: Original setup; wrong; unused?

    master_states = []
    master_actions = []
    rewards_after_five = []
    for _ in range(num_runs):
        if env == "FindAndDefeatZerglings":
            try:
                reward, replay_buffer = sc_episode(None, policy_agent, game_mode="FindAndDefeatZerglings")  # type: ignore[unbound]
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                logging.exception("Error in micro_episode")
                continue
        else:
            assert gym_env
            reward, replay_buffer = gym_episode(None, gym_env, policy_agent, render_mode=render_mode)
        master_states.extend(replay_buffer["states"])
        master_actions.extend(replay_buffer["actions_taken"])
        rewards_after_five.append(reward)
    mean_reward_after_five = np.mean(rewards_after_five).item()
    if verbose:
        print(f"Average reward after {num_runs} runs is {mean_reward_after_five:.3f}")

    # Run Discrete
    if run_discrete:
        master_states = torch.cat([state[0] for state in master_states], dim=0)
        if not classic_decision_tree:
            crispy_actor = convert_to_discrete(policy_agent.action_network)  # Discretize DDT
        else:
            ###### test with a DT #######
            x_train = [state.cpu().numpy().reshape(-1) for state in master_states]
            y_train = [action.cpu().numpy().reshape(-1) for action in master_actions]
            clf = DecisionTreeClassifier(max_depth=3)
            clf.fit(x_train, y_train)
            plt.figure(figsize=(20, 20))
            plot_tree(clf, filled=True)
            plt.savefig("tree.png")
            init_weights, init_comparators, init_leaves = ddt_init_from_dt(clf)
            crispy_actor = DDT(
                input_dim=len(x_train[0]),
                output_dim=len(np.unique(y_train)),
                weights=init_weights,
                comparators=init_comparators,
                leaves=init_leaves,
                alpha=99999.0,
                is_value=False,
                use_gpu=False,
            )
        if verbose:
            print("-----------\nCrispy:\n")

        policy_agent.action_network = crispy_actor
        crispy_reward = []
        if gym_env and seed is not None:
            gym_env.reset(seed=seed)  # NOTE: The observation from this reset is not used.
            seed_everything(env=None, seed=seed, torch_manual=True)
        for _ in range(num_runs):
            if env == "FindAndDefeatZerglings":
                try:
                    crispy_out, replay_buffer = sc_episode(None, policy_agent, "FindAndDefeatZerglings")  # type: ignore[unbound]
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    logging.exception("Error in micro_episode")
                    crispy_out = -3
                    continue
            else:
                assert gym_env
                crispy_out, replay_buffer = gym_episode(None, gym_env, policy_agent)

            crispy_reward.append(crispy_out)

        if verbose > 1:
            # For printing select the most chosen leaf
            leaves = crispy_actor.leaf_init_information
            for leaf_ind in range(len(leaves)):
                leaves[leaf_ind] = (*leaves[leaf_ind][:-1], np.argmax(leaves[leaf_ind][-1]).item())
            print(leaves)
            print(crispy_actor.comparators.detach().numpy().reshape(-1))
            ddt_weights = crispy_actor.layers.detach().numpy()
            print(np.argmax(np.abs(ddt_weights), axis=1))
    else:
        crispy_reward = None
    if verbose:
        msg = f"Average reward after {num_runs} runs is {mean_reward_after_five:.3f}\n"
        if run_discrete and crispy_reward is not None:
            msg += (
                f"Average reward for the crispy network after {num_runs} runs is {np.mean(crispy_reward)} "
                f"with std {np.std(crispy_reward):.3f}"
            )
        print(msg)
    elif parallel_count is not None:
        # not a precise but estimated progress count
        print(f"{'~'+str(parallel_count[0]):>9}/{parallel_count[1]}", end="\r", flush=True)
    else:
        print(".", end="", flush=True)
    if run_discrete and crispy_reward is not None:
        return ResultDict(
            fn=str(fn),
            fuzzy_reward=mean_reward_after_five,
            fuzzy_reward_std=np.std(rewards_after_five).item(),
            discrete_reward=np.mean(crispy_reward).item(),
            discrete_reward_std=np.std(crispy_reward).item(),
            discrete_model=policy_agent,
        )
    return mean_reward_after_five


def fc_state_dict(fn=""):
    fc_model = torch.load(fn)
    print(fc_model["actor"])


def test_model(
    discrete_fn: Path | str,
    *,
    seed: Optional[int] = None,
    verbose: int = True,
    count: Optional[tuple[int, int]] = None,
):
    """Allows parallel execution of run_a_model"""
    if verbose:
        print("\n------------------\nTesting", discrete_fn)
    elif count is not None:
        # not a precise but estimated progress count
        print(f"{'~'+str(count[0]):>9}/{count[1]}", end="\r", flush=True)
    else:
        print(".", end="", flush=True)
    filename = discrete_fn.name if isinstance(discrete_fn, Path) else discrete_fn
    # Run model
    result = evaluate_model(
        filename,
        env=None,
        seed=seed,
        verbose=verbose,
        run_discrete=True,
        classic_decision_tree=not args.discretize,
        parallel_count=None,  # print count here
    )
    # Gather results
    match = match_filename(filename)
    if match:
        header: dict[str, str] = match.groupdict()
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
    else:
        index = None
        raise ValueError(f"{filename} does not match pattern")
    return (index, result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--discretize", help="train sklearn tree or discretize ddt?", action="store_true")
    parser.add_argument("-env", "--env_type", help="FindAndDefeatZerglings, cart, or lunar", type=str, default="cart")
    parser.add_argument("-m", "--model_dir", help="where are models stored?", default="../models", type=str)
    parser.add_argument(
        "-f",
        "--find_model",
        nargs="?",
        const="DEFAULT",
        help="find the best models?",
    )
    parser.add_argument(
        "--csv",
        help="If not using find_model which csv to use",
        type=str,
        required=False,
        default="DEFAULT",
    )
    parser.add_argument("-r", "--run_model", help="run a model?", action="store_true")
    parser.add_argument("-n", "--model_fn", help="model filename for running", type=str, default="")
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed; use -1 for None",
        type=int,
        default=12496,
    )
    parser.add_argument(
        "-a",
        "--all",
        help="Test all models; not only the best",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose output. If more than 500 models are checked switches to False",
        action="store_true",
        default="auto",
    )
    # fmt: off
    parser.add_argument(
        "--silent", help="No output", action="store_true", default=False,
    )
    parser.add_argument(
        "-N", "--n_jobs", help="Number of jobs for parallel execution", type=int, default=25,
    )
    parser.add_argument(
        "--test", "--dry-run", help="Do not save any outputs", action="store_true", default=False,
    )
    # fmt: on

    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    SEED: Optional[int] = args.seed
    N_JOBS = args.n_jobs

    envir = args.env_type
    MODEL_DIR: str = args.model_dir
    # args.run_model = True
    # args.discretize = True

    if args.find_model:
        if args.find_model == "DEFAULT":
            outfile = f"../outputs/results_{envir}_{datetime.now():%Y-%m-%d %H:%M}.csv"
        else:
            outfile = f"../outputs/results_{envir}_{args.find_model}.csv"
        print("\nFinding model... storing results in", outfile if not args.test else "test_discrete.csv")
        *_, results_df = search_for_good_model(
            envir,
            n_jobs=N_JOBS,
            verbose=(not args.silent and (args.verbose or not args.all)),
        )
        Path("../outputs").mkdir(exist_ok=True)
        if not args.test:
            results_df.to_csv(outfile)
        else:
            results_df.to_csv("test_discrete.csv")
    else:
        if args.csv == "DEFAULT":
            # reuse saved data
            outfile = f"../outputs/results_{envir}.csv"
            logging.warning("Loading data from unspecific csv: %s", outfile)
        else:
            outfile = f"../outputs/{args.csv}"
        if args.test:
            outfile = "test_discrete.csv"
        print("\nLoading stored data from", outfile)
        results_df = pd.read_csv(outfile).set_index(
            ["env", "method", "sub-method", "capacity", "GPU", "version", "episode"],
            drop=True,
        )
    # Query df which is the best model
    best_disc_models = {}
    for sub_method in ["leaves", "rules"]:
        sub_df = results_df[results_df.index.get_level_values("sub-method") == sub_method]
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
            f"Best differentiable {sub_method} file: {best_fuzzy_fn} with {max_fuzzy_reward} reward "
            f"and {max_fuzzy_std} std",
        )
        print(
            f"Best discrete {sub_method} file: {best_fn} with {disc_reward} reward and {disc_std} std",
        )
        best_disc_models[sub_method] = os.path.join(MODEL_DIR, best_fn)
    if args.run_model:
        print("\nRunning model...")
        # preselected or best
        if args.model_fn:
            models = [args.model_fn]
        elif args.all:
            models = results_df.fn.to_numpy()
        else:  # only best
            models = best_disc_models.values()
        models = cast(Sequence[str], models)
        # cartpole random seeds include: [11421, 12494, 12495, 12496,
        # 30867, 30868, 30869, 30870, 30871, 30872, 34662, 38979, 38980, 45603, 45604, 45605, 45606, 46760, 46761,
        # 50266, 50267, 54857, 65926, 70614, 79986, 79987, 79988, 79989]
        best_results = []
        if args.all:  # execute parallel
            total = len(models)
            eval_functions = [
                delayed(test_model)(discrete_fn, seed=SEED, verbose=False, count=(i, total))
                for i, discrete_fn in enumerate(models, 1)
            ]
            results = Parallel(n_jobs=25)(eval_functions)
            # Assume Parallel return_as="list"
            results = cast(list[tuple[tuple[str, str, str, int, bool, int, int], ResultDict]], results)
            print("\n")
        else:
            results = [test_model(discrete_fn, seed=SEED) for discrete_fn in models]
        for index, result in results:
            results_df.loc[index, "test_diff_reward"] = round(result["fuzzy_reward"], 3)
            results_df.loc[index, "test_disc_reward"] = round(result["discrete_reward"], 3)
            results_df.loc[index, "test_disc_std"] = round(result["discrete_reward_std"], 3)
            best_results.append(index)
        if not args.test:
            results_df.to_csv(outfile)
        else:
            results_df.to_csv("../test_discrete.csv")
        if args.all:
            print(
                "\nAll results:\n",
                results_df[["fn", "test_diff_reward", "test_disc_reward", "test_disc_std"]].sort_values(
                    "test_disc_reward",
                    ascending=False,
                ),
            )
        else:
            print(
                "\nBest results:\n",
                results_df.loc[best_results, ["fn", "test_diff_reward", "test_disc_reward", "test_disc_std"]],
            )
