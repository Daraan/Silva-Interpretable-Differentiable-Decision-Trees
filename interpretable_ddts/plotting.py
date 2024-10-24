# %%
import re
from typing import Generator, Iterable, Union
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

sns.set_theme()

from interpretable_ddts import tools
from interpretable_ddts.tools import load_rewards

rewards_dir = Path('txts')
model_dir = Path('models')


version = -1
methods = "ddt", "mlp"
ENV = "lunar"

# %%

CART_CSV = "outputs/results_cart_recreation.csv"
LUNAR_CSV = "outputs/results_lunar_recreation.csv"
SC_CSV = None


def scatter_fuzzy_discrete(df: pd.DataFrame, ax=None, *, max_reward=500, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if "edgecolors" not in kwargs and mpl.__version__ < "3.7.":
        kwargs["edgecolors"] = "fill"
    sns.scatterplot(
        data=df,
        ax=ax,
        x="fuzzy_reward",
        y="discrete_reward",
        style="sub-method",
        #edgecolors="fill",
        linewidth=0,
        hue="capacity",
        palette="viridis_r",
        alpha=0.5,
        #size="episode",
        **kwargs,
    )
    min_reward = min(df["fuzzy_reward"].min(), df["discrete_reward"].min(), 0)
    
    ax.set_ylim(min_reward - 10, max_reward + 10)
    ax.set_xlim(min_reward - 10, max_reward + 10)
    
    return fig, ax

scatter_fuzzy_discrete(tools.load_output(CART_CSV, aggregate_version="mean"))
plt.show()
scatter_fuzzy_discrete(tools.load_output(CART_CSV, aggregate_version="max"))


# %%
files = list(rewards_dir.glob("*.txt"))
data = load_rewards(files)

envs = data.index.get_level_values("env").unique()

env_data = {
    env : data.xs(env, level="env")
    for env in envs
}

data = pd.DataFrame()
for method in methods:
    rewards = rewards_dir.glob(f"{method}{ENV}*rewards.txt")
    dfs = []
    for reward in rewards:
        df = pd.read_csv(reward, header=None).T
        version = int(reward.stem.split("_v")[-1].split("_")[0])
        df.set_index(pd.MultiIndex.from_tuples([(ENV, method, version)], 
                                               names=["env", "method", "version"]), 
                     inplace=True)
        dfs.append(df)
    method_df= pd.concat(dfs, axis=0)
    data = pd.concat([data, method_df], axis=0)

data_grouped = data.groupby("method")
avg_data: pd.DataFrame = data_grouped.mean()
# group in bins of 50
averages_grouped = avg_data.T.groupby(pd.cut(avg_data.columns, bins=1000//50, include_lowest=True)).mean()
errors = data_grouped.std()
errors_grouped = errors.T.groupby(pd.cut(errors.columns, bins=1000//50, include_lowest=True)).mean()

#SELECTION = range(0, 1000, 50)
#data_sel = averages_grouped[SELECTION]
#error_sel = errors[SELECTION].T
data_sel = averages_grouped
error_sel = errors_grouped
new_index = list(data_sel.index.map(lambda x: x.right))
data_sel = data_sel.reindex(new_index)
error_sel = error_sel.reindex(new_index)

fig, ax = plt.subplots()
# use scatter and no line
# average over 50 x values do have less random outliers that might lie
data_sel.plot(yerr=error_sel, ax=ax, marker="s", linestyle=":")
for method in methods:
    avg: "pd.Series[float]" = avg_data.T[method]
    err: "pd.Series[float] "= errors.T[method]
    ax.fill_between(avg_data.columns, avg - err, avg + err, alpha=0.2 if method =="mlp" else 0.8)  # pyright: ignore[reportArgumentType]


# Do without averaging
# take best mlp and best ddt

# %%