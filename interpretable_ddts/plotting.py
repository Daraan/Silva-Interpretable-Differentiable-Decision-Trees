# %%
import re
from typing import Generator, Iterable, Union
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

rewards_dir = Path('txts')
model_dir = Path('models')


version = -1
methods = "ddt", "mlp"
ENV = "lunar"

RE_PARSE_FILENAME = re.compile(
    r"(?P<method>ddt|mlp)"
    r"(?P<env>[^_]+?)(?P<GPU>GPU)?"
    r"_(?P<features>(?P<num>\d+)_(?P<typ>[^_]+))"
    r"_v(?P<version>\d+)"
)

def parse_filename(filename: Union[str, Path]):
    if isinstance(filename, Path):
        filename = filename.name
    filename = filename.split("/")[-1]
    result = RE_PARSE_FILENAME.match(filename)
    assert result is not None, f"Filename {filename} does not match pattern"
    data = result.groupdict()
    data["version"] = int(data["version"])
    data["num"] = int(data["num"])
    del data["features"]
    data["GPU"] = bool(data["GPU"])
    data["typ"] = "hidden layers" if data["typ"] == "hid" else data["typ"]
    return data

def load_data(files: Iterable[Union[str, Path]]):
    files = list(files)
    headers = list(map(parse_filename, files))
    print(files)
    objs = (
        pd.read_csv(file, header=None).T.set_index(
            # create_df_index
            pd.MultiIndex.from_tuples(
                [
                    (
                        header["env"],
                        header["method"],
                        header["typ"],
                        int(header["num"]),
                        bool(header["GPU"]),
                        int(header["version"]),
                    )
                ],
                names=["env", "method", "sub-method", "capacity", "GPU", "version"],
            )
        )
        for file, header in zip(files, headers)
    )
    data = pd.concat(objs)
    data.sort_index(inplace=True,)
    return data

files = list(rewards_dir.glob("*.txt"))
data = load_data(files)

envs = data.index.get_level_values("env").unique()

env_data = {
    env : data.xs(env, level="env")
    for env in envs
}

# %%

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