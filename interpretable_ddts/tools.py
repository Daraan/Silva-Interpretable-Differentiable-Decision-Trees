from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable, Literal, Union, Optional, TYPE_CHECKING
import pandas as pd
import random
import numpy as np
import torch
if TYPE_CHECKING:
    from pandas._typing import AggFuncTypeBase
    
import gymnasium as gym
    
from packaging.version import parse as parse_version, Version
GYM_VERSION = parse_version(gym.__version__)

GYM_VERSION = parse_version(gym.__version__)
GYM_V_0_26 = GYM_VERSION >= Version("0.26")
"""First gymnasium version and above"""
GYM_V1 = GYM_VERSION >= Version("1.0.0")

RE_PARSE_FILENAME = re.compile(
    r"(?P<parent_dir>.+/)?"  # likely for model files
    r"(?:(?P<episode>\d+)th)?"        # model files only
    r"(?P<method>ddt|mlp)"
    r"(?P<env>[^_]+?)(?P<GPU>GPU)?"
    r"_(?P<features>(?P<num>\d+)_(?P<typ>[^_]+))"
    r"(?(episode)_actor|)"
    r"_v(?P<version>\d+)"
)

RE_PARSE_FILENAME_OLD = re.compile(
    r"(?P<parent_dir>.+?/)?"  # likely for model files
    r"(?:(?P<episode>\d+)th)?"  # model files only
    r"(?P<method>ddt|mlp)"
    r"(?P<env>[^_]+?)(?P<GPU>GPU)?"
    r"_?(?P<features>(?P<num>\d+)_(?P<typ>[^_]+))"  # no _ before features in old format
    r"(?(episode)_actor_|)"
    r"(?:_v(?P<version>\d+))?"  # no version in old format
)

def match_filename(filename: str) -> "re.Match[str] | None":
    result = RE_PARSE_FILENAME.match(filename)
    if result is None:
        result = RE_PARSE_FILENAME_OLD.match(filename)
    return result

def parse_filename(filename: Union[str, Path]):
    if isinstance(filename, Path):
        filename = filename.name
    filename = filename.split("/")[-1]
    result = match_filename(filename)
    assert result is not None, f"Filename {filename} does not match pattern"
    data = result.groupdict()
    data["version"] = int(data["version"])
    data["num"] = int(data["num"])
    del data["features"]
    data["GPU"] = bool(data["GPU"])
    data["typ"] = "hidden layers" if data["typ"] == "hid" else data["typ"]
    return data


def load_rewards(files: Iterable[Union[str, Path]]):
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
    data.sort_index(
        inplace=True,
    )
    return data


def load_output(
    file: Union[str, Path],
    index=("env", "method", "sub-method", "capacity", "GPU", "version"),
    aggregate_version: Optional[Literal["max", "mean"] | AggFuncTypeBase] = None,
    aggregate_column: str="discrete_reward",
    **kwargs,
):
    df = pd.read_csv(file, index_col=index, **kwargs)
    if not aggregate_version:
        return df
    df_2 = df.reset_index().set_index([*index, "episode"])
    try:
        if "mean" in aggregate_version:
            df_2.drop(columns=["fn"], inplace=True)
    except TypeError:
        pass
    agg_df = (
        df_2.groupby(list(index)).aggregate(
            aggregate_version
        ).sort_values(aggregate_column, ascending=False)
    )
    return agg_df
    
    
def create_single_index(header: dict[str, str]):
    return pd.MultiIndex(
        (
                header["env"],
                header["method"],
                header["typ"],
                int(header["num"]),
                bool(header["GPU"]),
                int(header.get("version", 99) if header.get("version", 99) is not None else 99),  # no version in old format
                int(header["episode"]),
        ),
         names=[
            "env",
            "method",
            "sub-method",
            "capacity",
            "GPU",
            "version",
            "episode",
        ],
    )
    

def create_df_index(metadata: Iterable[dict[str, str]]):
    return pd.MultiIndex.from_tuples(
        tuples=[
            (
                header["env"],
                header["method"],
                header["typ"],
                int(header["num"]),
                bool(header["GPU"]),
                int(header.get("version", 99) if header.get("version", 99) is not None else 99),  # no version in old format
                int(header["episode"]),
            )
            for header in metadata
        ],
        names=[
            "env",
            "method",
            "sub-method",
            "capacity",
            "GPU",
            "version",
            "episode",
        ],
    )


def seed_everything(env, seed, torch_manual=False):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    if seed is None:
        torch.seed()
        torch.cuda.seed()
    elif torch_manual:
        torch.manual_seed(
            seed
        )  # setting torch manual seed causes bad models, # ok seed 124
        torch.cuda.manual_seed_all(seed)
    #
    if env:
        if not GYM_V_0_26:  # gymnasium does not have this
            env.seed(seed)
        env.action_space.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    
    if args.clean:
        print("Cleaning up...")
        rewards_dir = Path("txts/")
        model_dir = Path("models/")
        clean_files = []
        for file in rewards_dir.glob("*.txt"):
            txt = file.read_text().split("\n")
            if len(txt) < 1000:
                print(f"Removing {file}")
                stem = file.stem.split("_rewards")[0]
                models = list(model_dir.glob(f"*th{stem}*"))
                if models:
                    for model in models:
                        clean_files.append(model)
                clean_files.append(file)
        print("Files to remove:", clean_files)
        do_clean = input("Do you want to remove these files? (y/n): ").lower()
        if do_clean == "y":
            for file in clean_files:
                file.unlink()
        else:
            print("Aborted")
    
                