from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import re
import time
from typing import Iterable, Literal, TypeVar, Union, Optional, TYPE_CHECKING, overload
import pandas as pd
import random
import numpy as np
import torch
import torch.cuda
from tqdm import tqdm
from ray.experimental import tqdm_ray

from interpretable_ddts.runfiles.constants import COMET_OFFLINE_DIRECTORY

if TYPE_CHECKING:
    from ray.tune.callback import Trial
    from pandas._typing import AggFuncTypeBase
    from typing_extensions import TypeIs

import gymnasium as gym

from packaging.version import parse as parse_version, Version

GYM_VERSION = parse_version(gym.__version__)

GYM_VERSION = parse_version(gym.__version__)
GYM_V_0_26 = GYM_VERSION >= Version("0.26")
"""First gymnasium version and above"""
GYM_V1 = GYM_VERSION >= Version("1.0.0")

RE_PARSE_FILENAME = re.compile(
    r"(?P<parent_dir>.+/)?"  # likely for model files
    r"(?:(?P<episode>\d+)th)?"  # model files only
    r"(?P<method>ddt|mlp)"
    r"(?P<env>[^_]+?)(?P<GPU>GPU)?"
    r"_(?P<features>(?P<num>\d+)_(?P<typ>[^_]+))"
    r"(?(episode)_actor|)"
    r"_v(?P<version>\d+)",
)

RE_PARSE_FILENAME_OLD = re.compile(
    r"(?P<parent_dir>.+?/)?"  # likely for model files
    r"(?:(?P<episode>\d+)th)?"  # model files only
    r"(?P<method>ddt|mlp)"
    r"(?P<env>[^_]+?)(?P<GPU>GPU)?"
    r"_?(?P<features>(?P<num>\d+)_(?P<typ>[^_]+))"  # no _ before features in old format
    r"(?(episode)_actor_|)"
    r"(?:_v(?P<version>\d+))?",  # no version in old format
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
                    ),
                ],
                names=["env", "method", "sub-method", "capacity", "GPU", "version"],
            ),
        )
        for file, header in zip(files, headers)
    )
    data = pd.concat(objs).sort_index()
    return data


def load_output(
    file: Union[str, Path],
    index=("env", "method", "sub-method", "capacity", "GPU", "version"),
    aggregate_version: Optional[Literal["max", "mean"] | AggFuncTypeBase] = None,
    aggregate_column: str = "discrete_reward",
    **kwargs,
):
    df = pd.read_csv(file, index_col=index, **kwargs)
    if not aggregate_version:
        return df
    df_2 = df.reset_index().set_index([*index, "episode"])
    if isinstance(aggregate_version, (str, list, Iterable)) and "mean" in aggregate_version:
        df_2 = df_2.drop(columns=["fn"])
    agg_df = (  # noqa: RET504
        df_2.groupby(list(index))
        .aggregate(
            aggregate_version,
        )
        .sort_values(aggregate_column, ascending=False)
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
            int(
                header.get("version", 99) if header.get("version", 99) is not None else 99,
            ),  # no version in old format
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
                int(
                    header.get("version", 99) if header.get("version", 99) is not None else 99,
                ),  # no version in old format
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


@overload
def _split_seed(seed: None) -> tuple[None, None]: ...


@overload
def _split_seed(seed: int) -> tuple[int, int]: ...


def _split_seed(seed: Optional[int]) -> tuple[int, int] | tuple[None, None]:
    if seed is None:
        return None, None
    gen = random.Random(seed)
    return gen.randrange(2**32), gen.randrange(2**32)


def seed_everything(env, seed: Optional[int], *, torch_manual=False):
    """
    Args:
        torch_manual: If True, will set torch.manual_seed and torch.cuda.manual_seed_all
            In some cases setting this causes bad models, so it is False by default
    """
    # no not reuse seed if its not None
    seed, next_seed = _split_seed(seed)
    random.seed(seed)

    seed, next_seed = _split_seed(next_seed)
    np.random.seed(seed)

    # os.environ["PYTHONHASHSEED"] = str(seed)
    if next_seed is None:
        torch.seed()
        torch.cuda.seed()
    elif torch_manual:
        seed, next_seed = _split_seed(next_seed)
        torch.manual_seed(
            seed,
        )  # setting torch manual seed causes bad models, # ok seed 124
        seed, next_seed = _split_seed(next_seed)
        torch.cuda.manual_seed_all(seed)
    if env:
        if not GYM_V_0_26:  # gymnasium does not have this
            seed, next_seed = _split_seed(next_seed)
            env.seed(seed)
        seed, next_seed = _split_seed(next_seed)
        env.action_space.seed(seed)

    seed, next_seed = _split_seed(next_seed)
    return seed, next_seed


_T = TypeVar("_T")


def is_pbar(pbar: Iterable[_T]) -> TypeIs[tqdm_ray.tqdm | tqdm[_T]]:
    return isinstance(pbar, (tqdm_ray.tqdm, tqdm))


def comet_upload_offline_experiments():
    import comet_ml
    import logging

    archives = list(map(str, Path(COMET_OFFLINE_DIRECTORY).glob("*.zip")))
    if not archives:
        logging.info("No archives to upload")
        return
    logging.info("Uploading Archives: %s", archives)
    comet_ml.offline.main_upload(archives, force_upload=False)
    new_dir = Path(COMET_OFFLINE_DIRECTORY) / "uploaded"
    new_dir.mkdir(exist_ok=True)
    for path in Path(os.environ["COMET_OFFLINE_DIRECTORY"]).glob("*.zip"):
        path.rename(new_dir / path.name)


# ----

RUN_MIN_LENGTH = 1000
"""Amounts will lower lines are considered incomplete"""

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
            if len(txt) < RUN_MIN_LENGTH:
                print(f"Removing {file}")
                stem = file.stem.split("_rewards")[0]
                models = list(model_dir.glob(f"*th{stem}*"))
                clean_files.extend(models)
                clean_files.append(file)
        print("Files to remove:", clean_files)
        do_clean = input("Do you want to remove these files? (y/n): ").lower()
        if do_clean == "y":
            for file in clean_files:
                file.unlink()
        else:
            print("Aborted")

_SCRIPT_TIMESTAMP = time.time()


def trial_name_creator(trial: Trial) -> str:
    start_time = datetime.datetime.fromtimestamp(trial.run_metadata.start_time or _SCRIPT_TIMESTAMP)
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M")
    return "_".join(
        [
            trial.trainable_name,
            trial.evaluated_params["env"],
            trial.evaluated_params["module"],
            trial.trial_id,
            start_time_str,
        ]
    )
