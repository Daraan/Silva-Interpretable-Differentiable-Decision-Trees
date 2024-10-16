from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable, Union
import pandas as pd
import random
import numpy as np
import torch

RE_PARSE_FILENAME = re.compile(
    r"(?P<parent_dir>.+?/)?"  # likely for model files
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

def match_filename(filename: Union[str, Path]) -> "re.Match[str] | None":
    if isinstance(filename, Path):
        filename = filename.name
    result = RE_PARSE_FILENAME.match(filename)
    if result is None:
        result = RE_PARSE_FILENAME_OLD.match(filename)
    return result

def seed_everything(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    if seed is None:
        torch.seed()
        torch.cuda.seed()
    else:
        torch.manual_seed(
            seed
        )  # setting torch manual seed causes bad models, # ok seed 124
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    if env:
        env.seed(seed)
        env.action_space.seed(seed)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    
    if args.clean:
        print("Cleaning up...")
        rewards_dir = Path("txts/")
        model_dir = Path("models/")
        for file in rewards_dir.glob("*.txt"):
            txt = file.read_text().split("\n")
            if len(txt) < 1000:
                print(f"Removing {file}")
                stem = file.stem.split("_rewards")[0]
                models = list(model_dir.glob(f"*th{stem}*"))
                if models:
                    for model in models:
                        pass
                        model.unlink()
                file.unlink()
                