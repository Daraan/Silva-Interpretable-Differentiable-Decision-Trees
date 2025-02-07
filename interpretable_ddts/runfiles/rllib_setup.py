from __future__ import annotations

import logging
import os
import sys

import ray
from packaging.version import parse as parse_version
from ray.air.integrations.wandb import setup_wandb

# Import comet before
from ray_utilities.comet import comet_upload_offline_experiments  # fmt: skip
from interpretable_ddts.runfiles._ddt_trainable import build_and_train
from interpretable_ddts.runfiles.ddt_setup import DDTSetup

os.environ["RAY_COLOR_PREFIX"] = "1"

RAY_VERSION = parse_version(ray.__version__)

logger = logging.getLogger(__name__)


def main():
    # full parser example see: https://github.com/ray-project/ray/blob/master/rllib/utils/test_utils.py#L61

    setup = DDTSetup()
    args = setup.get_args()
    trainable = setup.trainable

    # Will use these resources per job
    # NOTE: Even if not used will allocate these resources per run
    # trainable_with_resources = tune.with_resources(trainable, tune.PlacementGroupFactory(
    #    [{'CPU': 1.0}] + [{'CPU': 1.0}] * (4 if args.parallel else 0),
    # ))
    # Use tune.with_parameters to pass large objects to the trainable

    # Callbacks

    # If videos are logged use custom callbacks for correct logging
    # NOTE: JSON, CSV, and Tensorboard loggers are created automatically by Tune if not disabled

    # -- Test --
    if args.test and args.not_parallel:
        # will spew some warnings about train.report
        print("-- TEST MODE --")
        if args.legacy:
            trainable({})
        else:
            results = build_and_train(setup.param_space, disable_report=True)
        sys.exit()

    tuner = setup.create_tuner()
    results = tuner.fit()
    if args.comet == "offline+upload":
        comet_upload_offline_experiments()
    return results


if __name__ == "__main__":
    results = main()
