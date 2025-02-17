from __future__ import annotations

from typing import TYPE_CHECKING

# Import comet before torch to allow monkey patching.
from ray_utilities import run_tune  # fmt: skip

from interpretable_ddts import DDTSetup

if TYPE_CHECKING:
    from ray_utilities.typing import FunctionalTrainable


def test_mode_func(trainable: FunctionalTrainable, setup: DDTSetup):
    if setup.args.legacy:
        # this is a partial of gym_runner.run_process with param_space already set to the second argument
        return trainable({})
    # Trainable is build_and_train but not with disabled report
    from interpretable_ddts.rllib_port import build_and_train

    return build_and_train(setup.param_space, disable_report=True)


if __name__ == "__main__":
    setup = DDTSetup()
    results = run_tune(setup, test_mode_func)
