from __future__ import annotations

from typing import TYPE_CHECKING

# Import comet before torch to allow monkey patching.
from ray_utilities import default_trainable, run_tune  # fmt: skip

from interpretable_ddts import DDTSetup

if TYPE_CHECKING:
    from ray_utilities.typing import FunctionalTrainable


def test_mode_func(trainable: FunctionalTrainable, setup: DDTSetup):
    if setup.args.legacy:
        # this is a partial of gym_runner.run_process with param_space already set to the second argument
        return trainable({})
    # Trainable is build_and_train but not with disabled report

    return trainable(setup.param_space)


if __name__ == "__main__":
    setup = DDTSetup()
    results = run_tune(setup, test_mode_func)
