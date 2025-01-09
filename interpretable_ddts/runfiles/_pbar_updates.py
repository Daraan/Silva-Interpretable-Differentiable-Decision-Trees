from typing import Optional, TypedDict
from typing_extensions import NotRequired


class TrainRewardMetrics(TypedDict, total=False):
    mean: float
    max: float
    roll: float


class EvalRewardMetrics(TypedDict):
    mean: float
    roll: NotRequired[float]


class DiscreteEvalRewardMetrics(TypedDict):
    mean: float
    roll: NotRequired[float]


def update_pbar(
    pbar,
    *,
    eval_results: EvalRewardMetrics,
    train_results: Optional[TrainRewardMetrics] = None,
    discrete_eval_results: Optional[DiscreteEvalRewardMetrics] = None,
):
    try:
        if train_results:
            if train_results and train_results["mean"] == train_results["max"]:
                train_results = train_results.copy()
                train_results.pop("max")
            lines = [
                f"Train R {key}: {value:>6.1f}" if key != "max" else f"Train R {key}: {value:>4.0f}"
                for key, value in train_results.items()
            ]
        else:
            lines = []
        lines += [f"Eval R {key}: {value:>6.1f}" for key, value in eval_results.items()]
        if discrete_eval_results:
            lines += [f"Disc Eval R {key}: {value:>6.1f}" for key, value in discrete_eval_results.items()]
        description = " |".join(lines)
    except KeyError as e:
        description = ""
        print("KeyError in update_pbar", e)
    pbar.set_description(description)
