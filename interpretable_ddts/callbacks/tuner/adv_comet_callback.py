from __future__ import annotations
import logging
import math
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING
from ray.air.integrations.comet import CometLoggerCallback
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
from ray.tune.experiment import Trial

from interpretable_ddts.runfiles.constants import DEFAULT_VIDEO_KEYS
from interpretable_ddts.tools import numpy_to_video


if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial
    from comet_ml import Experiment, OfflineExperiment


class AdvCometLoggerCallback(CometLoggerCallback):
    # Copy from parent for pylance
    """CometLoggerCallback for logging Tune results to Comet.

    Comet (https://comet.ml/site/) is a tool to manage and optimize the
    entire ML lifecycle, from experiment tracking, model optimization
    and dataset versioning to model production monitoring.

    This Ray Tune ``LoggerCallback`` sends metrics and parameters to
    Comet for tracking.

    In order to use the CometLoggerCallback you must first install Comet
    via ``pip install comet_ml``

    Then set the following environment variables
    ``export COMET_API_KEY=<Your API Key>``

    Alternatively, you can also pass in your API Key as an argument to the
    CometLoggerCallback constructor.

    ``CometLoggerCallback(api_key=<Your API Key>)``

    Args:
            online: Whether to make use of an Online or
                Offline Experiment. Defaults to True.
            tags: Tags to add to the logged Experiment.
                Defaults to None.
            save_checkpoints: If ``True``, model checkpoints will be saved to
                Comet ML as artifacts. Defaults to ``False``.
            **experiment_kwargs: Other keyword arguments will be passed to the
                constructor for comet_ml.Experiment (or OfflineExperiment if
                online=False).

    Please consult the Comet ML documentation for more information on the
    Experiment and OfflineExperiment classes: https://comet.ml/site/

    Example:

    .. code-block:: python

        from ray.air.integrations.comet import CometLoggerCallback
        tune.run(
            train,
            config=config
            callbacks=[CometLoggerCallback(
                True,
                ['tag1', 'tag2'],
                workspace='my_workspace',
                project_name='my_project_name'
                )]
        )

    """

    _trial_experiments: dict[Trial, Experiment | OfflineExperiment]

    def _cli_to_str(self, args: dict, prefix="--", sep=" ") -> str:
        return sep.join([f"{prefix}{k} {v}" for k, v in args.items()])

    def __init__(
        self,
        *,
        online: bool = True,
        tags: Optional[List[str]] = None,
        save_checkpoints: bool = False,
        # Note: maybe want to log these in an algorithm debugger
        exclude_metrics: Optional[Iterable[str]] = None,
        log_to_other: Optional[Iterable[str]] = ("comment", "cli_args/comment"),
        log_cli_args: bool = True,
        video_keys: Iterable[str] = DEFAULT_VIDEO_KEYS,  # NOTE: stored as string not list of keys
        **experiment_kwargs,
    ):
        super().__init__(online=online, tags=tags, save_checkpoints=save_checkpoints, **experiment_kwargs)  # pyright: ignore[reportArgumentType]

        exclude_video_keys = ["/".join(keys) for keys in video_keys]
        self._to_exclude.extend([*exclude_metrics, *exclude_video_keys] if exclude_metrics else exclude_video_keys)
        self._to_other.extend(log_to_other or [])
        self._cli_args = " ".join(sys.argv[1:]) if log_cli_args else None
        self._log_only_once = [*self._to_exclude, "config", *self._to_system]
        if "training_iteration" in self._log_only_once:
            self._log_only_once.remove("training_iteration")
            logging.warning("training_iteration must be in the results to log it")
        self._video_keys = video_keys

    def log_trial_start(self, trial: Trial, **kwargs):
        super().log_trial_start(trial, **kwargs)
        experiment = self._trial_experiments[trial]
        if self._cli_args:
            experiment.log_other("args", self._cli_args)

    def log_trial_result(self, iteration: int, trial: Trial, result: Dict):
        step = result["training_iteration"]
        videos: dict[str, dict[str, list | float]] = {k: v for k in self._video_keys if (v := result.get(k))}
        # Remove Video keys and NaN values which can cause problems in the Metrics Tab when logged
        if trial in self._trial_experiments:
            # log_trial_start was called already, do not log parameters again
            # NOTE: # WARNING: This prevents config_updates during the run!
            result = {
                k: v
                for k, v in result.items()
                if not (k in self._log_only_once or k in self._video_keys)
                and (not isinstance(v, float) or not math.isnan(v))
            }
        else:
            result = {
                k: v
                for k, v in result.items()
                if k not in self._video_keys and (not isinstance(v, float) or not math.isnan(v))
            }

        # Cannot remove this
        result["training_iteration"] = step
        # Log normal metrics and parameters
        super().log_trial_result(iteration, trial, result)
        if videos:
            experiment = self._trial_experiments[trial]
            # TODO: store in final file path; or extract video at the end to final path
            for key, data in videos.items():
                video: list[int] = data["video"]  # type: ignore[arg-type]
                stripped_key = (
                    key.replace(ENV_RUNNER_RESULTS + "/", "").replace("episode_videos_", "").replace("/", "_")
                )
                filename = f"videos/{stripped_key}.mp4"  # e.g. step0040_best.mp4
                with tempfile.NamedTemporaryFile(suffix=".mp4", dir="temp_dir") as temp:
                    # os.makedirs(os.path.dirname(filename), exist_ok=True)
                    numpy_to_video(video, video_filename=temp.name)
                    experiment.log_video(
                        temp.name,
                        name=filename,
                        step=step,
                        metadata={"reward": data["reward"], "discrete": "discrete" in key},
                    )
            experiment.log_other("hasVideo", value=True)
