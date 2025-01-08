from ray.rllib.algorithms import Algorithm
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_EPISODES,
    NUM_EPISODES_LIFETIME,
)
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interpretable_ddts.agents.ddt_ppo_module import DDTModule


def eval_with_discrete(self: Algorithm, eval_workers: EnvRunnerGroup) -> tuple[dict, int, int]:
    assert eval_workers == self.eval_env_runner_group
    local_runner = self.env_runner_group.local_env_runner
    module: DDTModule = local_runner.module
    if getattr(module, "CAN_USE_DISCRETE_EVAL", False):
        options = (False, True)
    else:
        options = (False,)

    combined_eval_results = {}
    for discrete in options:
        if discrete:
            metrics_backup = self.metrics
            if not hasattr(self, "_discrete_metrics"):
                self._discrete_metrics = MetricsLogger()
            self.metrics = self._discrete_metrics
            module.switch_mode(discrete=True)
            self.eval_env_runner_group.sync_weights(
                # policies=["discrete"], # can add different weight source
                from_worker_or_learner_group=self.env_runner_group.local_env_runner,
                inference_only=True,
            )
            # AttributeError: 'SingleAgentEnvRunner' object has no attribute 'foreach_env'

        if eval_workers is None:
            (
                eval_results,
                env_steps,
                agent_steps,
                batches,
            ) = self._evaluate_on_local_env_runner(self.env_runner_group.local_env_runner)
        elif eval_workers.num_healthy_remote_workers() == 0:
            (
                eval_results,
                env_steps,
                agent_steps,
                batches,
            ) = self._evaluate_on_local_env_runner(self.eval_env_runner)
        # There are healthy remote evaluation workers -> Run on these.
        elif self.eval_env_runner_group.num_healthy_remote_workers() > 0:
            # Cannot use this parallel to training
            assert self.config.evaluation_duration != "auto"
            (
                eval_results,
                env_steps,
                agent_steps,
                batches,
            ) = self._evaluate_with_fixed_duration()
        # Can't find a good way to run this evaluation -> Wait for next iteration.
        else:
            eval_results = {}
        if discrete:
            # Reduce discrete metrics
            eval_results = self.metrics.reduce(EVALUATION_RESULTS, return_stats_obj=True)
            combined_eval_results["discrete"] = eval_results
            # Update "normal" metrics with discrete results and revert
            self.metrics = metrics_backup
            metrics_backup.log_dict(
                eval_results,
                key=(EVALUATION_RESULTS, "discrete"),
            )
            metrics_backup.log_dict(
                {
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: env_steps,
                    NUM_AGENT_STEPS_SAMPLED_LIFETIME: agent_steps,
                    NUM_EPISODES_LIFETIME: self.metrics.peek(
                        (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, NUM_EPISODES),
                        default=0,
                    ),
                },
                key=(EVALUATION_RESULTS, "discrete"),
                reduce="sum",
            )
            module.switch_mode(discrete=False)
        else:
            if eval_results:
                combined_eval_results = {**combined_eval_results, **eval_results}
            agent_steps_normal = agent_steps
            env_steps_normal = env_steps
    return combined_eval_results, env_steps_normal, agent_steps_normal
