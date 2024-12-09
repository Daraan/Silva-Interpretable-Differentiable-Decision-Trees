import logging
from typing import Dict, TYPE_CHECKING, Any

import torch

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.utils.typing import ModuleID, TensorType
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY, VF_LOSS_KEY, ENTROPY_KEY
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_CURR_KL_COEFF_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
    PPOConfig,
)


logger = logging.getLogger(__name__)

class SilvaLearner(PPOTorchLearner):

    #@override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        # Note fwd_out["embeddings"] likely == batch
        module = self.module[module_id].unwrapped()
        use_silva_loss = self.config.learner_config_dict["use_silva_loss"]
        
        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]
            num_valid = torch.sum(mask)

            def possibly_masked_mean(data_):
                return torch.sum(data_[mask]) / num_valid

        else:
            possibly_masked_mean = torch.mean
            
        action_dist_class_train = module.get_train_action_dist_cls()
        action_dist_class_exploration = module.get_exploration_action_dist_cls()    
            
        # Silva originally used probs
        # From logits
        curr_action_dist = action_dist_class_train.from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        prev_action_dist = action_dist_class_exploration.from_logits(
            batch[Columns.ACTION_DIST_INPUTS]
        )

        # Silva used updated_log_probs from sample - action_probs of the samples
        logp_ratio = torch.exp(
            curr_action_dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP]
        )
        
        action_taken = batch[Columns.ACTIONS]
        #state = batch[Columns.OBS] # ?
        
        entropy_coef = self.entropy_coeff_schedulers_per_module[
            module_id
        ].get_current_value()
        
        # Silva
        # Forward pass
        #new_action_probs = curr_action_dist._dist.probs
        update_log_probs = curr_action_dist.logp(action_taken)
        entropy = (
            curr_action_dist.entropy().mean().mul(entropy_coef)
        )  # X: Here mean is taken
        
        #action_probs = torch.Tensor([sample['action_prob'] for sample in samples])
        action_probs = torch.exp(prev_action_dist.logp(action_taken))
        # Possible mistake, logit - prob
        # X: Silva uses ratio on probs; rrlib on logits
        ratio = torch.exp(update_log_probs) - action_probs

        # ----

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if config.use_kl_loss:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = possibly_masked_mean(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)
            
        # entropy
        curr_entropy = curr_action_dist.entropy()
        mean_entropy = possibly_masked_mean(curr_entropy)
        scaled_entropy = (
            entropy_coef
            * mean_entropy  # X: Silva used mean_entropy instead of current_entropy (PPO)
        )
        
        # Use negative later
        surrogate_loss = torch.min(
            # Surr1
            batch[Postprocessing.ADVANTAGES] * logp_ratio,
            # Surr2
            batch[Postprocessing.ADVANTAGES]
            * torch.clamp(logp_ratio, 1 - config.clip_param, 1 + config.clip_param),
        )
        # X Silva uses mean here
        
        # ----

        # Compute a value function loss.
        if config.use_critic:
            # If embeddings is not None, passes it trough self.vf; batch stays unused; which is equivalent
            value_fn_out = module.compute_values(
                batch, embeddings=fwd_out.get(Columns.EMBEDDINGS)
            )
            # Silva's model has 2 outputs, one for each action
            # Take value of action taken
            if use_silva_loss:
                original_value_fn_out = value_fn_out
                # If the value network has 2 outputs, take the one corresponding to the action taken
                if module.vf.output_dim != 1:  # type: ignore[attr-defined]
                    value_fn_out = value_fn_out[
                        torch.arange(0, len(value_fn_out)), batch[Columns.ACTIONS]
                    ]
                reward = batch[Columns.REWARDS]
                # Squared Error Loss
                vf_loss = torch.pow(value_fn_out - reward, 2.0)
                vf_loss_clipped = vf_loss
                mean_vf_loss = possibly_masked_mean(vf_loss_clipped)
                mean_vf_unclipped_loss = mean_vf_loss
            # Silva does not clip loss
            # X: Silva uses reward and not targets 
            # value_targets are discounted_returns when using critic
            # When using GAE they are calculated during postprocessing from advantages
            else:
                vf_loss = torch.pow(value_fn_out - batch[Postprocessing.VALUE_TARGETS], 2.0)
                vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)
                mean_vf_loss = possibly_masked_mean(vf_loss_clipped)
                mean_vf_unclipped_loss = possibly_masked_mean(vf_loss)
        # Ignore the value function -> Set all to 0.0.
        else:
            z = torch.tensor(0.0, device=surrogate_loss.device)
            value_fn_out = mean_vf_unclipped_loss = vf_loss_clipped = mean_vf_loss = z

        if use_silva_loss:
            action_loss = -surrogate_loss.mean()
            total_loss = action_loss - entropy + config.vf_loss_coeff * mean_vf_loss
        else:
            total_loss = possibly_masked_mean(
                -surrogate_loss + config.vf_loss_coeff * vf_loss_clipped - scaled_entropy
            )
        
        if config.use_kl_loss:
            total_loss += self.curr_kl_coeffs_per_module[module_id] * mean_kl_loss

        # Log important loss stats.
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: -possibly_masked_mean(surrogate_loss),
                VF_LOSS_KEY: mean_vf_loss,
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    batch[Postprocessing.VALUE_TARGETS], value_fn_out
                ),
                ENTROPY_KEY: mean_entropy,
                LEARNER_RESULTS_KL_KEY: mean_kl_loss,
            },
            key=module_id,
            window=1,  # <- single items (should not be mean/ema-reduced over time).
        )
        # Return the total loss.
        return total_loss