import torch
from transformers import PreTrainedModel

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import utils


def compute_loss(
    model: PreTrainedModel,
    base_model: PreTrainedModel,
    prompt_completion_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    args,
) -> tuple:
    """
    Compute GRPO loss. Identical to original main.py — zero changes.

    What it does conceptually:
        - Ask current model: how confident were you about each token you generated?
        - Ask frozen base model: how confident would YOU have been about those same tokens?
        - Tokens from good stories (high advantage) → reinforce
        - Tokens from bad stories (low advantage) → discourage
        - KL penalty → don't drift too far from base model

    Args:
        model                 : LoRA-wrapped policy model (being trained)
        base_model            : frozen reference model (never updated)
        prompt_completion_ids : (num_chains, prompt_len + completion_len)
        prompt_ids            : (num_chains, prompt_len)
        completion_ids        : (num_chains, completion_len)
        attention_mask        : (num_chains, prompt_len + completion_len)
        completion_mask       : (num_chains, completion_len) — 1=real, 0=padding
        advantages            : (num_chains,) — from scoring.py
        args                  : training args (kl_weight_beta)

    Returns:
        loss    : scalar tensor — backpropagated by PyTorch
        metrics : dict with kl and response_length for logging
    """

    # ------------------------------------------------------------------
    # Step 1: Per-token log probs from current model (gradients flow here)
    # ------------------------------------------------------------------
    logits_to_keep    = completion_ids.size(1)
    per_token_logps   = utils.get_per_token_logps(
        model, prompt_completion_ids, attention_mask, logits_to_keep
    )

    # ------------------------------------------------------------------
    # Step 2: Per-token log probs from frozen base model (no gradients)
    # ------------------------------------------------------------------
    with torch.inference_mode():
        ref_per_token_logps = utils.get_per_token_logps(
            base_model, prompt_completion_ids, attention_mask, logits_to_keep
        )

    # ------------------------------------------------------------------
    # Step 3: Per-token KL divergence
    #   kl = exp(ref - policy) - (ref - policy) - 1
    #   Always >= 0. Equals 0 when models are identical.
    # ------------------------------------------------------------------
    log_ratio = ref_per_token_logps - per_token_logps
    kl        = torch.exp(log_ratio) - log_ratio - 1

    # ------------------------------------------------------------------
    # Step 4: Importance sampling ratio
    #   ratio > 1 → current model more confident than base on this token
    #   ratio < 1 → current model less confident than base on this token
    # ------------------------------------------------------------------
    ratio = torch.exp(per_token_logps - ref_per_token_logps)

    # ------------------------------------------------------------------
    # Step 5: Per-token policy objective
    #   good story (advantage > 0) → positive objective → negative loss
    #   bad story  (advantage < 0) → negative objective → positive loss
    # ------------------------------------------------------------------
    policy_objective = ratio * advantages.unsqueeze(1)

    # ------------------------------------------------------------------
    # Step 6: Combine policy objective with KL penalty
    #   Negative sign: PyTorch minimises loss, we want to maximise objective
    # ------------------------------------------------------------------
    per_token_loss = -(policy_objective) + args.kl_weight_beta * kl

    # ------------------------------------------------------------------
    # Step 7: Mask padding, average over tokens, average over completions
    # ------------------------------------------------------------------
    masked_loss = (
        (per_token_loss * completion_mask).sum(dim=1)
        / completion_mask.sum(dim=1).clamp(min=1)
    )
    loss = masked_loss.mean()

    # ------------------------------------------------------------------
    # Step 8: Metrics for logging
    # ------------------------------------------------------------------
    mean_kl         = (kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
    response_length = completion_mask.sum(dim=1).float().mean().item()

    metrics = {
        "kl":              mean_kl.mean().item(),
        "response_length": response_length,
    }

    return loss, metrics