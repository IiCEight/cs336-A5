from __future__ import annotations

import torch


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum tensor elements where mask==1, then divide by normalize_constant.

    Args:
        tensor: tensor to sum.
        mask: same shape as tensor; positions with 1 are included.
        dim: dimension to sum along. If None, sum over all dimensions.
        normalize_constant: divisor for normalization.

    Returns:
        Normalized sum with masked elements excluded.
    """
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute SFT loss for a microbatch and backpropagate.

    Loss = -sum(log_probs * response_mask) / normalize_constant
    Scaled by 1/gradient_accumulation_steps for gradient accumulation.

    Args:
        policy_log_probs: shape (batch_size, sequence_length).
        response_mask: shape (batch_size, sequence_length), 1 for response tokens.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        normalize_constant: divisor for the masked sum.

    Returns:
        (loss, metadata) where loss is the scalar microbatch loss.
    """
    # NLL loss: negative masked sum, normalized by grad_accum * batch_size.
    # Backward is called on loss directly (not loss/grad_accum) — the batch_size
    # normalization ensures gradients are correctly scaled across microbatches.
    batch_size = policy_log_probs.shape[0]
    loss = -masked_normalize(
        policy_log_probs,
        response_mask,
        normalize_constant=normalize_constant * gradient_accumulation_steps * batch_size,
    )
    loss.backward()

    return loss, {}
