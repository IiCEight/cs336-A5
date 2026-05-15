from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy of next-token predictions.

    H(p) = -sum_x p(x) log p(x)

    Uses log_softmax for numerical stability (avoids overflow from raw softmax).

    Args:
        logits: shape (batch_size, sequence_length, vocab_size), unnormalized.

    Returns:
        shape (batch_size, sequence_length), entropy at each position.
    """
    log_probs = F.log_softmax(logits, dim=-1)           # numerically stable log p
    probs = torch.exp(log_probs)                         # p = exp(log p)
    return -(probs * log_probs).sum(dim=-1)              # H = -sum p log p


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities from a causal language model.

    For each position t, computes log p_theta(x_t = label_t | x_{<t}) = log_softmax(logits)[label_t].

    Args:
        model: HuggingFace causal LM.
        input_ids: shape (batch_size, sequence_length), prompt + response tokens.
        labels: shape (batch_size, sequence_length), shifted input_ids (from tokenize_prompt_and_output).
        return_token_entropy: if True, also return per-token entropy.

    Returns:
        dict with:
            "log_probs": shape (batch_size, sequence_length)
            "token_entropy": shape (batch_size, sequence_length), only if return_token_entropy=True
    """
    logits = model(input_ids).logits  # (batch, seq_len, vocab_size)

    # log p(x_t | x_{<t}) = log_softmax(logits)[label_t]
    log_probs_all = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
    # Gather the log-prob of each actual label token
    log_probs = log_probs_all.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq_len)

    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result
