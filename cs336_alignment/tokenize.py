from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Goal: given a prompt and an output, produce training-ready tensors where the
    model sees [prompt + output] as input, but loss is only computed on the output tokens.

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": shape (batch_size, max_len - 1)
            "labels":    shape (batch_size, max_len - 1), shifted input_ids
            "response_mask": shape (batch_size, max_len - 1), 1 for response tokens in labels
    """
    # Step 1 — Tokenize separately, no special tokens.
    # Tokenized separately so we know exactly where the prompt ends and output begins.
    #   prompt_ids = tokenizer.encode("Hello world")  # [15496, 995]
    #   output_ids = tokenizer.encode("foo bar")      # [21943, 2318]
    prompt_ids = [
        tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs
    ]
    output_ids = [
        tokenizer.encode(o, add_special_tokens=False) for o in output_strs
    ]

    # Step 2 — Concatenate and record the split point.
    #   full sequence: [15496, 995, 21943, 2318]
    #                   ^--- prompt ---^^-- output --^
    #   response_start = 2  (index where output begins)
    sequences = []
    response_starts = []
    for p_ids, o_ids in zip(prompt_ids, output_ids):
        sequences.append(p_ids + o_ids)
        response_starts.append(len(p_ids))

    # Step 3 — Pad to max length across the batch (right-pad).
    #   seq 0: [15496, 995, 21943, 2318]   len=4
    #   seq 1: [17250, 65,  50256]         len=3  <- padded
    max_len = max(len(s) for s in sequences)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_id] * pad_len)

    full_ids = torch.tensor(padded, dtype=torch.long)  # (batch, max_len)

    # Step 4 — Slice into input_ids and labels.
    # Standard language model training: at each position, predict the next token.
    #   full:      [A, B, C, D]
    #   input_ids: [A, B, C]      <- drop last token
    #   labels:    [B, C, D]      <- drop first token (shifted by 1)
    input_ids = full_ids[:, :-1]
    labels = full_ids[:, 1:]

    # Step 5 — Build response_mask.
    # labels[i, j] = full_ids[i, j+1], so output tokens in labels start at
    # position (response_start - 1):
    #   full:     [Hello, world, foo,  bar]
    #   labels:   [world, foo,   bar,  PAD]
    #   mask:     [  0,    1,     1,    0 ]
    #              ^prompt^ ^output^  ^pad^
    # The mask is 1 only where labels contains output tokens — used later to
    # compute loss only on those positions.
    batch_size, seq_len = labels.shape
    response_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for i, (r_start, seq) in enumerate(zip(response_starts, sequences)):
        out_len = len(seq) - r_start  # number of output tokens
        mask_start = max(r_start - 1, 0)  # shift by 1 because labels = full_ids[:, 1:]
        mask_end = mask_start + out_len
        response_mask[i, mask_start:mask_end] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
