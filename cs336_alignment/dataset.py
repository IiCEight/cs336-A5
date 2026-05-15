from __future__ import annotations

import json
import os
import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase
from xopen import xopen


class _PackedSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, Tensor]]):
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self._examples[idx]


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """Load a JSONL SFT dataset and pack examples into fixed-length sequences.

    Each JSONL line must have "prompt" and "response" keys. The text of each
    example is tokenized (prompt concatenated with response), then all token
    IDs are flattened into a single stream. The stream is chunked into windows
    of (seq_length + 1) tokens; each window produces one training example:
        input_ids = chunk[:seq_length]
        labels    = chunk[1:seq_length+1]   (standard LM next-token shift)

    If shuffle=True, the list of per-example token sequences is shuffled
    (document-level) before flattening.

    Args:
        tokenizer: Tokenizer used to encode text.
        dataset_path: Path to a JSONL file with "prompt" and "response" keys.
        seq_length: Number of tokens per packed sequence.
        shuffle: If True, shuffle documents before packing.

    Returns:
        A PyTorch Dataset where each item is a dict with:
            "input_ids": LongTensor of shape (seq_length,)
            "labels":    LongTensor of shape (seq_length,)
    """
    # Step 1 — tokenize each document
    token_seqs: list[list[int]] = []
    with xopen(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            text = example["prompt"] + example["response"]
            ids = tokenizer.encode(text, add_special_tokens=False)
            token_seqs.append(ids)

    # Step 2 — optionally shuffle at the document level
    if shuffle:
        random.shuffle(token_seqs)

    # Step 3 — flatten into a single token stream
    flat: list[int] = []
    for seq in token_seqs:
        flat.extend(seq)

    # Step 4 — chunk into (seq_length + 1) windows, drop the last incomplete chunk
    chunk_size = seq_length + 1
    packed_examples: list[dict[str, Tensor]] = []
    for start in range(0, len(flat) - seq_length, seq_length):
        chunk = flat[start : start + chunk_size]
        if len(chunk) < chunk_size:
            break
        packed_examples.append({
            "input_ids": torch.tensor(chunk[:seq_length], dtype=torch.long),
            "labels":    torch.tensor(chunk[1:chunk_size], dtype=torch.long),
        })

    return _PackedSFTDataset(packed_examples)


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Wrap a Dataset in a DataLoader for batch iteration.

    Args:
        dataset: PyTorch Dataset to iterate over.
        batch_size: Number of examples per batch.
        shuffle: If True, shuffle examples each epoch.

    Returns:
        DataLoader that yields batches of {"input_ids", "labels"} tensors.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
