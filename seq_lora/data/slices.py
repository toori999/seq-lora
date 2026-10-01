from __future__ import annotations

import random

from datasets import Dataset


def assign_random_slice_ids(train_raw: Dataset, num_slices: int, seed: int) -> Dataset:
    if num_slices <= 0:
        raise ValueError(f"num_slices must be > 0, got {num_slices}")
    n = len(train_raw)
    if n == 0:
        raise ValueError("Cannot assign random slice ids to an empty training set.")
    if num_slices > n:
        raise ValueError(f"num_slices={num_slices} exceeds train size={n}")

    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    slice_ids = [0] * n
    for rank, idx in enumerate(indices):
        slice_ids[idx] = int(rank % num_slices)
    return train_raw.add_column("slice_id", slice_ids)
