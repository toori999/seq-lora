from __future__ import annotations

import os
from typing import List

from datasets import Dataset, concatenate_datasets

import train_scienceqa_qwen35_9b_lora_map_leftpad as base


base.RUN_TAG = "scienceqa_text_closedchoice_grade2_11_reversegrade_qv_lmhead_leftpad"
base.OUTPUT_DIR = "./iid_qwen35_8b_scienceqa_lora_map_leftpad_reversegrade"
base.SLICE_OUT_DIR = f"./slice_data/{base.RUN_TAG}/kfac_balanced"
os.makedirs(base.OUTPUT_DIR, exist_ok=True)


def _order_train_by_grade_desc(train_raw: Dataset, seed: int) -> Dataset:
    parts: List[Dataset] = []
    for grade_num in range(base.GRADE_MAX, base.GRADE_MIN - 1, -1):
        idxs = [i for i, g in enumerate(train_raw["grade_num"]) if int(g) == grade_num]
        if not idxs:
            continue
        ds_g = train_raw.select(idxs).shuffle(seed=seed + grade_num)
        parts.append(ds_g)
    if not parts:
        raise RuntimeError("No training examples left after grade filtering.")
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


base.order_train_by_grade = _order_train_by_grade_desc


if __name__ == "__main__":
    base.main()
