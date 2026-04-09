from __future__ import annotations

import os

import train_scienceqa_qwen35_9b_lora_map_leftpad as base


base.RUN_TAG = "scienceqa_text_closedchoice_grade2_11_random_qv_lmhead_leftpad"
base.OUTPUT_DIR = "./iid_qwen35_8b_scienceqa_lora_map_leftpad_random"
base.SLICE_OUT_DIR = f"./slice_data/{base.RUN_TAG}/kfac_balanced"
os.makedirs(base.OUTPUT_DIR, exist_ok=True)


def _order_train_random(train_raw, seed: int):
    return train_raw.shuffle(seed=seed)


base.order_train_by_grade = _order_train_random


if __name__ == "__main__":
    base.main()
