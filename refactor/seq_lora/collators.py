from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer


@dataclass
class DynamicEvalCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
                for feature in features
            ],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(
            [int(feature["labels"]) for feature in features], dtype=torch.long
        )
        for key in features[0].keys():
            if key not in {"input_ids", "attention_mask", "labels"}:
                batch[key] = [feature[key] for feature in features]
        return batch


__all__ = ["DynamicEvalCollator"]
