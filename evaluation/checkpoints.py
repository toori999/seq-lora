from __future__ import annotations

import os
from typing import Dict, Tuple

import torch

try:
    from safetensors.torch import load_file as _load_safetensors_file
except Exception:
    _load_safetensors_file = None

Tensor = torch.Tensor


def load_adapter_checkpoint(adapter_dir: str) -> Dict[str, Tensor]:
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        if _load_safetensors_file is None:
            raise RuntimeError("safetensors is required to load adapter_model.safetensors")
        return _load_safetensors_file(safetensors_path)
    if os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError(f"Unexpected adapter checkpoint object type: {type(state)}")
        return state
    raise FileNotFoundError(
        f"Could not find adapter checkpoint under {adapter_dir}. "
        "Expected adapter_model.safetensors or adapter_model.bin."
    )


def remap_bayesian_peft_adapter_keys(state_dict: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], int]:
    remapped: Dict[str, Tensor] = {}
    num_changed = 0
    old_prefix = "base_model.model.base_model.model."
    new_prefix = "base_model.model."

    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith(old_prefix):
            new_key = new_prefix + new_key[len(old_prefix):]
        # Let PEFT inject the adapter name via set_peft_model_state_dict(..., adapter_name="default").
        # Appending ".default" here creates keys like "...lora_A.default.default.weight".
        if new_key != key:
            num_changed += 1
        remapped[new_key] = value
    return remapped, num_changed
