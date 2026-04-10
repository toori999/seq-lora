from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import importlib

MAP_VARIANT_MODULES = {
    "order": "train_scienceqa_qwen35_9b_lora_map_leftpad",
    "reverse": "train_scienceqa_qwen35_9b_lora_map_leftpad_grade_desc",
    "random": "train_scienceqa_qwen35_9b_lora_map_leftpad_random",
}


@dataclass(frozen=True)
class MapVariantConfig:
    order_key: str
    module_name: str
    output_dir: Path
    run_tag: str
    slice_dir: Path

    @property
    def map_root(self) -> Path:
        return self.output_dir / self.run_tag

    def run_dir(self, seed: int) -> Path:
        return self.map_root / f"seed_{seed}"

    def map_dir(self, seed: int) -> Path:
        return self.run_dir(seed) / "map_step_2000"

    def init_lora_path(self, seed: int) -> Path:
        return self.run_dir(seed) / "init_lora.pt"


def get_map_variant_module_name(order_key: str) -> str:
    if order_key not in MAP_VARIANT_MODULES:
        known = ", ".join(sorted(MAP_VARIANT_MODULES))
        raise KeyError(f"Unknown MAP variant {order_key!r}; expected one of: {known}")
    return MAP_VARIANT_MODULES[order_key]


def load_map_variant_module(order_key: str):
    module_name = get_map_variant_module_name(order_key)
    return importlib.import_module(module_name)


def load_map_variant_runner(order_key: str):
    mod = load_map_variant_module(order_key)
    return getattr(mod, "base", mod)


def load_map_variant_configs() -> Dict[str, MapVariantConfig]:
    out: Dict[str, MapVariantConfig] = {}
    for order_key, module_name in MAP_VARIANT_MODULES.items():
        runner_mod = load_map_variant_runner(order_key)
        out[order_key] = MapVariantConfig(
            order_key=order_key,
            module_name=module_name,
            output_dir=Path(str(runner_mod.OUTPUT_DIR)).resolve(),
            run_tag=str(runner_mod.RUN_TAG),
            slice_dir=Path(str(runner_mod.SLICE_OUT_DIR)).resolve(),
        )
    return out


__all__ = [
    "MAP_VARIANT_MODULES",
    "MapVariantConfig",
    "get_map_variant_module_name",
    "load_map_variant_configs",
    "load_map_variant_module",
    "load_map_variant_runner",
]
