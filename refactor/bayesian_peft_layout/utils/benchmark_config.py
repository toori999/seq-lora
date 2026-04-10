from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Sequence, Tuple

from .map_variants import MAP_VARIANT_MODULES, MapVariantConfig, load_map_variant_configs

SOURCE_TASK = "scienceqa_closedchoice_grade2_11"
DEFAULT_EVAL_TASKS = [
    "iid",
    "scienceqa_closedchoice_grade12",
    "obqa",
    "arc-c",
    "mmlu_science_high",
    "mmlu_science_college",
    "gpqa_main",
]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
POSTHOC_INTERNAL_SEED = 0
DEFAULT_ENSEMBLE_TOTAL_SEEDS = 20
DEFAULT_ENSEMBLE_GROUPS = 5
EXCLUDED_STATUS_PREFIXES: Tuple[str, ...] = ("laplace_order",)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_list(spec: str) -> List[int]:
    values: List[int] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(int(raw))
    if not values:
        raise ValueError("Expected at least one seed")
    return values


def expand_eval_tasks(spec_tasks: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for raw in spec_tasks:
        task = str(raw).strip().lower()
        if not task:
            continue
        if task == "mmlu":
            expanded.extend(["mmlu_science_high", "mmlu_science_college"])
        else:
            expanded.append(task)

    out: List[str] = []
    seen = set()
    for task in expanded:
        if task not in seen:
            seen.add(task)
            out.append(task)
    return out
def build_consecutive_ensemble_groups(total_seeds: int, num_groups: int) -> List[List[int]]:
    if total_seeds <= 0:
        raise ValueError("ensemble total_seeds must be positive")
    if num_groups <= 0:
        raise ValueError("ensemble num_groups must be positive")
    if total_seeds % num_groups != 0:
        raise ValueError(
            f"ensemble total_seeds={total_seeds} must be divisible by "
            f"num_groups={num_groups}"
        )
    seeds = list(range(total_seeds))
    group_size = total_seeds // num_groups
    return [seeds[i * group_size : (i + 1) * group_size] for i in range(num_groups)]


__all__ = [
    "DEFAULT_ENSEMBLE_GROUPS",
    "DEFAULT_ENSEMBLE_TOTAL_SEEDS",
    "DEFAULT_EVAL_TASKS",
    "DEFAULT_SEEDS",
    "EXCLUDED_STATUS_PREFIXES",
    "MAP_VARIANT_MODULES",
    "MapVariantConfig",
    "POSTHOC_INTERNAL_SEED",
    "SOURCE_TASK",
    "build_consecutive_ensemble_groups",
    "expand_eval_tasks",
    "load_map_variant_configs",
    "parse_int_list",
    "utc_now",
]
