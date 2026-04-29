#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-1 3 7 11 13}"
VARIANTS="${VARIANTS:-reverse_grade shuffled_grade full_shuffle base_nll_easyhard}"

TASK="${TASK:-scienceqa_closedchoice_grade2_11}"
EVAL_TASKS="${EVAL_TASKS:-iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu_science_high,mmlu_science_college,gpqa_main}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B-Base}"
MAP_BASE_DIR="${MAP_BASE_DIR:-iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad}"

SLICE_ROOT="${SLICE_ROOT:-slice_data/scienceqa_slice_ablation_controls}"
LOG_ROOT="${LOG_ROOT:-logs/scienceqa_slice_ablation_controls}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/.hf_datasets_cache}"

NUM_SLICES="${NUM_SLICES:-10}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-300}"
TOKENIZER_PADDING_SIDE="${TOKENIZER_PADDING_SIDE:-left}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

KFAC_BSZ="${KFAC_BSZ:-4}"
EVAL_BSZ="${EVAL_BSZ:-64}"
MC_EVAL_SAMPLES="${MC_EVAL_SAMPLES:-32}"
MC_EVAL_CHUNK="${MC_EVAL_CHUNK:-0}"
TAU_ANCHOR_SIZE="${TAU_ANCHOR_SIZE:-500}"
TAU_ANCHOR_BSZ="${TAU_ANCHOR_BSZ:-256}"
TAU_ANCHOR_N_SAMPLES="${TAU_ANCHOR_N_SAMPLES:-32}"
BASE_LOSS_BATCH_SIZE="${BASE_LOSS_BATCH_SIZE:-16}"

Q_MODE="${Q_MODE:-module_constant}"
KFAC_BACKEND="${KFAC_BACKEND:-asdl}"
OVERWRITE_SLICES="${OVERWRITE_SLICES:-0}"
USE_POSTERIOR_STATS_CACHE="${USE_POSTERIOR_STATS_CACHE:-0}"

export HF_DATASETS_CACHE

mkdir -p "$SLICE_ROOT" "$LOG_ROOT" caches/scienceqa_slice_ablation_controls

echo "[Config] root=$ROOT"
echo "[Config] seeds=$SEEDS"
echo "[Config] variants=$VARIANTS"
echo "[Config] slice_root=$SLICE_ROOT"
echo "[Config] log_root=$LOG_ROOT"

build_order_slices() {
  "$PYTHON_BIN" - "$SLICE_ROOT" "$NUM_SLICES" "$OVERWRITE_SLICES" "$SEEDS" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_eval_utils import (  # noqa: E402
    SCIENCEQA_DATASET_NAME,
    SCIENCEQA_GRADE_MAX,
    SCIENCEQA_GRADE_MIN,
)

slice_root = Path(sys.argv[1])
num_slices = int(sys.argv[2])
overwrite = str(sys.argv[3]).strip().lower() in {"1", "true", "yes", "y"}
seeds = [int(x) for x in sys.argv[4].split() if x.strip()]
cache_dir = os.environ.get("HF_DATASETS_CACHE", str(ROOT / ".hf_datasets_cache"))

def _parse_grade(ex: Dict) -> int:
    return int(str(ex["grade"]).strip().lower().replace("grade", ""))

def _choice_count(choices) -> int:
    if isinstance(choices, dict):
        return len(choices.get("text", []))
    if hasattr(choices, "tolist"):
        choices = choices.tolist()
    return len(choices) if isinstance(choices, (list, tuple)) else 0

def _remove_then_add(ds: Dataset, name: str, values: Sequence[int]) -> Dataset:
    if name in ds.column_names:
        ds = ds.remove_columns([name])
    return ds.add_column(name, [int(x) for x in values])

def _load_train() -> Dataset:
    ds = load_dataset(SCIENCEQA_DATASET_NAME, cache_dir=cache_dir)
    train = ds["train"]

    def keep(ex: Dict) -> bool:
        try:
            grade = _parse_grade(ex)
        except Exception:
            return False
        return (
            str(ex.get("task", "")).strip().lower() == "closed choice"
            and SCIENCEQA_GRADE_MIN <= grade <= SCIENCEQA_GRADE_MAX
        )

    def add_meta(ex: Dict) -> Dict:
        grade = _parse_grade(ex)
        return {
            "grade_num": grade,
            "source_slice_id": grade - SCIENCEQA_GRADE_MIN,
            "num_choices": _choice_count(ex.get("choices", [])),
        }

    train = train.filter(keep).map(add_meta)
    if "orig_idx" not in train.column_names:
        train = train.add_column("orig_idx", list(range(len(train))))
    if "slice_id" in train.column_names:
        train = train.remove_columns(["slice_id"])
    return train

def _grade_parts(train: Dataset, seed: int, grade_order: Sequence[int]) -> List[Dataset]:
    parts: List[Dataset] = []
    for grade in grade_order:
        idxs = [i for i, g in enumerate(train["grade_num"]) if int(g) == int(grade)]
        if not idxs:
            raise RuntimeError(f"No rows for grade {grade}")
        parts.append(train.select(idxs).shuffle(seed=int(seed) + int(grade)))
    return parts

def _concat_with_slice_ids(parts: Sequence[Dataset]) -> Dataset:
    out = []
    for sid, part in enumerate(parts):
        out.append(_remove_then_add(part, "slice_id", [sid] * len(part)))
    return concatenate_datasets(out)

def _build_reverse_grade(train: Dataset, seed: int):
    grades = list(range(SCIENCEQA_GRADE_MAX, SCIENCEQA_GRADE_MIN - 1, -1))
    return _concat_with_slice_ids(_grade_parts(train, seed, grades)), {
        "variant": "reverse_grade",
        "num_slices": num_slices,
        "seed": int(seed),
        "slice_grade_order": grades,
        "notes": "Grade buckets kept intact, ordered high-to-low grade.",
    }

def _build_shuffled_grade(train: Dataset, seed: int):
    grades = list(range(SCIENCEQA_GRADE_MIN, SCIENCEQA_GRADE_MAX + 1))
    perm = np.random.default_rng(int(seed)).permutation(len(grades)).tolist()
    shuffled_grades = [grades[i] for i in perm]
    return _concat_with_slice_ids(_grade_parts(train, seed, shuffled_grades)), {
        "variant": "shuffled_grade",
        "num_slices": num_slices,
        "seed": int(seed),
        "slice_grade_order": shuffled_grades,
        "slice_permutation": perm,
        "notes": "Grade buckets kept intact, but grade order is randomized per seed.",
    }

def _build_full_shuffle(train: Dataset, seed: int):
    shuffled = train.shuffle(seed=int(seed))
    parts = []
    sizes = []
    for sid, idxs in enumerate(np.array_split(np.arange(len(shuffled)), num_slices)):
        part = shuffled.select(idxs.astype(int).tolist())
        part = _remove_then_add(part, "slice_id", [sid] * len(part))
        parts.append(part)
        sizes.append(len(part))
    return concatenate_datasets(parts), {
        "variant": "full_shuffle",
        "num_slices": num_slices,
        "seed": int(seed),
        "slice_sizes": sizes,
        "notes": "Whole training set shuffled, then evenly split into 10 slices.",
    }

def _save(ds: Dataset, out_dir: Path, meta: Dict) -> None:
    if out_dir.exists():
        if not overwrite:
            print(f"[Skip] {out_dir}")
            return
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": ds}).save_to_disk(str(out_dir))
    (out_dir.parent / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[Save] {out_dir}")

train = _load_train()
print(f"[Load] ScienceQA train rows={len(train)}")
for seed in seeds:
    for name, builder in [
        ("reverse_grade", _build_reverse_grade),
        ("shuffled_grade", _build_shuffled_grade),
        ("full_shuffle", _build_full_shuffle),
    ]:
        ds_out, meta = builder(train, seed)
        _save(ds_out, slice_root / name / f"seed_{seed}" / "kfac_balanced", meta)
PY
}

build_base_nll_slices() {
  local out_dir="$SLICE_ROOT/base_nll_easyhard"
  local slice_dir="$out_dir/kfac_balanced"
  if [[ -d "$slice_dir" && "$OVERWRITE_SLICES" != "1" ]]; then
    echo "[Skip] $slice_dir"
    return
  fi
  if [[ "$OVERWRITE_SLICES" == "1" ]]; then
    rm -rf "$out_dir"
  fi
  "$PYTHON_BIN" mcqa_slices.py \
    --dataset_name scienceqa_closedchoice_grade2_11 \
    --split train \
    --out_dir "$out_dir" \
    --seed 0 \
    --num_slices "$NUM_SLICES" \
    --kfac_per_slice 0 \
    --slice_strategy quantile \
    --model_family custom \
    --score_with base \
    --base_model "$BASE_MODEL" \
    --tokenizer_padding_side "$TOKENIZER_PADDING_SIDE" \
    --trust_remote_code "$TRUST_REMOTE_CODE" \
    --local_files_only "$LOCAL_FILES_ONLY" \
    --batch_size "$BASE_LOSS_BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --save_full_train false \
    --save_kfac_balanced true
}

slice_dir_for_variant() {
  local variant="$1"
  local seed="$2"
  if [[ "$variant" == "base_nll_easyhard" ]]; then
    printf '%s\n' "$SLICE_ROOT/base_nll_easyhard/kfac_balanced"
  else
    printf '%s\n' "$SLICE_ROOT/$variant/seed_$seed/kfac_balanced"
  fi
}

run_one() {
  local variant="$1"
  local seed="$2"
  local slice_dir
  slice_dir="$(slice_dir_for_variant "$variant" "$seed")"
  local map_dir="$MAP_BASE_DIR/seed_${seed}/map_step_2000"
  local log_path="$LOG_ROOT/$variant/seed_${seed}.log"

  if [[ ! -d "$slice_dir" ]]; then
    echo "[Error] slice dir not found: $slice_dir" >&2
    exit 1
  fi
  if [[ ! -d "$map_dir" ]]; then
    echo "[Error] MAP dir not found: $map_dir" >&2
    exit 1
  fi

  local cache_args=()
  if [[ "$USE_POSTERIOR_STATS_CACHE" == "1" ]]; then
    cache_args=(
      --posterior_stats_cache_path
      "caches/scienceqa_slice_ablation_controls/${variant}_seed${seed}_kfac_stats.pt"
    )
  fi

  mkdir -p "$(dirname "$log_path")"
  echo "[Run] variant=$variant seed=$seed log=$log_path"
  "$PYTHON_BIN" seq_eval_iid_constantq.py \
    --task "$TASK" \
    --seed "$seed" \
    --slices_dir "$slice_dir" \
    --map_dir "$map_dir" \
    --eval_tasks "$EVAL_TASKS" \
    --trust_remote_code "$TRUST_REMOTE_CODE" \
    --tokenizer_padding_side "$TOKENIZER_PADDING_SIDE" \
    --kfac_backend "$KFAC_BACKEND" \
    --kfac_bsz "$KFAC_BSZ" \
    --eval_bsz "$EVAL_BSZ" \
    --q_mode "$Q_MODE" \
    --mc_eval_samples "$MC_EVAL_SAMPLES" \
    --mc_eval_chunk "$MC_EVAL_CHUNK" \
    --tau_mode auto \
    --tau_anchor_size "$TAU_ANCHOR_SIZE" \
    --tau_anchor_bsz "$TAU_ANCHOR_BSZ" \
    --tau_anchor_n_samples "$TAU_ANCHOR_N_SAMPLES" \
    "${cache_args[@]}" \
    2>&1 | tee "$log_path"
}

echo "[Build] reverse_grade / shuffled_grade / full_shuffle"
build_order_slices

echo "[Build] base_nll_easyhard"
build_base_nll_slices

for variant in $VARIANTS; do
  for seed in $SEEDS; do
    run_one "$variant" "$seed"
  done
done

echo "[Done] logs -> $LOG_ROOT"
