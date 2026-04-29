#!/usr/bin/env bash
set -euo pipefail

cd /home/tori/projects/Seq-LoRA/Seq_LoRA

RUN_DIR="/home/tori/projects/Seq-LoRA/Seq_LoRA/logs/map_mcdrop_ens_seedset_1_3_7_11_13_rerun_20260415_tmux"
PY="/home/tori/miniconda3_new/envs/seq-lora/bin/python"
TASK="scienceqa_closedchoice_grade2_11"
EVAL_TASKS="iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu_science_high,mmlu_science_college,gpqa_main"
BASE="/home/tori/projects/Seq-LoRA/Seq_LoRA/iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad"
SEEDS=(1 3 7 11 13)

mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/launcher.log") 2>&1

run_and_log() {
  local name="$1"
  shift
  echo "[RUN] ${name} at $(date -Is)" | tee -a "$RUN_DIR/run_all.log"
  "$@" 2>&1 | tee "$RUN_DIR/${name}.log"
  local rc=${PIPESTATUS[0]}
  echo "[DONE] ${name} rc=${rc} at $(date -Is)" | tee -a "$RUN_DIR/run_all.log"
  return "$rc"
}

echo "[START] $(date -Is)" | tee -a "$RUN_DIR/run_all.log"
echo "[INFO] logs_dir=$RUN_DIR" | tee -a "$RUN_DIR/run_all.log"

for seed in "${SEEDS[@]}"; do
  map_dir="$BASE/seed_${seed}/map_step_2000"
  run_and_log "map_eval_seed${seed}" \
    "$PY" map_eval.py \
      --task "$TASK" \
      --map_adapter_dir "$map_dir" \
      --eval_tasks "$EVAL_TASKS" \
      --seed "$seed"
done

for seed in "${SEEDS[@]}"; do
  map_dir="$BASE/seed_${seed}/map_step_2000"
  run_and_log "mcdrop_eval_seed${seed}" \
    "$PY" mcdrop_eval.py \
      --task "$TASK" \
      --map_adapter_dir "$map_dir" \
      --eval_tasks "$EVAL_TASKS" \
      --seed "$seed" \
      --mc_samples 32 \
      --temp 1.0
done

run_and_log "map_ensemble_eval_seedset_1_3_7_11_13" \
  "$PY" map_ensemble_eval.py \
    --task "$TASK" \
    --eval_tasks "$EVAL_TASKS" \
    --seed 0 \
    --map_adapter_dir "$BASE/seed_1/map_step_2000" \
    --map_adapter_dir "$BASE/seed_3/map_step_2000" \
    --map_adapter_dir "$BASE/seed_7/map_step_2000" \
    --map_adapter_dir "$BASE/seed_11/map_step_2000" \
    --map_adapter_dir "$BASE/seed_13/map_step_2000"

echo "[END] $(date -Is)" | tee -a "$RUN_DIR/run_all.log"
