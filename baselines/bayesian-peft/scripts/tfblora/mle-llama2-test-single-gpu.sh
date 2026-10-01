#!/usr/bin/env bash
set -euo pipefail

modelwrapper=tfblora
base_model=meta-llama/Llama-2-7b-hf
batch_size="${BATCH_SIZE:-16}"
beta="${BETA:-0.015}"
th="${THRESHOLD:-0.003}"
iter="${ITERATIONS:-5}"
sample="${N_SAMPLES:-10}"
gpu="${CUDA_DEVICE:-0}"

# These are the tasks whose local HF test split includes labels.
datasets=(
  ARC-Challenge
  ARC-Easy
  obqa
)

seeds=(1 3 7)

for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    load_path="checkpoints/mle/meta-llama/Llama-2-7b-hf/${dataset}/lora-${dataset}-lr5e-5-bs4-drop0.1-step2000-seed${seed}"
    if [ ! -d "${load_path}" ]; then
      echo "Skipping ${dataset} seed ${seed}: missing ${load_path}"
      continue
    fi

    name="${modelwrapper}-mle-llama2-test-${dataset}-sample${sample}-beta${beta}-th${th}-iter${iter}-seed${seed}"
    echo "Running ${name}"
    CUDA_VISIBLE_DEVICES="${gpu}" python run/main.py \
      --dataset-type mcdataset \
      --dataset "${dataset}" \
      --model-type causallm \
      --model "${base_model}" \
      --modelwrapper "${modelwrapper}" \
      --lr 1e-4 \
      --batch-size "${batch_size}" \
      --opt adamw \
      --warmup-ratio 0.06 \
      --max-seq-len 300 \
      --seed "${seed}" \
      --evaluate \
      --apply-classhead-lora \
      --lora-r 8 \
      --lora-alpha 16 \
      --lora-dropout 0 \
      --log-path "${name}" \
      --max-train-steps 10000 \
      --eval-per-steps 6000 \
      --bayes-klreweighting \
      --load-lora-path "${load_path}" \
      --testing-set train_train_test \
      --bayes-beta "${beta}" \
      --bayes-train-n-samples "${sample}" \
      --bayes-eval-n-samples "${sample}" \
      --bayes-eval-n-samples-final "${sample}" \
      --th "${th}" \
      --iter "${iter}" \
      --nowand
  done
done
