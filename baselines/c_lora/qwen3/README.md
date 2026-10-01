# C-LoRA  

## Seq-LoRA Qwen3 Baseline Notes

This directory adapts C-LoRA to the Qwen3 MCQA benchmark used by the Seq-LoRA
paper. It is the implementation behind the Qwen3 C-LoRA rows.

Formal reproduction entrypoints:

```bash
bash scripts/qwen3/scienceqa/scienceqa_main/run_clora.sh
bash scripts/qwen3/obqa_transfer/run_clora.sh
```

### File Map

| Path | Role |
| --- | --- |
| `run/main.py` | Qwen3 C-LoRA train/eval entrypoint. |
| `run/evaluation.py` | Evaluation helpers used by the runner. |
| `run/ood_eval_mdl.py` | Upstream-style OOD evaluation helper. |
| `modelwrappers/c_lora.py` | Contextual LoRA posterior wrapper. |
| `modelwrappers/wrapperbase_ece_prf.py` | Shared train/eval/calibration wrapper. |
| `models/causallm.py` | Qwen3 causal-LM MCQA wrapper. |
| `models/laplace_*`, `models/laplace_bayeslib/` | Upstream compatibility dependencies. |
| `dataset/benchmark_mcdataset.py` | Benchmark MCQA dataset adapter. |
| `dataset/S2*.py` | Upstream dataset adapters. |
| `utils/args.py` | CLI argument definitions. |
| `scripts/ctx/ctx_all.sh` | Upstream template; paper reproduction uses the root `scripts/`. |

### How The Paper Scripts Use This Code

ScienceQA C-LoRA:

```text
--dataset scienceqa_closedchoice_grade2_11
--modelwrapper c_lora
--shared-init-lora-path <MAP_ROOT>/seed_<seed>/init_lora.pt
--save-clora-dir clora_qwen3_8b_scienceqa_leftpad/seed_<seed>/clora
--eval-dataset iid,grade12,obqa,arcc,mmlu-h,mmlu-c,gpqa
--bayes-eval-n-samples-final 32
```

OBQA transfer C-LoRA:

```text
--dataset obqa
--modelwrapper c_lora
--shared-init-lora-path iid_qwen35_8b_obqa_lora_map_leftpad/obqa_qv_lmhead_leftpad/seed_<seed>/init_lora.pt
--save-clora-dir clora_qwen3_8b_obqa/seed_<seed>/clora
--eval-dataset iid,arc-e,arc-c,mmlu-h,mmlu-c
```

Logs are written to `logs/qwen3/scienceqa/scienceqa_main/clora/` and
`logs/qwen3/obqa_transfer/clora/`.

This repository contains codes for ***Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models***  
**NeurIPS 2025**

[Paper (arXiv)](https://arxiv.org/pdf/2505.17773)  

---

## Overview  

C-LoRA is a **parameter-efficient**, **uncertainty-aware** fine-tuning method for large language models (LLMs) in few-shot or data-scarce settings. The key idea is to introduce **input-dependent (contextual) uncertainty modeling** in the LoRA adapters, thereby yielding better calibrated predictive uncertainties and reducing overconfidence.  

Unlike prior Bayesian LoRA or mean-field approaches, C-LoRA (1) uses a **lightweight factorization** to reduce complexity, and (2) integrates a small **contextual module** that conditions the posterior distribution of adapter parameters on each input sample.  

The method achieves strong performance on calibration metrics (like ECE, NLL) while maintaining competitive accuracy across reasoning benchmarks and showing robustness under distribution shift. 

---

## Installation & Usage

Install the environment (env name need to be modified). 

```bash 
conda env create -f env.yaml
conda activate yourEnv
```

For experiments, use the script provided in scripts directory. 



---
## Citation
If you find this work useful, please cite our paper:
```bibtex
@inproceedings{
rahmati2025clora,
title={C-Lo{RA}: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models},
author={Amir Hossein Rahmati and Sanket Jantre and Weifeng Zhang and Yucheng Wang and Byung-Jun Yoon and Nathan Urban and Xiaoning Qian},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=siPeAstQLq}
}
```
