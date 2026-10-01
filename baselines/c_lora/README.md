# C-LoRA  

## Seq-LoRA Repository Notes

This directory is the upstream-compatible C-LoRA baseline subtree. In this
repository it is used for the Llama-2 OBQA transfer C-LoRA baseline. The Qwen3
adaptation used by the main ScienceQA and Qwen3 OBQA transfer tables lives in
`baselines/c_lora/qwen3/`.

Formal reproduction entrypoints:

```bash
bash scripts/llama2/obqa_transfer/run_clora.sh
bash scripts/qwen3/scienceqa/scienceqa_main/run_clora.sh
bash scripts/qwen3/obqa_transfer/run_clora.sh
```

### File Map

| Path | Role |
| --- | --- |
| `run/main.py` | C-LoRA train/eval entrypoint. |
| `run/evaluation.py` | Evaluation helpers used by the runner. |
| `run/ood_eval_mdl.py` | Upstream OOD evaluation helper. |
| `modelwrappers/c_lora.py` | Contextual LoRA uncertainty wrapper. |
| `modelwrappers/wrapperbase_ece_prf.py` | Shared training/eval/calibration wrapper. |
| `models/causallm.py` | Causal-LM wrapper for MCQA tasks. |
| `models/seqcls.py` | Sequence-classification wrapper. |
| `models/laplace_*`, `models/laplace_bayeslib/` | Upstream compatibility dependencies. |
| `dataset/` | Upstream dataset wrappers. |
| `utils/args.py` | CLI argument definitions. |
| `scripts/ctx/ctx_all.sh` | Upstream C-LoRA script template. |
| `qwen3/` | Qwen3-specific C-LoRA adaptation used by paper Qwen3 runs. |

For Llama-2 transfer, `scripts/llama2/obqa_transfer/run_clora.sh` changes into
this directory and calls `python run/main.py --modelwrapper c_lora ...`; logs
are written to `logs/llama2/obqa_transfer/clora/`.

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
