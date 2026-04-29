# ScienceQA Independent Slice Ensemble per3 vs T10 LGSSM

Mean ± sample SD over seeds 1, 3, 7, 11, 13. ACC and ECE are percentages; NLL is raw. Delta is Independent - LGSSM.

## Tau

| method | n | posterior_tau |
|---|---:|---:|
| T10 LGSSM final | 5 | 0.630995 ± 0.027703 |
| independent slice ensemble per3 | 5 | 0.917466 ± 0.054727 |

## ACC

| method | n | iid | grade12 | obqa | arc-c | mmlu-h | mmlu-c | gpqa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T10 LGSSM final | 5 | 93.85 ± 0.22 | 92.52 ± 0.84 | 87.44 ± 0.43 | 91.58 ± 0.13 | 81.71 ± 0.38 | 72.48 ± 0.75 | 35.49 ± 0.45 |
| independent slice ensemble per3 | 5 | 93.86 ± 0.12 | 90.00 ± 1.37 | 88.48 ± 0.59 | 91.72 ± 0.29 | 81.14 ± 0.46 | 73.24 ± 0.57 | 34.87 ± 1.36 |
| delta ind-lgssm | - | +0.01 | -2.52 | +1.04 | +0.14 | -0.56 | +0.75 | -0.62 |

## ECE

| method | n | iid | grade12 | obqa | arc-c | mmlu-h | mmlu-c | gpqa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T10 LGSSM final | 5 | 1.31 ± 0.10 | 3.70 ± 0.51 | 2.31 ± 0.53 | 1.46 ± 0.38 | 3.60 ± 0.38 | 5.85 ± 0.67 | 20.12 ± 0.51 |
| independent slice ensemble per3 | 5 | 1.25 ± 0.47 | 3.79 ± 1.09 | 2.43 ± 0.66 | 1.65 ± 0.35 | 4.06 ± 0.90 | 6.71 ± 1.62 | 20.48 ± 2.08 |
| delta ind-lgssm | - | -0.07 | +0.09 | +0.12 | +0.20 | +0.45 | +0.86 | +0.37 |

## NLL

| method | n | iid | grade12 | obqa | arc-c | mmlu-h | mmlu-c | gpqa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T10 LGSSM final | 5 | 0.1381 ± 0.0030 | 0.2349 ± 0.0057 | 0.3463 ± 0.0083 | 0.2407 ± 0.0032 | 0.5336 ± 0.0078 | 0.7336 ± 0.0063 | 1.5129 ± 0.0140 |
| independent slice ensemble per3 | 5 | 0.1456 ± 0.0038 | 0.2645 ± 0.0148 | 0.3473 ± 0.0036 | 0.2411 ± 0.0045 | 0.5503 ± 0.0137 | 0.7611 ± 0.0231 | 1.5376 ± 0.0336 |
| delta ind-lgssm | - | +0.0075 | +0.0296 | +0.0010 | +0.0004 | +0.0167 | +0.0275 | +0.0247 |

## Task Averages

| metric | T10 LGSSM final | independent per3 | delta ind-lgssm |
|---|---:|---:|---:|
| ACC all-task mean | 79.2957 | 79.0440 | -0.2517 |
| ECE all-task mean | 5.4783 | 5.7683 | +0.2900 |
| NLL all-task mean | 0.5343 | 0.5496 | +0.0153 |
| ACC near-OOD mean | 90.5133 | 90.0660 | -0.4473 |
| ACC far-OOD mean | 63.2253 | 63.0820 | -0.1433 |
| ECE near-OOD mean | 2.4887 | 2.6260 | +0.1373 |
| ECE far-OOD mean | 9.8560 | 10.4173 | +0.5613 |
| NLL near-OOD mean | 0.2740 | 0.2843 | +0.0103 |
| NLL far-OOD mean | 0.9267 | 0.9497 | +0.0230 |