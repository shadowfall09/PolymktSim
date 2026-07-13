# FutureX InfoDelphi S2 Shrinkage Ablation

Input: `data/results/futurex_infodelphi.jsonl`

Filter: `scenario == "s2"`

Transform:

```text
p' = 0.5 + alpha * (p_yes - 0.5)
```

## Summary

| alpha | Brier | Delta Brier | Accuracy | ECE | LogLoss | Avg Confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.2495 | +0.0000 | 0.6104 | 0.1383 | 0.7643 | 0.7417 |
| 0.8 | 0.2367 | -0.0127 | 0.6104 | 0.1212 | 0.6735 | 0.6934 |
| 0.6 | 0.2304 | -0.0190 | 0.6104 | 0.0569 | 0.6533 | 0.6450 |
| 0.5 | 0.2297 | -0.0198 | 0.6104 | 0.0441 | 0.6514 | 0.6209 |
| 0.4 | 0.2305 | -0.0189 | 0.6104 | 0.0505 | 0.6533 | 0.5967 |

The in-sample analytic optimum is `alpha = 0.5033`, with Brier `0.2297`.

## Finding

Simple probability shrinkage improves FutureX InfoDelphi S2 Brier substantially without changing accuracy. This supports the hypothesis that S2's main FutureX failure mode is overconfidence rather than worse label direction.

Use `alpha = 0.6` as a conservative fixed setting for the next real ablation. It improves Brier by `0.0190` while avoiding fitting the exact in-sample optimum too tightly.

## Reproduce

```bash
python calibrate_shrinkage.py data/results/futurex_infodelphi.jsonl --scenario s2 --output-csv data/results/futurex_infodelphi_s2_shrinkage_ablation.csv
```
