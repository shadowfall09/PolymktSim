# Benchmark summary — 2026-07-11

Each cell is `Brier score ↓ / accuracy ↑`. Bold marks the best value in its
dataset column for that metric; ties are both bold.

| Dataset | CoT / S0 | Standard Debate | InfoDelphi S1 | InfoDelphi S2 | MoA |
|---|---:|---:|---:|---:|---:|
| PolyMarket 375 | 0.2284 / 58.13% | **0.2171** / 60.00% | 0.2298 / 59.20% | 0.2191 / **60.27%** | 0.2279 / **60.27%** |
| FutureX 231 | 0.2611 / 60.61% | **0.2433** / 60.17% | 0.2735 / 56.71% | 0.2495 / **61.04%** | 0.2944 / 54.55% |
| ForecastBench 233 | 0.1945 / 67.38% | **0.1771** / **72.53%** | 0.1821 / 69.96% | 0.1829 / 69.53% | 0.1798 / 71.24% |

## Canonical result files

| Dataset | CoT / S0 | Standard Debate | InfoDelphi (contains S1 + S2) | MoA |
|---|---|---|---|---|
| PolyMarket 375 | `polymarket_375_cot.jsonl` | `polymarket_375_standard_debate.jsonl` | `polymarket_375_infodelphi.jsonl` | `polymarket_375_moa.jsonl` |
| FutureX 231 | `futurex_cot.jsonl` | `futurex_standard_debate.jsonl` | `futurex_infodelphi.jsonl` | `futurex_moa.jsonl` |
| ForecastBench 233 | `forecastbench_cot.jsonl` | `forecastbench_standard_debate.jsonl` | `forecastbench_infodelphi.jsonl` | `forecastbench_moa.jsonl` |

The two renamed InfoDelphi files retain matching `_detail.jsonl` companions.
