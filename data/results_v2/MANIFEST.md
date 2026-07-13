# Results v2

Smoke and default-limit gpt-4o-mini runs have been removed.

## Remaining Files

- futurex_s2_arguments_pooling_calibrated_5.4mini_full.jsonl
- futurex_s2_arguments_pooling_calibrated_5.4mini_full_detail.jsonl
- futurex_s2_arguments_pooling_calibrated_full.jsonl
- futurex_s2_arguments_pooling_calibrated_full_detail.jsonl
- leaderboard.csv
- runs.jsonl

## Registry Scope

`runs.jsonl` and `leaderboard.csv` keep the two full 231 FutureX runs currently present here:

| run_ts | model | n | brier | median brier | accuracy | retention |
|---|---|---:|---:|---:|---:|---:|
| 20260713_004825 | gpt-4o-mini | 231 | 0.23701 | 0.06250 | 0.6494 | 0.58 |
| 20260713_033120 | gpt-5.4-mini | 231 | 0.22116 | 0.06084 | 0.6883 | 0.38 |

Both use: s2, aggregator=mean, bm25=True, public_ratio=0.5, share_mode=arguments, evidence_pooling=True, calibrated_prompt=True, num_agents=3, num_rounds=2.
