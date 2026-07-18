# Results v2

Smoke and default-limit gpt-4o-mini runs have been removed.

## Files

- futurex_s2_arguments_pooling_calibrated_5.4mini_full.jsonl (+ _detail)
- futurex_s2_arguments_pooling_calibrated_full.jsonl (+ _detail) — gpt-4o-mini
- polymarket_250_s1s2_arguments_pooling_calibrated_5.4mini.jsonl (+ _detail) — prompt v1
- polymarket_250_s1s2_argpool_calv2_da_shrink_5.4mini.jsonl (+ _detail) — prompt v2 + devils-advocate + calibrated_shrink
- polymarket_250_s1s2_argpool_calv3_shrink_5.4mini.jsonl (+ _detail) — prompt v3 + calibrated_shrink
- leaderboard.csv, runs.jsonl

## FutureX 231 (s2, arguments+pooling+calibrated, 3 agents x 2 rounds)

| run_ts | model | n | brier | median brier | accuracy | retention |
|---|---|---:|---:|---:|---:|---:|
| 20260713_004825 | gpt-4o-mini | 231 | 0.23701 | 0.06250 | 0.6494 | 0.58 |
| 20260713_033120 | gpt-5.4-mini | 231 | 0.22116 | 0.06084 | 0.6883 | 0.38 |

Both use: s2, aggregator=mean, bm25=True, public_ratio=0.5, share_mode=arguments, evidence_pooling=True, calibrated_prompt=True, num_agents=3, num_rounds=2.

## Polymarket 250 s2 — 2026-07-17 calibration study (gpt-5.4-mini)

All runs: s2, bm25, public_ratio=0.5, share_mode=arguments, evidence_pooling, 3 agents x 2 rounds, temp 0.7.
"+shrink" = calibrated_shrink(p0=0.30, w_lo=0.8, w_hi=0.5); offline re-scoring via `postprocess_shrink.py`.

| config | raw Brier | +shrink | acc (raw) | notes |
|---|---:|---:|---:|---|
| prompt v1 (calibrated) | 0.1647 | **0.1575** | 0.748 | **winner**; shrink params cross-validated poly<->futurex |
| prompt v2 + devils-advocate | 0.1896 | 0.1779 | 0.760 | hard confidence cap over-suppresses the whole distribution |
| prompt v3 (soft criteria check) | 0.1780 | 0.1689 | 0.744 | fixes p>=0.6 hit rate (0.29->0.45) but drags mid-band true-YES down |

Same shrink on FutureX 231 gpt-5.4-mini: 0.2212 -> **0.2006**, accuracy 0.688 -> 0.701.
Reference points on polymarket_250: constant base-rate (0.232) Brier 0.1782; best prior
baseline (infodelphi_best, data/results) 0.1623.

Takeaway: overconfidence corrections belong in the aggregation layer (asymmetric shrink
toward a NO-leaning prior), not in the prompt — skepticism wording in the prompt cannot be
targeted at the high-confidence band and depresses everything (v2: mean_p 0.27->0.16;
v3: 0.27->0.20 vs base rate 0.232).

## Sports retrieval experiments (2026-07-18, both null results)

Paired on the 106 Sports questions, winner config (v1 + calibrated_shrink), vs the
original-evidence run. Slug metadata = opponent decoded from the market URL slug
(`augment_sports_slug_metadata.py`); opponent odds/form = targeted pre-cutoff Tavily
retrieval with the decoded opponent name (`augment_sports_opponent_evidence.py`,
three-layer leak control, 376 docs kept).

| evidence | Brier | AUC | paired diff vs orig (95% CI) |
|---|---:|---:|---|
| original | 0.1908 | 0.697 | — |
| + slug metadata | 0.1890 | 0.690 | -0.0018 (-0.010, +0.006) |
| + opponent odds/form | 0.1873 | 0.719 | -0.0035 (-0.012, +0.005) |

The info was consumed (52-53% of agent forecasts cite the new docs; 43% of retrieved
docs contain numeric odds) but does not convert into measurable skill. The A/A control
subsets moved as much as the treated subsets. Sports remains evidence-limited at the
model level, not the retrieval level.

Recommended command:

    python run_workflow.py --dataset-jsonl data/processed/polymarket_250_with_evidence.jsonl \
        --limit 250 --scenario s1s2 --aggregator calibrated_shrink \
        --calibrated-prompt --share-mode arguments --evidence-pooling --bm25
