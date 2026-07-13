# FutureX InfoDelphi Failure Mode Analysis

Input files:

- `data/results/futurex_infodelphi.jsonl`
- `data/results/futurex_infodelphi_detail.jsonl`
- `data/results/futurex_cot.jsonl`
- `data/results/futurex_standard_debate.jsonl`
- `data/results/futurex_moa.jsonl`

## Overall

| method | brier | accuracy |
| --- | --- | --- |
| cot | 0.2611 | 60.61% |
| standard | 0.2433 | 60.17% |
| s1 | 0.2735 | 56.71% |
| s2 | 0.2495 | 61.04% |
| moa | 0.2944 | 54.55% |

## Main Failure Modes

| mode | count | share |
| --- | --- | --- |
| final_correct | 130 | 56.28% |
| all_agents_wrong_from_start | 69 | 29.87% |
| debate_erased_correct_minority | 13 | 5.63% |
| debate_fixed | 11 | 4.76% |
| debate_broke_mean_label | 5 | 2.16% |
| wrong_majority_persisted | 3 | 1.30% |

## S2 Wrong Overlap

S2 wrong count: 90

| baseline condition | count among S2 wrong |
| --- | --- |
| cot | 25 |
| standard | 21 |
| s1 | 7 |
| moa | 14 |
| all four baselines wrong | 48 |
| cot or standard correct | 37 |

## By Domain

| domain | n | wrong | acc | round0_all_wrong | debate_broke | brier |
| --- | --- | --- | --- | --- | --- | --- |
| markets_finance_crypto | 43 | 21 | 51.16% | 16 | 2 | 0.2591 |
| media_entertainment | 3 | 0 | 100.00% | 0 | 0 | 0.0684 |
| other | 95 | 34 | 64.21% | 28 | 1 | 0.2322 |
| politics_geo | 30 | 9 | 70.00% | 4 | 1 | 0.2055 |
| sports | 38 | 16 | 57.89% | 12 | 1 | 0.2863 |
| tech_ai_product | 22 | 10 | 54.55% | 9 | 0 | 0.3260 |

## By Question Type

| qtype | n | wrong | acc | round0_all_wrong | debate_broke | brier |
| --- | --- | --- | --- | --- | --- | --- |
| deadline_event | 55 | 15 | 72.73% | 10 | 0 | 0.1899 |
| other_wording | 86 | 32 | 62.79% | 24 | 3 | 0.2475 |
| threshold_range | 65 | 30 | 53.85% | 24 | 1 | 0.2667 |
| winner_outcome | 25 | 13 | 48.00% | 11 | 1 | 0.3425 |

## Interpretation

The dominant FutureX failure is not debate corrupting initially correct forecasts. Among 90 final S2 wrong cases, 69 cases have all three S2 round-0 agents already on the wrong side. Only 5 cases are clear majority/mean-label debate regressions.

The debate step is usually beneficial or neutral at the label level: round-mean labels stay correct in 130 cases and are fixed in 11 cases, while only 5 cases are flipped from correct to wrong. The larger problem is shared evidence/interpretation failure before debate begins, followed by round-1 consensus that makes the same wrong answer more confident.

Aggregation is not the main issue: confidence-weighted aggregation almost never overturns the round-1 majority label. The Brier problem is mostly calibration/overconfidence on wrong consensus cases, which is consistent with the separate shrinkage ablation.

## Example Cases

### all_agents_wrong_from_start

| qid | y | s2_p | r0 | r1 | cot | std | question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| futurex_69cd100f84ea010067ba2288 | 0 | 1.00 | 0.99/0 | 1.00/0 | 0.99 | 1.00 | Will ICE kill a person by the End of April 2026? |
| futurex_6a144109314f08005cc4064a | 0 | 0.99 | 0.97/0 | 0.99/0 | 0.98 | 0.98 | Will both Aeonglass and Infested Prism receive balance changes in the next Slay the Spire  |
| futurex_69ba9b2453c457006694bc58 | 0 | 0.98 | 0.94/0 | 0.98/0 | 0.96 | 0.96 | Will Anthropic release a model with a higher version number 4.6 by April 15th? |
| futurex_69fb34fbc70fed005b77cd39 | 0 | 0.98 | 0.91/0 | 0.98/0 | 0.82 | 0.76 | Any new Claude Model Before May 16th? |
| futurex_69b2b226782d0900685d5fb4 | 0 | 0.98 | 0.97/0 | 0.97/0 | 0.97 | 0.02 | Assassin's Creed: Black Flag Resynced (remake) Announced March 20th |
| futurex_69a6d48ee78a390068a18736 | 0 | 0.96 | 0.95/0 | 0.96/0 | 0.18 | 0.36 | Will the Canada-India uranium supply agreement size exceed 2.5 billion USD on April 1, 202 |
| futurex_69a5830b7554ef0068e464be | 0 | 0.96 | 0.92/0 | 0.96/0 | 0.86 | 0.93 | Will foreign airlines resume commercial passenger flights to Aviv (TLV) by March 15, 2026? |
| futurex_69b403b749eb3f005a923d11 | 0 | 0.95 | 0.92/0 | 0.95/0 | 0.93 | 0.94 | will CS camp happen? |

### debate_broke_mean_label

| qid | y | s2_p | r0 | r1 | cot | std | question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| futurex_6a22c1969b0b8587160ddfcd | 1 | 0.14 | 0.55/2 | 0.15/0 | 0.03 | 0.05 | Will Spain defeat Uruguay in the opening round of World Cup 2026? |
| futurex_69d2560b92520400680b171f | 1 | 0.41 | 0.53/2 | 0.41/0 | 0.58 | 0.47 | Will a tornado outbreak occur on either April 12, 13, 14 or 15 in the USA? |
| futurex_6a26b60a9b0b8587160de16c | 1 | 0.44 | 0.53/2 | 0.45/0 | 0.57 | 0.46 | Will the US PCE annual inflation be greater than 3.8% in May? |
| futurex_69c527154b6f01005cae1313 | 0 | 0.54 | 0.50/1 | 0.52/0 | 0.61 | 0.56 | Will the Trump-Xi April 2026 summit produce a bilateral trade agreement? |
| futurex_69f5eea0b2a4f7005ec64538 | 0 | 0.53 | 0.46/1 | 0.52/1 | 0.38 | 0.60 | Bitcoin $65K in May? |

### debate_erased_correct_minority

| qid | y | s2_p | r0 | r1 | cot | std | question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| futurex_69f34e0e58ab86005d6f66cc | 1 | 0.17 | 0.22/1 | 0.20/0 | 0.14 | 0.64 | UEFA Champions League: Unbeaten Champion |
| futurex_69f891cbab4486005c6c382e | 1 | 0.30 | 0.37/1 | 0.30/0 | 0.12 | 0.27 | Will Terminator2 have at least 200 Moltbook followers by 2026-05-18? |
| futurex_6972189f11cfd2006997647f | 1 | 0.31 | 0.36/1 | 0.31/0 | 0.61 | 0.30 | New METR SOTA by end of January, 2026? |
| futurex_69a2e0165692ef005cdbf23e | 0 | 0.67 | 0.50/1 | 0.66/0 | 0.83 | 0.93 | Trump Sucks? |
| futurex_6a2bfc099b0b8587160de250 | 1 | 0.36 | 0.32/1 | 0.39/0 | 0.18 | 0.49 | Will Spain, Brazil, and France all finish first in their respective groups at the 2026 FIF |
| futurex_6964e98652029b005bc00997 | 1 | 0.37 | 0.48/1 | 0.38/0 | 0.73 | 0.64 | Global presence of algae in coral reefs to increase in comparison to 2021? |
| futurex_6964e98952029b005bc009b6 | 1 | 0.39 | 0.43/1 | 0.39/0 | 0.24 | 0.84 | The No-Bots Market. Which side will be automated-bots-free market close? |
| futurex_69f1fa089f7b49005d158498 | 0 | 0.59 | 0.59/1 | 0.59/0 | 0.57 | 0.69 | US unemployment rate above 4.3% in April? |

### debate_fixed

| qid | y | s2_p | r0 | r1 | cot | std | question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| futurex_6a0efb1f32b3e1005de2cd3b | 0 | 0.48 | 0.60/1 | 0.50/1 | 0.72 | 0.74 | Will Trailer 3 for Grand Theft Auto 6 be released by June 15, 2026? |
| futurex_69d3a7a3e53ae80066abf5b8 | 0 | 0.45 | 0.59/1 | 0.47/2 | 0.33 | 0.39 | Will the US-Iran conflict end by 30th April? |
| futurex_69f891cbab4486005c6c3837 | 0 | 0.43 | 0.56/1 | 0.44/2 | 0.18 | 0.11 | US Consumer Price Index (April '26) < 3.55% |
| futurex_69a2e39e5692ef005cdbf273 | 0 | 0.43 | 0.61/1 | 0.45/2 | 0.97 | 0.59 | FIFA World Cup 2026 Qualification Longshots Parlay |
| futurex_69de339aab5b1b0067f23c75 | 0 | 0.41 | 0.52/2 | 0.42/3 | 0.23 | 0.14 | Will a Iranian warship be destroyed in the ports blockade bef April 30th |
| futurex_6963980d5a6f9800684ed65f | 0 | 0.38 | 0.61/1 | 0.40/3 | 0.56 | 0.40 | Will Maryland change its congressional voting map for the 2026 elections? |
| futurex_69de36e8ab5b1b0067f23c80 | 0 | 0.29 | 0.52/1 | 0.31/3 | 0.36 | 0.26 | Did a crypto hedge fund blow up? |
| futurex_69a2e39e5692ef005cdbf2a8 | 0 | 0.22 | 0.55/1 | 0.23/3 | 0.78 | 0.89 | 50m views on a MrBeast video in the first day by March 31? |

