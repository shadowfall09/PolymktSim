# Paper Plan

**Title**: Diverse Evidence, Better Forecasts: Multi-Agent Deliberation Under Information Asymmetry
**Venue**: ACL/EMNLP (ACL LaTeX template — venue override; NOT ICLR)
**Type**: Method + theory + empirical
**Date**: 2026-07-19
**Page budget**: 8 pages main body (ACL long), references & appendix unlimited
**Section count**: 7
**Output dir**: `paper_v3/` (LaTeX), figures in `paper_v3/figures/`
**Old version**: `paper_v2/latex/acl_latex.tex` — structure/theory/framework figure reusable; ALL numbers stale

## Hard decisions (from user, do not revisit)

1. Main table has **9 baselines** (no zeroshot): CoT, Self-Consistency, Superforecaster, Halawi, Bayesian-k5 (single-agent); Standard Debate, MoA, Crowd Ensemble, AIA (multi-agent). All gpt-5.4-mini, same fixed evidence.
2. **calibrated_shrink is a method component** (aggregation layer), applied only to our method in the main table. Formula: p' = p0 + w·(p−p0), w = w_hi if p>p0 else w_lo; (p0=0.30, w_lo=0.8, w_hi=0.5), params cross-validated poly↔futurex.
3. **Shrink-on-baselines control goes to appendix**, reported honestly: ranking preserved, ours still best, but several gaps not significant.
4. Prompt v1/v2/v3 calibration study = main-text analysis section.
5. Sports retrieval double null result = analysis/limitations.
6. Cross-model: only futurex gpt-4o-mini run (0.2370) exists → one footnote, no section.
7. Abstract must NOT say "375 questions" or "12–18% over strongest baselines" from old version. Datasets: Polymarket 250 + FutureX 231.

## Verified numbers (recomputed 2026-07-19 from data/results_v2 jsonl; do not trust old tex)

### Polymarket-250 (Brier / Acc)
| Method | Brier | Acc |
|---|---|---|
| CoT | 0.2063 | 0.696 |
| Self-Consistency | 0.1965 | 0.712 |
| Superforecaster | 0.1937 | 0.700 |
| Halawi et al. | 0.1944 | 0.716 |
| Sequential Bayesian (k=5) | 0.2523 | 0.540 |
| Standard Debate | 0.1815 | 0.720 |
| MoA | 0.1905 | 0.720 |
| Crowd Ensemble | 0.1992 | 0.700 |
| AIA Forecaster | 0.1897 | 0.704 |
| InfoDelphi w/o shrink | 0.1647 | 0.748 |
| **InfoDelphi (full)** | **0.1575** | (recompute shrink acc in Phase 2) |

### FutureX-231 (Brier / Acc)
| Method | Brier | Acc |
|---|---|---|
| CoT | 0.2611 | 0.606 |
| Self-Consistency | 0.2713 | 0.550 |
| Superforecaster | 0.2596 | 0.636 |
| Halawi et al. | 0.2654 | 0.567 |
| Sequential Bayesian (k=5) | 0.3043 | 0.485 |
| Standard Debate | 0.2433 | 0.602 |
| MoA | 0.2944 | 0.545 |
| Crowd Ensemble | 0.2888 | 0.524 |
| AIA Forecaster | 0.2764 | 0.550 |
| InfoDelphi w/o shrink | 0.2212 | 0.688 |
| **InfoDelphi (full)** | **0.2006** | 0.701 |

### Significance (paired bootstrap, 5000 resamples, 95% CI, full method vs each baseline)
Poly: all 9 CIs exclude 0 (e.g., vs Standard Debate −0.0240 [−0.0419,−0.0076]; vs CoT −0.0488 [−0.0762,−0.0225]).
FutureX: all 9 exclude 0 (e.g., vs Standard Debate −0.0427 [−0.0748,−0.0099]).
→ regenerate exact CI table with seeded script in Phase 2 (seed 0).

### Ablations available at paper scale (only these — old dev-set ρ/share-mode sweeps are NOT paper-grade, drop them)
- Deliberation vs independent (paired, same run): poly s1 0.1698 → s2 0.1647; second run 0.1694 → 0.1623; futurex earlier config s1 0.2735 → s2 0.2495.
- Homogeneous input (Standard Debate = all agents see all evidence) vs partitioned: 0.1815 vs 0.1647 (poly), 0.2433 vs 0.2212 (futurex) — both without shrink, apples-to-apples.
- Aggregation layer: raw mean → calibrated_shrink: 0.1647→0.1575 (poly), 0.2212→0.2006 (futurex); shrink params transferred across datasets.
- Prompt calibration study (poly, s2, raw / +shrink): v1 0.1647/0.1575; v2+devils-advocate 0.1896/0.1779; v3 soft-criteria 0.1780/0.1689. Mechanism: v2 mean_p 0.27→0.16, v3 →0.20 vs base rate 0.232 (whole distribution depressed; cannot target the overconfident band).
- Run variance note: two identical-config runs 0.1623 vs 0.1647 (Δ≈0.002).
- NEW (compute in Phase 2 from *_detail.jsonl per-agent forecasts): inter-agent error correlation, InfoDelphi vs Standard Debate — empirical support for Prop. 1.

### Sports null results (106 sports questions, s2, paired vs original evidence)
original 0.1908 (AUC 0.697); +slug metadata 0.1890, diff −0.0018 (95% CI −0.010,+0.006); +opponent odds/form 0.1873 (AUC 0.719), diff −0.0035 (−0.012,+0.005). 52–53% of forecasts cite new docs; 43% of retrieved docs contain numeric odds. A/A controls moved as much as treated subsets.

### Reference points (for text, not main table)
Constant base-rate predictor (0.232) on poly: Brier 0.1782. FutureX gpt-4o-mini InfoDelphi: 0.2370 (footnote only).

## Claims-Evidence Matrix

| # | Claim | Evidence | Status | Section |
|---|---|---|---|---|
| C1 | Designed information asymmetry + rationale deliberation + calibrated aggregation beats 9 single-/multi-agent baselines on both datasets, significantly (paired bootstrap) | Main tables + CI table | Supported | §5 |
| C2 | Homogeneous input induces correlated errors; partitioning decorrelates | Prop. 1 + Cor. 1 (reuse) + NEW empirical inter-agent correlation from detail files + Standard Debate contrast | Supported (empirical part to compute) | §3, §6.1 |
| C3 | Deliberation round improves over independent forecasting | s1 vs s2 paired (both datasets) | Supported (small effect; report honestly with paired CI) | §6.1 |
| C4 | Overconfidence must be fixed in the aggregation layer, not the prompt | v1/v2/v3 study + shrink transfer poly↔futurex | Supported | §6.2 |
| C5 | Sports questions are evidence-limited at the model level, not retrieval level | two targeted-retrieval nulls with paired CIs + consumption stats | Supported (null) | §6.3 |

## Structure

### §0 Abstract (~180 words)
Problem: multi-agent forecasting overlooks *what information each agent receives*; identical evidence → herding. Approach: designed information asymmetry (public/private partition) + rationale deliberation + calibrated aggregation = InfoDelphi. Key result: on 250 Polymarket + 231 FutureX questions with fixed pre-resolution evidence, InfoDelphi attains Brier 0.1575/0.2006, outperforming all nine single- and multi-agent baselines under paired bootstrap. Also: prompt-level skepticism cannot fix overconfidence — the aggregation layer can (calibration study). NO "375", NO "12–18%"; if a relative number is used: "13–18% lower Brier than the strongest baseline on each dataset" (vs Standard Debate: 13.2% poly, 17.6% futurex) — verify wording against table.

### §1 Introduction (~1.25 pp)
Hook: LLM deliberation assumed to help; with identical input it collapses into herding. Gap: information allocation is a neglected design axis. Contributions: (i) information-asymmetry framing + theory; (ii) InfoDelphi framework incl. calibrated-shrink aggregation; (iii) evaluation on two benchmarks vs 9 baselines with significance; (iv) analysis: prompt-vs-aggregation calibration, deliberation dynamics, sports nulls. Hero figure: reuse framework.png (fig spans page top). Reuse intro.pdf motivation figure if it still matches numbers — CHECK before reuse; drop if stale.

### §2 Related Work (~0.75 pp)
Reuse old §2 (LLM forecasting; multi-agent debate; ensemble diversity/calibration) — verify citations, add calibration-of-LLM-forecasters refs. Position: prior debate work varies *protocol*, we vary *information structure*.

### §3 Method (~2 pp)
3.1 Problem formulation (reuse). 3.2 Theory (reuse Prop. 1 + proof sketch, Cor. 1; full proofs appendix). 3.3 InfoDelphi: evidence routing (BM25 + round-robin), iterative deliberation with rationale sharing (arguments mode + evidence pooling + calibrated prompt), **NEW: calibrated-shrink aggregation** — asymmetric shrinkage toward NO-leaning prior p0; motivation: LLM forecasters systematically overconfident on YES side; params fit on one dataset, validated on the other (state protocol precisely).

### §4 Experimental Setup (~0.75 pp)
Datasets: PolyGym-250 (Polymarket, temporally filtered evidence ≤30 docs/question) + FutureX-231 (past split). Baselines: 9, two categories, all gpt-5.4-mini, same evidence pool, appendix details. Metrics: Brier, accuracy; paired bootstrap protocol (5000, seed, 95% CI). Implementation: J=3, R=2, ρ=0.5, temp 0.7.

### §5 Main Results (~1.25 pp)
Table 1 (both datasets, 9 baselines + ours w/o shrink + ours full; bold best, † = significant vs ours). Text: multi-agent baselines with homogeneous input often *underperform* single-agent prompting (MoA 0.1905/0.2944 vs CoT 0.2063/0.2611 — mixed on poly, clear on futurex; phrase carefully per-dataset). Ours w/o shrink already beats all 9 baselines on both datasets; shrink adds further calibration gain. Footnote: gpt-4o-mini futurex run 0.2370 suggests architecture transfers across models (no baseline suite → footnote only).

### §6 Analysis (~1.5 pp)
6.1 Why asymmetry helps: inter-agent correlation (NEW fig) InfoDelphi vs Standard Debate; s1→s2 paired deliberation gain; homogeneous-input contrast.
6.2 Calibration study: Table/fig of v1/v2/v3 × {raw, +shrink}; mechanism (mean_p depression); takeaway C4; shrink cross-dataset transfer.
6.3 What deliberation cannot fix: sports nulls (Table with paired CIs + consumption stats).

### §7 Conclusion + Limitations (~0.5 pp)
Limitations (honest): single primary model (gpt-5.4-mini); binary questions only; fixed evidence pools (no live retrieval); shrink params dataset-pair validated, not universal; sports/evidence-limited domains; effect vs strongest debate baseline is modest on Polymarket (appendix control).

### Appendices
A. PolyGym-250 dataset details (reuse old App. A, fix counts: 250 not 375; category breakdown — recompute from data). B. Baseline descriptions (reuse, drop zeroshot entry). C. Proofs (reuse). D. **Shrink-on-baselines control**: full table both datasets, raw/+shrink for all 9 baselines, paired CIs vs ours (both shrunk); state plainly ranking preserved, ours best, some gaps n.s. E. Prompts (v1/v2/v3 + devils-advocate). F. Case study (reuse if consistent). G. Computational cost (reuse structure, update).

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|---|---|---|---|---|
| Fig 1 | Architecture (manual, exists) | framework.png from paper_v2 | copy | HIGH |
| Tab 1 | Main results | 2 datasets × 11 rows, Brier/Acc, sig markers | results_v2 jsonl | HIGH |
| Fig 2 | Inter-agent error correlation | InfoDelphi vs Standard Debate, per-question corr of agent errors (violin/bar), both datasets | *_detail.jsonl | HIGH |
| Fig 3 | Calibration study | grouped bars v1/v2/v3 × raw/+shrink + mean_p vs base rate inset; or reliability diagram | results_v2 jsonl | HIGH |
| Tab 2 | Ablation | s1 vs s2, homogeneous vs partitioned, mean vs shrink | results_v2 jsonl | HIGH |
| Tab 3 | Sports nulls | 3 evidence conditions × Brier/AUC/paired diff CI | sports jsonl + MANIFEST | MED |
| Tab 4 (App D) | Shrink-on-baselines control | 9 baselines × raw/+shrink × 2 datasets + CIs | results_v2 jsonl | HIGH |
| Tab 5 (App) | Significance CIs full | ours vs each baseline, both datasets | script | MED |
| Fig 4 (App, optional) | Reliability diagrams before/after shrink | detail/main jsonl | LOW |

All generated by scripts in `paper_v3/figures/gen_*.py` with seed 0; numbers must match Verified-numbers section above (assert in scripts).

## Citation Plan

Reuse `paper_v2/latex/custom.bib` — verify each key used. §1: du2023improving, halawi2024approaching, schoenegger2024wisdom. §2: debate (du2023improving, liang2023encouraging…), forecasting (halawi2024approaching, karger2024forecastbench, turtel2025llms, alur2025aia), diversity/calibration (wood2023unified, kim2025correlated, turkmen2026don, 10.1214/ss/1177013825). §3: shin2026reasoning, wangself, wei2022chain, kojima2022large (CoT/SC), shi2023language, wangmixture. New candidates [VERIFY]: LLM calibration/overconfidence refs if custom.bib lacks them. Never invent BibTeX; only reuse verified entries or mark [VERIFY].

## Reviewer Feedback

External GPT-5.4 review skipped: Codex MCP not available in this session. Mitigation: Phase 5 improvement loop will use an available reviewer route; self-check against Claims-Evidence Matrix before writing.

## Next Steps
- [ ] /paper-figure → `paper_v3/figures/` (scripts + PDFs + latex_includes.tex; recompute shrink acc, corr fig, CI tables)
- [ ] /paper-write → `paper_v3/` LaTeX (ACL template from paper_v2)
- [ ] /paper-compile → `paper_v3/main.pdf`
- [ ] improvement loop ×2
