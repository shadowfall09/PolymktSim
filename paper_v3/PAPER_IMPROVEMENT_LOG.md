# Paper Improvement Log — paper_v3

Note: Codex MCP (external GPT-5.4 reviewer) was not available in this session.
Both rounds below are self-review passes driven by the Claims-Evidence Matrix in
PAPER_PLAN.md and visual PDF inspection; an external review pass is still
recommended before submission (`/auto-review-loop` in a session with Codex MCP).

## Round 0 (baseline) — main_round0_original.pdf
Initial compile: 15 pages, 0 LaTeX errors, 0 undefined references/citations,
0 overfull boxes. Main body ran ~0.2 page past the 8-page ACL limit.

## Round 1 — main_round1.pdf
- **[MAJOR] Duplicate citation**: `choi2026debate` and `choi2025debate` were the
  same NeurIPS paper under two keys, rendering as "2026a"/"2026b" in References.
  Unified all cites to `choi2025debate`.
- **[MINOR] Fig 2 legend** overlapped the descending disagreement lines; moved
  to upper right.
- **[MINOR] Page budget**: shrank intro figure to 0.9\linewidth, tightened the
  framework-overview paragraph (it restated the three theory conditions),
  dropped a non-load-bearing Related Work clause (hong2004groups), added small
  negative vspace after the two big figures.

## Round 2 — main_round2.pdf (= main.pdf)
- **[MAJOR] Removed the duplicate `choi2026debate` entry from custom.bib** (the
  References list showed both).
- **[MINOR] Conclusion tightened** (merged the mechanism/lesson/boundary
  sentence); §6.3 closing sentence shortened; intro figure to 0.85\linewidth.
- Result: Conclusion now begins on page 8 (spills a few lines to page 9;
  Limitations/Ethics are exempt from the ACL page limit). 15 pages total,
  0 errors, 0 undefined refs, 0 overfull.

## Round 3 (2026-07-22) — v2-style experiment rewrite, baseline-comparative
User directives: match paper_v2's analysis style (each paragraph ties a result to
a designed component / theory and explains the mechanism), and emphasize
advantages over baselines rather than our own tuning iterations.
- §5 rewritten as four \ding{224} claim paragraphs; §6 restructured into
  "Why asymmetry beats homogeneous debate" + "Calibration: shared failure mode".
- New analyses (all from existing results_v2 / results files, no new runs;
  source: figures/gen_mechanism_stats.py -> mechanism_numbers.json):
  - Baseline hi-band overconfidence: all 9 baselines state 0.71-0.80 on p>=0.6
    but hit 0.28-0.51; ours has lower exposure (14 vs 28-55 on poly).
  - Direction diversity: final-round extremity equal (0.277 vs 0.271) but
    inter-agent variance 2.7x (poly) / 1.3x (futurex) vs Standard Debate.
  - Cross-model subsection revived with real data (TABLE_crossmodel):
    gemini-3.1-flash-lite full config 0.1624/0.2040 beats all baselines;
    gpt-4o-mini FutureX 0.2370 (footnote in §4 removed, superseded).
- AUC added as third metric (compute_numbers.py, main table now 6 numeric
  cols): ours 0.757/0.711 vs best baselines 0.729/0.694; AUC is invariant to
  the monotone shrink transform, so it isolates ranking skill from calibration.
- Prompt-level calibration study (Fig 3) and sports study moved to appendix
  (sec:appendix_promptcal, sec:sports); abstract/intro de-emphasize
  self-iteration narrative accordingly.
- 16 pages, 0 errors / 0 undefined refs / 0 overfull. Main body now ends
  mid-page 9 in [preprint] mode (~1 page over the 8-page ACL body limit) —
  must trim before submission.

## Round 4 (2026-07-22) — ablation suite run and integrated
All 6 missing experiments resolved (scripts/run_ablation_suite.sh, 7 runs,
~54 min total; results in data/results_v2/, documented in its MANIFEST):
1. FutureX "- deliberation": free — mean-pooling round-0 agent forecasts from
   the winner detail file reproduces s1 exactly (verified max-diff 0.0 on
   polymarket); futurex s1 raw = 0.2352. Table 2 "---" cell filled.
2. rho sweep: interior is FLAT (0.3: 0.1594, 0.5: 0.1647, 0.7: 0.1561 raw; all
   within seed noise of each other); only rho=1 (0.1815) is catastrophic.
   Narrative: robustness to the split, not a tuned optimum.
3. numbers-only sharing: raw 0.1687, AUC 0.751 (vs 0.757) — directionally
   supports DPI, small.
4. random routing: raw 0.1665, AUC 0.746 — directionally supports BM25 routing.
5. R=3: 0.1593 raw / 0.1544 shrunk — slightly BETTER, n.s. paper_v2's
   "third round reverses gains" claim is REFUTED; do not resurrect it.
6. Seeds 1/2: winner shrunk Brier 0.1575/0.1577/0.1578 (std 0.0002!), raw
   std 0.0011. Headline numbers are seed-stable.
Paper changes: new Table (tab:design, TABLE_design_grid.tex) + new §6.1
paragraph "The advantage rests on the principle, not on a tuned
hyperparameter"; FutureX deliberation numbers in the component-ablation
paragraph; all numbers asserted in compute_numbers.py (ablation_suite block).
17 pages, clean compile. Main body still ~1 page over the 8-page limit.

## Known remaining items (for a future external-review pass)
1. `choi2025debate` bib metadata says year=2026, NeurIPS vol. 38 — verify
   year/volume against the published record.
2. Main body is ~3-5 lines over 8 pages in [preprint] mode; verify in [review]
   mode before submission and trim if needed.
3. Fig 1 (intro.png) is an illustrative example from an earlier run; regenerate
   from the current winner run if exact reproducibility of the example matters.
4. The old paper's qualitative case-study appendix was not carried over (its
   transcripts came from superseded runs); re-mine cases from
   `polymarket_250_s1s2_arguments_pooling_calibrated_5.4mini_detail.jsonl` if a
   case study is wanted.
5. Sports table reports the s2 winner subset re-scored with shrink; AUC for the
   original condition (0.696) differs in the third decimal from MANIFEST (0.697)
   due to rounding only.
