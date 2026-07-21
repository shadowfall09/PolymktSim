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
