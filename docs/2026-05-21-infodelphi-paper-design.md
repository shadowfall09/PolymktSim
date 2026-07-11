# InfoDelphi Paper Design — EMNLP Submission

## Core Claim

**Information asymmetry promotes reasoning.** When agents each hold different private evidence, deliberation produces genuine information gain — unlike standard multi-agent debate where identical inputs lead to herding.

## Framework Name

- **InfoDelphi** — the full framework (evidence routing + iterative deliberation + confidence-weighted aggregation)
- Internal protocol names (for paper):
  - Single-Agent baseline (code: S0)
  - Independent aggregation (code: S1)
  - Iterative deliberation / InfoDelphi (code: S2)

## Title Direction

"Information Asymmetry as a Design Principle for Multi-Agent Forecasting"

(Alternatives: "InfoDelphi: Deliberation under Designed Information Asymmetry for LLM Forecasting")

---

## Paper Structure

### §1 Introduction (5 paragraphs)

1. Prediction markets are important + LLM forecasting is rising
2. Multi-agent debate seems promising but has a fundamental flaw: identical inputs → homogeneous reasoning → herding (martingale, Choi et al. 2025)
3. Our insight: deliberately creating information asymmetry — like expert panels where each expert brings unique knowledge
4. Method overview + main results (Brier -17.6%, Accuracy +8pp on 375 Polymarket questions)
5. Contributions:
   - (1) Information asymmetry as a design principle for multi-agent reasoning, with theoretical justification
   - (2) PolyGym benchmark (375 temporally-filtered binary prediction questions + web evidence)
   - (3) Comprehensive ablation study revealing the contribution of each design choice

### §2 Related Work (3 subsections)

1. **LLM-based Forecasting** — ForecastBench, Halawi et al. 2024, Turtel et al. 2025, PolyBench
2. **Multi-Agent Reasoning & Debate** — Du et al. 2023, MoA, Choi martingale result, AceMAD, self-consistency
3. **Information Aggregation & Collective Intelligence** — Condorcet, Surowiecki conditions, DeGroot, opinion pools, extremizing

### §3 Methodology

#### §3.1 Theoretical Motivation (~2 pages)

**Narrative: "Negative result → Our fix"**

Step 1 — Establish why debate under homogeneous input fails:
- Cite Choi et al. (NeurIPS 2025): standard MAD is a martingale (zero expected improvement)
- Cite MAP Saturation (arXiv 2602.08003): under correlated errors, ensemble MSE has a non-zero floor: MSE_ensemble ≥ ρ · MSE_individual
- Cite Correlated Errors (ICML 2025): LLM errors are 60%+ correlated

Step 2 — **Proposition 1 (Diversity-Induced Decorrelation)**:
- Decompose agent prediction error into public-information component and private-information component
- Public component: correlated across agents (shared input)
- Private component: independent across agents (disjoint evidence)
- Prove: ρ_asymmetric ≤ (public_ratio)² · ρ_symmetric
- i.e., information asymmetry reduces error correlation by (public_ratio)² factor
- Proof technique: linear signal model, bias-variance-covariance decomposition (Wood et al. JMLR 2023 style)
- Full proof in Appendix

Step 3 — **Corollary 1 (Optimal public ratio trade-off)**:
- r too small → agents lack common ground, cannot interpret each other's reasoning
- r too large → errors re-correlate, aggregation loses power
- Predicts existence of optimal r* ∈ (0, 1), validated by ablation

Step 4 — Rationale sharing breaks DPI:
- Cite Reasoning Trap (arXiv 2605.01704): closed-system debate satisfies Markov property → DPI → information can only decrease
- Our design is open-system: rationale sharing transmits partial information from E_priv^j to agent i
- Therefore H(Y | context_i, R_j) < H(Y | context_i) — information strictly increases
- Semi-formal argument, no full proof needed

#### §3.2 Design Space & Instantiation

Three-dimensional design space table:

| Dimension | Options | Our Choice | Justification |
|-----------|---------|------------|---------------|
| Evidence allocation | All-shared / Random split / Relevance-based | BM25 routing | Satisfies desiderata 1+2 (common ground + exclusive expertise) |
| Interaction topology | None / One-shot / Iterative | 2-round deliberation with rationale | Satisfies desideratum 3 (effective communication) |
| Aggregation | Mean / Voting / Weighted | Confidence-weighted (logit-space) | Exploits extremity ≈ evidence quality |

#### §3.3 Evidence Routing

- BM25 relevance scoring of evidence items to question
- Top public_ratio fraction → public pool (shared by all agents)
- Remainder → round-robin assignment by relevance rank → private subsets
- Effect: public pool = most relevant (common ground); private = diverse peripheral evidence

#### §3.4 Iterative Deliberation Protocol

- Round 1: agent_k receives E_pub ∪ E_priv^k → outputs (p_yes, rationale)
- Round 2: agent_k additionally receives {(p_j, rationale_excerpt_j)}_{j≠k} → re-predicts
- Key: sharing rationale (not just p_yes) is what enables information transfer from private evidence

#### §3.5 Confidence-Weighted Aggregation

- Weight w_i = |logit(p_i)| (agents with extreme predictions get higher weight)
- p_agg = σ(Σ w_i · logit(p_i) / Σ w_i)
- Intuition: agents with strong private evidence tend to be more confident → confidence is a proxy for information quality

### §4 PolyGym Benchmark

- Source: 500 Polymarket binary questions
- Temporal filter: endDate ≥ 2025-09-01 → 375 effective questions (prevents LLM prior knowledge contamination)
- Evidence: ~20 web search results per question (Tavily API, title + URL + snippet ≤ 500 chars)
- Topics: Crypto, Sports, SciTech, Politics, Business
- Ground truth: market resolution (Yes/No)

### §5 Experiments

#### §5.1 Setup
- Primary model: gpt-4o-mini (via OpenRouter)
- Cross-model: gemini-flash, deepseek-v3
- Metrics: Brier score (primary), Accuracy (secondary)
- Baselines: Zero-shot (no evidence), Direct (no CoT), Single-Agent (RAG+CoT), Sequential Bayesian

#### §5.2 Main Results
Main comparison table (all methods on 375 questions)

#### §5.3 Ablation: Validating the Theory

Each ablation validates a theoretical prediction:

| Ablation | Validates | Expected finding |
|----------|-----------|-----------------|
| Public ratio (0.3/0.5/0.7) | Corollary 1 (optimal r* exists) | 0.5 is optimal |
| BM25 vs Random routing | Proposition 1 (quality of split matters) | BM25 reduces correlation more |
| With vs Without rationale sharing | DPI argument (open vs closed system) | Without rationale, debate is nearly useless |
| Rounds (1/2/3) | Submartingale convergence | 2 rounds optimal, 3 shows diminishing returns |
| Agents (2/3/5/7) | Diversity scaling | 3 optimal, more agents → redundancy |
| Aggregation (Mean vs CW) | Confidence = info quality | CW outperforms Mean |

#### §5.4 Cross-Model Generalization
- 3 different LLM backends
- Show consistent improvement pattern → not model-specific

#### §5.5 Analysis
- Herding quantification: agent prediction variance (Random vs BM25, round 1 vs round 2)
- Extremity-accuracy relationship: scatter plot showing confident predictions are more accurate
- Error analysis / case study (optional)

### §6 Conclusion & Limitations

Limitations to acknowledge:
- Linear model assumption in Proposition 1 (LLMs are non-linear)
- Binary questions only
- Static evidence (no live retrieval)
- Cost increases with agents × rounds

---

## Figures Plan

| # | Content | Section | Purpose |
|---|---------|---------|---------|
| 1 | InfoDelphi framework diagram (evidence → BM25 routing → agents with pub/priv → debate rounds → CW aggregation) | §3 | Visual overview |
| 2 | Public ratio ablation curve (Brier vs public_ratio) | §5.3 | Validate Corollary 1 |
| 3 | Agent variance distribution (Random vs BM25; Round 1 vs Round 2) | §5.5 | Show anti-herding effect |
| 4 | Extremity vs Brier scatter | §5.5 | Justify CW aggregation |

---

## Key References to Cite

### Theory support
- Choi et al. 2025 (NeurIPS) — martingale result for MAD
- AceMAD 2025 — breaking martingale via asymmetry
- Reasoning Trap 2025 — DPI bound on closed-system debate
- Kim et al. 2025 (ICML) — correlated LLM errors
- MAP Saturation 2026 — error floor under correlation
- Wood et al. 2023 (JMLR) — bias-variance-diversity decomposition

### Forecasting
- Halawi et al. 2024, Turtel et al. 2025, ForecastBench, PolyBench

### Multi-agent
- Du et al. 2023, Mixture-of-Agents, self-consistency

### Collective intelligence
- Surowiecki 2004, Condorcet, DeGroot 1974, Hong-Page 2004

---

## Experimental Status

Results already available for all main experiments (375 questions):
- All baselines ✓
- Main comparison (Single-Agent / Independent / InfoDelphi) ✓
- All ablations ✓
- Cross-model (3 backends) ✓

May need to supplement:
- Additional LLM backends for robustness (optional)
- Formal herding metrics (can compute from existing JSONL data)
- Case studies (select from existing results)
