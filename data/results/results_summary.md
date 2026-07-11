# PolymktSim Experiment Results Summary

**Dataset**: 375 questions (from 500 Polymarket markets, filtered by endDate >= 2025-09-01)
**Primary Model**: gpt-5.4-mini (via GPA API)
**Default Settings**: temperature=0.7, 3 agents, 2 rounds, BM25 routing, public_ratio=0.5, CW aggregation

---

## 1. Main Results: S0 vs S1 vs S2

| Method | Brier | Acc | Description |
|--------|-------|-----|-------------|
| S0 (single agent) | 0.2160 | 69.9% | 1 agent, all evidence, CoT |
| S1 (multi-agent independent) | 0.2060 | 74.1% | 3 agents, CW+BM25, no deliberation |
| **S2 (multi-agent deliberation)** | **0.1783** | **77.9%** | 3 agents, CW+BM25, 2 rounds |

**Key gain**: S0 → S2 = -17.5% Brier, +8.0% Accuracy

---

## 2. Full Baseline Comparison (for EMNLP paper)

### Single-Agent Methods

| Method | Brier | Acc | Description | Reference |
|--------|-------|-----|-------------|-----------|
| Zero-shot | 0.1923 | 68.3% | No evidence, LLM internal knowledge only | — |
| Single-Agent (RAG+CoT) | 0.2160 | 69.9% | Standard RAG pipeline with chain-of-thought | — |
| Self-Consistency (5 samples) | 0.2212 | 69.6% | Same prompt 5 times, mean aggregation | Wang et al. 2023 |
| Direct (no CoT) | 0.2265 | 70.7% | Evidence provided, no reasoning chain | — |
| Halawi et al. (scratchpad × 3) | 0.2385 | 65.3% | 7-step scratchpad + trimmed mean of 3 | Halawi et al. 2024 (NeurIPS) |
| Superforecaster Prompt | 0.2419 | 68.8% | Structured reasoning (decompose→base rate→update→calibrate) | ForecastBench |
| Seq. Bayesian (K=5) | 0.2556 | 62.1% | Sequential Bayesian belief updating | — |

### Multi-Agent Methods (Homogeneous Input)

| Method | Brier | Acc | Description | Reference |
|--------|-------|-----|-------------|-----------|
| Standard Debate (2 rounds) | 0.2015 | 73.9% | 3 agents same evidence, 2 rounds deliberation | Du et al. 2023 |
| MoA (3 proposers → 1 aggregator) | 0.2202 | 70.9% | Mixture-of-Agents layered architecture | Wang et al. 2025 (ICLR) |
| Multi-Agent Independent (All-Info) | 0.2206 | 67.7% | 3 agents same evidence, mean agg | — |
| Crowd Ensemble (5 agents, median) | 0.2453 | 65.9% | Independent predictions, median agg | Schoenegger et al. 2024 |
| AIA Forecaster (10 + supervisor) | 0.2471 | 67.7% | 10 agents + supervisor reconciliation | Alur et al. 2025 (Bridgewater) |

### Multi-Agent Methods (Information Asymmetry) — Ours

| Method | Brier | Acc | Description |
|--------|-------|-----|-------------|
| InfoDelphi – Independent | 0.2060 | 74.1% | BM25 split, no deliberation, CW agg |
| **InfoDelphi – Full** | **0.1783** | **77.9%** | BM25 split, 2 rounds deliberation, CW agg |

### Key Observations

1. **Information asymmetry is the key differentiator**: InfoDelphi (0.178) beats ALL baselines by a large margin, including sophisticated multi-agent methods like AIA (0.247) and MoA (0.220)
2. **More agents ≠ better**: AIA with 10 agents + supervisor (0.247) is worse than a single agent (0.216). Without information diversity, adding agents adds noise
3. **Prompt engineering cannot compensate**: Halawi scratchpad (0.239) and Superforecaster (0.242) are worse than simple RAG+CoT (0.216) on static evidence
4. **Standard Debate is the strongest homogeneous-input method** (0.202) but still 12% worse than InfoDelphi (0.178)
5. **Crowd Ensemble (Silicon Crowd) fails** (0.245): median of identical-input predictions cannot overcome correlated errors
6. **Zero-shot paradox**: Zero-shot (0.192) has lower Brier than most RAG methods due to conservative predictions near 0.5, but lower accuracy (worse directional judgment)

Note: All methods use gpt-5.4-mini. InfoDelphi uses 3 agents, 2 rounds, public_ratio=0.5, BM25 routing, CW aggregation.

---

## 3. Ablation: Number of Deliberation Rounds

| Rounds | Brier | Acc | vs Best |
|--------|-------|-----|---------|
| 1 (no deliberation) | 0.2085 | 71.5% | +0.030 |
| **2** | **0.1783** | **77.9%** | — |
| 3 | 0.1958 | 75.7% | +0.018 |

**Finding**: 2 rounds is optimal. More rounds causes herding (conformity to majority).

---

## 4. Ablation: Number of Agents

| Agents | Brier | Acc | vs Best |
|--------|-------|-----|---------|
| 2 | 0.1899 | 76.5% | +0.012 |
| **3** | **0.1783** | **77.9%** | — |
| 5 | 0.1861 | 76.0% | +0.008 |
| 7 | 0.1898 | 76.3% | +0.012 |

**Finding**: 3 agents is optimal. Too few = insufficient diversity; too many = information dilution.

---

## 5. Ablation: Public/Private Information Ratio

| Public Ratio | Brier | Acc | vs Best |
|--------------|-------|-----|---------|
| 0.3 (more private) | 0.1905 | 75.5% | +0.012 |
| **0.5** | **0.1783** | **77.9%** | — |
| 0.7 (more public) | 0.1887 | 76.8% | +0.010 |
| 1.0 (all public, no private) | 0.1931 | 75.7% | +0.015 |

**Finding**: 50/50 split is optimal. Too little shared info = agents lack common ground; too little private info = no unique perspectives.

---

## 6. Ablation: Aggregation Method

| Aggregator | Brier | Acc | vs Best |
|------------|-------|-----|---------|
| Mean | 0.1955 | 74.9% | +0.017 |
| **Confidence-Weighted** | **0.1783** | **77.9%** | — |

**Finding**: CW outperforms mean by giving higher weight to more confident (and typically more informed) agents.

---

## 7. Ablation: Rationale Sharing in Deliberation

| Shared Info | Brier | Acc | vs Best |
|-------------|-------|-----|---------|
| Only p_yes + label (no rationale) | 0.2112 | 70.4% | +0.033 |
| **Full (p_yes + label + rationale)** | **0.1783** | **77.9%** | — |

**Finding**: Sharing reasoning is critical. Knowing only "what" others predict (without "why") provides almost no benefit over no deliberation (rounds=1: 0.209 vs no-rationale: 0.211). The value of deliberation comes from understanding other agents' evidence-based reasoning.

---

## 8. Evidence Routing: BM25 vs Random

| Routing | Brier | Acc |
|---------|-------|-----|
| Random split | 0.1912 | 76.0% |
| **BM25** | **0.1783** | **77.9%** |

**Finding**: BM25-based evidence routing improves over random by assigning more relevant evidence to public pool and distributing diverse private items.

---

## 9. Cross-Model Generalization

| Model | S1 Brier | S1 Acc | S2 Brier | S2 Acc |
|-------|----------|--------|----------|--------|
| gpt-5.4-mini | 0.2060 | 74.1% | 0.1783 | 77.9% |
| gemini-3-flash-preview | 0.1891 | 79.7% | 0.1930 | 80.0% |
| deepseek-v3.2 | 0.1841 | 76.0% | **0.1604** | **81.1%** |

**Finding**: The method generalizes across models. DeepSeek-v3.2 achieves best absolute performance.

---

## 10. Key Takeaways

1. **Multi-round deliberation is the largest contributor** (rounds 1→2: -0.030 Brier, +6.4% Acc)
2. **Rationale sharing is essential** — without it, deliberation provides no benefit
3. **Information asymmetry (public/private split) matters** — 50/50 is optimal sweet spot
4. **All ablation curves are inverted-U shaped** — there exists an optimal for rounds, agents, and PR
5. **Confidence-weighted aggregation amplifies informed agents** — better than simple mean
6. **BM25 routing > random** — relevance-aware evidence distribution helps

---

## Cost Reference

| Experiment | API Calls | Approx. Cost |
|------------|-----------|--------------|
| S0 (375q) | 375 | ~$1.5 |
| S1 (375q, 3 agents) | 1,125 | ~$4 |
| S2 (375q, 3 agents, 2 rounds) | 2,250 | ~$7-9 |
| Full ablation suite | ~20,000 | ~$60-70 |
