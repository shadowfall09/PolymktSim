# PolymktSim

LLM-based multi-agent framework for Polymarket-style binary event forecasting.
Implements and compares three interaction protocols (S0/S1/S2) with configurable
evidence routing and probabilistic aggregation strategies.

## Setup

```bash
pip install -r requirements.txt
```

Set environment variable:
```
ANTHROPIC_AUTH_TOKEN=sk-...
```

Optional overrides in `.env`:
```
LLM_MODEL_NAME=gpt-5.4-mini
LLM_BASE_URL=https://gpa-models.genai.prd.aws.saccap.int/v1
```

---

## Method

### 1. Retrieval-Augmented Forecasting

Each question is paired with up to 20 evidence documents retrieved via Tavily web search.
Evidence items include title, source URL, and content snippet (≤ 500 chars).
Agents are grounded to only the provided evidence; no external knowledge is assumed.

### 2. Interaction Protocols

Three protocols control how agents access evidence and interact:

**S0 — Single Agent**
A single LLM agent receives all evidence and produces one probabilistic forecast
`p_yes ∈ [0, 1]`. Serves as the single-agent baseline.

**S1 — Multi-Agent Independent**
`N` agents (default 3) each receive a *public* evidence pool (shared) plus a
disjoint *private* subset. Agents predict independently with no communication.
Final forecast is the aggregate of all individual predictions.

**S2 — Multi-Agent Iterative Deliberation**
Same evidence split as S1, but across `R` rounds (default 2).
After each round, agents receive a *history summary* containing every other agent's
`p_yes`, label, and rationale excerpt (≤ 300 chars) before re-predicting.
Final forecast aggregates the last-round predictions.
Instantiates a structured Delphi process adapted for LLMs.

When S1 and S2 are requested together (`--scenario s1s2` or `all`), S2 reuses
the per-agent S1 forecasts as round 0 because the evidence routing and initial
prompt are identical. Only the subsequent deliberation rounds make new API calls.

### 3. Evidence Routing

**Random split (default)**
Evidence items are shuffled with a fixed seed, then the top `public_ratio` fraction
is designated as public (shared by all agents); the remainder is partitioned evenly
into private subsets.

**BM25 routing (`--bm25`)**
Evidence is ranked by BM25 relevance score against the question text.
The top `public_ratio` fraction (most relevant) becomes the shared public pool.
The remaining items are assigned round-robin by relevance rank, so each agent
receives the most relevant non-overlapping private evidence.

### 4. Probabilistic Aggregation

**Mean (linear opinion pool)**
Simple arithmetic average of individual `p_yes` values.

**Confidence-weighted**
Logit-space weighted average where each agent's weight is its expressed confidence
|logit(pᵢ)|, so more-certain agents dominate the aggregate:
```
p_agg = sigmoid(Σ wᵢ·logit(pᵢ) / Σ wᵢ),   wᵢ = |logit(pᵢ)|
```

### 5. Evaluation

- **Brier score** — primary metric: `BS = (p_yes - outcome)²`, lower is better.
- **Accuracy** — fraction of questions where `(p_yes ≥ 0.5) == outcome`.
- **Log loss** — cross-entropy, reported but not primary.

---

## Dataset

- **Source**: 500 Polymarket binary markets (`data/outputs/final_markets_500.csv`)
- **Temporal filter**: Only markets with `endDate >= 2025-09-01` are used (375 questions), to avoid LLM prior knowledge contamination.
- **Topics**: Crypto, Sports, ScienceTech, Politics, Business, etc.
- **Evidence**: ~20 Tavily web search results per question (`data/evidences/`)

---

## Results (375 questions, gpt-5.4-mini)

### Main Comparison

| Method | Brier | Acc |
|--------|-------|-----|
| Sequential Bayesian (K=5) | 0.256 | 62.1% |
| Direct (no CoT) | 0.227 | 70.7% |
| S0 (RAG + CoT) | 0.216 | 69.9% |
| Zero-shot (no evidence) | 0.192 | 68.3% |
| S1 (CW + BM25) | 0.198 | 69.3% |
| **S2 (CW + BM25)** | **0.178** | **77.9%** |

### Key Ablations

| Ablation | Brier | Acc |
|----------|-------|-----|
| Rounds=1 (no deliberation) | 0.209 | 71.5% |
| **Rounds=2** | **0.178** | **77.9%** |
| Rounds=3 | 0.196 | 75.7% |
| PR=1.0 (all public, no private info) | 0.193 | 75.7% |
| **PR=0.5** | **0.178** | **77.9%** |
| PR=0.3 | 0.191 | 75.5% |
| No rationale sharing | 0.211 | 70.4% |
| **With rationale sharing** | **0.178** | **77.9%** |
| Mean aggregator | 0.196 | 74.9% |
| **CW aggregator** | **0.178** | **77.9%** |

### Cross-Model

| Model | S2 Brier | S2 Acc |
|-------|----------|--------|
| gpt-5.4-mini | 0.178 | 77.9% |
| gemini-3-flash | 0.193 | 80.0% |
| deepseek-v3.2 | **0.160** | **81.1%** |

---

## CLI Reference

```bash
python run_workflow.py [options]
```

| Flag | Default | Description |
|---|---|---|
| `--scenario {s0,s1,s2,all,s1s2}` | `s1s2` | Which protocol(s) to run |
| `--limit N` | `10` | Number of rows to process |
| `--qid-file PATH` | — | Text file of qids to run |
| `--temperature F` | `0.7` | Sampling temperature |
| `--num-agents N` | `3` | Number of agents in S1/S2 |
| `--num-rounds N` | `2` | Deliberation rounds in S2 |
| `--public-ratio F` | `0.5` | Fraction of evidence shared publicly |
| `--bm25` | off | BM25 relevance-based evidence routing |
| `--aggregator {mean,extremizing,confidence_weighted}` | `mean` | Aggregation method |
| `--max-workers N` | `1` | Parallel workers for concurrent questions |
| `--no-rationale-sharing` | off | S2: only share p_yes+label, not rationale |
| `--output PATH` | auto-timestamped | Output JSONL path |
| `--dry-run` | off | Use stub agent (no API calls) |

### Baseline Scripts

```bash
# Zero-shot, Direct, Self-consistency
python run_baselines.py --baseline all --max-workers 15

# Sequential Bayesian update
python run_baseline_bayesian.py --k 5 --max-workers 15

# Analysis on 375-question filtered set
python analyze_375.py
```

---

## Best Configuration

```bash
python run_workflow.py --scenario s2 --temperature 0.7 --max-workers 15 \
  --aggregator confidence_weighted --bm25 \
  --num-agents 3 --num-rounds 2 --public-ratio 0.5 \
  --limit 500
```

---

## Project Structure

```
src/
  agents/         # LLMAgent, prompts, multi-round protocol
  aggregation/    # mean, extremizing, confidence_weighted
  data/           # schema, loader, random splitter, bm25_splitter
  runner/         # run_s0 / run_s1 / run_s2 (parallel support)
  evaluation/     # brier_score, log_loss, accuracy, calibration
  utils/

data/
  outputs/        # final_markets_500.csv (500 Polymarket questions)
  evidences/      # per-question Tavily evidence (JSON)
  results/        # experiment outputs, RESULTS_SUMMARY.md
```
