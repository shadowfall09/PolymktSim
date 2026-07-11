#!/usr/bin/env python3
"""Run additional baselines for EMNLP paper:
  1. superforecaster  - Structured reasoning with superforecaster-style prompt
  2. crowd-ensemble   - N independent agents, median aggregation (Silicon Crowd)
  3. moa              - Mixture-of-Agents: 3 proposers → 1 aggregator

Usage:
    python run_baselines_extra.py --baseline superforecaster --max-workers 15
    python run_baselines_extra.py --baseline crowd-ensemble --max-workers 15 --num-agents 5
    python run_baselines_extra.py --baseline moa --max-workers 15
    python run_baselines_extra.py --baseline all --max-workers 15
"""
import argparse
import csv
import json
import logging
import sys
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents.llm_agent import LLMAgent, _parse_forecast, _fallback_forecast
from src.agents.prompts import format_evidence, OUTPUT_INSTRUCTIONS
from src.data.schema import EvidenceItem, Forecast
from src.data.baseline_loader import load_processed_baseline_questions
from src.evaluation.metrics import brier_score
from src.utils.logger import setup_logging

EVIDENCES_DIR = Path("data/evidences")
MARKETS_CSV = Path("data/outputs/final_markets_500.csv")
RESULTS_DIR = Path("data/results")
CUTOFF = date(2025, 9, 1)

logger = logging.getLogger(__name__)
_write_lock = threading.Lock()


# ============================================
# Prompts
# ============================================

SYSTEM_PROMPT_SUPERFORECASTER = """You are a superforecaster, trained in the art of calibrated probabilistic reasoning. You follow a structured analytical process:

1. DECOMPOSE: Break the question into key factors that would need to be true for a YES outcome.
2. BASE RATE: Estimate the prior probability based on reference classes and historical frequencies.
3. EVIDENCE UPDATE: For each piece of evidence, assess whether it updates toward YES or NO and by how much.
4. SYNTHESIS: Combine your base rate with evidence updates, being careful about double-counting.
5. CALIBRATION CHECK: Ask yourself "If I made 100 predictions at this probability, would roughly that many resolve YES?"

Be precise. Avoid round numbers. Express genuine uncertainty. You must respond with valid JSON only after your reasoning."""

SYSTEM_PROMPT_MOA_AGGREGATOR = """You are a meta-forecaster. You will be given a binary prediction market question, evidence, and forecasts from three independent analysts who each had access to different subsets of the evidence.

Your job is to synthesize their predictions and reasoning into a single, well-calibrated probability estimate. Consider:
- Where do analysts agree? This strengthens confidence.
- Where do they disagree? Examine their reasoning to determine who has stronger evidence.
- Are any analysts clearly wrong or clearly right based on their stated reasoning?

Produce your final probability estimate. You must respond with valid JSON only, no other text before or after."""


def build_superforecaster_prompt(question: str, evidence_text: str) -> str:
    parts = [
        f"Question: {question}",
        "",
        "Evidence:",
        evidence_text,
        "",
        "Follow the structured superforecasting process (decompose → base rate → evidence update → synthesis → calibration check), then provide your final estimate.",
        "",
        OUTPUT_INSTRUCTIONS,
    ]
    return "\n".join(parts)


def build_moa_aggregator_prompt(question: str, evidence_text: str, proposals: list) -> str:
    parts = [
        f"Question: {question}",
        "",
        "Evidence summary:",
        evidence_text[:2000],
        "",
        "Independent analyst forecasts:",
    ]
    for i, (p_yes, rationale) in enumerate(proposals, 1):
        parts.append(f"  Analyst {i}: p(YES) = {p_yes:.3f}")
        parts.append(f"    Reasoning: {rationale[:200]}")
        parts.append("")

    parts.extend([
        "Based on all analysts' predictions and reasoning, provide your synthesized probability estimate.",
        "",
        OUTPUT_INSTRUCTIONS,
    ])
    return "\n".join(parts)


# ============================================
# Data loading (same as run_baselines.py)
# ============================================

def load_questions():
    questions = []
    with open(MARKETS_CSV) as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            endiso = row.get("endDateIso") or row.get("endDate") or ""
            if not endiso:
                continue
            try:
                ed = date.fromisoformat(str(endiso)[:10])
            except (ValueError, TypeError):
                continue
            if ed < CUTOFF:
                continue

            market_id = row["id"]
            question = row["question"]
            qid = f"row_{row_idx:04d}_{market_id}"
            p_yes = float(row.get("p_yes", "0.5"))
            outcome = None
            if p_yes == 1.0:
                outcome = True
            elif p_yes == 0.0:
                outcome = False

            ev_path = EVIDENCES_DIR / f"row_{row_idx:04d}_{market_id}.json"
            evidence = []
            if ev_path.exists():
                with open(ev_path) as ef:
                    d = json.load(ef)
                for i, e in enumerate(d.get("evidences", [])):
                    evidence.append(EvidenceItem(
                        doc_id=f"doc_{i+1:03d}",
                        source=e.get("query", ""),
                        title=e.get("title", ""),
                        content=e.get("content", ""),
                        timestamp=e.get("published_date"),
                    ))

            questions.append({
                "qid": qid,
                "question": question,
                "evidence": evidence[:20],
                "outcome": outcome,
                "topic": row.get("topic", ""),
            })
    return questions


# ============================================
# Baselines
# ============================================

def run_superforecaster(questions, max_workers=15, out_path=None):
    """Superforecaster: structured reasoning prompt (single agent)."""
    print(f"=== SUPERFORECASTER baseline (n={len(questions)}, workers={max_workers}) ===")

    def _predict(q):
        agent = LLMAgent(temperature=0.7, max_tokens=2048)
        evidence_text = format_evidence(q["evidence"], max_chars=500)
        prompt = build_superforecaster_prompt(q["question"], evidence_text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SUPERFORECASTER},
            {"role": "user", "content": prompt},
        ]
        raw, usage = agent._call_api(messages)
        forecast = _parse_forecast(raw)
        if forecast is None:
            forecast = _fallback_forecast()
        return q["qid"], forecast

    return _run_parallel(questions, _predict, "superforecaster", max_workers, out_path)


def run_crowd_ensemble(questions, num_agents=5, max_workers=15, out_path=None):
    """Crowd Ensemble: N independent agents, median aggregation (Silicon Crowd style)."""
    print(f"=== CROWD ENSEMBLE baseline (n={len(questions)}, agents={num_agents}, workers={max_workers}) ===")

    SYSTEM_PROMPT = """You are a forecaster for binary (Yes/No) prediction markets. Use only the evidence provided. Prefer high-credibility sources when they conflict. Reflect uncertainty with probabilities near 0.5. You must respond with valid JSON only, no other text before or after."""

    def _predict(q):
        preds = []
        evidence_text = format_evidence(q["evidence"], max_chars=500)
        prompt_parts = [
            f"Question: {q['question']}",
            "",
            "Evidence:",
            evidence_text,
            "",
            OUTPUT_INSTRUCTIONS,
        ]
        prompt = "\n".join(prompt_parts)

        for _ in range(num_agents):
            agent = LLMAgent(temperature=0.7, max_tokens=2048)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            raw, usage = agent._call_api(messages)
            forecast = _parse_forecast(raw)
            if forecast:
                preds.append(forecast.p_yes)

        if not preds:
            return q["qid"], Forecast(p_yes=0.5, label="NO", rationale="[all failed]", evidence_used=[])

        median_p = float(np.median(preds))
        label = "YES" if median_p >= 0.5 else "NO"
        return q["qid"], Forecast(
            p_yes=median_p, label=label,
            rationale=f"[crowd ensemble median of {len(preds)} agents: {[f'{p:.3f}' for p in preds]}]",
            evidence_used=[],
        )

    return _run_parallel(questions, _predict, "crowd_ensemble", max_workers, out_path)


def run_moa(questions, max_workers=15, out_path=None):
    """Mixture-of-Agents: 3 proposers generate forecasts → 1 aggregator synthesizes."""
    print(f"=== MIXTURE-OF-AGENTS baseline (n={len(questions)}, workers={max_workers}) ===")

    SYSTEM_PROMPT_PROPOSER = """You are a forecaster for binary (Yes/No) prediction markets. Use only the evidence provided. Prefer high-credibility sources when they conflict. Reflect uncertainty with probabilities near 0.5. You must respond with valid JSON only, no other text before or after."""

    def _predict(q):
        evidence_text = format_evidence(q["evidence"], max_chars=500)
        prompt_parts = [
            f"Question: {q['question']}",
            "",
            "Evidence:",
            evidence_text,
            "",
            OUTPUT_INSTRUCTIONS,
        ]
        prompt = "\n".join(prompt_parts)

        # Stage 1: 3 proposers independently forecast
        proposals = []
        for _ in range(3):
            agent = LLMAgent(temperature=0.7, max_tokens=2048)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_PROPOSER},
                {"role": "user", "content": prompt},
            ]
            raw, usage = agent._call_api(messages)
            forecast = _parse_forecast(raw)
            if forecast:
                proposals.append((forecast.p_yes, forecast.rationale))

        if not proposals:
            return q["qid"], Forecast(p_yes=0.5, label="NO", rationale="[all proposers failed]", evidence_used=[])

        # Stage 2: aggregator synthesizes
        agg_prompt = build_moa_aggregator_prompt(q["question"], evidence_text, proposals)
        agg_agent = LLMAgent(temperature=0.3, max_tokens=2048)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_MOA_AGGREGATOR},
            {"role": "user", "content": agg_prompt},
        ]
        raw, usage = agg_agent._call_api(messages)
        forecast = _parse_forecast(raw)
        if forecast is None:
            # Fallback: mean of proposals
            mean_p = sum(p for p, _ in proposals) / len(proposals)
            forecast = Forecast(p_yes=mean_p, label="YES" if mean_p >= 0.5 else "NO",
                              rationale="[aggregator failed, using mean]", evidence_used=[])

        return q["qid"], forecast

    return _run_parallel(questions, _predict, "moa", max_workers, out_path)


# ============================================
# Parallel runner (same as run_baselines.py)
# ============================================

def _run_parallel(questions, predict_fn, scenario_name, max_workers, out_path):
    if out_path is None:
        out_path = RESULTS_DIR / f"baseline_{scenario_name}.jsonl"
    else:
        out_path = Path(out_path)

    # Clear output file
    out_path.write_text("")

    results = {}
    completed = 0

    def _on_done(qid, forecast, q):
        nonlocal completed
        with _write_lock:
            completed += 1
            outcome = q["outcome"]
            record = {
                "qid": qid,
                "question": q["question"],
                "topic": q["topic"],
                "scenario": scenario_name,
                "p_yes": forecast.p_yes,
                "label": forecast.label,
                "rationale": forecast.rationale,
                "evidence_used": forecast.evidence_used,
                "outcome": outcome,
                "brier": brier_score(forecast.p_yes, outcome) if outcome is not None else None,
                "model": "gpt-5.4-mini",
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if completed % 50 == 0:
                print(f"  Progress: {completed}/{len(questions)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for q in questions:
            fut = executor.submit(predict_fn, q)
            futures[fut] = q

        for fut in as_completed(futures):
            q = futures[fut]
            try:
                qid, forecast = fut.result()
                _on_done(qid, forecast, q)
                results[qid] = forecast
            except Exception as e:
                logger.error(f"Error on {q['qid']}: {e}")
                fallback = Forecast(p_yes=0.5, label="NO", rationale=f"[error: {e}]", evidence_used=[])
                _on_done(q["qid"], fallback, q)
                results[q["qid"]] = fallback

    # Print metrics
    briers, accs = [], []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if r["brier"] is not None:
                briers.append(r["brier"])
                accs.append((r["p_yes"] >= 0.5) == r["outcome"])

    if briers:
        print(f"\n  {scenario_name} Results (n={len(briers)}):")
        print(f"    Brier: {np.mean(briers):.4f}")
        print(f"    Acc:   {np.mean(accs)*100:.1f}%")
        print(f"    Output: {out_path}")

    return results


# ============================================
# Main
# ============================================

# ============================================
# Halawi et al. (2024) - Scratchpad + Trimmed Mean
# ============================================

SYSTEM_PROMPT_HALAWI = """You are an expert superforecaster, familiar with the work of Tetlock and others. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. You must respond with valid JSON only, no other text before or after."""

HALAWI_SCRATCHPAD_PROMPT = """Question: {question}
Today's date: 2025-08-01
Question close date: 2025-12-31

We have retrieved the following information for this question:
{evidence}

Instructions:
1. Given the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.
2. Using your knowledge of the world and topic, as well as the information provided, provide a few reasons why the answer might be no. Rate the strength of each reason.
3. Using your knowledge of the world and topic, as well as the information provided, provide a few reasons why the answer might be yes. Rate the strength of each reason.
4. Aggregate your considerations. Think like a superforecaster (e.g. Nate Silver).
5. Output an initial probability (prediction) given steps 1-4.
6. Evaluate whether your calculated probability is excessively confident or not confident enough. Also, consider anything else that might affect the forecast that you did not before consider (e.g. base rate of the event).
7. Output your final prediction as a JSON object.

{output_instructions}"""


def run_halawi(questions, num_samples=3, max_workers=15, out_path=None):
    """Halawi et al. (2024): Scratchpad prompting + trimmed mean of 3."""
    print(f"=== HALAWI et al. baseline (n={len(questions)}, samples={num_samples}, workers={max_workers}) ===")

    def _trimmed_mean(preds):
        """Halawi's custom trimmed mean: reduce weight of furthest-from-median by 50%."""
        if len(preds) <= 1:
            return preds[0] if preds else 0.5
        median_val = float(np.median(preds))
        # Find the prediction furthest from median
        distances = [abs(p - median_val) for p in preds]
        furthest_idx = int(np.argmax(distances))
        # Assign weights: all get 1.0, furthest gets 0.5
        weights = [1.0] * len(preds)
        weights[furthest_idx] = 0.5
        # Redistribute the 0.5 removed weight uniformly to others
        redistribute = 0.5 / (len(preds) - 1)
        for i in range(len(preds)):
            if i != furthest_idx:
                weights[i] += redistribute
        # Weighted average
        return sum(w * p for w, p in zip(weights, preds)) / sum(weights)

    def _predict(q):
        preds = []
        evidence_text = format_evidence(q["evidence"], max_chars=500)
        prompt = HALAWI_SCRATCHPAD_PROMPT.format(
            question=q["question"],
            evidence=evidence_text,
            output_instructions=OUTPUT_INSTRUCTIONS,
        )

        for _ in range(num_samples):
            agent = LLMAgent(temperature=0.7, max_tokens=2048)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_HALAWI},
                {"role": "user", "content": prompt},
            ]
            raw, usage = agent._call_api(messages)
            forecast = _parse_forecast(raw)
            if forecast:
                preds.append(forecast.p_yes)

        if not preds:
            return q["qid"], Forecast(p_yes=0.5, label="NO", rationale="[all failed]", evidence_used=[])

        final_p = _trimmed_mean(preds)
        label = "YES" if final_p >= 0.5 else "NO"
        return q["qid"], Forecast(
            p_yes=final_p, label=label,
            rationale=f"[halawi trimmed mean of {len(preds)}: {[f'{p:.3f}' for p in preds]}]",
            evidence_used=[],
        )

    return _run_parallel(questions, _predict, "halawi", max_workers, out_path)


# ============================================
# AIA Forecaster (simplified, no live retrieval)
# ============================================

SYSTEM_PROMPT_AIA_AGENT = """You are an independent forecasting analyst. Given a binary prediction market question and evidence, provide your probability estimate with detailed reasoning. Focus on the strongest signals in the evidence. You must respond with valid JSON only, no other text before or after."""

SYSTEM_PROMPT_AIA_SUPERVISOR = """You are a senior forecasting supervisor at a research firm. You receive forecasts from 10 independent analysts who each analyzed the same question and evidence.

Your job:
1. IDENTIFY the key points of DISAGREEMENT among analysts.
2. For each disagreement, determine which side has stronger evidence-based reasoning.
3. Produce your own final probability estimate that resolves the disagreements.
4. Rate your CONFIDENCE in your final estimate as "high", "medium", or "low".
   - "high": you are confident the disagreements are clearly resolvable
   - "medium": some ambiguity remains
   - "low": fundamental uncertainty, analysts may all be guessing

You must respond with valid JSON only: {"p_yes": <0-1>, "label": "YES"|"NO", "confidence": "high"|"medium"|"low", "rationale": "..."}"""


def build_aia_supervisor_prompt(question, evidence_text, agent_outputs):
    parts = [
        f"Question: {question}",
        "",
        "Evidence:",
        evidence_text[:3000],
        "",
        f"=== Analyst Forecasts ({len(agent_outputs)} analysts) ===",
        "",
    ]
    for i, (p, rationale) in enumerate(agent_outputs, 1):
        parts.append(f"Analyst {i}: p(YES) = {p:.3f}")
        parts.append(f"  Reasoning: {rationale[:150]}")
        parts.append("")

    parts.extend([
        "=== Your Task ===",
        "1. Identify key disagreements among analysts.",
        "2. Determine which side has stronger evidence.",
        "3. Output your final probability estimate with confidence level.",
        "",
        '{"p_yes": <0-1>, "label": "YES"|"NO", "confidence": "high"|"medium"|"low", "rationale": "..."}',
    ])
    return "\n".join(parts)


def run_aia(questions, num_agents=10, max_workers=15, out_path=None):
    """AIA Forecaster (simplified): 10 independent agents + supervisor with confidence gating."""
    print(f"=== AIA FORECASTER baseline (n={len(questions)}, agents={num_agents}, workers={max_workers}) ===")

    def _predict(q):
        evidence_text = format_evidence(q["evidence"], max_chars=500)
        prompt_parts = [
            f"Question: {q['question']}",
            "",
            "Evidence:",
            evidence_text,
            "",
            OUTPUT_INSTRUCTIONS,
        ]
        prompt = "\n".join(prompt_parts)

        # Stage 1: N independent agents forecast
        agent_outputs = []  # list of (p_yes, rationale)
        for _ in range(num_agents):
            agent = LLMAgent(temperature=0.7, max_tokens=2048)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_AIA_AGENT},
                {"role": "user", "content": prompt},
            ]
            raw, usage = agent._call_api(messages)
            forecast = _parse_forecast(raw)
            if forecast:
                agent_outputs.append((forecast.p_yes, forecast.rationale))

        if not agent_outputs:
            return q["qid"], Forecast(p_yes=0.5, label="NO", rationale="[all agents failed]", evidence_used=[])

        # Stage 2: Supervisor reconciles
        sup_prompt = build_aia_supervisor_prompt(q["question"], evidence_text, agent_outputs)
        sup_agent = LLMAgent(temperature=0.3, max_tokens=2048)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_AIA_SUPERVISOR},
            {"role": "user", "content": sup_prompt},
        ]
        raw, usage = sup_agent._call_api(messages)

        # Parse supervisor output (includes confidence)
        import re
        sup_forecast = _parse_forecast(raw)

        # Extract confidence from raw output
        confidence = "low"
        if raw:
            raw_lower = raw.lower()
            if '"confidence"' in raw_lower:
                if '"high"' in raw_lower:
                    confidence = "high"
                elif '"medium"' in raw_lower:
                    confidence = "medium"

        # Confidence gating: only use supervisor if high confidence
        agent_mean = sum(p for p, _ in agent_outputs) / len(agent_outputs)

        if sup_forecast and confidence == "high":
            final_p = sup_forecast.p_yes
            rationale = f"[AIA supervisor (high conf): {final_p:.3f}, agents mean: {agent_mean:.3f}]"
        else:
            final_p = agent_mean
            rationale = f"[AIA fallback to mean ({confidence} conf): {agent_mean:.3f}]"

        label = "YES" if final_p >= 0.5 else "NO"
        return q["qid"], Forecast(p_yes=final_p, label=label, rationale=rationale, evidence_used=[])

    return _run_parallel(questions, _predict, "aia", max_workers, out_path)


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run extra baselines for EMNLP paper")
    parser.add_argument("--baseline", required=True,
                       choices=["superforecaster", "crowd-ensemble", "moa", "halawi", "aia", "all"],
                       help="Which baseline to run")
    parser.add_argument("--max-workers", type=int, default=15)
    parser.add_argument("--num-agents", type=int, default=5,
                       help="Number of agents for crowd-ensemble (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                       help="Custom output path")
    parser.add_argument("--dataset-jsonl", type=str, default=None,
                        help="Processed JSONL with embedded evidence; bypasses legacy CSV/cache loading")
    args = parser.parse_args()

    setup_logging()
    questions = (load_processed_baseline_questions(args.dataset_jsonl)
                 if args.dataset_jsonl else load_questions())
    source = args.dataset_jsonl or "legacy CSV/cache (temporal filter applied)"
    print(f"Loaded {len(questions)} questions from {source}")

    if args.baseline in ("superforecaster", "all"):
        run_superforecaster(questions, max_workers=args.max_workers, out_path=args.output)

    if args.baseline in ("crowd-ensemble", "all"):
        run_crowd_ensemble(questions, num_agents=args.num_agents,
                          max_workers=args.max_workers, out_path=args.output)

    if args.baseline in ("moa", "all"):
        run_moa(questions, max_workers=args.max_workers, out_path=args.output)

    if args.baseline in ("halawi", "all"):
        run_halawi(questions, num_samples=3, max_workers=args.max_workers, out_path=args.output)

    if args.baseline in ("aia", "all"):
        run_aia(questions, num_agents=10, max_workers=args.max_workers, out_path=args.output)
