#!/usr/bin/env python3
"""Sequential Bayesian Update baseline.

Each question starts with prior p=0.5, then processes evidence items one at a time.
At each step the LLM sees the current belief + one new evidence item and outputs
an updated probability.

Usage:
    python run_baseline_bayesian.py --max-workers 15 --k 5
    python run_baseline_bayesian.py --max-workers 15 --k 10
"""
import argparse
import csv
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents.llm_agent import LLMAgent, _parse_forecast
from src.agents.prompts import format_evidence
from src.data.schema import EvidenceItem, Forecast
from src.evaluation.metrics import brier_score
from src.utils.logger import setup_logging

EVIDENCES_DIR = Path("data/evidences")
MARKETS_CSV = Path("data/outputs/final_markets_500.csv")
RESULTS_DIR = Path("data/results")
CUTOFF = date(2025, 9, 1)

logger = logging.getLogger(__name__)
_write_lock = threading.Lock()

SYSTEM_PROMPT_BAYESIAN = """You are a Bayesian forecaster for binary (Yes/No) prediction markets. You will be shown your current belief (probability) and one new piece of evidence. Update your probability based on this evidence. If the evidence supports YES, increase your probability. If it supports NO, decrease it. If it is irrelevant or ambiguous, keep your probability roughly the same. You must respond with valid JSON only, no other text before or after."""

UPDATE_TEMPLATE = """Question: {question}

Your current belief: p_yes = {current_p:.3f}

New evidence:
{evidence}

Based on this new evidence, update your probability estimate.
Respond with exactly one JSON object: {{"p_yes": <0-1>, "label": "YES"|"NO", "rationale": "brief reason for update"}}"""


def load_questions():
    """Load 375 filtered questions with evidence."""
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
                "evidence": evidence,
                "outcome": outcome,
                "topic": row.get("topic", ""),
            })
    return questions


def sequential_bayesian_predict(q, k=5):
    """Run sequential Bayesian updates on one question."""
    evidence_items = q["evidence"][:k]
    current_p = 0.5
    update_history = [current_p]

    for ev_item in evidence_items:
        agent = LLMAgent(temperature=0.0, max_tokens=512)
        ev_text = format_evidence([ev_item], max_chars=500)
        prompt = UPDATE_TEMPLATE.format(
            question=q["question"],
            current_p=current_p,
            evidence=ev_text,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_BAYESIAN},
            {"role": "user", "content": prompt},
        ]
        try:
            raw, usage = agent._call_api(messages)
            forecast = _parse_forecast(raw)
            if forecast:
                current_p = forecast.p_yes
            # If parse fails, keep current_p unchanged
        except Exception as e:
            logger.warning("Bayesian update failed for %s: %s", q["qid"], e)
            # Keep current_p unchanged on error

        update_history.append(current_p)

    label = "YES" if current_p >= 0.5 else "NO"
    return q["qid"], Forecast(
        p_yes=current_p,
        label=label,
        rationale=f"[bayesian K={len(evidence_items)} updates: {' → '.join(f'{p:.3f}' for p in update_history)}]",
        evidence_used=[e.doc_id for e in evidence_items],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="Number of evidence items to process sequentially")
    ap.add_argument("--max-workers", type=int, default=15)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output) if args.output else RESULTS_DIR / f"baseline_bayesian_k{args.k}.jsonl"
    if out_path.exists():
        out_path.unlink()

    questions = load_questions()
    print(f"Loaded {len(questions)} questions")
    print(f"Sequential Bayesian Update: K={args.k}, workers={args.max_workers}")
    print(f"Output: {out_path}\n")

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
                "scenario": f"bayesian_k{args.k}",
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

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for q in questions:
            future = executor.submit(sequential_bayesian_predict, q, args.k)
            futures[future] = q
        for future in as_completed(futures):
            q = futures[future]
            try:
                qid, forecast = future.result()
                results[qid] = forecast
                _on_done(qid, forecast, q)
            except Exception as e:
                logger.error("Failed qid=%s: %s", q["qid"], e)
                fallback = Forecast(p_yes=0.5, label="NO", rationale=f"[error: {e}]", evidence_used=[])
                results[q["qid"]] = fallback
                _on_done(q["qid"], fallback, q)

    # Final metrics
    briers, accs = [], []
    for q in questions:
        if q["outcome"] is None or q["qid"] not in results:
            continue
        f = results[q["qid"]]
        truth = 1.0 if q["outcome"] else 0.0
        briers.append((f.p_yes - truth) ** 2)
        accs.append(int((f.p_yes >= 0.5) == q["outcome"]))

    if briers:
        print(f"\n  [bayesian_k{args.k}] n={len(briers)}  Brier={sum(briers)/len(briers):.4f}  Acc={sum(accs)/len(accs):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
