#!/usr/bin/env python3
"""Run baseline experiments: zero-shot, direct (no CoT), self-consistency.

Usage:
    python run_baselines.py --baseline zeroshot --max-workers 15
    python run_baselines.py --baseline direct --max-workers 15
    python run_baselines.py --baseline self-consistency --max-workers 15 --num-samples 3
"""
import argparse
import csv
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents.llm_agent import LLMAgent
from src.agents.prompts import (
    SYSTEM_PROMPT, SYSTEM_PROMPT_DIRECT, SYSTEM_PROMPT_ZEROSHOT,
    OUTPUT_INSTRUCTIONS, OUTPUT_INSTRUCTIONS_DIRECT,
    build_forecast_prompt, build_zeroshot_prompt, build_direct_prompt,
    format_evidence,
)
from src.data.schema import EvidenceItem, Forecast
from src.data.baseline_loader import load_processed_baseline_questions
from src.evaluation.metrics import brier_score, log_loss, accuracy
from src.utils.logger import setup_logging

EVIDENCES_DIR = Path("data/evidences")
MARKETS_CSV = Path("data/outputs/final_markets_500.csv")
RESULTS_DIR = Path("data/results")
CUTOFF = date(2025, 9, 1)

logger = logging.getLogger(__name__)
_write_lock = threading.Lock()


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

            # Load evidence
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


def run_zeroshot(questions, max_workers=15, out_path=None):
    """Zero-shot: no evidence, LLM uses only internal knowledge."""
    print(f"=== ZERO-SHOT baseline (n={len(questions)}, workers={max_workers}) ===")

    def _predict(q):
        agent = LLMAgent(temperature=0.7, max_tokens=1024)
        prompt = build_zeroshot_prompt(q["question"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_ZEROSHOT},
            {"role": "user", "content": prompt},
        ]
        raw, usage = agent._call_api(messages)
        from src.agents.llm_agent import _parse_forecast, _fallback_forecast
        forecast = _parse_forecast(raw)
        if forecast is None:
            forecast = _fallback_forecast()
        return q["qid"], forecast

    return _run_parallel(questions, _predict, "zeroshot", max_workers, out_path)


def run_direct(questions, max_workers=15, out_path=None):
    """Direct: evidence provided but no chain-of-thought (no rationale)."""
    print(f"=== DIRECT (no CoT) baseline (n={len(questions)}, workers={max_workers}) ===")

    def _predict(q):
        agent = LLMAgent(temperature=0.7, max_tokens=512)
        evidence_text = format_evidence(q["evidence"], max_chars=500)
        prompt = build_direct_prompt(q["question"], evidence_text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
            {"role": "user", "content": prompt},
        ]
        raw, usage = agent._call_api(messages)
        from src.agents.llm_agent import _parse_forecast, _fallback_forecast
        forecast = _parse_forecast(raw)
        if forecast is None:
            forecast = _fallback_forecast()
        return q["qid"], forecast

    return _run_parallel(questions, _predict, "direct", max_workers, out_path)


def run_self_consistency(questions, num_samples=3, max_workers=15, out_path=None):
    """Self-consistency: same prompt N times, aggregate by mean."""
    print(f"=== SELF-CONSISTENCY baseline (n={len(questions)}, samples={num_samples}, workers={max_workers}) ===")

    def _predict(q):
        preds = []
        for _ in range(num_samples):
            agent = LLMAgent(temperature=0.7, max_tokens=2048)
            evidence_text = format_evidence(q["evidence"], max_chars=500)
            prompt = build_forecast_prompt(q["question"], evidence_text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            raw, usage = agent._call_api(messages)
            from src.agents.llm_agent import _parse_forecast
            forecast = _parse_forecast(raw)
            if forecast:
                preds.append(forecast.p_yes)

        if not preds:
            return q["qid"], Forecast(p_yes=0.5, label="NO", rationale="[all failed]", evidence_used=[])

        mean_p = sum(preds) / len(preds)
        label = "YES" if mean_p >= 0.5 else "NO"
        return q["qid"], Forecast(
            p_yes=mean_p, label=label,
            rationale=f"[self-consistency mean of {len(preds)} samples: {[f'{p:.3f}' for p in preds]}]",
            evidence_used=[],
        )

    return _run_parallel(questions, _predict, "self_consistency", max_workers, out_path)


def _run_parallel(questions, predict_fn, scenario_name, max_workers, out_path):
    """Generic parallel runner with streaming write."""
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
            future = executor.submit(predict_fn, q)
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

    # Compute metrics
    briers, accs = [], []
    for q in questions:
        if q["outcome"] is None or q["qid"] not in results:
            continue
        f = results[q["qid"]]
        truth = 1.0 if q["outcome"] else 0.0
        briers.append((f.p_yes - truth) ** 2)
        accs.append(int((f.p_yes >= 0.5) == q["outcome"]))

    if briers:
        print(f"  [{scenario_name}] n={len(briers)}  Brier={sum(briers)/len(briers):.4f}  Acc={sum(accs)/len(accs):.3f}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["zeroshot", "direct", "self-consistency", "all"], default="all")
    ap.add_argument("--max-workers", type=int, default=15)
    ap.add_argument("--num-samples", type=int, default=3, help="Number of samples for self-consistency")
    ap.add_argument("--output-dir", type=str, default="data/results")
    ap.add_argument("--dataset-jsonl", type=str, default=None,
                    help="Processed JSONL with embedded evidence; bypasses legacy CSV/cache loading")
    args = ap.parse_args()

    setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    questions = (load_processed_baseline_questions(args.dataset_jsonl)
                 if args.dataset_jsonl else load_questions())
    source = args.dataset_jsonl or "legacy CSV/cache (after temporal filter)"
    print(f"Loaded {len(questions)} questions from {source}\n")

    if args.baseline in ("zeroshot", "all"):
        out = Path(args.output_dir) / "baseline_zeroshot.jsonl"
        if out.exists():
            out.unlink()
        run_zeroshot(questions, max_workers=args.max_workers, out_path=out)
        print()

    if args.baseline in ("direct", "all"):
        out = Path(args.output_dir) / "baseline_direct.jsonl"
        if out.exists():
            out.unlink()
        run_direct(questions, max_workers=args.max_workers, out_path=out)
        print()

    if args.baseline in ("self-consistency", "all"):
        out = Path(args.output_dir) / "baseline_self_consistency.jsonl"
        if out.exists():
            out.unlink()
        run_self_consistency(questions, num_samples=args.num_samples,
                           max_workers=args.max_workers, out_path=out)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
