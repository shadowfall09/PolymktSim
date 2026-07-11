#!/usr/bin/env python3
"""DCV (Decomposer + Verifier) prototype runner.

Pipeline per question:
  1. Decomposer LLM -> K atomic sub-claims (each tagged YES/NO + weight).
  2. K verifier agents -> each verifies one sub-claim against FULL evidence.
  3. Confidence-weighted log-pool combiner -> final p_yes.

Usage:
  python run_dcv_prototype.py --qid-file data/dcv_test_qids.txt --k 3 \\
      --output data/results/dcv_proto.jsonl

Compares against existing CW+BM25 S2 numbers for the same qids when --compare-run is set.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents.dcv_agent import DCVAgent
from src.aggregation.dcv_combiner import combine_dcv
from src.data.schema import EvidenceItem, Forecast, QuestionExample
from src.evaluation.metrics import accuracy, brier_score, log_loss
from src.utils.logger import setup_logging

EVIDENCES_DIR = Path("data/evidences")
MARKETS_CSV = Path("data/outputs/final_markets_500.csv")
RESULTS_DIR = Path("data/results")
logger = logging.getLogger(__name__)


def load_evidence_file(row_idx: int, market_id: str) -> tuple[list[EvidenceItem], date | None]:
    path = EVIDENCES_DIR / f"row_{row_idx:04d}_{market_id}.json"
    if not path.exists():
        return [], None
    with open(path) as f:
        d = json.load(f)
    items = []
    for i, e in enumerate(d.get("evidences", [])):
        items.append(EvidenceItem(
            doc_id=f"doc_{i+1:03d}",
            source=e.get("query", ""),
            title=e.get("title"),
            url=e.get("url"),
            content=e.get("content", ""),
            retrieval_score=e.get("score"),
        ))
    resolution_date = None
    raw = d.get("end_date_limit")
    if raw:
        try:
            resolution_date = date.fromisoformat(str(raw)[:10])
        except ValueError:
            pass
    return items, resolution_date


def load_examples(selected_qids: set[str]) -> list[tuple[QuestionExample, str]]:
    examples = []
    remaining = set(selected_qids)
    with open(MARKETS_CSV) as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            qid = f"row_{row_idx:04d}_{row['id']}"
            if qid not in remaining:
                continue
            evidence, resdate = load_evidence_file(row_idx, row["id"])
            p_str = row.get("p_yes", "")
            mp = float(p_str) if p_str else None
            outcome = True if mp == 1.0 else False if mp == 0.0 else None
            ex = QuestionExample(
                qid=qid,
                question=row["question"],
                evidence=evidence,
                outcome=outcome,
                resolution_date=resdate,
            )
            examples.append((ex, row.get("topic", "")))
            remaining.discard(qid)
            if not remaining:
                break
    return examples


def run_dcv(
    examples: list[tuple[QuestionExample, str]],
    decomposer: DCVAgent,
    verifier_factory,
    k: int,
    evidence_max_items: int,
    out_path: Path,
    detail_path: Path,
    meta: dict,
) -> tuple[list[tuple[str, Forecast]], list[dict]]:
    """Save each question's result incrementally so partial runs aren't lost."""
    results: list[tuple[str, Forecast]] = []
    detail: list[dict] = []
    for idx, (ex, topic) in enumerate(examples, 1):
        ev = ex.evidence[:evidence_max_items]
        print(f"\n[{idx}/{len(examples)}] {ex.qid} | outcome={ex.outcome} | {ex.question[:90]}", flush=True)

        # 1. Decompose
        sub_claims = decomposer.decompose(ex.question, k=k)
        print(f"  decomposed into {len(sub_claims)} sub-claim(s):", flush=True)
        for sc in sub_claims:
            print(f"    [{sc.supports} w={sc.weight:.2f}] {sc.claim[:120]}", flush=True)

        # 2. Verify each (one verifier per sub-claim, full evidence)
        verifications = []
        q_detail = []
        for ci, sc in enumerate(sub_claims):
            v = verifier_factory(ci)
            res = v.verify(sc.claim, ev)
            res.supports = sc.supports
            res.weight = sc.weight
            verifications.append(res)
            print(f"    -> p_true={res.p_true:.3f} conf={res.confidence}  ({res.rationale[:90]})", flush=True)
            rec = {
                "qid": ex.qid,
                "scenario": "dcv",
                "claim_idx": ci,
                "claim": sc.claim,
                "supports": sc.supports,
                "weight": sc.weight,
                "p_true": res.p_true,
                "confidence": res.confidence,
                "rationale": res.rationale,
                "evidence_used": res.evidence_used,
            }
            q_detail.append(rec)
            detail.append(rec)

        # 3. Combine
        forecast = combine_dcv(verifications)
        bri = brier_score(forecast.p_yes, ex.outcome) if ex.outcome is not None else None
        print(f"  => p_yes={forecast.p_yes:.3f} label={forecast.label} brier={bri}", flush=True)
        results.append((ex.qid, forecast))

        # Incremental save: append this question's records right now
        with open(out_path, "a") as fout:
            fout.write(json.dumps({
                "qid": ex.qid,
                "question": ex.question,
                "topic": topic,
                "scenario": "dcv",
                "p_yes": forecast.p_yes,
                "label": forecast.label,
                "rationale": forecast.rationale,
                "evidence_used": forecast.evidence_used,
                "outcome": ex.outcome,
                "brier": bri,
                **meta,
            }, ensure_ascii=False) + "\n")
        with open(detail_path, "a") as fdet:
            for rec in q_detail:
                fdet.write(json.dumps({
                    **rec,
                    "question": ex.question,
                    "topic": topic,
                    "outcome": ex.outcome,
                    **meta,
                }, ensure_ascii=False) + "\n")
    return results, detail


def print_eval(results: list[tuple[str, Forecast]], outcome_map: dict[str, bool | None], tag: str):
    evaluated = [(qid, f, outcome_map[qid]) for qid, f in results if outcome_map.get(qid) is not None]
    if not evaluated:
        print(f"  [{tag}] No resolved outcomes to evaluate.")
        return
    n = len(evaluated)
    bs = sum(brier_score(f.p_yes, o) for _, f, o in evaluated) / n
    ll = sum(log_loss(f.p_yes, o) for _, f, o in evaluated) / n
    acc = sum(accuracy(f.p_yes, o) for _, f, o in evaluated) / n
    print(f"  [{tag}] n={n}  Brier={bs:.4f}  LogLoss={ll:.4f}  Acc={acc:.3f}")


def compare_with_baseline(qids: list[str], baseline_path: str, dcv_results: list[tuple[str, Forecast]],
                          outcome_map: dict[str, bool | None]):
    """Print side-by-side baseline (CW+BM25 S2) vs DCV per qid + aggregate."""
    base = {}
    with open(baseline_path) as h:
        for line in h:
            r = json.loads(line)
            if r.get("scenario") != "s2": continue
            if r.get("rationale") != "confidence_weighted": continue
            if r["qid"] not in qids: continue
            base[r["qid"]] = (r["p_yes"], r.get("brier"))
    dcv_map = {qid: f for qid, f in dcv_results}
    print("\n=== Per-qid comparison (CW+BM25 S2 vs DCV) ===")
    print(f"{'qid':<28} {'y':<2} {'CW p':<8} {'CW B':<8} {'DCV p':<8} {'DCV B':<8} {'Δ Brier':<8}")
    n = bs_base = bs_dcv = 0
    acc_base = acc_dcv = 0
    fix_count = break_count = 0
    for qid in qids:
        y = outcome_map.get(qid)
        if y is None: continue
        bf = base.get(qid)
        df = dcv_map.get(qid)
        if not bf or not df: continue
        p_b, br_b = bf
        p_d = df.p_yes
        br_d = brier_score(p_d, y)
        delta = br_d - br_b
        b_ok = (p_b >= 0.5) == y
        d_ok = (p_d >= 0.5) == y
        if b_ok and not d_ok: break_count += 1
        if not b_ok and d_ok: fix_count += 1
        bs_base += br_b; bs_dcv += br_d
        acc_base += int(b_ok); acc_dcv += int(d_ok)
        n += 1
        flag = " *FIX" if (not b_ok and d_ok) else (" *BREAK" if (b_ok and not d_ok) else "")
        print(f"{qid:<28} {int(y):<2} {p_b:<8.3f} {br_b:<8.3f} {p_d:<8.3f} {br_d:<8.3f} {delta:+8.3f}{flag}")
    if n:
        print(f"\nSummary on n={n}:")
        print(f"  CW+BM25 S2:  Brier={bs_base/n:.4f}  Acc={acc_base/n:.3%}")
        print(f"  DCV:         Brier={bs_dcv/n:.4f}  Acc={acc_dcv/n:.3%}")
        print(f"  Fixed by DCV: {fix_count}    Broken by DCV: {break_count}    Net: {fix_count - break_count:+d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid-file", required=True, help="One qid per line")
    ap.add_argument("--k", type=int, default=3, help="Number of sub-claims (default 3)")
    ap.add_argument("--evidence-max-items", type=int, default=20)
    ap.add_argument("--evidence-max-chars", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--decomposer-temp", type=float, default=0.3,
                    help="Lower temperature for decomposer (default 0.3 for stability)")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--compare-run", type=str, default="data/results/20260413_215746.jsonl",
                    help="Baseline aggregated jsonl to compare DCV against (default: CW+BM25 S2)")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N qids (for smoke test)")
    ap.add_argument("--model", type=str, default=None,
                    help="Override LLM_MODEL_NAME env (e.g. openai/gpt-5.4-mini for OpenRouter)")
    ap.add_argument("--base-url", type=str, default=None,
                    help="Override LLM_BASE_URL env (e.g. https://openrouter.ai/api/v1)")
    ap.add_argument("--api-key-env", type=str, default=None,
                    help="Name of env var holding the API key (e.g. OPENROUTER_API_KEY)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Do not skip qids already present in --output (default: resume)")
    ap.add_argument("--prompt-version", choices=["v1", "v2"], default="v2",
                    help="Prompt set: v1=basic, v2=stricter with CoT and anti-hallucination")
    args = ap.parse_args()

    setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Override env vars early so DCVAgent default constructor picks them up
    if args.base_url:
        os.environ["LLM_BASE_URL"] = args.base_url
    if args.model:
        os.environ["LLM_MODEL_NAME"] = args.model
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"WARNING: env var {args.api_key_env} is not set")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"dcv_{ts}.jsonl"
    detail_path = out_path.with_name(out_path.stem + "_detail.jsonl")

    with open(args.qid_file) as f:
        qids = [line.strip() for line in f if line.strip()]
    if args.limit:
        qids = qids[: args.limit]
    print(f"Loading {len(qids)} qids from {args.qid_file}")

    # Resume: read already-completed qids from --output
    done_qids: set[str] = set()
    if not args.no_resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_qids.add(json.loads(line)["qid"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_qids:
            print(f"Resume: skipping {len(done_qids)} qids already in {out_path}")
    qids_to_run = [q for q in qids if q not in done_qids]

    examples = load_examples(set(qids_to_run))
    qids_loaded = [ex.qid for ex, _ in examples]
    # Preserve original ordering
    order = {q: i for i, q in enumerate(qids_to_run)}
    examples.sort(key=lambda x: order.get(x[0].qid, 1e9))
    print(f"Loaded {len(examples)} examples to run  output -> {out_path}")
    for ex, _ in examples:
        print(f"  {ex.qid}: {ex.question[:80]}  evidence={len(ex.evidence)}  outcome={ex.outcome}")

    _model = args.model or os.environ.get("LLM_MODEL_NAME", "openai/gpt-5.4-mini")
    _base_url = args.base_url or os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    decomposer = DCVAgent(model_name=_model,
                          base_url=_base_url,
                          temperature=args.decomposer_temp,
                          api_key=api_key,
                          evidence_max_chars_per_item=args.evidence_max_chars,
                          prompt_version=args.prompt_version)
    print(f"Using model={decomposer.model_name}  base_url={decomposer._client.base_url}  prompt_version={args.prompt_version}")
    verifiers: list[DCVAgent] = []

    def verifier_factory(_i):
        v = DCVAgent(model_name=_model,
                     base_url=_base_url,
                     temperature=args.temperature,
                     api_key=api_key,
                     evidence_max_chars_per_item=args.evidence_max_chars,
                     prompt_version=args.prompt_version)
        verifiers.append(v)
        return v

    meta = {
        "model": decomposer.model_name,
        "k": args.k,
        "decomposer_temp": args.decomposer_temp,
        "verifier_temp": args.temperature,
        "run_ts": ts,
        "method": "dcv",
        "prompt_version": args.prompt_version,
    }

    results: list[tuple[str, Forecast]] = []
    details: list[dict] = []
    if examples:
        results, details = run_dcv(examples, decomposer, verifier_factory,
                                   k=args.k, evidence_max_items=args.evidence_max_items,
                                   out_path=out_path, detail_path=detail_path, meta=meta)
    else:
        print("Nothing to run (all qids already done).")

    # For comparison/eval, reload everything ever written to out_path that's in the requested qid set
    all_results: list[tuple[str, Forecast]] = []
    outcome_map: dict[str, bool | None] = {}
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                if r["qid"] not in qids:
                    continue
                all_results.append((r["qid"], Forecast(
                    p_yes=r["p_yes"], label=r.get("label", "YES" if r["p_yes"] >= 0.5 else "NO"),
                    rationale=r.get("rationale", ""), evidence_used=r.get("evidence_used", []),
                )))
                outcome_map[r["qid"]] = r.get("outcome")

    print()
    print_eval(all_results, outcome_map, "DCV (all)")

    if args.compare_run and Path(args.compare_run).exists():
        compare_with_baseline([qid for qid, _ in all_results], args.compare_run, all_results, outcome_map)

    total_pt = sum(a.total_prompt_tokens for a in [decomposer] + verifiers)
    total_ct = sum(a.total_completion_tokens for a in [decomposer] + verifiers)
    print(f"\n=== Tokens (this run only) ===  prompt={total_pt:,}  completion={total_ct:,}  total={total_pt+total_ct:,}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
