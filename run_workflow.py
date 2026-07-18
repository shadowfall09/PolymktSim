#!/usr/bin/env python3
from __future__ import annotations
"""Run S0/S1/S2 workflow on real dataset.

Examples:
  # dry-run (no API), first 5 rows, S0 only
  python run_workflow.py --dry-run --limit 5 --scenario s0

  # real LLM, first 20 rows, all scenarios
  python run_workflow.py --limit 20

  # specific rows
  python run_workflow.py --start 10 --limit 5 --scenario s1
"""
import argparse
import csv
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import load_dataset
from src.data.schema import EvidenceItem, Forecast, QuestionExample
from src.aggregation.mean import MeanAggregator
from src.aggregation.extremizing import ExtremizingAggregator
from src.aggregation.confidence_weighted import ConfidenceWeightedAggregator
from src.aggregation.adaptive_extremizing import AdaptiveExtremizingAggregator
from src.aggregation.temporal_reliability import TemporalReliabilityAggregator
from src.aggregation.calibrated_shrink import CalibratedShrinkAggregator
from src.runner.experiment import run_s0, run_s1, run_s2
from src.evaluation.metrics import brier_score, log_loss, accuracy
from src.utils.logger import setup_logging

EVIDENCES_DIR = Path("data/evidences")
MARKETS_CSV = Path("data/outputs/final_markets_500.csv")
RESULTS_DIR = Path("data/results")


def load_evidence_file(row_idx: int, market_id: str) -> tuple[list[EvidenceItem], date | None]:
    """Return (evidence_items, resolution_date). resolution_date may be None."""
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
    # New evidence caches distinguish the true market date from the stricter
    # retrieval cutoff. Fall back for historical cache files.
    raw = d.get("resolution_date") or d.get("end_date_limit")
    if raw:
        try:
            resolution_date = date.fromisoformat(str(raw)[:10])
        except ValueError:
            pass
    return items, resolution_date


def iter_market_rows():
    """Yield rows from MARKETS_CSV.

    Default behavior: only yield markets whose `endDateIso`/`endDate` is on or
    after 2025-09-01. Rows without a parseable end date are skipped.
    """
    cutoff = date(2025, 9, 1)
    with open(MARKETS_CSV) as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            endiso = row.get('endDateIso') or row.get('endDate') or ''
            if not endiso:
                # skip markets without an end date by default
                continue
            try:
                ed = date.fromisoformat(str(endiso)[:10])
            except Exception:
                # skip rows with unparseable dates
                continue
            if ed >= cutoff:
                yield row_idx, row


def make_example(row_idx: int, row: dict[str, str]) -> tuple[QuestionExample, float | None, str]:
    """Build one workflow example from one CSV row."""
    market_id = row["id"]
    question = row["question"]
    qid = f"row_{row_idx:04d}_{market_id}"
    topic = row.get("topic", "")

    p_yes_str = row.get("p_yes", "")
    market_p_yes = float(p_yes_str) if p_yes_str else None

    outcome = None
    if market_p_yes == 1.0:
        outcome = True
    elif market_p_yes == 0.0:
        outcome = False

    evidence, resolution_date = load_evidence_file(row_idx, market_id)
    ex = QuestionExample(qid=qid, question=question, evidence=evidence, outcome=outcome,
                         resolution_date=resolution_date)
    return ex, market_p_yes, topic


def load_examples(start: int, limit: int | None, selected_qids: set[str] | None = None) -> list[tuple[QuestionExample, float | None, str]]:
    """Returns list of (example, market_p_yes, topic)."""
    examples = []
    remaining_qids = set(selected_qids) if selected_qids is not None else None
    for row_idx, row in iter_market_rows():
        qid = f"row_{row_idx:04d}_{row['id']}"
        if remaining_qids is not None:
            if qid not in remaining_qids:
                continue
        else:
            if row_idx < start:
                continue
            if limit is not None and len(examples) >= limit:
                break

        examples.append(make_example(row_idx, row))
        if remaining_qids is not None:
            remaining_qids.remove(qid)
            if not remaining_qids:
                break
    return examples


def load_jsonl_examples(
    path: str | Path,
    start: int,
    limit: int | None,
    selected_qids: set[str] | None = None,
    topic: str = "",
) -> list[tuple[QuestionExample, float | None, str]]:
    """Load a processed QuestionExample JSONL dataset."""
    row_topics: dict[str, str] = {}
    if not topic:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = str(row.get("qid") or "")
                row_topic = str(row.get("topic") or "")
                if qid and row_topic:
                    row_topics[qid] = row_topic

    loaded = load_dataset(path)
    examples: list[tuple[QuestionExample, float | None, str]] = []
    remaining_qids = set(selected_qids) if selected_qids is not None else None

    for row_idx, ex in enumerate(loaded, start=1):
        if remaining_qids is not None:
            if ex.qid not in remaining_qids:
                continue
        else:
            if row_idx < start:
                continue
            if limit is not None and len(examples) >= limit:
                break

        examples.append((ex, None, topic or row_topics.get(ex.qid, "")))
        if remaining_qids is not None:
            remaining_qids.remove(ex.qid)
            if not remaining_qids:
                break

    return examples


def print_eval(results: list[tuple[str, Forecast]], outcome_map: dict[str, bool | None], scenario: str):
    evaluated = [(qid, f, outcome_map[qid]) for qid, f in results if outcome_map.get(qid) is not None]
    if not evaluated:
        print(f"  [{scenario}] No resolved outcomes to evaluate.")
        return
    bs = sum(brier_score(f.p_yes, o) for _, f, o in evaluated) / len(evaluated)
    ll = sum(log_loss(f.p_yes, o) for _, f, o in evaluated) / len(evaluated)
    acc = sum(accuracy(f.p_yes, o) for _, f, o in evaluated) / len(evaluated)
    print(f"  [{scenario}] n={len(evaluated)}  Brier={bs:.4f}  LogLoss={ll:.4f}  Acc={acc:.3f}")


def save_results(
    results: list[tuple[str, Forecast]],
    scenario: str,
    meta: dict,
    outcome_map: dict[str, bool | None],
    topic_map: dict[str, str],
    question_map: dict[str, str],
    out_path: Path,
):
    """Append results to a JSONL file."""
    with open(out_path, "a") as f:
        for qid, forecast in results:
            outcome = outcome_map.get(qid)
            record = {
                "qid": qid,
                "question": question_map.get(qid, ""),
                "topic": topic_map.get(qid, ""),
                "scenario": scenario,
                "p_yes": forecast.p_yes,
                "label": forecast.label,
                "rationale": forecast.rationale,
                "evidence_used": forecast.evidence_used,
                "outcome": outcome,
                "brier": brier_score(forecast.p_yes, outcome) if outcome is not None else None,
                **meta,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_detail_records(
    records: list[dict],
    meta: dict,
    outcome_map: dict[str, bool | None],
    question_map: dict[str, str],
    topic_map: dict[str, str],
    out_path: Path,
):
    """Append per-agent per-round records to a JSONL file."""
    with open(out_path, "a") as f:
        for rec in records:
            qid = rec["qid"]
            f.write(json.dumps({
                **rec,
                "question": question_map.get(qid, ""),
                "topic": topic_map.get(qid, ""),
                "outcome": outcome_map.get(qid),
                **meta,
            }, ensure_ascii=False) + "\n")


class StubAgent:
    def predict(self, qid, question, public_evidence, private_evidence, history_summary):
        n = len(public_evidence) + len(private_evidence)
        p = 0.55 if n > 10 else 0.5
        return Forecast.from_p_yes(p, rationale="[dry-run stub]", evidence_used=[])


LEADERBOARD_FIELDS = [
    "run_ts", "dataset", "scenario", "model", "temperature", "aggregator",
    "extremizing_alpha", "public_ratio", "bm25", "num_agents", "num_rounds",
    "n", "brier_mean", "brier_median", "accuracy",
]


def _compute_scenario_metrics(out_path: Path, scenario: str, filter_qids: set | None) -> dict | None:
    from collections import defaultdict
    records = defaultdict(dict)
    for line in out_path.open():
        r = json.loads(line)
        if filter_qids and r["qid"] not in filter_qids:
            continue
        records[r["qid"]][r["scenario"]] = r
    briers, hits, n = [], 0, 0
    for by_sc in records.values():
        rec = by_sc.get(scenario)
        if not rec:
            continue
        b = rec.get("brier")
        if b is not None:
            briers.append(b)
        o, p = rec.get("outcome"), rec.get("p_yes")
        if o is not None and p is not None:
            hits += (p >= 0.5) == o
            n += 1
    if not briers:
        return None
    briers.sort()
    nb = len(briers)
    median = briers[nb // 2] if nb % 2 else (briers[nb // 2 - 1] + briers[nb // 2]) / 2
    return dict(n=nb, brier_mean=round(sum(briers) / nb, 5),
                brier_median=round(median, 5),
                accuracy=round(hits / n, 4) if n else None)


def _update_registry(args, ts: str, out_path: Path, selected_qids: set | None):
    """Append this run to runs.jsonl and leaderboard.csv."""
    from src.agents.llm_agent import LLMAgent
    model_name = LLMAgent().model_name if not args.dry_run else "stub"

    sample_set_name = Path(args.qid_file).name if args.qid_file else None
    if args.qid_file:
        n_total = sum(1 for l in open(args.qid_file) if l.strip())
        dataset = f"custom_{n_total}"
    elif args.dataset_jsonl:
        dataset = Path(args.dataset_jsonl).stem
    else:
        dataset = f"rows_{args.start}-{args.start + (args.limit or 0) - 1}"

    run_entry = {
        "run_ts": ts,
        "dataset": dataset,
        "sample_set": sample_set_name,
        "model": model_name,
        "temperature": args.temperature,
        "num_agents": args.num_agents,
        "num_rounds": args.num_rounds,
        "aggregator": args.aggregator,
        "public_ratio": args.public_ratio,
        "bm25": args.bm25,
        "extremizing_alpha": args.extremizing_alpha if args.aggregator == "extremizing" else None,
        "share_mode": "numbers" if args.no_rationale_sharing else args.share_mode,
        "evidence_pooling": args.evidence_pooling,
        "calibrated_prompt": ("v3" if args.calibrated_prompt_v3 else "v2" if args.calibrated_prompt_v2 else args.calibrated_prompt),
        "devils_advocate": args.devils_advocate,
        "notes": "",
    }

    runs_path = RESULTS_DIR / "runs.jsonl"
    with runs_path.open("a") as f:
        f.write(json.dumps(run_entry, ensure_ascii=False) + "\n")

    leaderboard_path = RESULTS_DIR / "leaderboard.csv"
    write_header = not leaderboard_path.exists()
    with leaderboard_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        if write_header:
            writer.writeheader()
        for sc in ("s0", "s1", "s2"):
            sc_was_run = (
                args.scenario == "all"
                or args.scenario == sc
                or (args.scenario == "s1s2" and sc in ("s1", "s2"))
            )
            if not sc_was_run:
                continue
            m = _compute_scenario_metrics(out_path, sc, selected_qids)
            if m is None:
                continue
            writer.writerow({
                "run_ts": ts, "dataset": dataset, "scenario": sc,
                "model": model_name,
                "temperature": args.temperature,
                "aggregator": args.aggregator,
                "extremizing_alpha": args.extremizing_alpha if args.aggregator == "extremizing" else "",
                "public_ratio": args.public_ratio,
                "bm25": args.bm25,
                "num_agents": args.num_agents,
                "num_rounds": args.num_rounds,
                **m,
            })

    print(f"Registry updated: {runs_path.name}, {leaderboard_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Use stub agent, no API call")
    ap.add_argument("--dataset-jsonl", type=str, default=None,
                    help="Processed QuestionExample JSONL dataset; bypasses Polymarket CSV/evidence loading")
    ap.add_argument("--dataset-topic", type=str, default="",
                    help="Topic label to attach when using --dataset-jsonl")
    ap.add_argument("--start", type=int, default=1, help="Start row (1-based, default 1)")
    ap.add_argument("--limit", type=int, default=10, help="Number of rows to process (default 10)")
    ap.add_argument("--qid-file", type=str, default=None, help="Optional text file containing one qid per line to run a curated subset")
    ap.add_argument("--scenario", choices=["s0", "s1", "s2", "all", "s1s2"], default="s1s2",
                    help="Scenarios to run: s1s2 (default, skips S0), all (includes S0), or individual")
    ap.add_argument("--num-agents", type=int, default=3)
    ap.add_argument("--num-rounds", type=int, default=2)
    ap.add_argument("--evidence-max-items", type=int, default=20)
    ap.add_argument("--evidence-max-chars", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7, help="Temperature for S1/S2 agents (default 0.7); S0 always uses 0")
    ap.add_argument("--public-ratio", type=float, default=0.5, help="Fraction of evidence shared publicly across S1/S2 agents (default 0.5)")
    ap.add_argument("--aggregator",
                    choices=["mean", "extremizing", "confidence_weighted", "adaptive_extremizing",
                             "temporal_reliability", "calibrated_shrink"],
                    default="mean", help="Aggregation method for S1/S2 (default: mean)")
    ap.add_argument("--shrink-p0", type=float, default=0.30,
                    help="calibrated_shrink prior (default 0.30)")
    ap.add_argument("--shrink-w-lo", type=float, default=0.8,
                    help="calibrated_shrink weight below prior (default 0.8)")
    ap.add_argument("--shrink-w-hi", type=float, default=0.5,
                    help="calibrated_shrink weight above prior (default 0.5)")
    ap.add_argument("--extremizing-alpha", type=float, default=2.5,
                    help="Extremizing factor alpha (only used when --aggregator=extremizing, default 2.5)")
    ap.add_argument("--bm25", action="store_true", help="Use BM25 relevance-based evidence routing instead of random split")
    ap.add_argument("--share-mode", choices=["full", "arguments", "numbers"], default="full",
                    help="S2 between-round sharing: full (legacy: p_yes+rationale+round mean), "
                         "arguments (rationale+citations only, no numeric anchors), "
                         "numbers (p_yes+label only)")
    ap.add_argument("--evidence-pooling", action="store_true",
                    help="S2 only: private docs cited in a round are disclosed in full to all agents in later rounds")
    ap.add_argument("--calibrated-prompt", action="store_true",
                    help="Use the base-rate-anchored system prompt for all agents")
    ap.add_argument("--calibrated-prompt-v2", action="store_true",
                    help="v2 calibrated prompt: adds a resolution-criteria step and a "
                         "high-confidence gate for p_yes >= 0.7 (overrides --calibrated-prompt). "
                         "NOTE: measured net-negative on polymarket_250 (over-shrinks the low side); "
                         "prefer v3")
    ap.add_argument("--calibrated-prompt-v3", action="store_true",
                    help="v3 calibrated prompt: v1 plus a resolution-criteria step and a soft "
                         "high-confidence re-check, without v2's hard cap (overrides v2/v1). "
                         "NOTE: fixes high-confidence misses (p>=0.6 hit rate 0.29 -> 0.45) but "
                         "suppresses mid-band true-YES questions; net-negative Brier on "
                         "polymarket_250 (0.1780 vs v1's 0.1647 raw) — prefer plain "
                         "--calibrated-prompt with --aggregator calibrated_shrink")
    ap.add_argument("--devils-advocate", action="store_true",
                    help="S2 only: when a round is unanimous, instruct agents to build the "
                         "strongest opposing case before finalizing")
    ap.add_argument("--no-rationale-sharing", action="store_true",
                    help="[deprecated: use --share-mode numbers] S2 only: only share p_yes and label between rounds")
    ap.add_argument("--selective-update", action="store_true",
                    help="S2 only: revert herding agents to round-1 prediction")
    ap.add_argument("--herding-threshold", type=float, default=0.7,
                    help="Herding ratio threshold for selective update (default 0.7)")
    ap.add_argument("--max-workers", type=int, default=1, help="Parallel workers for concurrent question processing (default 1 = sequential)")
    ap.add_argument("--output", type=str, default=None, help="Output JSONL path (default: data/results/<timestamp>.jsonl)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume an existing --output JSONL: skip qids already written and append only missing results")
    args = ap.parse_args()

    setup_logging()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"{ts}.jsonl"

    if args.resume and not args.output:
        ap.error("--resume requires an explicit --output path")

    completed_qids: set[str] = set()
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Cannot resume from malformed JSONL at {out_path}:{line_number}"
                    ) from exc
                qid = str(record.get("qid", "")).strip()
                if qid:
                    completed_qids.add(qid)
        print(f"Resume enabled: found {len(completed_qids)} completed qids in {out_path}")

    selected_qids = None
    if args.qid_file:
        with open(args.qid_file) as f:
            selected_qids = {line.strip() for line in f if line.strip()}
        print(f"Loading {len(selected_qids)} qids from {args.qid_file}")
        if args.dataset_jsonl:
            rows = load_jsonl_examples(args.dataset_jsonl, args.start, args.limit,
                                       selected_qids=selected_qids, topic=args.dataset_topic)
        else:
            rows = load_examples(args.start, args.limit, selected_qids=selected_qids)
    else:
        if args.dataset_jsonl:
            print(f"Loading rows {args.start} to {args.start + args.limit - 1} from {args.dataset_jsonl}")
            rows = load_jsonl_examples(args.dataset_jsonl, args.start, args.limit,
                                       topic=args.dataset_topic)
        else:
            print(f"Loading rows {args.start} to {args.start + args.limit - 1} from {MARKETS_CSV}")
            rows = load_examples(args.start, args.limit)

    if completed_qids:
        before_resume_filter = len(rows)
        rows = [row for row in rows if row[0].qid not in completed_qids]
        print(f"Resume enabled: skipped {before_resume_filter - len(rows)} completed rows; {len(rows)} remain")
    print(f"Loaded {len(rows)} examples  output -> {out_path}\n")

    examples = [ex for ex, _, _ in rows]
    outcome_map = {ex.qid: ex.outcome for ex in examples}
    topic_map = {ex.qid: topic for ex, _, topic in rows}
    question_map = {ex.qid: ex.question for ex in examples}

    for ex in examples:
        print(f"  {ex.qid}: {ex.question[:80]}  evidence={len(ex.evidence)}  outcome={ex.outcome}")
    print()

    all_agents: list = []

    share_mode = "numbers" if args.no_rationale_sharing else args.share_mode

    if args.dry_run:
        model_name = "stub"
        agent = StubAgent()
        agent_factory = lambda _: StubAgent()
    else:
        from src.agents.llm_agent import LLMAgent
        from src.agents.prompts import (
            SYSTEM_PROMPT_CALIBRATED,
            SYSTEM_PROMPT_CALIBRATED_V2,
            SYSTEM_PROMPT_CALIBRATED_V3,
        )
        model_name = LLMAgent().model_name
        if args.calibrated_prompt_v3:
            system_prompt = SYSTEM_PROMPT_CALIBRATED_V3
        elif args.calibrated_prompt_v2:
            system_prompt = SYSTEM_PROMPT_CALIBRATED_V2
        elif args.calibrated_prompt:
            system_prompt = SYSTEM_PROMPT_CALIBRATED
        else:
            system_prompt = None
        # Citations are what carry private evidence between rounds, so require
        # them whenever a mechanism downstream consumes them.
        require_citations = share_mode == "arguments" or args.evidence_pooling

        def _new_agent():
            a = LLMAgent(temperature=args.temperature, max_tokens=2048,
                         system_prompt=system_prompt, require_citations=require_citations)
            all_agents.append(a)
            return a

        agent = _new_agent()

        def agent_factory(i):
            return _new_agent()

    meta = {
        "model": model_name,
        "num_agents": args.num_agents,
        "num_rounds": args.num_rounds,
        "run_ts": ts,
        "dry_run": args.dry_run,
        "aggregator": args.aggregator,
        "public_ratio": args.public_ratio,
        "bm25": args.bm25,
        "temperature": args.temperature,
        "share_mode": share_mode,
        "evidence_pooling": args.evidence_pooling,
        "calibrated_prompt": ("v3" if args.calibrated_prompt_v3 else "v2" if args.calibrated_prompt_v2 else args.calibrated_prompt),
        "devils_advocate": args.devils_advocate,
    }

    if args.aggregator == "extremizing":
        aggregator = ExtremizingAggregator(alpha=args.extremizing_alpha)
    elif args.aggregator == "confidence_weighted":
        aggregator = ConfidenceWeightedAggregator()
    elif args.aggregator == "adaptive_extremizing":
        aggregator = AdaptiveExtremizingAggregator()
    elif args.aggregator == "temporal_reliability":
        aggregator = TemporalReliabilityAggregator()
    elif args.aggregator == "calibrated_shrink":
        aggregator = CalibratedShrinkAggregator(p0=args.shrink_p0, w_lo=args.shrink_w_lo,
                                                w_hi=args.shrink_w_hi)
    else:
        aggregator = MeanAggregator()
    run_s0_ = args.scenario in ("s0", "all")
    run_s1_ = args.scenario in ("s1", "all", "s1s2")
    run_s2_ = args.scenario in ("s2", "all", "s1s2")

    detail_path = out_path.with_name(out_path.stem + "_detail.jsonl")
    # When S1 and S2 run together with the same routing configuration, S2's
    # first round is identical to S1's independent-agent round. Cache those
    # forecasts so S2 can deliberate from them without repeating API calls.
    s1_forecast_cache: dict[str, list[Forecast]] = {}

    # Streaming write callback: writes each question's result immediately (thread-safe)
    def _make_on_complete(scenario: str):
        def _on_complete(qid: str, forecast, detail_records_local: list[dict]):
            outcome = outcome_map.get(qid)
            record = {
                "qid": qid,
                "question": question_map.get(qid, ""),
                "topic": topic_map.get(qid, ""),
                "scenario": scenario,
                "p_yes": forecast.p_yes,
                "label": forecast.label,
                "rationale": forecast.rationale,
                "evidence_used": forecast.evidence_used,
                "outcome": outcome,
                "brier": brier_score(forecast.p_yes, outcome) if outcome is not None else None,
                **meta,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if detail_records_local:
                with open(detail_path, "a") as f:
                    for rec in detail_records_local:
                        f.write(json.dumps({
                            **rec,
                            "question": question_map.get(qid, ""),
                            "topic": topic_map.get(qid, ""),
                            "outcome": outcome,
                            **meta,
                        }, ensure_ascii=False) + "\n")
        return _on_complete

    if run_s0_:
        print(f"=== S0: Single agent, full evidence (workers={args.max_workers}) ===")
        s0_agent_factory = _new_agent if not args.dry_run else None
        results = run_s0(examples, agent, args.evidence_max_items, args.evidence_max_chars,
                         max_workers=args.max_workers, agent_factory=s0_agent_factory,
                         on_complete=_make_on_complete("s0"))
        print_eval(results, outcome_map, "S0")
        print()

    if run_s1_:
        print(f"=== S1: Multi-agent independent (workers={args.max_workers}) ===")
        results, details = run_s1(examples, agent_factory, aggregator,
                                  num_agents=args.num_agents, seed=args.seed,
                                  evidence_max_items=args.evidence_max_items,
                                  evidence_max_chars=args.evidence_max_chars,
                                  public_ratio=args.public_ratio,
                                  use_bm25=args.bm25,
                                  max_workers=args.max_workers,
                                  on_complete=_make_on_complete("s1"),
                                  forecast_cache=s1_forecast_cache if run_s2_ else None)
        print_eval(results, outcome_map, "S1")
        print()

    if run_s2_:
        print(f"=== S2: Multi-agent iterative (workers={args.max_workers}) ===")
        results, details = run_s2(examples, agent_factory, aggregator,
                                  num_agents=args.num_agents, num_rounds=args.num_rounds, seed=args.seed,
                                  evidence_max_items=args.evidence_max_items,
                                  evidence_max_chars=args.evidence_max_chars,
                                  public_ratio=args.public_ratio,
                                  use_bm25=args.bm25,
                                  use_selective_update=args.selective_update,
                                  herding_threshold=args.herding_threshold,
                                  max_workers=args.max_workers,
                                  on_complete=_make_on_complete("s2"),
                                  show_rationale=not args.no_rationale_sharing,
                                  share_mode=share_mode,
                                  evidence_pooling=args.evidence_pooling,
                                  devils_advocate=args.devils_advocate,
                                  initial_forecasts_by_qid=s1_forecast_cache if run_s1_ else None)
        print_eval(results, outcome_map, "S2")
        print()

    if all_agents:
        total_prompt = sum(a.total_prompt_tokens for a in all_agents)
        total_completion = sum(a.total_completion_tokens for a in all_agents)
        total_cost = sum(a.total_cost_usd for a in all_agents)
        print(f"=== Token summary ===")
        print(f"  prompt tokens:     {total_prompt:,}")
        print(f"  completion tokens: {total_completion:,}")
        print(f"  total tokens:      {total_prompt + total_completion:,}")
        print(f"  total cost:        ${total_cost:.4f}")
        print()

    print(f"Results saved to {out_path}")

    if not args.dry_run:
        _update_registry(args, ts, out_path, selected_qids)

    print("Done." + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
