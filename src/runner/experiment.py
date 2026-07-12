from __future__ import annotations
"""Run one experiment (S0/S1/S2) over dataset."""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from src.agents.base import BaseAgent
from src.agents.protocol import summarize_round
from src.aggregation.base import BaseAggregator
from src.data.schema import EvidenceItem, Forecast, QuestionExample
from src.data.splitter import split
from src.data.bm25_splitter import bm25_split
from src.runner.belief_revision import selective_update

logger = logging.getLogger(__name__)

# Thread-safe write lock for parallel result writing
_write_lock = threading.Lock()


def run_s0(
    examples: list[QuestionExample],
    agent: BaseAgent,
    evidence_max_items: int,
    evidence_max_chars: int,
    max_workers: int = 1,
    agent_factory: Any = None,
    on_complete: Callable[[str, Forecast, list[dict]], None] | None = None,
) -> list[tuple[str, Forecast]]:
    """Single agent, full evidence. Supports parallel execution via max_workers.

    on_complete: optional callback(qid, forecast, detail_records) called per question (thread-safe).
    """
    if max_workers <= 1 or agent_factory is None:
        # Sequential (original behavior)
        results = []
        for ex in examples:
            ev = ex.evidence[:evidence_max_items]
            f = agent.predict(ex.qid, ex.question, ev, [], "")
            results.append((ex.qid, f))
            if on_complete:
                on_complete(ex.qid, f, [])
        return results

    # Parallel execution
    def _predict_one(ex: QuestionExample) -> tuple[str, Forecast]:
        a = agent_factory()
        ev = ex.evidence[:evidence_max_items]
        f = a.predict(ex.qid, ex.question, ev, [], "")
        return (ex.qid, f)

    results_map: dict[str, Forecast] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_predict_one, ex): ex.qid for ex in examples}
        for future in as_completed(futures):
            qid = futures[future]
            try:
                qid_out, forecast = future.result()
                results_map[qid_out] = forecast
                if on_complete:
                    with _write_lock:
                        on_complete(qid_out, forecast, [])
            except Exception as e:
                logger.error("S0 parallel qid=%s failed: %s", qid, e)
                results_map[qid] = Forecast(p_yes=0.5, label="NO", rationale=f"[error: {e}]", evidence_used=[])

    # Preserve original order
    return [(ex.qid, results_map[ex.qid]) for ex in examples if ex.qid in results_map]


def run_s1(
    examples: list[QuestionExample],
    agent_factory: Any,
    aggregator: BaseAggregator,
    num_agents: int,
    seed: int,
    evidence_max_items: int,
    evidence_max_chars: int,
    public_ratio: float = 0.5,
    use_bm25: bool = False,
    max_workers: int = 1,
    on_complete: Callable[[str, Forecast, list[dict]], None] | None = None,
    forecast_cache: dict[str, list[Forecast]] | None = None,
) -> tuple[list[tuple[str, Forecast]], list[dict]]:
    """Multi agent, independent; optionally cache agent forecasts for S2 round 0."""
    results = []
    detail_records = []

    def _run_one_question(ex: QuestionExample) -> tuple[str, Forecast, list[dict]]:
        ev = ex.evidence[:evidence_max_items]
        sp = (bm25_split(ev, ex.question, num_agents, public_ratio=public_ratio)
              if use_bm25 else split(ev, num_agents, seed, public_ratio=public_ratio))
        forecasts = []
        agent_evidence: list[list[EvidenceItem]] = []
        local_details = []
        for i in range(num_agents):
            agent = agent_factory(i)
            priv = sp["private_map"].get(i, [])
            full_ev = sp["public_evidence"] + priv
            f = agent.predict(ex.qid, ex.question, sp["public_evidence"], priv, "")
            forecasts.append(f)
            agent_evidence.append(full_ev)
            logger.info("S1 qid=%s agent_id=%d p_yes=%.3f label=%s", ex.qid, i, f.p_yes, f.label)
            local_details.append({
                "qid": ex.qid, "scenario": "s1", "agent_id": i, "round_id": 0,
                "p_yes": f.p_yes, "label": f.label,
                "rationale": f.rationale, "evidence_used": f.evidence_used,
            })
        agg = aggregator.aggregate(forecasts, evidence_sets=agent_evidence,
                                   resolution_date=ex.resolution_date)
        if forecast_cache is not None:
            with _write_lock:
                forecast_cache[ex.qid] = list(forecasts)
        return ex.qid, agg, local_details

    if max_workers <= 1:
        # Sequential
        for ex in examples:
            qid, agg, local_details = _run_one_question(ex)
            results.append((qid, agg))
            detail_records.extend(local_details)
            if on_complete:
                on_complete(qid, agg, local_details)
            print(f"  {qid}  p_yes={agg.p_yes:.3f}  label={agg.label}")
    else:
        # Parallel
        results_map: dict[str, tuple[Forecast, list[dict]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one_question, ex): ex.qid for ex in examples}
            for future in as_completed(futures):
                qid = futures[future]
                try:
                    qid_out, agg, local_details = future.result()
                    results_map[qid_out] = (agg, local_details)
                    if on_complete:
                        with _write_lock:
                            on_complete(qid_out, agg, local_details)
                    print(f"  {qid_out}  p_yes={agg.p_yes:.3f}  label={agg.label}")
                except Exception as e:
                    logger.error("S1 parallel qid=%s failed: %s", qid, e)
                    fallback = Forecast(p_yes=0.5, label="NO", rationale=f"[error: {e}]", evidence_used=[])
                    results_map[qid] = (fallback, [])
        # Preserve original order
        for ex in examples:
            if ex.qid in results_map:
                agg, local_details = results_map[ex.qid]
                results.append((ex.qid, agg))
                detail_records.extend(local_details)

    return results, detail_records


def run_s2(
    examples: list[QuestionExample],
    agent_factory: Any,
    aggregator: BaseAggregator,
    num_agents: int,
    num_rounds: int,
    seed: int,
    evidence_max_items: int,
    evidence_max_chars: int,
    public_ratio: float = 0.5,
    use_bm25: bool = False,
    use_selective_update: bool = False,
    herding_threshold: float = 0.7,
    max_workers: int = 1,
    on_complete: Callable[[str, Forecast, list[dict]], None] | None = None,
    show_rationale: bool = True,
    share_mode: str = "full",
    evidence_pooling: bool = False,
    initial_forecasts_by_qid: dict[str, list[Forecast]] | None = None,
) -> tuple[list[tuple[str, Forecast]], list[dict]]:
    """Multi agent deliberation, optionally reusing S1 forecasts as round 0.

    evidence_pooling: private docs cited by their owner (via evidence_used) are
    disclosed in full to all agents in subsequent rounds, so private signal
    travels as original evidence instead of a truncated rationale excerpt.
    """
    results = []
    detail_records = []

    def _run_one_question(ex: QuestionExample) -> tuple[str, Forecast, list[dict]]:
        ev = ex.evidence[:evidence_max_items]
        sp = (bm25_split(ev, ex.question, num_agents, public_ratio=public_ratio)
              if use_bm25 else split(ev, num_agents, seed, public_ratio=public_ratio))
        history = ""
        forecasts: list[Forecast] = []
        round1_forecasts: list[Forecast] = []
        agent_evidence: list[list[EvidenceItem]] = [
            sp["public_evidence"] + sp["private_map"].get(i, [])
            for i in range(num_agents)
        ]
        local_details = []
        promoted: list[EvidenceItem] = []  # cited private docs, now visible to everyone
        promoted_ids: set[str] = set()
        reusable = (initial_forecasts_by_qid or {}).get(ex.qid)
        if reusable is not None and len(reusable) != num_agents:
            logger.warning(
                "S2 qid=%s cannot reuse S1: expected %d forecasts, got %d",
                ex.qid, num_agents, len(reusable),
            )
            reusable = None
        for round_id in range(num_rounds):
            pooled_by_agent: dict[int, list[str]] = {i: [] for i in range(num_agents)}
            if round_id == 0 and reusable is not None:
                forecasts = list(reusable)
                logger.info("S2 qid=%s round=1 reused %d S1 forecasts", ex.qid, len(forecasts))
            else:
                forecasts = []
                for i in range(num_agents):
                    agent = agent_factory(i)
                    priv = sp["private_map"].get(i, [])
                    pooled: list[EvidenceItem] = []
                    if evidence_pooling and promoted:
                        own_ids = {d.doc_id for d in priv}
                        pooled = [d for d in promoted if d.doc_id not in own_ids]
                        pooled_by_agent[i] = [d.doc_id for d in pooled]
                    f = agent.predict(ex.qid, ex.question, sp["public_evidence"] + pooled, priv, history)
                    forecasts.append(f)
                    logger.info("S2 qid=%s round=%d agent_id=%d p_yes=%.3f label=%s", ex.qid, round_id + 1, i, f.p_yes, f.label)
            for i, f in enumerate(forecasts):
                local_details.append({
                    "qid": ex.qid, "scenario": "s2", "agent_id": i, "round_id": round_id,
                    "p_yes": f.p_yes, "label": f.label,
                    "rationale": f.rationale, "evidence_used": f.evidence_used,
                    "reused_from_s1": round_id == 0 and reusable is not None,
                    "pooled_evidence": pooled_by_agent.get(i, []),
                })
            if evidence_pooling:
                for i, f in enumerate(forecasts):
                    cited = {str(x).strip() for x in (f.evidence_used or [])}
                    for d in sp["private_map"].get(i, []):
                        if d.doc_id in cited and d.doc_id not in promoted_ids:
                            promoted.append(d)
                            promoted_ids.add(d.doc_id)
            if round_id == 0:
                round1_forecasts = list(forecasts)
            history = summarize_round(forecasts, show_rationale=show_rationale, share_mode=share_mode)

        # Selective update: revert herding agents to round-1 predictions
        if use_selective_update and len(round1_forecasts) == len(forecasts):
            forecasts, br_stats = selective_update(
                round1_forecasts, forecasts, herding_threshold=herding_threshold
            )
            logger.info(
                "S2 qid=%s herding: %d/%d agents reverted (ratios=%s)",
                ex.qid, br_stats.herding_count, num_agents,
                [f"{r:.2f}" for r in br_stats.herding_ratios],
            )

        agg = aggregator.aggregate(forecasts, evidence_sets=agent_evidence,
                                   resolution_date=ex.resolution_date)
        return ex.qid, agg, local_details

    if max_workers <= 1:
        # Sequential
        for ex in examples:
            print(f"  [S2] {ex.qid}:")
            qid, agg, local_details = _run_one_question(ex)
            results.append((qid, agg))
            detail_records.extend(local_details)
            if on_complete:
                on_complete(qid, agg, local_details)
            print(f"    final p_yes={agg.p_yes:.3f}  label={agg.label}")
    else:
        # Parallel (different questions in parallel; rounds within a question are still sequential)
        results_map: dict[str, tuple[Forecast, list[dict]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one_question, ex): ex.qid for ex in examples}
            for future in as_completed(futures):
                qid = futures[future]
                try:
                    qid_out, agg, local_details = future.result()
                    results_map[qid_out] = (agg, local_details)
                    if on_complete:
                        with _write_lock:
                            on_complete(qid_out, agg, local_details)
                    print(f"  {qid_out}  p_yes={agg.p_yes:.3f}  label={agg.label}")
                except Exception as e:
                    logger.error("S2 parallel qid=%s failed: %s", qid, e)
                    fallback = Forecast(p_yes=0.5, label="NO", rationale=f"[error: {e}]", evidence_used=[])
                    results_map[qid] = (fallback, [])
        # Preserve original order
        for ex in examples:
            if ex.qid in results_map:
                agg, local_details = results_map[ex.qid]
                results.append((ex.qid, agg))
                detail_records.extend(local_details)

    return results, detail_records
