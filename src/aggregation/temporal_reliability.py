"""Temporal Reliability-Weighted Aggregator (TERS).

For each agent, compute a reliability score r_i ∈ [0, 1] based on whether
their evidence items contain dates that fall after the question's resolution
date (possible temporal leakage).

Adjustment:
    p_adj_i = 0.5 + (p_i - 0.5) * r_i

Agents with low reliability (suspicious evidence) have their predictions
shrunk toward 0.5 (maximum uncertainty) before aggregation.
Final output is a mean over the adjusted predictions.

This combats the failure mode where all agents see misleading future-dated
evidence and collectively produce high-confidence wrong forecasts.
"""
from datetime import date
from typing import Optional

from src.data.schema import EvidenceItem, Forecast
from src.data.temporal_scorer import agent_reliability
from .base import BaseAggregator

_EPS = 1e-6


class TemporalReliabilityAggregator(BaseAggregator):
    """Aggregate with per-agent confidence discount based on evidence temporal quality."""

    def aggregate(
        self,
        forecasts: list[Forecast],
        evidence_sets: Optional[list[list[EvidenceItem]]] = None,
        resolution_date: Optional[date] = None,
    ) -> Forecast:
        if not forecasts:
            return Forecast(p_yes=0.5, label="NO", rationale="no forecasts", evidence_used=[])

        # Compute per-agent reliability
        reliabilities: list[float] = []
        for i, f in enumerate(forecasts):
            ev = evidence_sets[i] if evidence_sets and i < len(evidence_sets) else []
            r = agent_reliability(ev, resolution_date)
            reliabilities.append(r)

        # Adjust predictions: shrink toward 0.5 proportional to unreliability
        adjusted: list[float] = []
        for f, r in zip(forecasts, reliabilities):
            p_adj = 0.5 + (f.p_yes - 0.5) * r
            adjusted.append(p_adj)

        p_mean = sum(adjusted) / len(adjusted)
        avg_r = sum(reliabilities) / len(reliabilities)

        rationale = (
            f"temporal_reliability(avg_r={avg_r:.2f}, "
            f"reliabilities={[round(r,2) for r in reliabilities]})"
        )
        return Forecast.from_p_yes(p_mean, rationale=rationale, evidence_used=[])
