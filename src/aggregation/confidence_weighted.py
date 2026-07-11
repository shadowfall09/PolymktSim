"""Confidence-weighted aggregator: weight each forecast by |logit(p_i)|.

Rationale: an agent expressing p=0.95 is making a stronger claim than
p=0.6. Weighting by logit magnitude lets more-certain agents dominate.
This mirrors Chen et al. (2024) Reconcile's confidence-weighted voting.

Formula:  w_i = |logit(p_i)|
          p_agg = sigmoid( sum(w_i * logit(p_i)) / sum(w_i) )
"""
import math
from src.data.schema import Forecast
from .base import BaseAggregator

_EPS = 1e-6


def _logit(p: float) -> float:
    p = max(_EPS, min(1 - _EPS, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class ConfidenceWeightedAggregator(BaseAggregator):
    """Aggregate in logit space, weighted by each agent's confidence."""

    def aggregate(self, forecasts: list[Forecast], evidence_sets=None, resolution_date=None) -> Forecast:
        if not forecasts:
            return Forecast(p_yes=0.5, label="NO", rationale="no forecasts", evidence_used=[])
        logits = [_logit(f.p_yes) for f in forecasts]
        weights = [abs(l) for l in logits]
        total_w = sum(weights)
        if total_w < _EPS:
            # All agents near 0.5 — fall back to mean
            p = sum(f.p_yes for f in forecasts) / len(forecasts)
        else:
            weighted_logit = sum(w * l for w, l in zip(weights, logits)) / total_w
            p = _sigmoid(weighted_logit)
        return Forecast.from_p_yes(p, rationale="confidence_weighted", evidence_used=[])
