"""Adaptive extremizing aggregator: α* = 1/J where J is mean pairwise Jaccard
of agent evidence sets.

Theoretical derivation
----------------------
Each agent's forecast can be decomposed as:
    logit(p_i) = logit(p_prior) + LLR_i
where LLR_i = log P(evidence_i|YES) / P(evidence_i|NO) is the private signal.

When evidence sets are fully independent (J=0), the Bayesian-optimal combination
is the product of likelihoods:
    logit(p_agg) = logit(p_prior) + Σ LLR_i
                 = logit(p_prior) + N × (mean_logit - logit(p_prior))

When sets fully overlap (J=1), each agent holds the same information and a
simple average suffices (α=1).

For intermediate overlap J, the effective number of independent signals is 1/J,
giving the optimal extremizing factor α* = 1/J.

This extends Satopaa et al. (2014) and Abernethy & Frongillo (2014) by providing
a closed-form α derived from measurable evidence overlap rather than heuristic
tuning.
"""
import math
from typing import Optional
from src.data.schema import EvidenceItem, Forecast
from .base import BaseAggregator

_EPS = 1e-6
_ALPHA_MIN = 1.0
_ALPHA_MAX = 6.0


def _logit(p: float) -> float:
    p = max(_EPS, min(1 - _EPS, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _mean_pairwise_jaccard(evidence_sets: list[list[EvidenceItem]]) -> float:
    """Compute mean pairwise Jaccard similarity of evidence sets by doc_id."""
    id_sets = [frozenset(e.doc_id for e in ev) for ev in evidence_sets]
    n = len(id_sets)
    if n < 2:
        return 1.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            union = id_sets[i] | id_sets[j]
            j_val = len(id_sets[i] & id_sets[j]) / len(union) if union else 1.0
            total += j_val
            count += 1
    return total / count if count else 1.0


class AdaptiveExtremizingAggregator(BaseAggregator):
    """Extremize by α* = 1/J, where J is mean pairwise Jaccard of agent evidence."""

    def __init__(
        self,
        fallback_alpha: float = 2.0,
        alpha_min: float = _ALPHA_MIN,
        alpha_max: float = _ALPHA_MAX,
    ):
        self.fallback_alpha = fallback_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def _compute_alpha(self, evidence_sets: Optional[list]) -> tuple:
        """Return (alpha, jaccard). Uses fallback_alpha if evidence_sets unavailable."""
        if not evidence_sets:
            return self.fallback_alpha, float("nan")
        J = _mean_pairwise_jaccard(evidence_sets)
        if J < _EPS:
            alpha = self.alpha_max
        else:
            alpha = min(self.alpha_max, max(self.alpha_min, 1.0 / J))
        return alpha, J

    def aggregate(
        self,
        forecasts: list[Forecast],
        evidence_sets: Optional[list] = None,
        resolution_date=None,
    ) -> Forecast:
        if not forecasts:
            return Forecast(p_yes=0.5, label="NO", rationale="no forecasts", evidence_used=[])
        alpha, J = self._compute_alpha(evidence_sets)
        mean_logit = sum(_logit(f.p_yes) for f in forecasts) / len(forecasts)
        p = _sigmoid(alpha * mean_logit)
        j_str = f"{J:.3f}" if not math.isnan(J) else "N/A"
        return Forecast.from_p_yes(
            p,
            rationale=f"adaptive_extremizing(alpha={alpha:.2f}, J={j_str})",
            evidence_used=[],
        )
