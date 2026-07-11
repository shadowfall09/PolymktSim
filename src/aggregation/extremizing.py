"""Extremizing aggregator: logit-space mean then re-sharpened by factor alpha.

Theoretical basis: Satopaa et al. (2014) show that linear opinion pools
are systematically underconfident when forecasters share information.
Abernethy & Frongillo (2014) show optimal alpha increases as information
overlap decreases — which is exactly our split-evidence setup.

Formula:  p_ext = sigmoid(alpha * mean(logit(p_i)))
  alpha = 1  →  reduces to mean aggregation (linear opinion pool)
  alpha > 1  →  extremizes toward 0/1 (sharper)
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


class ExtremizingAggregator(BaseAggregator):
    """Logit-average then extremize by alpha."""

    def __init__(self, alpha: float = 2.5):
        self.alpha = alpha

    def aggregate(self, forecasts: list[Forecast], evidence_sets=None, resolution_date=None) -> Forecast:
        if not forecasts:
            return Forecast(p_yes=0.5, label="NO", rationale="no forecasts", evidence_used=[])
        mean_logit = sum(_logit(f.p_yes) for f in forecasts) / len(forecasts)
        p = _sigmoid(self.alpha * mean_logit)
        return Forecast.from_p_yes(p, rationale=f"extremizing(alpha={self.alpha})", evidence_used=[])
