"""Calibrated-shrink aggregator: mean, then asymmetric shrinkage toward a prior.

Motivation (2026-07 diagnosis on polymarket_250 + futurex s2 runs): the
aggregate forecast is systematically overconfident, and much more so on the
YES side — p>=0.6 predictions resolved YES only ~29% of the time on
polymarket_250 while the low side was nearly calibrated. Shrinking the mean
toward a NO-leaning prior, with a stronger pull above the prior than below,
transfers across both datasets:

    fixed (p0=0.30, w_lo=0.8, w_hi=0.5)
    polymarket_250 s2: 0.1647 -> 0.1575 Brier
    futurex 231   s2: 0.2212 -> 0.2006 Brier, accuracy 0.688 -> 0.706

Formula:  m = mean(p_i)
          w = w_hi if m > p0 else w_lo
          p_agg = p0 + w * (m - p0)
"""
from src.data.schema import Forecast
from .base import BaseAggregator


class CalibratedShrinkAggregator(BaseAggregator):
    def __init__(self, p0: float = 0.30, w_lo: float = 0.8, w_hi: float = 0.5):
        if not 0.0 < p0 < 1.0:
            raise ValueError(f"p0 must be in (0,1), got {p0}")
        if not (0.0 <= w_lo <= 1.0 and 0.0 <= w_hi <= 1.0):
            raise ValueError(f"weights must be in [0,1], got w_lo={w_lo}, w_hi={w_hi}")
        self.p0 = p0
        self.w_lo = w_lo
        self.w_hi = w_hi

    def aggregate(self, forecasts: list[Forecast], evidence_sets=None, resolution_date=None) -> Forecast:
        if not forecasts:
            return Forecast(p_yes=0.5, label="NO", rationale="no forecasts", evidence_used=[])
        m = sum(f.p_yes for f in forecasts) / len(forecasts)
        w = self.w_hi if m > self.p0 else self.w_lo
        p = self.p0 + w * (m - self.p0)
        return Forecast.from_p_yes(
            p,
            rationale=f"calibrated_shrink(p0={self.p0}, w_lo={self.w_lo}, w_hi={self.w_hi}) from mean={m:.4f}",
            evidence_used=[],
        )
