"""Meta-model aggregator stub: train logistic regression on p_yes features."""
from src.data.schema import Forecast
from .base import BaseAggregator


class MetaModelAggregator(BaseAggregator):
    """Stub: fits on dev set later; predict returns mean for now."""

    def aggregate(self, forecasts: list[Forecast]) -> Forecast:
        if not forecasts:
            return Forecast(p_yes=0.5, label="NO", rationale="meta (no forecasts)", evidence_used=[])
        p = sum(f.p_yes for f in forecasts) / len(forecasts)
        return Forecast.from_p_yes(p, rationale="meta-model stub", evidence_used=[])
