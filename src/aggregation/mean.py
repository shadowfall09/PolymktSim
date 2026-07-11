"""Mean aggregation: average p_yes."""
from src.data.schema import Forecast
from .base import BaseAggregator


class MeanAggregator(BaseAggregator):
    def aggregate(self, forecasts: list[Forecast], evidence_sets=None, resolution_date=None) -> Forecast:
        if not forecasts:
            return Forecast(p_yes=0.5, label="NO", rationale="no forecasts", evidence_used=[])
        p = sum(f.p_yes for f in forecasts) / len(forecasts)
        return Forecast.from_p_yes(p, rationale="mean aggregation", evidence_used=[])
