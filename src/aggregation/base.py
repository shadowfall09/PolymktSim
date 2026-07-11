"""Aggregator interface: combine multiple forecasts into one."""
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

from src.data.schema import EvidenceItem, Forecast


class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        forecasts: list[Forecast],
        evidence_sets: list[list[EvidenceItem]] | None = None,
        resolution_date: Optional[date] = None,
    ) -> Forecast:
        """Produce a single forecast from many.

        Args:
            forecasts:        Individual agent predictions.
            evidence_sets:    Optional per-agent evidence lists (public+private).
            resolution_date:  Market resolution date for temporal reliability scoring.
        """
        ...
