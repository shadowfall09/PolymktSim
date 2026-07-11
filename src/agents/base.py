"""Agent interface: all agents implement predict()."""
from abc import ABC, abstractmethod

from src.data.schema import EvidenceItem, Forecast


class BaseAgent(ABC):
    @abstractmethod
    def predict(
        self,
        qid: str,
        question: str,
        public_evidence: list[EvidenceItem],
        private_evidence: list[EvidenceItem],
        history_summary: str,
    ) -> Forecast:
        """Produce a single forecast. Deterministic decoding recommended."""
        ...
