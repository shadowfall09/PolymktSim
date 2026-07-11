from abc import ABC, abstractmethod

from src.data.schema import EvidenceItem


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, question: str, qid: str, top_k: int = 20) -> list[EvidenceItem]:
        """Return top_k evidence items for the question."""
        ...
