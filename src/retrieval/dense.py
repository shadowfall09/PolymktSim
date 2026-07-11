"""Dense retriever stub."""
from .base import BaseRetriever
from src.data.schema import EvidenceItem


class DenseRetriever(BaseRetriever):
    def retrieve(self, question: str, qid: str, top_k: int = 20) -> list[EvidenceItem]:
        return []
