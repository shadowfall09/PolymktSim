"""BM25-based evidence splitter.

Instead of random shuffling, rank evidence by BM25 relevance to the
question, then assign:
  - public_evidence: top-scoring docs shared by all agents
  - private_map[i]: the next tier of docs, distributed round-robin by
    relevance rank so each agent gets a complementary relevant subset.

This implements the information-diversity hypothesis from Abernethy &
Frongillo (2014): lower evidence overlap → stronger extremization is
warranted. With BM25 routing, private evidence is both relevant and
non-overlapping, maximizing the signal each agent contributes.
"""
import re
from typing import TypedDict

from rank_bm25 import BM25Okapi

from .schema import EvidenceItem
from .splitter import SplitResult


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def bm25_split(
    evidence: list[EvidenceItem],
    question: str,
    num_agents: int,
    public_ratio: float = 0.5,
) -> SplitResult:
    """Rank evidence by BM25 relevance, then split into public + private pools.

    Public pool  = top `public_ratio` fraction (shared by all agents).
    Private pool = remaining docs assigned round-robin by rank, so agent 0
                   gets rank-1, rank-4, …; agent 1 gets rank-2, rank-5, …
                   Each agent's private docs are as relevant as possible
                   while staying complementary.
    """
    if not evidence:
        return {"public_evidence": [], "private_map": {i: [] for i in range(num_agents)}}

    corpus = [_tokenize((e.title or "") + " " + e.content) for e in evidence]
    query = _tokenize(question)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)

    # Sort by descending relevance score
    ranked = sorted(range(len(evidence)), key=lambda i: -scores[i])

    n_public = max(0, int(len(ranked) * public_ratio))
    public_evidence = [evidence[i] for i in ranked[:n_public]]
    private_pool = [evidence[i] for i in ranked[n_public:]]

    # Round-robin assignment preserves relevance ordering per agent
    private_map: dict[int, list[EvidenceItem]] = {i: [] for i in range(num_agents)}
    for slot, item in enumerate(private_pool):
        private_map[slot % num_agents].append(item)

    return {"public_evidence": public_evidence, "private_map": private_map}
