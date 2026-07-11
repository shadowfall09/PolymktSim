"""Split evidence into public vs per-agent private. """
import random
from typing import TypedDict

from .schema import EvidenceItem


class SplitResult(TypedDict):
    public_evidence: list[EvidenceItem]
    private_map: dict[int, list[EvidenceItem]]  # agent_id -> private evidence


def split(
    evidence: list[EvidenceItem],
    num_agents: int,
    seed: int,
    public_ratio: float = 0.0,
) -> SplitResult:
    """Random split: optional public subset, rest partitioned across agents."""
    rng = random.Random(seed)
    shuffled = evidence.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_public = int(n * public_ratio) if public_ratio else 0
    public_evidence = shuffled[:n_public]
    private_pool = shuffled[n_public:]
    private_map: dict[int, list[EvidenceItem]] = {}
    for i in range(num_agents):
        start = i * len(private_pool) // num_agents
        end = (i + 1) * len(private_pool) // num_agents
        private_map[i] = private_pool[start:end]
    return {"public_evidence": public_evidence, "private_map": private_map}
