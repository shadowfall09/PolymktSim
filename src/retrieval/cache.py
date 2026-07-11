"""Evidence cache: load/save by qid."""
from __future__ import annotations
import json
from pathlib import Path

from src.data.schema import EvidenceItem


def get_cache_path(cache_dir: Path, qid: str) -> Path:
    return cache_dir / f"qid_{qid}.json"


def load_cached(cache_dir: Path, qid: str) -> list[EvidenceItem] | None:
    p = get_cache_path(cache_dir, qid)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]
    return [EvidenceItem(**e) if isinstance(e, dict) else e for e in items]


def save_cache(cache_dir: Path, qid: str, evidence: list[EvidenceItem]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = get_cache_path(cache_dir, qid)
    with open(p, "w") as f:
        json.dump([e.model_dump(mode="json") for e in evidence], f, indent=0)
