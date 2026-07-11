"""Load dataset and evidence cache."""
import json
from pathlib import Path

from .schema import EvidenceItem, QuestionExample


def load_dataset(path: str | Path) -> list[QuestionExample]:
    """Load processed dataset (JSONL)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.setdefault("evidence", [])
            d["evidence"] = [EvidenceItem(**e) if isinstance(e, dict) else e for e in d["evidence"]]
            out.append(QuestionExample(**d))
    return out


def load_evidence_cache(path: str | Path) -> list[EvidenceItem]:
    """Load evidence cache for a question (e.g. qid_xxx.json)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [EvidenceItem(**e) if isinstance(e, dict) else e for e in data]
    return [EvidenceItem(**data)]
