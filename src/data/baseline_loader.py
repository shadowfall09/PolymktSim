"""Load processed JSONL datasets into the dict shape used by baseline scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import EvidenceItem


def load_processed_baseline_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load a processed JSONL dataset with embedded evidence for baseline runs."""
    questions: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for row_index, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid") or f"row_{row_index:04d}").strip()
            question = str(row.get("question") or "").strip()
            if not qid or not question:
                raise ValueError(f"row {row_index} requires qid and question")
            outcome = row.get("outcome")
            if not isinstance(outcome, bool):
                outcome = None
            evidence = [
                EvidenceItem(**item)
                for item in row.get("evidence") or []
                if isinstance(item, dict)
            ]
            questions.append({
                "qid": qid,
                "question": question,
                "evidence": evidence,
                "outcome": outcome,
                "topic": str(row.get("topic") or ""),
            })
    return questions
