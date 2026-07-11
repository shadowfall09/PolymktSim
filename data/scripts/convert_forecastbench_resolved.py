#!/usr/bin/env python3
"""Convert resolved ForecastBench questions into PolymktSim JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_bool(value: Any) -> bool:
    if value == 1 or value == 1.0:
        return True
    if value == 0 or value == 0.0:
        return False
    raise ValueError(f"Only binary resolved_to values are supported, got: {value!r}")


def _question_text(question: dict[str, Any]) -> str:
    text = str(question.get("question") or "").strip()
    background = str(question.get("background") or "").strip()
    criteria = str(question.get("resolution_criteria") or "").strip()

    details = []
    if criteria:
        details.append(f"Resolution criteria: {criteria}")
    if background:
        details.append(f"Background: {background}")
    if details:
        return text + "\n\n" + "\n\n".join(details)
    return text


def convert(question_set: Path, resolution_set: Path) -> list[dict[str, Any]]:
    questions_doc = _read_json(question_set)
    resolutions_doc = _read_json(resolution_set)

    questions_by_id = {
        str(item.get("id")): item
        for item in questions_doc.get("questions", [])
        if item.get("id") is not None
    }

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for resolution in resolutions_doc.get("resolutions", []):
        if not resolution.get("resolved"):
            continue
        resolved_to = resolution.get("resolved_to")
        if resolved_to not in (0, 0.0, 1, 1.0):
            continue

        qid = str(resolution.get("id") or "").strip()
        question = questions_by_id.get(qid)
        if question is None:
            missing.append(qid)
            continue

        rows.append(
            {
                "qid": f"forecastbench_2026_06_21_{qid}",
                "question": _question_text(question),
                "evidence": [],
                "outcome": _as_bool(resolved_to),
                "resolution_date": str(resolution.get("resolution_date") or "").strip(),
                "source": question.get("source") or resolution.get("source"),
                "url": question.get("url"),
                "forecast_due_date": questions_doc.get("forecast_due_date"),
                "forecastbench_id": qid,
            }
        )

    if missing:
        raise ValueError(f"{len(missing)} resolved ids were not found in question set")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-set", type=Path, required=True)
    parser.add_argument("--resolution-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = convert(args.question_set, args.resolution_set)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    yes_count = sum(1 for row in rows if row["outcome"] is True)
    no_count = sum(1 for row in rows if row["outcome"] is False)
    print(f"wrote={len(rows)} yes={yes_count} no={no_count} output={args.output}")


if __name__ == "__main__":
    main()
