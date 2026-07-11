#!/usr/bin/env python3
"""Convert FutureX past Yes/No JSONL into PolymktSim QuestionExample JSONL."""
from __future__ import annotations

import argparse
import ast
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _parse_yes_no(value: Any) -> bool | None:
    if isinstance(value, list) and value:
        value = value[0]
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list) and parsed:
                    value = parsed[0]
                else:
                    value = text
            except Exception:
                value = text

    normalized = str(value or "").strip().lower()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    return None


def _parse_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    # FutureX uses both ISO dates and slash-separated, non-zero-padded dates
    # (for example, 2026/5/1).
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        pass
    try:
        return datetime.strptime(text.split()[0], "%Y/%m/%d").date().isoformat()
    except ValueError:
        return None


def convert_row(row: dict[str, Any]) -> dict[str, Any] | None:
    raw_id = str(row.get("id") or "").strip()
    title = str(row.get("title") or "").strip()
    if not raw_id or not title:
        return None

    outcome = _parse_yes_no(row.get("answer") or row.get("ground_truth"))
    resolution_date = _parse_date(row.get("end_time"))

    return {
        "qid": f"futurex_{raw_id}",
        "question": title,
        "evidence": [],
        "outcome": outcome,
        "resolution_date": resolution_date,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../work/futurex-past/futurex_past_yes_no.jsonl"),
        help="FutureX past Yes/No JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/futurex_past_yes_no.jsonl"),
        help="Output PolymktSim JSONL path",
    )
    args = parser.parse_args()

    converted = []
    skipped = 0
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = convert_row(json.loads(line))
            if item is None:
                skipped += 1
                continue
            converted.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    yes_count = sum(1 for item in converted if item["outcome"] is True)
    no_count = sum(1 for item in converted if item["outcome"] is False)
    unresolved = sum(1 for item in converted if item["outcome"] is None)
    print(f"Wrote {len(converted)} rows to {args.output}")
    print(f"Skipped {skipped} rows")
    print(f"Outcomes: yes={yes_count} no={no_count} unresolved={unresolved}")


if __name__ == "__main__":
    main()
