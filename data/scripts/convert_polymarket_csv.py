#!/usr/bin/env python3
"""Convert a Polymarket CSV into PolymktSim processed JSONL.

The output schema is accepted by collect_processed_jsonl_evidences_tavily.py:
{"qid": "...", "question": "...", "evidence": [], "outcome": true, "resolution_date": "..."}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _parse_bool_from_probability(value: str) -> bool | None:
    value = _clean(value)
    if not value:
        return None
    try:
        prob = float(value)
    except ValueError:
        return None
    if prob >= 0.999:
        return True
    if prob <= 0.001:
        return False
    return None


def _resolution_date(row: dict[str, str]) -> str:
    """Use the earliest known end/close date as the forecasting boundary.

    A market can close before its configured end date. Choosing the first CSV
    field (the old behavior) can therefore put a post-settlement date into a
    processed dataset and allow evidence leakage during collection.
    """
    candidates = []
    for key in ("endDateIso", "endDate", "closedTime", "umaEndDateIso", "umaEndDate"):
        value = _clean(row.get(key))
        date_value = value[:10]
        if len(date_value) == 10 and date_value[4] == "-" and date_value[7] == "-":
            candidates.append(date_value)
    return min(candidates) if candidates else ""


def _question_text(row: dict[str, str], include_description: bool) -> str:
    question = _clean(row.get("question"))
    if not include_description:
        return question

    parts = [question]
    description = _clean(row.get("description"))
    if description:
        parts.append(f"Description: {description}")
    resolution_source = _clean(row.get("resolutionSource"))
    if resolution_source:
        parts.append(f"Resolution source: {resolution_source}")
    return "\n\n".join(part for part in parts if part)


def convert(csv_path: Path, include_description: bool, require_binary_outcome: bool) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market_id = _clean(row.get("id"))
            question = _question_text(row, include_description)
            if not market_id or not question:
                skipped += 1
                continue

            outcome = _parse_bool_from_probability(row.get("p_yes", ""))
            if outcome is None and require_binary_outcome:
                skipped += 1
                continue

            rows.append(
                {
                    "qid": f"mkt_{market_id}",
                    "question": question,
                    "evidence": [],
                    "outcome": outcome,
                    "resolution_date": _resolution_date(row),
                    "market_id": market_id,
                    "market_end_date": _clean(row.get("endDateIso") or row.get("endDate")) or None,
                    "market_closed_time": _clean(row.get("closedTime")) or None,
                    "resolution_date_policy": "min(endDate/endDateIso, closedTime, umaEndDate)",
                    "url": f"https://polymarket.com/market/{_clean(row.get('slug'))}" if _clean(row.get("slug")) else None,
                    "topic": _clean(row.get("topic")) or None,
                }
            )
    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/outputs/final_markets_500.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/polymarket_500.jsonl"))
    parser.add_argument(
        "--title-only",
        action="store_true",
        help="Use only the Polymarket question/title instead of appending description and resolution source.",
    )
    parser.add_argument(
        "--keep-nonbinary-outcome",
        action="store_true",
        help="Keep rows whose p_yes is not effectively 0 or 1, with outcome=null.",
    )
    args = parser.parse_args()

    rows, skipped = convert(
        csv_path=args.input,
        include_description=not args.title_only,
        require_binary_outcome=not args.keep_nonbinary_outcome,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    yes_count = sum(1 for row in rows if row["outcome"] is True)
    no_count = sum(1 for row in rows if row["outcome"] is False)
    null_count = sum(1 for row in rows if row["outcome"] is None)
    print(
        f"wrote={len(rows)} skipped={skipped} yes={yes_count} no={no_count} "
        f"null={null_count} output={args.output}"
    )


if __name__ == "__main__":
    main()
