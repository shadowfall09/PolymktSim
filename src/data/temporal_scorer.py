"""Temporal Evidence Reliability Scoring (TERS).

Algorithm:
  For each evidence item, extract dates mentioned in content via regex.
  Compare against the question's resolution_date (market end date).
  If evidence contains dates clearly AFTER resolution_date → score 0.0 (leaky).
  If evidence dates are all before or ambiguous → score 1.0 / 0.5.
  Per-agent reliability = mean score across their evidence items.

Motivation:
  dev_targeted hard questions fail because evidence is temporally misaligned:
  e.g. "BTC above $28k in Aug 2023?" but evidence shows BTC at $67k in 2024.
  Standard aggregation amplifies this overconfidence; TERS discounts it.
"""
import re
from datetime import date
from typing import Optional

from .schema import EvidenceItem

# ── Month name → number ─────────────────────────────────────────────────────
_MONTH = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Regex patterns: each group captures (year, month, day) in some order
_PATTERNS = [
    # ISO: 2025-10-22
    re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
    # US: Oct 22, 2025  |  October 22, 2025
    re.compile(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
               r'Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|'
               r'Dec(?:ember)?)[.\s]+(\d{1,2}),?\s+(\d{4})\b', re.IGNORECASE),
    # EU: 22 October 2025
    re.compile(r'\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
               r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|'
               r'Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\b', re.IGNORECASE),
    # Year-only: standalone 4-digit year (coarser signal)
    re.compile(r'\b(20\d{2})\b'),
]


def _parse_match(m: re.Match, pattern_idx: int) -> Optional[date]:
    """Convert a regex match to a date. Returns None if parsing fails."""
    try:
        g = m.groups()
        if pattern_idx == 0:  # ISO
            return date(int(g[0]), int(g[1]), int(g[2]))
        elif pattern_idx == 1:  # US month-first
            mon = _MONTH.get(g[0].lower()[:3])
            if not mon:
                return None
            return date(int(g[2]), mon, int(g[1]))
        elif pattern_idx == 2:  # EU day-first
            mon = _MONTH.get(g[1].lower()[:3])
            if not mon:
                return None
            return date(int(g[2]), mon, int(g[0]))
        elif pattern_idx == 3:  # year-only → use Jan 1 as proxy
            return date(int(g[0]), 1, 1)
    except (ValueError, TypeError):
        return None
    return None


def extract_dates_from_text(text: str) -> list[date]:
    """Return all dates found in text, deduplicated and sorted."""
    found: set[date] = set()
    for idx, pat in enumerate(_PATTERNS):
        for m in pat.finditer(text):
            d = _parse_match(m, idx)
            if d:
                found.add(d)
    return sorted(found)


def score_evidence_temporal(ev: EvidenceItem, resolution_date: date) -> float:
    """Score one evidence item against the resolution date.

    Returns:
        0.0  — evidence contains dates clearly after resolution_date (leaky)
        0.5  — no datable content found (unknown)
        1.0  — all extracted dates are on or before resolution_date (valid)
    """
    dates = extract_dates_from_text(ev.content)
    if not dates:
        return 0.5  # no dates → uncertain, apply mild discount

    # Year-only matches are coarse; only use them if no precise dates found
    precise = [d for d in dates if d.month != 1 or d.day != 1]
    if precise:
        dates = precise

    if any(d > resolution_date for d in dates):
        return 0.0
    return 1.0


def agent_reliability(
    evidence_list: list[EvidenceItem],
    resolution_date: Optional[date],
) -> float:
    """Mean temporal reliability score across an agent's evidence set.

    Returns 1.0 (no discount) when resolution_date is unknown.
    """
    if not resolution_date or not evidence_list:
        return 1.0
    scores = [score_evidence_temporal(ev, resolution_date) for ev in evidence_list]
    return sum(scores) / len(scores)
