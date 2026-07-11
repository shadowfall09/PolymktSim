#!/usr/bin/env python3
import argparse
import csv
import importlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha1
from typing import Any, Dict, List, Optional


class TavilySearchError(RuntimeError):
    """A fatal Tavily search failure; callers must stop the collection run."""


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def compact_text(text: str, limit: int = 420) -> str:
    t = normalize_ws(text)
    if len(t) <= limit:
        return t
    return t[: limit - 3].rstrip() + "..."


def slugify(text: str, max_len: int = 80) -> str:
    s = normalize_ws(text).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if not s:
        return "row"
    return s[:max_len]


def normalize_end_date(raw_value: str) -> str:
    # Tavily end_date expects YYYY-MM-DD. Convert ISO datetime when needed.
    v = normalize_ws(raw_value)
    if not v:
        return ""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", v)
    return m.group(1) if m else ""


def evidence_cutoff_date(resolution_date: str, days_before: int = 1) -> str:
    """Return the latest date that may be used as evidence.

    Evidence must predate the market resolution day itself: same-day reporting
    can already disclose the outcome.
    """
    if not resolution_date:
        return ""
    try:
        return (date.fromisoformat(resolution_date) - timedelta(days=days_before)).isoformat()
    except ValueError:
        return ""


_ISO_DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-/](\d{2})[-/](\d{2})(?!\d)")
_MONTH_FIRST_DATE_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\.?\s+(\d{1,2}),?\s+(20\d{2})\b",
    re.IGNORECASE,
)
_DAY_FIRST_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+(20\d{2})\b",
    re.IGNORECASE,
)
_MONTH_NUMBERS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def extract_explicit_dates(text: str) -> List[date]:
    """Extract complete calendar dates without treating standalone years as dates."""
    found = set()
    for year, month, day in _ISO_DATE_RE.findall(text or ""):
        try:
            found.add(date(int(year), int(month), int(day)))
        except ValueError:
            pass
    for month_name, day, year in _MONTH_FIRST_DATE_RE.findall(text or ""):
        try:
            found.add(date(int(year), _MONTH_NUMBERS[month_name[:3].lower()], int(day)))
        except (KeyError, ValueError):
            pass
    for day, month_name, year in _DAY_FIRST_DATE_RE.findall(text or ""):
        try:
            found.add(date(int(year), _MONTH_NUMBERS[month_name[:3].lower()], int(day)))
        except (KeyError, ValueError):
            pass
    return sorted(found)


def quarantine_late_dates_for_rewrite(items: List["SearchItem"], cutoff_date: str) -> int:
    """Mark late-dated evidence so it can only be rewritten or dropped."""
    if not cutoff_date:
        return 0
    try:
        cutoff = date.fromisoformat(cutoff_date)
    except ValueError:
        return 0

    quarantined = 0
    for item in items:
        searchable_text = " ".join((item.title, item.url, item.content))
        late_dates = [d for d in extract_explicit_dates(searchable_text) if d > cutoff]
        if late_dates:
            item.requires_temporal_rewrite = True
            quarantined += 1
            print(
                f"hard_time_quarantine title={compact_text(item.title, 80)!r} "
                f"late_date={late_dates[0].isoformat()} cutoff={cutoff_date}"
            )
    return quarantined


def clamp_query(query: str, max_len: int = 400) -> str:
    q = normalize_ws(query)
    if len(q) <= max_len:
        return q
    return q[: max_len - 3].rstrip() + "..."


def tavily_search(client: Any, search_kwargs: Dict[str, object]) -> Dict[str, object]:
    try:
        return client.search(**search_kwargs)
    except Exception as exc:
        query = compact_text(str(search_kwargs.get("query", "")), 160)
        raise TavilySearchError(f"Tavily search failed for query {query!r}: {exc}") from exc


def _extract_text_response(response) -> str:
    choice = (response.choices or [None])[0]
    if not choice or not getattr(choice, "message", None):
        return ""
    content = getattr(choice.message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _fallback_queries(question: str, description: str, event_title: str) -> List[str]:
    anchor = event_title or question or description
    candidates = [
        f"{anchor} official source rules primary data",
        f"{anchor} historical baseline prior comparable events",
        f"{anchor} named entities organizations people background",
        f"{anchor} schedule deadlines constraints incentives",
        f"{anchor} independent expert analysis risks forecasts",
    ]
    out: List[str] = []
    seen = set()
    for q in candidates:
        qn = normalize_ws(q)
        if not qn or qn in seen:
            continue
        seen.add(qn)
        out.append(qn)
        if len(out) == 5:
            break
    return out


@dataclass
class SearchItem:
    title: str
    url: str
    content: str
    score: Optional[float]
    query: str
    search_index: int
    rewritten_for_leakage: bool = False
    requires_temporal_rewrite: bool = False

    def dedupe_key(self) -> str:
        core = "|".join(
            [
                normalize_ws(self.url).lower(),
                normalize_ws(self.title).lower(),
                compact_text(self.content, 220).lower(),
            ]
        )
        return sha1(core.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "query": self.query,
            "search_index": self.search_index,
        }
        if self.rewritten_for_leakage:
            payload["rewritten_for_leakage"] = True
        return payload


def build_queries(
    row: Dict[str, str],
    llm_client: Optional[Any],
    llm_model: str,
    cutoff_date: str = "",
) -> List[str]:
    question = normalize_ws(row.get("question", ""))
    description = normalize_ws(row.get("description", "") or row.get("event_description", ""))
    event_title = normalize_ws(row.get("event_title", ""))
    if not question:
        return _fallback_queries(question, description, event_title)

    if llm_client is None:
        return _fallback_queries(question, description, event_title)

    prompt = (
        "You generate exactly 5 web search queries to build a broad, pre-resolution evidence base for a forecaster.\n"
        "Return strict JSON only in this format: {\"queries\":[\"...\",\"...\",\"...\",\"...\",\"...\"]}.\n"
        "Rules:\n"
        "- Each query must be concise, topic-specific, and seek evidence useful before resolution.\n"
        "- The 5 queries must be substantially different from each other. Do not make shallow paraphrases,\n"
        "  reorder the same keywords, or repeat the same source type with slightly different wording.\n"
        "- Produce one query for each non-overlapping evidence lens: (1) official rules, definitions,\n"
        "  filings, primary data, or resolution source; (2) historical baseline or comparable prior cases;\n"
        "  (3) relevant named people, organizations, assets, teams, countries, or institutions; (4) schedule,\n"
        "  deadlines, constraints, incentives, market mechanics, or leading indicators; and (5) independent\n"
        "  expert, industry, academic, analyst, or domain analysis.\n"
        "- Make the set broad: at least one query must seek recent, pre-cutoff news or developments from roughly\n"
        "  the preceding month, while the other queries collect durable rules, historical context, primary data,\n"
        "  and distinct decision-relevant background. Recent news is a lens, not a request for the outcome.\n"
        "- Only seek material available on or before the evidence cutoff. When useful, put a concise date or\n"
        "  one-month time window in the query; never search for a post-cutoff update.\n"
        "- Make each query's wording reveal its distinct lens; avoid generic catch-all queries.\n"
        "- Prefer named entities, official organizations, and stable source types over generic keywords.\n"
        "- The purpose is to collect broad supporting evidence, NOT to find the final answer.\n"
        "- Never ask whether the event happened, who won, what the final score was, whether a threshold\n"
        "  was reached, or whether the market resolved Yes/No. Do not search for results, recaps,\n"
        "  retrospectives, postmortems, current status, or post-resolution summaries.\n"
        "- Do not smuggle the answer check into a paraphrase. Use neutral, pre-event framing.\n"
        "- No markdown, no extra fields, no explanation.\n\n"
        f"Question: {question}\n"
        f"Description: {description}\n"
        f"Event title: {event_title}\n"
        f"Evidence cutoff: {cutoff_date or 'unknown'}\n"
    )

    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = _extract_text_response(response)
        parsed = json.loads(raw)
        queries = parsed.get("queries", []) if isinstance(parsed, dict) else []
        cleaned: List[str] = []
        seen = set()
        for q in queries:
            qn = normalize_ws(str(q))
            if not qn or qn in seen:
                continue
            seen.add(qn)
            cleaned.append(qn)
            if len(cleaned) == 5:
                break
        if len(cleaned) == 5:
            return cleaned
    except Exception as exc:
        print(f"build_queries llm failed, fallback to heuristic queries: {exc}")

    return _fallback_queries(question, description, event_title)


def _supplemental_fallback_queries(
    row: Dict[str, str],
    existing_queries: List[str],
    count: int,
) -> List[str]:
    question = normalize_ws(row.get("question", ""))
    description = normalize_ws(row.get("description", "") or row.get("event_description", ""))
    event_title = normalize_ws(row.get("event_title", ""))
    anchor = event_title or question or description
    candidates = [
        f"{anchor} official primary source methodology rules",
        f"{anchor} historical baseline comparable prior events",
        f"{anchor} independent expert analysis leading indicators",
        f"{anchor} relevant entities schedule constraints incentives",
        f"{anchor} domain background statistics context",
    ]
    seen = {normalize_ws(q).lower() for q in existing_queries if normalize_ws(q)}
    out: List[str] = []
    for candidate in candidates:
        query = clamp_query(candidate, 400)
        key = normalize_ws(query).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(query)
        if len(out) == count:
            break
    return out


def build_supplemental_queries(
    row: Dict[str, str],
    existing_evidence: List[SearchItem],
    existing_queries: List[str],
    llm_client: Optional[Any],
    llm_model: str,
    cutoff_date: str = "",
) -> List[str]:
    """Create three additional, non-overlapping pre-resolution evidence queries."""
    question = normalize_ws(row.get("question", ""))
    event_title = normalize_ws(row.get("event_title", ""))
    existing_titles = [compact_text(item.title, 160) for item in existing_evidence[:20] if item.title]
    fallback = _supplemental_fallback_queries(row, existing_queries, count=3)
    if llm_client is None:
        return fallback

    prompt = (
        "Generate exactly 3 supplemental web search queries for a forecaster whose evidence set is too small.\n"
        "Return strict JSON only: {\"queries\":[\"...\",\"...\",\"...\"]}.\n"
        "The 3 queries must be completely different from each other and from the existing queries.\n"
        "Find missing, useful pre-resolution evidence angles not already covered by the retained titles.\n"
        "Prefer primary sources, official data, historical analogues, constraints, leading indicators, or\n"
        "domain analysis. Include a recent-news/leading-indicator angle from roughly the month before the\n"
        "evidence cutoff when that angle is not already covered. This is broad decision-support evidence, not an answer search. Never ask whether\n"
        "the event happened, who won, what the final result was, or whether the market resolved Yes/No.\n"
        f"Question: {question}\n"
        f"Event title: {event_title}\n"
        f"Evidence cutoff: {cutoff_date or 'unknown'}\n"
        f"Existing queries: {json.dumps(existing_queries, ensure_ascii=False)}\n"
        f"Retained titles: {json.dumps(existing_titles, ensure_ascii=False)}"
    )
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        parsed = json.loads(_extract_text_response(response))
        raw_queries = parsed.get("queries", []) if isinstance(parsed, dict) else []
        if isinstance(raw_queries, str):
            raw_queries = [raw_queries]
        seen = {normalize_ws(q).lower() for q in existing_queries if normalize_ws(q)}
        queries: List[str] = []
        for raw_query in raw_queries:
            query = clamp_query(str(raw_query), 400)
            key = normalize_ws(query).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            queries.append(query)
            if len(queries) == 3:
                break
        for query in fallback:
            key = normalize_ws(query).lower()
            if len(queries) == 3:
                break
            if key and key not in seen:
                seen.add(key)
                queries.append(query)
        if queries:
            return queries[:3]
    except Exception as exc:
        print(f"build_supplemental_queries llm failed, using fallback: {exc}")
    return fallback


def _leak_filter_prompt(
    question: str,
    resolution_date: str,
    cutoff_date: str,
    items: List[SearchItem],
) -> str:
    evidence = [
        {
            "index": index,
            "title": item.title,
            "url": item.url,
            "snippet": compact_text(item.content, 900),
            "requires_temporal_rewrite": item.requires_temporal_rewrite,
        }
        for index, item in enumerate(items)
    ]
    return (
        "You are a strict temporal-leakage reviewer for a forecasting dataset.\n"
        "Decide whether each retrieved web snippet leaks information unavailable before the evidence cutoff.\n"
        "Return strict JSON only: {\"decisions\":[{\"index\":0,\"leak\":\"yes\",\"action\":\"drop\",\"reason\":\"...\",\"rewritten_title\":\"\",\"rewritten_content\":\"\"}]}.\n"
        "The cutoff is a non-negotiable historical boundary. Judge the title, URL, and snippet together.\n"
        "Use leak=no for useful background and genuinely prospective pre-cutoff evidence: announcements,\n"
        "schedules, planned airings, filings, odds, forecasts, intentions, previews, and statements that an\n"
        "event WILL happen. A plan published before the cutoff remains safe even when its planned date is later.\n"
        "Use leak=yes whenever title, URL, or snippet explicitly says it was published, released, updated,\n"
        "posted, broadcast, streamed, or reported AFTER the cutoff; gives a concrete post-cutoff page state;\n"
        "or reveals the answer directly or indirectly. Also mark leak=yes for completed/result language such as\n"
        "'won', 'lost', 'ended', 'aired', 'was broadcast', 'was revealed', 'has happened', final score, final\n"
        "result, current status, 'today', 'latest', 'at close', or 'after hours' when it supplies a concrete\n"
        "value, status, or outcome. This remains true even if the item contains useful background.\n"
        "Distinguish prospective from retrospective wording carefully: an earlier preview saying 'will air on\n"
        "January 18' can be safe; a page marked 'Release 01/18/2026' after a 01/17/2026 cutoff, or one stating\n"
        "the event airs/has aired, is leakage. A future date alone is not leakage. Unknown publication time alone\n"
        "is not a reason to drop neutral background, but it never excuses an explicit completed outcome, current\n"
        "state, or post-cutoff fact. When the supplied text is ambiguous about whether it is prospective or\n"
        "retrospective, fail closed and use leak=yes.\n"
        "Items marked requires_temporal_rewrite=true contain an explicit post-cutoff date. They must be\n"
        "leak=yes and can only be action=rewrite or action=drop; action=keep is forbidden.\n"
        "For leak=no, action must be keep and rewritten fields must be empty. For leak=yes, choose rewrite only\n"
        "when you can produce a standalone, useful pre-cutoff background summary after removing every outcome,\n"
        "post-cutoff fact, hindsight claim, and direct answer. Otherwise action must be drop.\n"
        "A rewrite must be a faithful, concise rewrite of the provided snippet only; do not introduce new facts.\n"
        "Judge the snippet itself, including title and URL; do not infer a publication date solely from an\n"
        "unrelated historical date mentioned in a valid snippet.\n"
        f"Question: {question}\n"
        f"Resolution date: {resolution_date or 'unknown'}\n"
        f"Evidence cutoff date: {cutoff_date or 'unknown'}\n"
        f"Items: {json.dumps(evidence, ensure_ascii=False)}"
    )


def _rewrite_verification_prompt(question: str, cutoff_date: str, items: List[SearchItem]) -> str:
    evidence = [
        {
            "index": index,
            "title": item.title,
            "content": compact_text(item.content, 900),
        }
        for index, item in enumerate(items)
    ]
    return (
        "You are the final temporal-leakage gate for a forecasting dataset.\n"
        "These snippets were rewritten after their originals leaked. Return strict JSON only: \n"
        "{\"decisions\":[{\"index\":0,\"leak\":\"no\"}]}.\n"
        "Return leak=no only when the rewritten title and content contain no outcome, no post-cutoff fact,\n"
        "no hindsight, no completed-result language, and no concrete current page state such as 'today',\n"
        "'latest', 'at close', or 'after hours'. A future plan may remain if it is explicitly prospective\n"
        "rather than a later release or completed event. Treat title and content as evidence; when uncertain,\n"
        "return leak=yes.\n"
        f"Question: {question}\nEvidence cutoff date: {cutoff_date or 'unknown'}\n"
        f"Rewritten items: {json.dumps(evidence, ensure_ascii=False)}"
    )


def _verify_rewrites(
    items: List[SearchItem],
    question: str,
    cutoff_date: str,
    llm_client: Any,
    llm_model: str,
    batch_size: int,
) -> tuple[List[SearchItem], int, int]:
    """Return only rewritten evidence that passes an independent leak check."""
    kept: List[SearchItem] = []
    rejected = 0
    failed_batches = 0
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": _rewrite_verification_prompt(question, cutoff_date, batch)}],
                temperature=0,
            )
            parsed = json.loads(_extract_text_response(response))
            decisions = parsed.get("decisions", []) if isinstance(parsed, dict) else []
            by_index = {
                int(decision["index"]): normalize_ws(str(decision.get("leak", "yes"))).lower()
                for decision in decisions
                if isinstance(decision, dict)
                and isinstance(decision.get("index"), int)
                and normalize_ws(str(decision.get("leak", "")).lower()) in {"yes", "no"}
            }
            if set(by_index) != set(range(len(batch))):
                raise ValueError("rewrite verification returned incomplete decisions")
        except Exception as exc:
            failed_batches += 1
            rejected += len(batch)
            print(f"rewrite_verification failed; dropped {len(batch)} rewritten items: {exc}")
            continue

        for index, item in enumerate(batch):
            if by_index[index] == "no":
                kept.append(item)
            else:
                rejected += 1
    return kept, rejected, failed_batches


def llm_post_filter_leaks(
    items: List[SearchItem],
    question: str,
    resolution_date: str,
    cutoff_date: str,
    llm_client: Any,
    llm_model: str,
    batch_size: int = 10,
) -> tuple[List[SearchItem], Dict[str, int]]:
    """Keep only evidence explicitly judged non-leaky by the LLM.

    A failed or malformed batch is dropped rather than passed through. This
    makes failures visible in stats while preserving the no-leak invariant.
    """
    if llm_client is None:
        raise ValueError("LLM post-filter requires --llm-api-key.")

    kept: List[SearchItem] = []
    filtered = 0
    failed_batches = 0
    batches = 0
    direct_kept = 0
    rewrite_candidates: List[SearchItem] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        batches += 1
        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": _leak_filter_prompt(question, resolution_date, cutoff_date, batch),
                    }
                ],
                temperature=0,
            )
            parsed = json.loads(_extract_text_response(response))
            decisions = parsed.get("decisions", []) if isinstance(parsed, dict) else []
            by_index = {
                int(decision["index"]): {
                    "leak": normalize_ws(str(decision.get("leak", "yes"))).lower(),
                    "action": normalize_ws(str(decision.get("action", "drop"))).lower(),
                    "rewritten_title": compact_text(str(decision.get("rewritten_title", "")), 300),
                    "rewritten_content": compact_text(str(decision.get("rewritten_content", "")), 1200),
                }
                for decision in decisions
                if isinstance(decision, dict)
                and isinstance(decision.get("index"), int)
                and normalize_ws(str(decision.get("leak", "")).lower()) in {"yes", "no"}
            }
            if set(by_index) != set(range(len(batch))):
                raise ValueError("LLM post-filter returned incomplete decisions")
        except Exception as exc:
            failed_batches += 1
            filtered += len(batch)
            print(f"llm_post_filter failed; dropped {len(batch)} evidence items: {exc}")
            continue

        for index, item in enumerate(batch):
            decision = by_index[index]
            if decision["leak"] == "no" and decision["action"] == "keep":
                if item.requires_temporal_rewrite:
                    # A hard temporal signal can never pass through unchanged.
                    filtered += 1
                else:
                    kept.append(item)
                    direct_kept += 1
            elif (
                decision["leak"] == "yes"
                and decision["action"] == "rewrite"
                and decision["rewritten_title"]
                and decision["rewritten_content"]
            ):
                rewrite_candidates.append(
                    SearchItem(
                        title=decision["rewritten_title"],
                        # A URL slug can itself reveal the outcome, so a
                        # salvaged snippet must not retain the original URL.
                        url="",
                        content=decision["rewritten_content"],
                        score=item.score,
                        query=item.query,
                        search_index=item.search_index,
                        rewritten_for_leakage=True,
                    )
                )
            else:
                filtered += 1

        batch_kept = sum(
            1
            for index, item in enumerate(batch)
            if not item.requires_temporal_rewrite
            and by_index[index]["leak"] == "no"
            and by_index[index]["action"] == "keep"
        )
        batch_rewrites = sum(
            1 for decision in by_index.values()
            if decision["leak"] == "yes" and decision["action"] == "rewrite"
            and decision["rewritten_title"] and decision["rewritten_content"]
        )
        print(
            f"llm_post_filter batch={batches} kept_leak_no={batch_kept} "
            f"rewrite_candidates={batch_rewrites} removed_leak_yes={len(batch) - batch_kept - batch_rewrites}"
        )

    verified_rewrites, rewrite_rejected, rewrite_failed_batches = _verify_rewrites(
        items=rewrite_candidates,
        question=question,
        cutoff_date=cutoff_date,
        llm_client=llm_client,
        llm_model=llm_model,
        batch_size=batch_size,
    )
    kept.extend(verified_rewrites)
    filtered += rewrite_rejected

    return kept, {
        "llm_filter_batch_count": batches,
        "llm_filter_failed_batch_count": failed_batches,
        "llm_filtered_leak_count": filtered,
        "llm_kept_count": len(kept),
        "llm_kept_leak_no_count": direct_kept,
        "llm_rewrite_candidate_count": len(rewrite_candidates),
        "llm_rewritten_kept_count": len(verified_rewrites),
        "llm_rewrite_rejected_count": rewrite_rejected,
        "llm_rewrite_verification_failed_batch_count": rewrite_failed_batches,
    }


def output_filename(row_index: int, row: Dict[str, str]) -> str:
    market_id = normalize_ws(row.get("id", ""))
    slug = normalize_ws(row.get("slug", ""))
    if market_id:
        suffix = market_id
    elif slug:
        suffix = slugify(slug)
    else:
        suffix = f"row-{row_index:04d}"
    return f"row_{row_index:04d}_{suffix}.json"


def collect_for_row(
    client,
    row_index: int,
    row: Dict[str, str],
    sleep_seconds: float,
    llm_client: Optional[Any],
    llm_model: str,
    evidence_cutoff_days: int = 1,
) -> Dict[str, object]:
    row_end_date_raw = normalize_ws(row.get("endDateIso", "") or row.get("endDate", ""))
    resolution_date = normalize_end_date(row_end_date_raw)
    end_date = evidence_cutoff_date(resolution_date, evidence_cutoff_days)
    queries = build_queries(
        row, llm_client=llm_client, llm_model=llm_model, cutoff_date=end_date
    )
    all_items: List[SearchItem] = []
    raw_responses: List[Dict[str, object]] = []
    query_log: List[str] = []

    for i, query in enumerate(queries, start=1):
        safe_query = clamp_query(query, 400)
        print(f"[{row_index}] query#{i} len={len(safe_query)}: {safe_query}")

        search_kwargs = {
            "query": safe_query,
            "search_depth": "advanced",
            "max_results": 10,
        }
        if end_date:
            search_kwargs["end_date"] = end_date

        response = tavily_search(client, search_kwargs)
        query_log.append(safe_query)
        raw_responses.append(response)

        for result in response.get("results", []) or []:
            all_items.append(
                SearchItem(
                    title=normalize_ws(str(result.get("title", ""))),
                    url=normalize_ws(str(result.get("url", ""))),
                    content=compact_text(str(result.get("content", "")), 1200),
                    score=float(result.get("score")) if result.get("score") is not None else None,
                    query=safe_query,
                    search_index=i,
                )
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    deduped: Dict[str, SearchItem] = {}
    for item in all_items:
        key = item.dedupe_key()
        current = deduped.get(key)
        if current is None:
            deduped[key] = item
            continue
        current_score = current.score if current.score is not None else -1.0
        next_score = item.score if item.score is not None else -1.0
        if next_score > current_score:
            deduped[key] = item

    sorted_items = sorted(
        deduped.values(),
        key=lambda x: (x.score is not None, x.score if x.score is not None else -1.0),
        reverse=True,
    )
    hard_quarantined_count = quarantine_late_dates_for_rewrite(sorted_items, end_date)
    filtered_items, filter_stats = llm_post_filter_leaks(
        items=sorted_items,
        question=normalize_ws(row.get("question", "")),
        resolution_date=resolution_date,
        cutoff_date=end_date,
        llm_client=llm_client,
        llm_model=llm_model,
    )
    supplemental_queries: List[str] = []
    supplemental_raw_count = 0
    supplemental_deduped_count = 0
    supplemental_filter_stats: Dict[str, int] = {}

    # Three extra retrieval passes improve coverage without turning collection
    # into an unbounded search loop.
    if len(filtered_items) < 20:
        supplemental_queries = build_supplemental_queries(
            row=row,
            existing_evidence=filtered_items,
            existing_queries=query_log,
            llm_client=llm_client,
            llm_model=llm_model,
            cutoff_date=end_date,
        )
        for supplemental_query in supplemental_queries:
            supplemental_index = len(query_log) + 1
            print(
                f"[{row_index}] supplemental_query#{len(query_log) + 1} "
                f"len={len(supplemental_query)}: {supplemental_query}"
            )
            search_kwargs = {
                "query": supplemental_query,
                "search_depth": "advanced",
                "max_results": 10,
            }
            if end_date:
                search_kwargs["end_date"] = end_date
            response = tavily_search(client, search_kwargs)
            query_log.append(supplemental_query)
            raw_responses.append(response)
            supplemental_raw_count += len(response.get("results", []) or [])

            supplemental_items: List[SearchItem] = []
            for result in response.get("results", []) or []:
                item = SearchItem(
                    title=normalize_ws(str(result.get("title", ""))),
                    url=normalize_ws(str(result.get("url", ""))),
                    content=compact_text(str(result.get("content", "")), 1200),
                    score=float(result.get("score")) if result.get("score") is not None else None,
                    query=supplemental_query,
                    search_index=supplemental_index,
                )
                key = item.dedupe_key()
                if key in deduped:
                    continue
                deduped[key] = item
                supplemental_items.append(item)

            supplemental_deduped_count += len(supplemental_items)
            supplemental_items.sort(
                key=lambda x: (x.score is not None, x.score if x.score is not None else -1.0),
                reverse=True,
            )
            supplemental_quarantined_count = quarantine_late_dates_for_rewrite(supplemental_items, end_date)
            supplemental_filtered, supplemental_filter_stats = llm_post_filter_leaks(
                items=supplemental_items,
                question=normalize_ws(row.get("question", "")),
                resolution_date=resolution_date,
                cutoff_date=end_date,
                llm_client=llm_client,
                llm_model=llm_model,
            )
            filtered_items.extend(supplemental_filtered)
            filtered_items.sort(
                key=lambda x: (x.score is not None, x.score if x.score is not None else -1.0),
                reverse=True,
            )
            hard_quarantined_count += supplemental_quarantined_count
            for key, value in supplemental_filter_stats.items():
                filter_stats[key] = filter_stats.get(key, 0) + value

    return {
        "row_index": row_index,
        "market_id": row.get("id", ""),
        "slug": row.get("slug", ""),
        "question": row.get("question", ""),
        "event_title": row.get("event_title", ""),
        "resolution_date": resolution_date,
        "evidence_cutoff_date": end_date,
        "end_date_limit": end_date,
        "row_end_date_raw": row_end_date_raw,
        "expanded_queries": queries,
        "supplemental_query": supplemental_queries[0] if supplemental_queries else None,
        "supplemental_queries": supplemental_queries,
        "stats": {
            "search_calls": len(query_log),
            "raw_result_count": len(all_items) + supplemental_raw_count,
            "deduped_result_count": len(deduped),
            "hard_quarantined_late_date_count": hard_quarantined_count,
            "llm_filter_input_count": len(sorted_items) + supplemental_deduped_count,
            "supplemental_search_triggered": bool(supplemental_queries),
            "supplemental_search_count": len(supplemental_queries),
            "supplemental_raw_result_count": supplemental_raw_count,
            "supplemental_deduped_result_count": supplemental_deduped_count,
            "final_evidence_count": len(filtered_items),
            **filter_stats,
        },
        "evidences": [x.to_dict() for x in filtered_items],
        "raw_response_meta": [
            {
                "search_index": idx + 1,
                "query": query_log[idx],
                "response_keys": sorted(list(resp.keys())),
                "results_count": len(resp.get("results", []) or []),
            }
            for idx, resp in enumerate(raw_responses)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read final_markets_500.csv and collect 5x pre-resolution Tavily evidences per row."
    )
    parser.add_argument(
        "--input",
        default="../outputs/final_markets_500.csv",
        help="CSV file to process",
    )
    parser.add_argument(
        "--output-dir",
        default="../evidences",
        help="Directory for per-row evidence JSON files",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("TAVILY_API_KEY", ""),
        help="Tavily API key (or use env TAVILY_API_KEY)",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based start row",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max CSV rows to process (0 means all)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep between search calls to reduce rate-limit risk",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent questions to collect (default: 4; use 1 for serial collection)",
    )
    parser.add_argument(
        "--evidence-cutoff-days",
        type=int,
        default=1,
        help="Search this many calendar days before resolution (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing row evidence files",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.getenv("CMU_API_KEY", ""),
        help="OpenAI-compatible API key used for queries and required leakage filtering",
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://ai-gateway.andrew.cmu.edu"),
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        help="Chat completion model used for query generation",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing Tavily API key. Pass --api-key or set TAVILY_API_KEY.")

    try:
        from tavily import TavilyClient
    except Exception as exc:
        raise RuntimeError("tavily-python is not installed. Install via: pip install tavily-python") from exc

    if args.llm_api_key:
        openai_module = importlib.import_module("openai")
    else:
        raise ValueError("Missing --llm-api-key: LLM post-filter is required to prevent leakage.")

    if args.evidence_cutoff_days < 1:
        raise ValueError("--evidence-cutoff-days must be at least 1.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total_rows = len(rows)
    start_idx = max(args.start_row - 1, 0)
    end_idx = total_rows if args.limit <= 0 else min(total_rows, start_idx + args.limit)

    print(f"total_rows={total_rows} start={start_idx + 1} end={end_idx}")

    work_items: List[tuple[int, Dict[str, str], str]] = []
    skipped = 0
    for i in range(start_idx, end_idx):
        row_num = i + 1
        row = rows[i]
        out_name = output_filename(row_num, row)
        out_path = os.path.join(args.output_dir, out_name)

        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            print(f"[{row_num}] skip existing: {out_name}")
            continue

        work_items.append((row_num, row, out_path))

    def collect_one(row_num: int, row: Dict[str, str]) -> Dict[str, object]:
        # Keep HTTP clients task-local; some provider clients do not guarantee
        # their connection pools are safe to share across worker threads.
        task_tavily_client = TavilyClient(args.api_key)
        task_llm_client = openai_module.OpenAI(
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
        )
        return collect_for_row(
            client=task_tavily_client,
            row_index=row_num,
            row=row,
            sleep_seconds=args.sleep_seconds,
            llm_client=task_llm_client,
            llm_model=args.llm_model,
            evidence_cutoff_days=args.evidence_cutoff_days,
        )

    pending: Dict[object, tuple[int, Dict[str, str], str]] = {}
    fatal_tavily_error: tuple[int, Exception] | None = None
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row_num, row, out_path in work_items:
            future = executor.submit(collect_one, row_num, row)
            pending[future] = (row_num, row, out_path)

        processed = 0
        failed = 0
        for future in as_completed(pending):
            row_num, row, out_path = pending[future]
            try:
                payload = future.result()
                with open(out_path, "w", encoding="utf-8") as wf:
                    json.dump(payload, wf, ensure_ascii=False, indent=2)
                processed += 1
                print(
                    f"[{row_num}] ok: {os.path.basename(out_path)} "
                    f"raw={payload['stats']['raw_result_count']} "
                    f"deduped={payload['stats']['deduped_result_count']} "
                    f"quarantined_hard_time={payload['stats']['hard_quarantined_late_date_count']} "
                    f"kept_leak_no={payload['stats']['llm_kept_leak_no_count']} "
                    f"rewritten_kept={payload['stats']['llm_rewritten_kept_count']} "
                    f"dropped_leak_yes={payload['stats']['llm_filtered_leak_count']}"
                )
            except TavilySearchError as exc:
                fatal_tavily_error = (row_num, exc)
                for pending_future in pending:
                    pending_future.cancel()
                print(f"[{row_num}] fatal Tavily failure; current row discarded: {exc}")
                break
            except Exception as exc:
                failed += 1
                err_name = os.path.join(args.output_dir, f"row_{row_num:04d}_ERROR.json")
                with open(err_name, "w", encoding="utf-8") as ef:
                    json.dump(
                        {
                            "row_index": row_num,
                            "market_id": row.get("id", ""),
                            "slug": row.get("slug", ""),
                            "error": str(exc),
                        },
                        ef,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"[{row_num}] failed: {exc}")

    if fatal_tavily_error is not None:
        row_num, exc = fatal_tavily_error
        raise RuntimeError(f"Aborted after Tavily failure at row {row_num}; no cache was written for that row.") from exc

    print(
        "done "
        f"processed={processed} skipped={skipped} failed={failed} output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
