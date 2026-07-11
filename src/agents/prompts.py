"""Prompt building for agent: evidence list + JSON output contract."""
from src.data.schema import EvidenceItem


def format_evidence(items: list[EvidenceItem], max_chars: int = 500) -> str:
    lines = []
    for i, e in enumerate(items, 1):
        content = (e.content or "")[:max_chars]
        parts = [f"[{i}] doc_id={e.doc_id}", f"source={e.source}"]
        if e.title:
            parts.append(f"title={e.title}")
        if e.timestamp:
            parts.append(f"timestamp={e.timestamp}")
        parts.append(f"content: {content}")
        lines.append(" | ".join(parts))
    return "\n".join(lines) if lines else "(no evidence)"


OUTPUT_INSTRUCTIONS = (
    "Respond with exactly one JSON object, no other text. Schema: "
    '{"p_yes": <0-1>, "label": "YES"|"NO", "rationale": "...", "evidence_used": ["doc_id", ...]}'
)

OUTPUT_INSTRUCTIONS_DIRECT = (
    "Respond with exactly one JSON object, no other text. Schema: "
    '{"p_yes": <0-1>, "label": "YES"|"NO"}'
)

SYSTEM_PROMPT = """You are a forecaster for binary (Yes/No) prediction markets. Use only the evidence provided. Prefer high-credibility sources when they conflict. Reflect uncertainty with probabilities near 0.5. You must respond with valid JSON only, no other text before or after."""

SYSTEM_PROMPT_ZEROSHOT = """You are a forecaster for binary (Yes/No) prediction markets. Based on your general knowledge, estimate the probability of the event occurring. Reflect uncertainty with probabilities near 0.5. You must respond with valid JSON only, no other text before or after."""

SYSTEM_PROMPT_DIRECT = """You are a forecaster for binary (Yes/No) prediction markets. Use only the evidence provided. Output your probability estimate directly without explanation. You must respond with valid JSON only, no other text before or after."""


def build_forecast_prompt(
    question: str,
    evidence_text: str,
    history_summary: str = "",
) -> str:
    parts = [f"Question: {question}", "", "Evidence:", evidence_text]
    if history_summary:
        parts.extend(["", "Previous round summary (other agents):", history_summary, ""])
    parts.extend(["", OUTPUT_INSTRUCTIONS])
    return "\n".join(parts)


def build_zeroshot_prompt(question: str) -> str:
    """Zero-shot: no evidence, just the question."""
    parts = [
        f"Question: {question}",
        "",
        "Based on your knowledge, estimate the probability that the answer is YES.",
        "",
        OUTPUT_INSTRUCTIONS,
    ]
    return "\n".join(parts)


def build_direct_prompt(question: str, evidence_text: str) -> str:
    """Direct: evidence provided but no rationale required."""
    parts = [
        f"Question: {question}",
        "",
        "Evidence:",
        evidence_text,
        "",
        "Output your probability estimate directly.",
        "",
        OUTPUT_INSTRUCTIONS_DIRECT,
    ]
    return "\n".join(parts)
