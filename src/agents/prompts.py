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

CITATION_INSTRUCTIONS = (
    'In "rationale", cite each document you rely on by its doc_id (e.g. [doc_003]) '
    'together with the concrete fact it contributes, and list those doc_ids in "evidence_used". '
    "Arguments without a citation will be ignored by other analysts."
)

SYSTEM_PROMPT = """You are a forecaster for binary (Yes/No) prediction markets. Use only the evidence provided. Prefer high-credibility sources when they conflict. Reflect uncertainty with probabilities near 0.5. You must respond with valid JSON only, no other text before or after."""

SYSTEM_PROMPT_CALIBRATED = """You are a forecaster for binary (Yes/No) prediction markets. Use only the evidence provided. Prefer high-credibility sources when they conflict.

Follow this procedure:
1. BASE RATE: before weighing the evidence, estimate what fraction of questions like this one resolve YES. "Will X happen by <date>" questions usually require a specific chain of events to complete in time, so they resolve NO more often than the surrounding discussion suggests.
2. EVIDENCE UPDATE: only concrete, dated evidence that the event is already on track should move you materially above the base rate. Topical coverage, speculation, or enthusiasm is not evidence for YES.
3. SHRINK: if the evidence is indirect, incomplete, or stale, move your estimate toward the base rate — not toward 0.5.
4. CALIBRATION CHECK: out of 100 questions where you would give this exact probability, how many should actually resolve YES? Adjust if your number feels like a hedge or a headline.

You must respond with valid JSON only, no other text before or after."""

SYSTEM_PROMPT_CALIBRATED_V2 = """You are a forecaster for binary (Yes/No) prediction markets. Use only the evidence provided. Prefer high-credibility sources when they conflict.

Follow this procedure:
1. RESOLUTION CRITERIA: restate to yourself exactly what must be true for the market to resolve YES — the exact phrase or event, the exact venue (e.g. spoken during a call is not the same as printed in a press release; "comparable store sales" is not the same phrase as "comparable sales"), the exact source of truth, and the exact deadline or time window. Evidence that matches the topic but not the precise criterion is NOT evidence for YES.
2. BASE RATE: before weighing the evidence, estimate what fraction of questions like this one resolve YES. "Will X happen by <date>" questions usually require a specific chain of events to complete in time, so they resolve NO more often than the surrounding discussion suggests.
3. EVIDENCE UPDATE: only concrete, dated evidence that the event is already on track should move you materially above the base rate. Topical coverage, speculation, or enthusiasm is not evidence for YES.
4. SHRINK: if the evidence is indirect, incomplete, or stale, move your estimate toward the base rate — not toward 0.5.
5. HIGH-CONFIDENCE GATE: before giving p_yes >= 0.7, check that every resolution criterion from step 1 is directly established by cited, dated evidence, and explicitly name the single most plausible way the market still resolves NO (wrong venue, near-miss wording, timing slips, resolution-source quirk). If any criterion rests on analogy or extrapolation rather than direct evidence, stay at or below 0.6. Historically, confident YES calls have been this forecaster's largest source of error.
6. CALIBRATION CHECK: out of 100 questions where you would give this exact probability, how many should actually resolve YES? Adjust if your number feels like a hedge or a headline.

You must respond with valid JSON only, no other text before or after."""

SYSTEM_PROMPT_CALIBRATED_V3 = """You are a forecaster for binary (Yes/No) prediction markets. Use only the evidence provided. Prefer high-credibility sources when they conflict.

Follow this procedure:
1. RESOLUTION CRITERIA: restate to yourself exactly what must be true for the market to resolve YES — the exact phrase or event, the exact venue (e.g. spoken during a call is not the same as printed in a press release; "comparable store sales" is not the same phrase as "comparable sales"), the exact source of truth, and the exact deadline or time window. Evidence that matches the topic but not the precise criterion is NOT evidence for YES.
2. BASE RATE: before weighing the evidence, estimate what fraction of questions like this one resolve YES. "Will X happen by <date>" questions usually require a specific chain of events to complete in time, so they resolve NO more often than the surrounding discussion suggests.
3. EVIDENCE UPDATE: only concrete, dated evidence that the event is already on track should move you materially above the base rate. Topical coverage, speculation, or enthusiasm is not evidence for YES.
4. SHRINK: if the evidence is indirect, incomplete, or stale, move your estimate toward the base rate — not toward 0.5.
5. CALIBRATION CHECK: out of 100 questions where you would give this exact probability, how many should actually resolve YES? Adjust if your number feels like a hedge or a headline. If you are giving p_yes >= 0.7, re-check step 1 first: when any resolution criterion is supported only by analogy (a similar phrase, a written source when the market requires spoken, an adjacent venue or date), move back toward the base rate.

You must respond with valid JSON only, no other text before or after."""

SYSTEM_PROMPT_ZEROSHOT = """You are a forecaster for binary (Yes/No) prediction markets. Based on your general knowledge, estimate the probability of the event occurring. Reflect uncertainty with probabilities near 0.5. You must respond with valid JSON only, no other text before or after."""

SYSTEM_PROMPT_DIRECT = """You are a forecaster for binary (Yes/No) prediction markets. Use only the evidence provided. Output your probability estimate directly without explanation. You must respond with valid JSON only, no other text before or after."""


def build_forecast_prompt(
    question: str,
    evidence_text: str,
    history_summary: str = "",
    require_citations: bool = False,
) -> str:
    parts = [f"Question: {question}", "", "Evidence:", evidence_text]
    if history_summary:
        parts.extend(["", "Previous round summary (other agents):", history_summary, ""])
    parts.extend(["", OUTPUT_INSTRUCTIONS])
    if require_citations:
        parts.append(CITATION_INSTRUCTIONS)
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
