"""Decomposer + Verifier (DCV) agent.

Two roles, sharing the same OpenAI-compatible API client:

  decompose(question, k) -> list[SubClaim]
      Split the question into K atomic sub-claims, each with a YES/NO polarity
      and a weight indicating how decisive it is.

  verify(claim, evidence) -> SubClaimResult
      Verify a single sub-claim against the FULL evidence corpus, returning
      P(claim is true | evidence) plus a discrete confidence and a rationale.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

from src.data.schema import EvidenceItem
from src.agents.prompts import format_evidence

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class SubClaim:
    claim: str
    supports: str  # "YES" or "NO"
    weight: float  # 0..1, expected to roughly sum to 1 across claims


@dataclass
class SubClaimResult:
    claim: str
    supports: str
    weight: float
    p_true: float
    confidence: int  # 0/1/2
    rationale: str = ""
    evidence_used: list[str] = field(default_factory=list)


DECOMPOSER_SYSTEM = (
    "You are an expert prediction market analyst. Decompose a binary forecasting "
    "question into atomic, verifiable sub-claims. Output strict JSON only."
)

VERIFIER_SYSTEM = (
    "You are a careful fact verifier. Given a single factual claim and an evidence "
    "corpus, estimate the probability the claim is TRUE. Use only the provided "
    "evidence. Output strict JSON only."
)


def _decomposer_prompt_v1(question: str, k: int) -> str:
    return f"""Decompose the following forecasting question into exactly {k} atomic sub-claims whose truth values jointly determine the answer.

Each sub-claim must:
- Be a CONCRETE FACTUAL STATEMENT verifiable from evidence (NOT a probability or a hypothetical).
- Reference specific numbers, dates, named entities, or measurable conditions when possible.
- Be as INDEPENDENT from the other sub-claims as you can make it.

For each sub-claim, also provide:
- "supports": "YES" if the sub-claim being TRUE pushes the answer toward YES; "NO" if TRUE pushes toward NO.
- "weight": how decisive the sub-claim is (a number in [0,1]); the {k} weights should roughly sum to 1.

Question: {question}

Respond with EXACTLY one JSON object, no other text:
{{"sub_claims": [{{"claim": "...", "supports": "YES" or "NO", "weight": <0-1>}}, ... ({k} entries)]}}"""


def _verifier_prompt_v1(claim: str, evidence_text: str) -> str:
    return f"""Verify the following factual claim against the evidence below.

Claim: {claim}

Evidence:
{evidence_text}

Estimate:
- p_true: probability the claim is true given the evidence (0 to 1).
- confidence: 0 = evidence is sparse, ambiguous, or contradictory; 1 = moderate; 2 = clear and direct evidence.
- Use 0.5 if evidence is insufficient — do NOT invent facts.

Respond with EXACTLY one JSON object, no other text:
{{"p_true": <0-1>, "confidence": <0|1|2>, "rationale": "...", "evidence_used": ["doc_id", ...]}}"""


def _decomposer_prompt_v2(question: str, k: int) -> str:
    return f"""Decompose the following forecasting question into exactly {k} atomic sub-claims whose truth values jointly determine the answer.

Each sub-claim MUST satisfy ALL of:
1. CONCRETE & FACTUAL: a verifiable statement about an event, number, date, or entity. NOT a probability, prediction, hypothetical, or interpretation rule.
2. ORTHOGONAL to the others: covers a DIFFERENT aspect of the question. Do NOT include logical complements (e.g. "X above Y" and "X at or below Y" — those are the same fact restated).
3. NON-META: do NOT include claims about how the market resolves, what counts as "dip"/"hit"/"reach", or what reference price source is used. Stick to direct facts that the evidence can confirm or refute.

For each sub-claim, also provide:
- "supports": "YES" if the sub-claim being TRUE pushes the answer toward YES; "NO" if TRUE pushes toward NO.
- "weight": how decisive the sub-claim is (a number in [0,1]); the {k} weights should roughly sum to 1.

Good example for "Will BTC be above $96,000 on Nov 30?":
  [supports YES] BTC's spot close on Nov 30 (per major USD exchanges) is above $96,000.
  [supports YES] BTC's market trend during the week leading up to Nov 30 is upward (latest weekly change is positive).
  [supports YES] No major bearish catalyst (e.g. exchange collapse, regulatory ban) was reported between Nov 25–30.

Bad examples (do NOT do this):
  - "BTC trades at or below $96K on Nov 30" (logical complement of another sub-claim).
  - "The market resolves based on the closing price rather than intraday high" (meta/interpretive).
  - "There is a 60% probability that BTC reaches $96K" (a probability, not a fact).

Question: {question}

Respond with EXACTLY one JSON object, no other text:
{{"sub_claims": [{{"claim": "...", "supports": "YES" or "NO", "weight": <0-1>}}, ... ({k} entries)]}}"""


def _verifier_prompt_v2(claim: str, evidence_text: str) -> str:
    return f"""Verify the following factual claim against the evidence below.

Claim: {claim}

Evidence:
{evidence_text}

Procedure (you MUST follow):
1. In "rationale", FIRST quote the SINGLE most relevant fact verbatim from the evidence (e.g. an exact price, date, score, or named entity). If no relevant fact exists, say so explicitly.
2. THEN compare that quoted fact to the claim's threshold/condition.
3. THEN set p_true so that it is CONSISTENT with that comparison:
   - If the quoted fact directly satisfies the claim → p_true ≥ 0.85.
   - If the quoted fact directly refutes the claim → p_true ≤ 0.15.
   - If the evidence is silent or only weakly indicative → p_true near 0.5.
4. Set confidence: 0 = no clear fact found / contradictory; 1 = some support; 2 = direct, unambiguous fact quoted.

Sanity check: BEFORE outputting, reread your rationale. The numerical relation you stated MUST match the direction of p_true (e.g. if rationale says "BTC was at $71K, well below $118K", then p_true for "BTC above $118K" MUST be near 0, not near 1). Self-correct if inconsistent.

Respond with EXACTLY one JSON object, no other text:
{{"p_true": <0-1>, "confidence": <0|1|2>, "rationale": "...", "evidence_used": ["doc_id", ...]}}"""


_DECOMP_PROMPTS = {"v1": _decomposer_prompt_v1, "v2": _decomposer_prompt_v2}
_VERIFY_PROMPTS = {"v1": _verifier_prompt_v1, "v2": _verifier_prompt_v2}


def _extract_json(raw: str) -> str | None:
    raw = raw.strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[-1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[-1].rsplit("```", 1)[0].strip()
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None


class DCVAgent:
    def __init__(
        self,
        model_name: str = os.environ.get("LLM_MODEL_NAME", "openai/gpt-5.4-mini"),
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key: str | None = None,
        base_url: str = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        evidence_max_chars_per_item: int = 500,
        prompt_version: str = "v2",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.evidence_max_chars = evidence_max_chars_per_item
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        if prompt_version not in _DECOMP_PROMPTS:
            raise ValueError(f"prompt_version must be 'v1' or 'v2', got {prompt_version}")
        self.prompt_version = prompt_version
        self._client = OpenAI(
            api_key=api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url,
        )

    def _call(self, system: str, user: str, retries: int = 3) -> str:
        last_err = None
        for attempt in range(retries):
            r = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
            content = None
            if r.choices:
                msg = r.choices[0].message
                content = msg.content
                if content is None:
                    content = getattr(msg, "reasoning", None)
            if r.usage:
                self.total_prompt_tokens += getattr(r.usage, "prompt_tokens", 0) or 0
                self.total_completion_tokens += getattr(r.usage, "completion_tokens", 0) or 0
            if content:
                return content.strip()
            last_err = f"attempt {attempt+1}: empty response"
            logger.warning("DCV %s", last_err)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        raise RuntimeError(f"DCV API returned no content after {retries} retries. Last: {last_err}")

    def decompose(self, question: str, k: int = 3) -> list[SubClaim]:
        raw = self._call(DECOMPOSER_SYSTEM, _DECOMP_PROMPTS[self.prompt_version](question, k))
        js = _extract_json(raw)
        if not js:
            logger.warning("decompose: no JSON found, raw=%s", raw[:200])
            return [SubClaim(claim=question, supports="YES", weight=1.0)]
        try:
            d = json.loads(js)
        except json.JSONDecodeError as e:
            logger.warning("decompose: JSON parse failed: %s", e)
            return [SubClaim(claim=question, supports="YES", weight=1.0)]
        out = []
        for c in d.get("sub_claims", [])[:k]:
            try:
                claim = str(c.get("claim", "")).strip()
                if not claim:
                    continue
                sup = str(c.get("supports", "YES")).upper()
                if sup not in ("YES", "NO"):
                    sup = "YES"
                w = float(c.get("weight", 1.0 / k))
                w = max(0.0, min(1.0, w))
                out.append(SubClaim(claim=claim, supports=sup, weight=w))
            except (TypeError, ValueError):
                continue
        if not out:
            return [SubClaim(claim=question, supports="YES", weight=1.0)]
        # Renormalise weights so they sum to 1
        total = sum(c.weight for c in out)
        if total > 0:
            for c in out:
                c.weight = c.weight / total
        return out

    def verify(self, claim: str, evidence: list[EvidenceItem]) -> SubClaimResult:
        evidence_text = format_evidence(evidence, max_chars=self.evidence_max_chars)
        raw = self._call(VERIFIER_SYSTEM, _VERIFY_PROMPTS[self.prompt_version](claim, evidence_text))
        js = _extract_json(raw)
        if not js:
            logger.warning("verify: no JSON found, raw=%s", raw[:200])
            return SubClaimResult(claim=claim, supports="YES", weight=0.0,
                                  p_true=0.5, confidence=0, rationale="[parse error]")
        try:
            d = json.loads(js)
        except json.JSONDecodeError as e:
            logger.warning("verify: JSON parse failed: %s", e)
            return SubClaimResult(claim=claim, supports="YES", weight=0.0,
                                  p_true=0.5, confidence=0, rationale="[parse error]")
        try:
            p = float(d.get("p_true", 0.5))
            p = max(0.0, min(1.0, p))
        except (TypeError, ValueError):
            p = 0.5
        try:
            conf = int(d.get("confidence", 1))
            conf = max(0, min(2, conf))
        except (TypeError, ValueError):
            conf = 1
        ev_used = d.get("evidence_used", [])
        if not isinstance(ev_used, list):
            ev_used = []
        return SubClaimResult(
            claim=claim, supports="YES", weight=0.0,
            p_true=p, confidence=conf,
            rationale=str(d.get("rationale", "")),
            evidence_used=[str(x) for x in ev_used],
        )
