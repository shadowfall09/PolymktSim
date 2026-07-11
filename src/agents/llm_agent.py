"""LLM agent: real API call, JSON parse, single retry, token logging."""
from __future__ import annotations
import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from src.data.schema import EvidenceItem, Forecast
from .base import BaseAgent
from .prompts import OUTPUT_INSTRUCTIONS, SYSTEM_PROMPT, build_forecast_prompt, format_evidence

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (input, output) — update as needed
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 0.80),
    "gpt-5.4": (2.50, 10.00),
    "gpt-5.4-pro": (5.00, 20.00),
    "gpt-5-mini": (0.40, 1.60),
    "gpt-5": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-3-flash-preview": (0.10, 0.40),
    "gemini-2.5-flash": (0.15, 0.60),
    "deepseek-r1": (0.55, 2.19),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
}


def _calc_cost(model: str, prompt_tokens: int, completion_tokens: int, **_) -> float:
    pricing = _MODEL_PRICING.get(model)
    if pricing is None:
        # Try partial match
        for key, val in _MODEL_PRICING.items():
            if key in model or model in key:
                pricing = val
                break
    if pricing is None:
        return 0.0
    inp_per_m, out_per_m = pricing
    return prompt_tokens * inp_per_m / 1_000_000 + completion_tokens * out_per_m / 1_000_000


def _extract_json(raw: str) -> str | None:
    """Try to extract a single JSON object from model output."""
    raw = raw.strip()
    # Strip markdown code block if present
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


def _parse_forecast(raw: str) -> Forecast | None:
    try:
        json_str = _extract_json(raw)
        if not json_str:
            return None
        d = json.loads(json_str)
        p = float(d.get("p_yes", 0.5))
        p = max(0.0, min(1.0, p))
        label = "YES" if p >= 0.5 else "NO"
        evidence_used = d.get("evidence_used", [])
        if not isinstance(evidence_used, list):
            evidence_used = []
        return Forecast(
            p_yes=p,
            label=label,
            rationale=str(d.get("rationale", "")),
            evidence_used=[str(x) for x in evidence_used],
        )
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Parse forecast failed: %s", e)
        return None


def _fallback_forecast() -> Forecast:
    return Forecast(p_yes=0.5, label="NO", rationale="[parse error]", evidence_used=[])


REPAIR_TEMPLATE = """Your previous response was invalid (must be exactly one JSON object). Raw output:

---
%s
---

Please try again. Output ONLY one JSON object with keys: p_yes (number 0-1), label ("YES" or "NO"), rationale (string), evidence_used (list of doc_id strings). No other text."""


class LLMAgent(BaseAgent):
    """Calls OpenAI-compatible API; enforces JSON output, single retry, logs usage."""

    def __init__(
        self,
        model_name: str = os.environ.get("LLM_MODEL_NAME", "gpt-5.4-mini"),
        temperature: float = 0,
        max_tokens: int = 2048,
        api_key: str | None = None,
        base_url: str = os.environ.get("LLM_BASE_URL", "https://gpa-models.genai.prd.aws.saccap.int/v1"),
        evidence_max_chars_per_item: int = 500,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.evidence_max_chars = evidence_max_chars_per_item
        self.base_url = base_url
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self._client = OpenAI(
            api_key=api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
            base_url=self.base_url,
        )

    def _call_api(self, messages: list[dict], _retries: int = 3) -> tuple[str, dict]:
        """Returns (content, usage_dict). Retries up to _retries times on empty/None response."""
        import time
        last_err = None
        for attempt in range(_retries):
            r = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
            # Extract content — some models return None content (e.g. mid-stream failures)
            content = None
            if r.choices:
                msg = r.choices[0].message
                content = msg.content
                if content is None:
                    # Fallback: reasoning field (some thinking models)
                    content = getattr(msg, "reasoning", None)
            if content:
                content = content.strip()
                break
            last_err = f"attempt {attempt+1}: empty response — {r.model_dump()}"
            logger.warning("qid=? model=%s %s", self.model_name, last_err)
            if attempt < _retries - 1:
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"API returned no content after {_retries} retries. Last: {last_err}")
        usage = {}
        if r.usage:
            pt = getattr(r.usage, "prompt_tokens", 0) or 0
            ct = getattr(r.usage, "completion_tokens", 0) or 0
            cost = _calc_cost(self.model_name, pt, ct)
            self.total_prompt_tokens += pt
            self.total_completion_tokens += ct
            self.total_cost_usd += cost
            usage = {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
                "cost_usd": cost,
            }
        return content, usage

    def predict(
        self,
        qid: str,
        question: str,
        public_evidence: list[EvidenceItem],
        private_evidence: list[EvidenceItem],
        history_summary: str,
    ) -> Forecast:
        evidence = public_evidence + private_evidence
        evidence_text = format_evidence(evidence, max_chars=self.evidence_max_chars)
        prompt = build_forecast_prompt(question, evidence_text, history_summary)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw, usage = self._call_api(messages)
        logger.info(
            "qid=%s model=%s tokens=%s cost=$%.6f cumulative=$%.4f",
            qid,
            self.model_name,
            usage.get("total_tokens", "?"),
            usage.get("cost_usd", 0.0),
            self.total_cost_usd,
        )
        forecast = _parse_forecast(raw)
        if forecast is not None:
            return forecast
        # Retry once with repair prompt
        logger.warning("qid=%s parse failed, retrying once", qid)
        repair = REPAIR_TEMPLATE % raw[:2000]
        messages_retry = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw[:2000]},
            {"role": "user", "content": repair},
        ]
        raw2, usage2 = self._call_api(messages_retry)
        total_tokens = usage.get("total_tokens", 0) + usage2.get("total_tokens", 0)
        logger.info("qid=%s retry tokens=%s", qid, total_tokens)
        forecast = _parse_forecast(raw2)
        if forecast is not None:
            return forecast
        logger.error("qid=%s parse failed after retry, using fallback", qid)
        return _fallback_forecast()
