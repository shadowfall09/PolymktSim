"""Summarize a round of forecasts into history_summary for next round."""
import re
from src.data.schema import Forecast

_RATIONALE_MAX_CHARS = 300


def _sanitize(text: str) -> str:
    """Remove control characters and null bytes that break JSON serialization."""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return text.strip()


def summarize_round(forecasts: list[Forecast], show_rationale: bool = True, **_) -> str:
    """Include each agent's verdict (and optionally rationale) so next round can deliberate."""
    if not forecasts:
        return ""
    lines = ["Other agents' assessments from the previous round:"]
    for i, f in enumerate(forecasts):
        if show_rationale:
            raw = _sanitize(f.rationale or "")
            rationale = raw[:_RATIONALE_MAX_CHARS]
            if len(raw) > _RATIONALE_MAX_CHARS:
                rationale += "..."
            lines.append(f"  Agent {i + 1}: p_yes={f.p_yes:.3f} ({f.label}) — {rationale}")
        else:
            lines.append(f"  Agent {i + 1}: p_yes={f.p_yes:.3f} ({f.label})")
    ps = [f.p_yes for f in forecasts]
    mean = sum(ps) / len(ps)
    lines.append(f"Round mean p_yes={mean:.3f}. Consider whether you agree or disagree with the above.")
    return "\n".join(lines)
