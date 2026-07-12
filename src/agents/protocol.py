"""Summarize a round of forecasts into history_summary for next round."""
import re
from src.data.schema import Forecast

_RATIONALE_MAX_CHARS = 300
# Arguments mode carries the private-evidence signal between rounds, so it
# gets a wider budget than the legacy 300-char excerpt.
_ARGUMENT_MAX_CHARS = 600
_MAX_CITED_DOCS = 8


def _sanitize(text: str) -> str:
    """Remove control characters and null bytes that break JSON serialization."""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return text.strip()


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def summarize_round(
    forecasts: list[Forecast],
    show_rationale: bool = True,
    share_mode: str = "full",
    **_,
) -> str:
    """Build the between-round history summary.

    share_mode:
      full      — p_yes + label + rationale + round mean (legacy; the round
                  mean is a strong numeric anchor and drives convergence)
      numbers   — p_yes + label only (legacy --no-rationale-sharing)
      arguments — rationale + cited doc_ids only; no numeric estimates and no
                  round mean, so agents can only converge through evidence
    """
    if not forecasts:
        return ""
    if not show_rationale and share_mode == "full":
        share_mode = "numbers"

    if share_mode == "arguments":
        lines = [
            "Colleagues who saw partially different evidence shared the arguments below.",
            "Weigh each argument by the concrete evidence it cites; ignore unsupported claims.",
        ]
        for i, f in enumerate(forecasts):
            arg = _clip(_sanitize(f.rationale or ""), _ARGUMENT_MAX_CHARS) or "(no rationale provided)"
            cited = ", ".join(f.evidence_used[:_MAX_CITED_DOCS]) if f.evidence_used else "none cited"
            lines.append(f"  Colleague {i + 1} (cites: {cited}): {arg}")
        lines.append(
            "Re-examine your own evidence in light of these arguments, then give your own "
            "independent probability. Do not converge for the sake of agreement."
        )
        return "\n".join(lines)

    lines = ["Other agents' assessments from the previous round:"]
    for i, f in enumerate(forecasts):
        if share_mode == "numbers":
            lines.append(f"  Agent {i + 1}: p_yes={f.p_yes:.3f} ({f.label})")
        else:
            rationale = _clip(_sanitize(f.rationale or ""), _RATIONALE_MAX_CHARS)
            lines.append(f"  Agent {i + 1}: p_yes={f.p_yes:.3f} ({f.label}) — {rationale}")
    ps = [f.p_yes for f in forecasts]
    mean = sum(ps) / len(ps)
    lines.append(f"Round mean p_yes={mean:.3f}. Consider whether you agree or disagree with the above.")
    return "\n".join(lines)
