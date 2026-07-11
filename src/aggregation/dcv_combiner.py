"""DCV combiner: confidence-weighted log-pooling over sub-claim verifications."""
from __future__ import annotations

import math

from src.agents.dcv_agent import SubClaimResult
from src.data.schema import Forecast


def combine_dcv(results: list[SubClaimResult]) -> Forecast:
    """Combine sub-claim verifications into a single p_yes via weighted logit pooling.

    For each sub-claim:
      - if it supports YES: p_yes_contribution = p_true
      - if it supports NO:  p_yes_contribution = 1 - p_true
      - per-claim weight = sub_claim.weight * (confidence + 1)

    Final p_yes = sigmoid( sum(w_i * logit(p_yes_contribution_i)) / sum(w_i) ).
    """
    if not results:
        return Forecast.from_p_yes(0.5, rationale="no sub-claims")
    eps = 1e-3
    total_w = 0.0
    weighted_logit = 0.0
    parts = []
    for r in results:
        contrib = r.p_true if r.supports == "YES" else (1.0 - r.p_true)
        contrib = max(eps, min(1.0 - eps, contrib))
        w = max(0.0, r.weight) * (r.confidence + 1)
        if w == 0:
            continue
        logit = math.log(contrib / (1.0 - contrib))
        weighted_logit += w * logit
        total_w += w
        parts.append(
            f"[{r.supports} w={r.weight:.2f} conf={r.confidence} p_true={r.p_true:.2f}]"
        )
    if total_w == 0:
        return Forecast.from_p_yes(0.5, rationale="all weights zero")
    avg_logit = weighted_logit / total_w
    p_yes = 1.0 / (1.0 + math.exp(-avg_logit))
    p_yes = max(0.0, min(1.0, p_yes))
    rationale = "DCV log-pool: " + " | ".join(parts)
    evidence_used: list[str] = []
    seen = set()
    for r in results:
        for d in r.evidence_used:
            if d not in seen:
                seen.add(d)
                evidence_used.append(d)
    return Forecast(
        p_yes=p_yes,
        label="YES" if p_yes >= 0.5 else "NO",
        rationale=rationale,
        evidence_used=evidence_used,
    )
