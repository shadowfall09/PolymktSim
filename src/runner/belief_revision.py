"""Belief revision analysis for S2 multi-round forecasting.

Tracks how agent beliefs change between rounds and detects herding — the
phenomenon where agents abandon correct private signals under social pressure
(Lorenz et al. 2011; Bikhchandani et al. 1992).

Herding metric
--------------
For each agent i, define:
    social_pressure_i  = logit(group_mean_round1) - logit(p_round1_i)
        (how far agent i was from the group consensus in round 1)
    delta_i            = logit(p_round2_i) - logit(p_round1_i)
        (how much the agent moved in round 2)
    herding_ratio_i    = delta_i / social_pressure_i
        (fraction of the gap that was closed toward the group mean)

herding_ratio ≈ 1 → agent fully converged to group (herding)
herding_ratio ≈ 0 → agent did not change (held their ground)
herding_ratio < 0 → agent moved away from the group (counter-update)

An agent is flagged as herding when herding_ratio > herding_threshold AND
|delta_i| > min_delta (non-trivial change).

Selective update
----------------
Flagged agents revert to their round-1 forecast.
Non-flagged agents keep their round-2 forecast.
This preserves genuine belief updates while blocking social capitulation.
"""
import math
from dataclasses import dataclass

from src.data.schema import Forecast

_EPS = 1e-6


def _logit(p: float) -> float:
    p = max(_EPS, min(1 - _EPS, p))
    return math.log(p / (1 - p))


@dataclass
class BeliefRevisionStats:
    """Per-question belief revision statistics."""
    deltas: list[float]           # logit(p_round2_i) - logit(p_round1_i)
    herding_ratios: list[float]   # delta_i / social_pressure_i
    herding_flags: list[bool]     # True if agent i herded
    mean_delta: float
    variance_delta: float
    mean_abs_delta: float
    herding_count: int


def compute_belief_revision(
    round1: list[Forecast],
    round2: list[Forecast],
    herding_threshold: float = 0.7,
    min_delta: float = 0.2,
) -> BeliefRevisionStats:
    """Compute per-agent belief revision statistics."""
    logits_r1 = [_logit(f.p_yes) for f in round1]
    logits_r2 = [_logit(f.p_yes) for f in round2]
    group_mean_r1 = sum(logits_r1) / len(logits_r1)

    deltas, herding_ratios, herding_flags = [], [], []
    for l1, l2 in zip(logits_r1, logits_r2):
        d = l2 - l1
        social_pressure = group_mean_r1 - l1
        ratio = d / social_pressure if abs(social_pressure) > _EPS else 0.0
        herded = ratio > herding_threshold and abs(d) > min_delta
        deltas.append(d)
        herding_ratios.append(ratio)
        herding_flags.append(herded)

    n = len(deltas)
    mean_d = sum(deltas) / n
    var_d = sum((d - mean_d) ** 2 for d in deltas) / n
    return BeliefRevisionStats(
        deltas=deltas,
        herding_ratios=herding_ratios,
        herding_flags=herding_flags,
        mean_delta=mean_d,
        variance_delta=var_d,
        mean_abs_delta=sum(abs(d) for d in deltas) / n,
        herding_count=sum(herding_flags),
    )


def selective_update(
    round1: list[Forecast],
    round2: list[Forecast],
    herding_threshold: float = 0.7,
    min_delta: float = 0.2,
) -> tuple[list[Forecast], BeliefRevisionStats]:
    """Return corrected forecasts: herding agents revert to round-1 prediction.

    Returns (corrected_forecasts, stats) so callers can log herding diagnostics.
    """
    stats = compute_belief_revision(round1, round2, herding_threshold, min_delta)
    corrected = []
    for f1, f2, herded in zip(round1, round2, stats.herding_flags):
        corrected.append(f1 if herded else f2)
    return corrected, stats
