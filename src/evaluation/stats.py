"""Paired t-test and bootstrap CI stubs."""


def paired_ttest(scores_a: list[float], scores_b: list[float]) -> float:
    """Stub: return 0.0 (no scipy dependency). Implement with scipy.stats.ttest_rel."""
    return 0.0


def bootstrap_ci(scores: list[float], n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """Stub: return (mean, mean). Implement proper bootstrap."""
    if not scores:
        return (0.0, 0.0)
    mean = sum(scores) / len(scores)
    return (mean, mean)
