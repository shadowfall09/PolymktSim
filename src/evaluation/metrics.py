"""Brier, log loss, accuracy."""
import math


def brier_score(p_yes: float, outcome: bool) -> float:
    """Brier score; outcome True=Yes."""
    y = 1.0 if outcome else 0.0
    return (p_yes - y) ** 2


def log_loss(p_yes: float, outcome: bool, eps: float = 1e-15) -> float:
    p = max(eps, min(1 - eps, p_yes))
    y = 1.0 if outcome else 0.0
    return -((y * math.log(p)) + (1 - y) * math.log(1 - p))


def accuracy(p_yes: float, outcome: bool) -> float:
    pred = p_yes >= 0.5
    return 1.0 if pred == outcome else 0.0
