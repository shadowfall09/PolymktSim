"""ECE and reliability bins stub."""


def reliability_bins(
    p_yes_list: list[float], outcomes: list[bool], n_bins: int = 10
) -> list[tuple[float, float, float]]:
    """Return list of (bin_mean_prob, accuracy, count) per bin."""
    if not p_yes_list or len(p_yes_list) != len(outcomes):
        return []
    paired = sorted(zip(p_yes_list, outcomes), key=lambda x: x[0])
    n = len(paired)
    bin_size = max(1, n // n_bins)
    result = []
    for i in range(0, n, bin_size):
        chunk = paired[i : i + bin_size]
        if not chunk:
            continue
        probs, outs = zip(*chunk)
        result.append((sum(probs) / len(probs), sum(outs) / len(outs), float(len(chunk))))
    return result


def ece(p_yes_list: list[float], outcomes: list[bool], n_bins: int = 10) -> float:
    """Equal-width bin ECE."""
    bins = reliability_bins(p_yes_list, outcomes, n_bins)
    if not bins:
        return 0.0
    n = sum(b[2] for b in bins)
    return sum(b[2] * abs(b[0] - b[1]) for b in bins) / n if n else 0.0
