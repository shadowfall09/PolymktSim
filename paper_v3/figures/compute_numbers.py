#!/usr/bin/env python3
"""Single source of truth for every number in the paper.

Reads data/results_v2/*.jsonl, computes all metrics, CIs, and analysis
quantities, asserts them against the values recorded in PAPER_PLAN.md,
and writes paper_numbers.json for the table/figure scripts.

Run from repo root:  python paper_v3/figures/compute_numbers.py
"""
import json
import math
import random
from pathlib import Path

RV2 = Path("data/results_v2")
OUT = Path("paper_v3/figures/paper_numbers.json")
SEED = 0
B = 5000  # bootstrap resamples

P0, W_LO, W_HI = 0.30, 0.8, 0.5

BASELINES = [
    "cot", "self_consistency", "superforecaster", "halawi", "bayesian_k5",
    "standard_debate", "moa", "crowd_ensemble", "aia",
]

WINNER = {
    "polymarket_250": ("polymarket_250_s1s2_arguments_pooling_calibrated_5.4mini.jsonl", "s2"),
    "futurex": ("futurex_s2_arguments_pooling_calibrated_5.4mini_full.jsonl", None),
}


def shrink(p):
    w = W_HI if p > P0 else W_LO
    return P0 + w * (p - P0)


def load(fname, scenario=None):
    d = {}
    for line in open(RV2 / fname):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if scenario and r.get("scenario") != scenario:
            continue
        d[r["qid"]] = (float(r["p_yes"]), 1.0 if r["outcome"] else 0.0)
    return d


def metrics(d, sh=False):
    n = len(d)
    briers = [((shrink(p) if sh else p) - y) ** 2 for p, y in d.values()]
    acc = sum(((shrink(p) if sh else p) >= 0.5) == (y >= 0.5) for p, y in d.values()) / n
    return {"n": n, "brier": sum(briers) / n, "acc": acc}


def paired_boot(a, b, sh_a=False, sh_b=False):
    rng = random.Random(SEED)
    qids = sorted(set(a) & set(b))
    diffs = [
        ((shrink(a[q][0]) if sh_a else a[q][0]) - a[q][1]) ** 2
        - ((shrink(b[q][0]) if sh_b else b[q][0]) - b[q][1]) ** 2
        for q in qids
    ]
    n = len(diffs)
    mean = sum(diffs) / n
    boots = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(B))
    return {"n": n, "diff": mean, "lo": boots[int(0.025 * B)], "hi": boots[int(0.975 * B)]}


def auc(d, sh=False):
    pairs = [((shrink(p) if sh else p), y) for p, y in d.values()]
    pos = [p for p, y in pairs if y == 1.0]
    neg = [p for p, y in pairs if y == 0.0]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if pp > pn else 0.5 if pp == pn else 0.0) for pp in pos for pn in neg)
    return wins / (len(pos) * len(neg))


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def agent_corr(detail_fname, scenario="s2"):
    """Mean pairwise inter-agent error correlation per round."""
    rows = {}
    for line in open(RV2 / detail_fname):
        r = json.loads(line)
        if scenario and r.get("scenario") != scenario:
            continue
        if r.get("outcome") is None:
            continue
        key = (r["round_id"], r["agent_id"])
        rows.setdefault(key, {})[r["qid"]] = float(r["p_yes"]) - (1.0 if r["outcome"] else 0.0)
    out = {}
    rounds = sorted({k[0] for k in rows})
    for rd in rounds:
        agents = sorted(a for (r_, a) in rows if r_ == rd)
        cors = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                ei, ej = rows[(rd, agents[i])], rows[(rd, agents[j])]
                qids = sorted(set(ei) & set(ej))
                cors.append(pearson([ei[q] for q in qids], [ej[q] for q in qids]))
        out[str(rd)] = sum(cors) / len(cors)
    return out


def agent_std(detail_fname, scenario="s2"):
    """Mean per-question std of agent forecasts, per round."""
    rows = {}
    for line in open(RV2 / detail_fname):
        r = json.loads(line)
        if scenario and r.get("scenario") != scenario:
            continue
        rows.setdefault((r["round_id"], r["qid"]), []).append(float(r["p_yes"]))
    out = {}
    for (rd, _), ps in rows.items():
        m = sum(ps) / len(ps)
        sd = math.sqrt(sum((p - m) ** 2 for p in ps) / len(ps))
        out.setdefault(rd, []).append(sd)
    return {str(rd): sum(v) / len(v) for rd, v in sorted(out.items())}


def main():
    R = {"seed": SEED, "bootstrap": B, "shrink_params": {"p0": P0, "w_lo": W_LO, "w_hi": W_HI}}

    # ---------- main results ----------
    for ds in ["polymarket_250", "futurex"]:
        wf, scen = WINNER[ds]
        W = load(wf, scen)
        block = {"ours_raw": metrics(W), "ours_shrink": metrics(W, sh=True), "baselines": {}, "sig_vs_baselines": {}, "control_both_shrunk": {}}
        for b in BASELINES:
            D = load(f"{ds}_{b}.jsonl")
            block["baselines"][b] = {"raw": metrics(D), "shrink": metrics(D, sh=True)}
            block["sig_vs_baselines"][b] = paired_boot(W, D, sh_a=True, sh_b=False)
            block["control_both_shrunk"][b] = paired_boot(W, D, sh_a=True, sh_b=True)
        R[ds] = block

    # sanity: match PAPER_PLAN verified numbers
    assert abs(R["polymarket_250"]["ours_raw"]["brier"] - 0.1647) < 5e-4
    assert abs(R["polymarket_250"]["ours_shrink"]["brier"] - 0.1575) < 5e-4
    assert abs(R["futurex"]["ours_raw"]["brier"] - 0.2212) < 5e-4
    assert abs(R["futurex"]["ours_shrink"]["brier"] - 0.2006) < 5e-4
    assert abs(R["polymarket_250"]["baselines"]["standard_debate"]["raw"]["brier"] - 0.1815) < 5e-4
    assert all(v["hi"] < 0 for v in R["polymarket_250"]["sig_vs_baselines"].values())
    assert all(v["hi"] < 0 for v in R["futurex"]["sig_vs_baselines"].values())

    # ---------- ablations ----------
    ab = {}
    # deliberation: s1 vs s2 within the same run (paired)
    for ds, fname in [
        ("polymarket_250", "polymarket_250_s1s2_arguments_pooling_calibrated_5.4mini.jsonl"),
    ]:
        s1, s2 = load(fname, "s1"), load(fname, "s2")
        ab[f"{ds}_s1"] = metrics(s1)
        ab[f"{ds}_s2"] = metrics(s2)
        ab[f"{ds}_s2_vs_s1"] = paired_boot(s2, s1)
    # second identical-config run (seed variance)
    ab["polymarket_250_best_rerun_s2"] = metrics(load("polymarket_250_infodelphi_best.jsonl", "s2"))
    ab["polymarket_250_best_rerun_s1"] = metrics(load("polymarket_250_infodelphi_s1_reused.jsonl", "s1"))
    # futurex earlier-config s1/s2
    ab["futurex_infodelphi_s1"] = metrics(load("futurex_infodelphi.jsonl", "s1"))
    ab["futurex_infodelphi_s2"] = metrics(load("futurex_infodelphi.jsonl", "s2"))
    # homogeneous input: standard debate vs ours raw (both no shrink)
    for ds in ["polymarket_250", "futurex"]:
        wf, scen = WINNER[ds]
        ab[f"{ds}_ours_vs_stddebate_raw"] = paired_boot(load(wf, scen), load(f"{ds}_standard_debate.jsonl"))
    # aggregation: shrink vs raw (paired, same forecasts)
    for ds in ["polymarket_250", "futurex"]:
        wf, scen = WINNER[ds]
        W = load(wf, scen)
        ab[f"{ds}_shrink_vs_raw"] = paired_boot(W, W, sh_a=True, sh_b=False)
    R["ablation"] = ab

    # ---------- calibration study (poly s2) ----------
    cal = {}
    variants = {
        "v1": "polymarket_250_s1s2_arguments_pooling_calibrated_5.4mini.jsonl",
        "v2_da": "polymarket_250_s1s2_argpool_calv2_da_shrink_5.4mini.jsonl",
        "v3": "polymarket_250_s1s2_argpool_calv3_shrink_5.4mini.jsonl",
    }
    # NOTE: v2/v3 files already contain shrunk p_yes? verify via aggregator field
    for name, fname in variants.items():
        first = json.loads(open(RV2 / fname).readline())
        d = load(fname, "s2")
        mean_p_raw = sum(p for p, _ in d.values()) / len(d)
        cal[name] = {
            "aggregator_field": first.get("aggregator"),
            "as_stored": metrics(d),
            "mean_p_stored": mean_p_raw,
        }
    # v1 stored raw; v2/v3 stored WITH shrink (check plan: v2 raw 0.1896 shrunk 0.1779; v3 raw 0.1780 shrunk 0.1689)
    d1 = load(variants["v1"], "s2")
    cal["v1"]["raw_brier"] = metrics(d1)["brier"]
    cal["v1"]["shrunk_brier"] = metrics(d1, sh=True)["brier"]
    base_rate = sum(y for _, y in d1.values()) / len(d1)
    cal["base_rate"] = base_rate
    cal["const_baserate_brier"] = sum((base_rate - y) ** 2 for _, y in d1.values()) / len(d1)
    R["calibration_study"] = cal

    # ---------- inter-agent correlation & diversity ----------
    corr = {}
    for ds, ours_detail, sd_detail in [
        ("polymarket_250",
         "polymarket_250_s1s2_arguments_pooling_calibrated_5.4mini_detail.jsonl",
         "polymarket_250_standard_debate_detail.jsonl"),
        ("futurex",
         "futurex_s2_arguments_pooling_calibrated_5.4mini_full_detail.jsonl",
         "futurex_standard_debate_detail.jsonl"),
    ]:
        for label, fname in [("infodelphi", ours_detail), ("standard_debate", sd_detail)]:
            scen = "s2"
            corr[f"{ds}_{label}_corr"] = agent_corr(fname, scen)
            corr[f"{ds}_{label}_std"] = agent_std(fname, scen)
    R["agent_analysis"] = corr

    # ---------- sports ----------
    sports_files = {
        "slugmeta": "polymarket_sports106_slugmeta_cal_shrink_5.4mini.jsonl",
        "slugodds": "polymarket_sports106_slugodds_cal_shrink_5.4mini.jsonl",
    }
    sp = {}
    sq = set(load(sports_files["slugmeta"], "s2"))
    wf, scen = WINNER["polymarket_250"]
    orig_full = load(wf, scen)
    orig = {q: v for q, v in orig_full.items() if q in sq}
    sp["n"] = len(orig)
    sp["original"] = {**metrics(orig, sh=True), "auc": auc(orig, sh=True)}
    for name, fname in sports_files.items():
        d = load(fname, "s2")
        stored_shrunk = json.loads(open(RV2 / fname).readline()).get("aggregator") == "calibrated_shrink"
        m = metrics(d, sh=not stored_shrunk)
        pb = paired_boot(d, orig, sh_a=not stored_shrunk, sh_b=True)
        sp[name] = {**m, "auc": auc(d, sh=not stored_shrunk), "vs_original": pb, "stored_shrunk": stored_shrunk}
    R["sports"] = sp

    OUT.write_text(json.dumps(R, indent=1))
    print(f"Wrote {OUT}")

    # print summary for eyeballing
    for ds in ["polymarket_250", "futurex"]:
        print(f"\n{ds}: ours raw {R[ds]['ours_raw']['brier']:.4f}/{R[ds]['ours_raw']['acc']:.3f}  "
              f"full {R[ds]['ours_shrink']['brier']:.4f}/{R[ds]['ours_shrink']['acc']:.3f}")
    print("\ncalibration:", {k: (v["as_stored"]["brier"] if isinstance(v, dict) and "as_stored" in v else v) for k, v in R["calibration_study"].items() if k != "base_rate"})
    print("\nagent corr:", json.dumps(R["agent_analysis"], indent=1)[:600])
    print("\nsports:", {k: (v if not isinstance(v, dict) else {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items() if kk in ('brier', 'auc')}) for k, v in R["sports"].items()})


if __name__ == "__main__":
    main()
