"""Mechanism-level stats for the v2-style analysis rewrite.

Computes, from existing results_v2 detail files (no new LLM calls):
  1. reliability   -- per-band hit rates of the raw (pre-shrink) ensemble forecast,
                      to substantiate the affirmative-side overconfidence that
                      motivates asymmetric shrinkage.
  2. extremity     -- accuracy / Brier by |p-0.5| quartile of the raw forecast
                      (v2-style: extreme forecasts carry real signal).
  3. diversity     -- final-round per-agent extremity and inter-agent variance,
                      ours vs Standard Debate ("confident about different things").
  4. crossmodel    -- InfoDelphi full-config results on other backbones
                      (gemini-3.1-flash-lite both datasets, gpt-4o-mini FutureX).

Writes mechanism_numbers.json next to paper_numbers.json.
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RV2 = os.path.join(ROOT, "data", "results_v2")
RV1 = os.path.join(ROOT, "data", "results")


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def brier(rows):
    return sum((r["p_yes"] - r["outcome"]) ** 2 for r in rows) / len(rows)


def acc(rows):
    return sum(1 for r in rows if (r["p_yes"] >= 0.5) == (r["outcome"] == 1)) / len(rows)


# ---------------------------------------------------------------- raw winner runs
WINNER_RAW = {
    "polymarket_250": os.path.join(RV2, "polymarket_250_s1s2_arguments_pooling_calibrated_5.4mini.jsonl"),
    "futurex_231": os.path.join(RV2, "futurex_s2_arguments_pooling_calibrated_5.4mini_full.jsonl"),
}
WINNER_DETAIL = {
    "polymarket_250": os.path.join(RV2, "polymarket_250_s1s2_arguments_pooling_calibrated_5.4mini_detail.jsonl"),
    "futurex_231": os.path.join(RV2, "futurex_s2_arguments_pooling_calibrated_5.4mini_full_detail.jsonl"),
}
DEBATE_DETAIL = {
    "polymarket_250": os.path.join(RV2, "polymarket_250_standard_debate_detail.jsonl"),
    "futurex_231": os.path.join(RV2, "futurex_standard_debate_detail.jsonl"),
}

out = {}

# 1) reliability bands + 2) extremity quartiles, on raw s2 ensemble forecasts
for ds, path in WINNER_RAW.items():
    rows = [r for r in load(path) if r.get("scenario") == "s2"
            and r.get("p_yes") is not None and r.get("outcome") is not None]
    assert len(rows) in (231, 250), (ds, len(rows))

    bands = {"p<0.30": [], "0.30<=p<0.60": [], "p>=0.60": []}
    for r in rows:
        p = r["p_yes"]
        key = "p<0.30" if p < 0.30 else ("0.30<=p<0.60" if p < 0.60 else "p>=0.60")
        bands[key].append(r)
    out.setdefault("reliability", {})[ds] = {
        k: {
            "n": len(v),
            "mean_p": sum(r["p_yes"] for r in v) / len(v),
            "yes_rate": sum(r["outcome"] for r in v) / len(v),
        }
        for k, v in bands.items() if v
    }

    ranked = sorted(rows, key=lambda r: abs(r["p_yes"] - 0.5))
    qs = [ranked[i * len(ranked) // 4:(i + 1) * len(ranked) // 4] for i in range(4)]
    out.setdefault("extremity", {})[ds] = [
        {"quartile": i + 1, "n": len(q), "acc": acc(q), "brier": brier(q),
         "mean_extremity": sum(abs(r["p_yes"] - 0.5) for r in q) / len(q)}
        for i, q in enumerate(qs)
    ]

# 3) final-round diversity: ours vs standard debate
def final_round_stats(path):
    rows = load(path)
    last = max(r["round_id"] for r in rows)
    per_q = defaultdict(dict)
    for r in rows:
        if r["round_id"] == last and r.get("p_yes") is not None:
            per_q[r["qid"]][r["agent_id"]] = r["p_yes"]
    ext, var = [], []
    for ps in per_q.values():
        vals = list(ps.values())
        if len(vals) < 2:
            continue
        ext.extend(abs(p - 0.5) for p in vals)
        m = sum(vals) / len(vals)
        var.append(sum((p - m) ** 2 for p in vals) / len(vals))
    return {
        "mean_extremity": sum(ext) / len(ext),
        "mean_interagent_var": sum(var) / len(var),
        "n_questions": len(var),
    }

for ds in WINNER_DETAIL:
    ours = final_round_stats(WINNER_DETAIL[ds])
    deb = final_round_stats(DEBATE_DETAIL[ds])
    out.setdefault("diversity", {})[ds] = {
        "ours": ours, "debate": deb,
        "var_ratio": ours["mean_interagent_var"] / deb["mean_interagent_var"],
    }

# 3b) baseline reliability: high-confidence band (p >= 0.6) stated vs hit rate
BASELINES = ["cot", "superforecaster", "standard_debate", "self_consistency",
             "halawi", "moa", "crowd_ensemble", "aia", "bayesian_k5"]
for ds, pat in [("polymarket_250", os.path.join(RV2, "polymarket_250_%s.jsonl")),
                ("futurex_231", os.path.join(RV2, "futurex_%s.jsonl"))]:
    for b in BASELINES:
        rows = [r for r in load(pat % b)
                if r.get("p_yes") is not None and r.get("outcome") is not None]
        hi = [r for r in rows if r["p_yes"] >= 0.6]
        out.setdefault("baseline_hiband", {}).setdefault(ds, {})[b] = {
            "n": len(hi),
            "stated": sum(r["p_yes"] for r in hi) / len(hi),
            "hit": sum(r["outcome"] for r in hi) / len(hi),
        }

# 4) cross-model: full winner config on other backbones
gm_poly = [r for r in load(os.path.join(RV1, "polymarket_250_best_20260719_083632-gemini.jsonl"))
           if r["scenario"] == "s2"]
gm_fx = [r for r in load(os.path.join(RV1, "futurex_best_20260719_083718.jsonl"))
         if r["scenario"] == "s2"]
g4_fx = [r for r in load(os.path.join(RV2, "futurex_s2_arguments_pooling_calibrated_full.jsonl"))
         if r["scenario"] == "s2"]
out["crossmodel"] = {
    "gemini-3.1-flash-lite": {
        "polymarket": {"n": len(gm_poly), "brier": brier(gm_poly), "acc": acc(gm_poly)},
        "futurex": {"n": len(gm_fx), "brier": brier(gm_fx), "acc": acc(gm_fx)},
        "note": "full config incl. calibrated_shrink (aggregator field); poly run covers 248/250 questions",
    },
    "gpt-4o-mini": {
        "futurex": {"n": len(g4_fx), "brier": brier(g4_fx), "acc": acc(g4_fx)},
        "note": "mean aggregation, no shrink (matches the raw / w-o-shrink row)",
    },
}

with open(os.path.join(HERE, "mechanism_numbers.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
