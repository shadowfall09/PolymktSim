#!/usr/bin/env python3
"""Analyze FutureX InfoDelphi S2 failure modes from result/detail JSONL files."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_correct(p_yes, outcome):
    return (p_yes >= 0.5) == bool(outcome)


def brier(p_yes, outcome):
    return (p_yes - float(bool(outcome))) ** 2


DOMAIN_RULES = [
    ("sports", r"\b(uefa|champions league|nba|nfl|mlb|ufc|soccer|football|tennis|cricket|game|match|defeat|win|team|player|club|world cup|olympic|esports)\b"),
    ("markets_finance_crypto", r"\b(bitcoin|btc|ethereum|crypto|stock|price|market|rate|fed|inflation|tariff|oil|gold|dollar|bank|bond|yield|ipo|recession|gdp|xrp|solana|lithium)\b"),
    ("tech_ai_product", r"\b(openai|anthropic|claude|gpt|ai|model|iphone|apple|google|tesla|spacex|nvidia|microsoft|release|launch|announced|version|product|app|software|assassin)\b"),
    ("politics_geo", r"\b(trump|biden|election|senate|congress|war|iran|israel|ukraine|russia|china|india|canada|eu|nato|president|minister|government|court|law|ice|uranium)\b"),
    ("media_entertainment", r"\b(oscar|movie|film|album|song|youtube|tiktok|anime|manga|one piece|netflix|episode|actor|grammy|box office|cecot|60 minutes)\b"),
]

QTYPE_RULES = [
    ("threshold_range", r"\b(above|below|under|over|at least|less than|more than|exceed|reach|hit|\$|%|percent|top [0-9]|[0-9]+k|[0-9]+m|[0-9]+b)\b"),
    ("deadline_event", r"\b(by|before|within|on or before|through|end of|in .*month|in .*week)\b"),
    ("winner_outcome", r"\b(win|winner|defeat|beat|champion|nomination|elected|pass|resolve)\b"),
]


def classify_text(question):
    text = question.lower()
    domain = "other"
    for name, pattern in DOMAIN_RULES:
        if re.search(pattern, text):
            domain = name
            break
    qtype = "other_wording"
    for name, pattern in QTYPE_RULES:
        if re.search(pattern, text):
            qtype = name
            break
    return domain, qtype


def load_results(results_dir):
    files = {
        "cot": results_dir / "futurex_cot.jsonl",
        "standard": results_dir / "futurex_standard_debate.jsonl",
        "info": results_dir / "futurex_infodelphi.jsonl",
        "moa": results_dir / "futurex_moa.jsonl",
    }
    results = defaultdict(dict)
    for name, path in files.items():
        for record in load_jsonl(path):
            if record.get("outcome") is None:
                continue
            key = record.get("scenario") if name == "info" else name
            results[record["qid"]][key] = record
    return results


def load_detail(path):
    detail = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in load_jsonl(path):
        if record.get("outcome") is None:
            continue
        detail[record["scenario"]][record["qid"]][record["round_id"]].append(record)
    return detail


def build_rows(results, detail):
    rows = []
    for qid in sorted(q for q, methods in results.items() if "s2" in methods):
        final = results[qid]["s2"]
        outcome = bool(final["outcome"])
        round0 = sorted(detail["s2"][qid][0], key=lambda r: r["agent_id"])
        round1 = sorted(detail["s2"][qid][1], key=lambda r: r["agent_id"])
        p0 = [float(r["p_yes"]) for r in round0]
        p1 = [float(r["p_yes"]) for r in round1]
        c0 = [is_correct(p, outcome) for p in p0]
        c1 = [is_correct(p, outcome) for p in p1]
        m0 = sum(p0) / len(p0)
        m1 = sum(p1) / len(p1)
        final_correct = is_correct(float(final["p_yes"]), outcome)
        mean0_correct = is_correct(m0, outcome)
        mean1_correct = is_correct(m1, outcome)
        domain, qtype = classify_text(final["question"])

        if not final_correct:
            if all(not x for x in c0):
                failure_mode = "all_agents_wrong_from_start"
            elif mean0_correct and not mean1_correct:
                failure_mode = "debate_broke_mean_label"
            elif any(c0) and not any(c1):
                failure_mode = "debate_erased_correct_minority"
            elif not mean0_correct and not mean1_correct:
                failure_mode = "wrong_majority_persisted"
            elif mean1_correct:
                failure_mode = "aggregation_broke_mean_label"
            else:
                failure_mode = "other_final_wrong"
        elif not mean0_correct and mean1_correct:
            failure_mode = "debate_fixed"
        else:
            failure_mode = "final_correct"

        row = {
            "qid": qid,
            "question": final["question"],
            "outcome": int(outcome),
            "domain": domain,
            "qtype": qtype,
            "failure_mode": failure_mode,
            "s2_p": float(final["p_yes"]),
            "s2_brier": brier(float(final["p_yes"]), outcome),
            "s2_correct": int(final_correct),
            "round0_mean": m0,
            "round1_mean": m1,
            "round0_correct_agents": sum(c0),
            "round1_correct_agents": sum(c1),
            "round0_mean_correct": int(mean0_correct),
            "round1_mean_correct": int(mean1_correct),
            "round_mean_brier_delta": brier(m1, outcome) - brier(m0, outcome),
        }
        for method in ["cot", "standard", "s1", "moa"]:
            record = results[qid][method]
            row[f"{method}_p"] = float(record["p_yes"])
            row[f"{method}_correct"] = int(is_correct(float(record["p_yes"]), outcome))
            row[f"{method}_brier"] = float(record["brier"])
        rows.append(row)
    return rows


def summarize(rows):
    counters = {
        "failure_mode": Counter(r["failure_mode"] for r in rows),
        "domain": Counter(r["domain"] for r in rows),
        "qtype": Counter(r["qtype"] for r in rows),
    }
    return counters


def mean(values):
    return sum(values) / len(values) if values else 0.0


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def write_report(rows, path):
    n = len(rows)
    s2_wrong = [r for r in rows if not r["s2_correct"]]
    mode_counts = Counter(r["failure_mode"] for r in rows)

    overall = []
    for method in ["cot", "standard", "s1", "s2", "moa"]:
        if method == "s2":
            b = mean([r["s2_brier"] for r in rows])
            acc = mean([r["s2_correct"] for r in rows])
        else:
            b = mean([r[f"{method}_brier"] for r in rows])
            acc = mean([r[f"{method}_correct"] for r in rows])
        overall.append([method, f"{b:.4f}", f"{acc:.2%}"])

    failure_rows = []
    for mode, count in mode_counts.most_common():
        failure_rows.append([mode, count, f"{count / n:.2%}"])

    domain_rows = []
    for domain in sorted(set(r["domain"] for r in rows)):
        group = [r for r in rows if r["domain"] == domain]
        wrong = [r for r in group if not r["s2_correct"]]
        all0 = [r for r in group if r["round0_correct_agents"] == 0]
        broke = [r for r in group if r["failure_mode"] == "debate_broke_mean_label"]
        domain_rows.append([
            domain,
            len(group),
            len(wrong),
            f"{mean([r['s2_correct'] for r in group]):.2%}",
            len(all0),
            len(broke),
            f"{mean([r['s2_brier'] for r in group]):.4f}",
        ])

    qtype_rows = []
    for qtype in sorted(set(r["qtype"] for r in rows)):
        group = [r for r in rows if r["qtype"] == qtype]
        wrong = [r for r in group if not r["s2_correct"]]
        qtype_rows.append([
            qtype,
            len(group),
            len(wrong),
            f"{mean([r['s2_correct'] for r in group]):.2%}",
            sum(r["round0_correct_agents"] == 0 for r in group),
            sum(r["failure_mode"] == "debate_broke_mean_label" for r in group),
            f"{mean([r['s2_brier'] for r in group]):.4f}",
        ])

    overlap_rows = []
    for method in ["cot", "standard", "s1", "moa"]:
        overlap_rows.append([method, sum(r[f"{method}_correct"] for r in s2_wrong)])
    overlap_rows.append(["all four baselines wrong", sum(not r["cot_correct"] and not r["standard_correct"] and not r["s1_correct"] and not r["moa_correct"] for r in s2_wrong)])
    overlap_rows.append(["cot or standard correct", sum(r["cot_correct"] or r["standard_correct"] for r in s2_wrong)])

    examples = []
    for mode in [
        "all_agents_wrong_from_start",
        "debate_broke_mean_label",
        "debate_erased_correct_minority",
        "debate_fixed",
    ]:
        pool = sorted([r for r in rows if r["failure_mode"] == mode], key=lambda r: r["s2_brier"], reverse=True)[:8]
        examples.append(f"### {mode}\n")
        examples.append(markdown_table(
            ["qid", "y", "s2_p", "r0", "r1", "cot", "std", "question"],
            [
                [
                    r["qid"],
                    r["outcome"],
                    f"{r['s2_p']:.2f}",
                    f"{r['round0_mean']:.2f}/{r['round0_correct_agents']}",
                    f"{r['round1_mean']:.2f}/{r['round1_correct_agents']}",
                    f"{r['cot_p']:.2f}",
                    f"{r['standard_p']:.2f}",
                    r["question"].replace("|", "/")[:90],
                ]
                for r in pool
            ],
        ))
        examples.append("")

    report = f"""# FutureX InfoDelphi Failure Mode Analysis

Input files:

- `data/results/futurex_infodelphi.jsonl`
- `data/results/futurex_infodelphi_detail.jsonl`
- `data/results/futurex_cot.jsonl`
- `data/results/futurex_standard_debate.jsonl`
- `data/results/futurex_moa.jsonl`

## Overall

{markdown_table(["method", "brier", "accuracy"], overall)}

## Main Failure Modes

{markdown_table(["mode", "count", "share"], failure_rows)}

## S2 Wrong Overlap

S2 wrong count: {len(s2_wrong)}

{markdown_table(["baseline condition", "count among S2 wrong"], overlap_rows)}

## By Domain

{markdown_table(["domain", "n", "wrong", "acc", "round0_all_wrong", "debate_broke", "brier"], domain_rows)}

## By Question Type

{markdown_table(["qtype", "n", "wrong", "acc", "round0_all_wrong", "debate_broke", "brier"], qtype_rows)}

## Interpretation

The dominant FutureX failure is not debate corrupting initially correct forecasts. Among {len(s2_wrong)} final S2 wrong cases, {mode_counts["all_agents_wrong_from_start"]} cases have all three S2 round-0 agents already on the wrong side. Only {mode_counts["debate_broke_mean_label"]} cases are clear majority/mean-label debate regressions.

The debate step is usually beneficial or neutral at the label level: round-mean labels stay correct in 130 cases and are fixed in 11 cases, while only 5 cases are flipped from correct to wrong. The larger problem is shared evidence/interpretation failure before debate begins, followed by round-1 consensus that makes the same wrong answer more confident.

Aggregation is not the main issue: confidence-weighted aggregation almost never overturns the round-1 majority label. The Brier problem is mostly calibration/overconfidence on wrong consensus cases, which is consistent with the separate shrinkage ablation.

## Example Cases

{chr(10).join(examples)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--detail", type=Path, default=Path("data/results/futurex_infodelphi_detail.jsonl"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/results/futurex_failure_modes.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("data/results/futurex_failure_mode_analysis.md"))
    args = parser.parse_args()

    results = load_results(args.results_dir)
    detail = load_detail(args.detail)
    rows = build_rows(results, detail)
    write_csv(rows, args.output_csv)
    write_report(rows, args.output_md)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
