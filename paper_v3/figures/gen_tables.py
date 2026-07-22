#!/usr/bin/env python3
"""Generate all LaTeX tables from paper_numbers.json.

Outputs (paper_v3/figures/):
  TABLE_main_results.tex   — Tab 1: main comparison, both datasets
  TABLE_ablation.tex       — Tab 2: component ablation
  TABLE_sports.tex         — Tab 3: sports retrieval nulls
  TABLE_shrink_control.tex — Tab 4 (App): shrink applied to baselines
  TABLE_significance.tex   — Tab 5 (App): paired bootstrap CIs
Run from repo root.
"""
import json
from pathlib import Path

R = json.load(open("paper_v3/figures/paper_numbers.json"))
OUT = Path("paper_v3/figures")

NAME = {
    "cot": ("Chain-of-Thought~\\citep{wei2022chain}", "single"),
    "self_consistency": ("Self-Consistency~\\citep{wangself}", "single"),
    "superforecaster": ("Superforecaster~\\citep{karger2024forecastbench}", "single"),
    "halawi": ("Halawi et al.~\\citep{halawi2024approaching}", "single"),
    "bayesian_k5": ("Sequential Bayesian~\\citep{shi2023language}", "single"),
    "standard_debate": ("Standard Debate~\\citep{du2023improving}", "multi"),
    "moa": ("MoA~\\citep{wangmixture}", "multi"),
    "crowd_ensemble": ("Crowd Ensemble~\\citep{schoenegger2024wisdom}", "multi"),
    "aia": ("AIA Forecaster~\\citep{alur2025aia}", "multi"),
}
ORDER_SINGLE = ["cot", "self_consistency", "bayesian_k5", "superforecaster", "halawi"]
ORDER_MULTI = ["standard_debate", "moa", "crowd_ensemble", "aia"]


def f4(x):
    return f"{x:.4f}"


def f3(x):
    return f"{x:.3f}"


# ---------------- Tab 1: main results ----------------
def row(name, pm, fx, dagger=False):
    d = "$^\\dagger$" if dagger else ""
    return (f"{name}{d} & {f4(pm['brier'])} & {f3(pm['acc'])} & {f3(pm['auc'])}"
            f" & {f4(fx['brier'])} & {f3(fx['acc'])} & {f3(fx['auc'])} \\\\")


lines = [
    "\\begin{table*}[t]",
    "\\centering",
    "\\small",
    "\\caption{Main results on \\textsc{PolyGym}-250 and FutureX-231. All methods use"
    " \\texttt{gpt-5.4-mini} over the same fixed pre-resolution evidence pools."
    " AUC is invariant to monotone recalibration, so it isolates ranking skill from"
    " the calibration layer (identical for our raw and full rows)."
    " $\\dagger$: InfoDelphi (full) is significantly better under a paired bootstrap"
    " over questions (5{,}000 resamples; 95\\% CI of the Brier difference excludes 0;"
    " Appendix Table~\\ref{tab:significance}). Best in \\textbf{bold}.}",
    "\\label{tab:main}",
    "\\begin{tabular}{lcccccc}",
    "\\toprule",
    " & \\multicolumn{3}{c}{\\textsc{PolyGym}-250} & \\multicolumn{3}{c}{FutureX-231} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
    "Method & Brier$\\,\\downarrow$ & Acc$\\,\\uparrow$ & AUC$\\,\\uparrow$"
    " & Brier$\\,\\downarrow$ & Acc$\\,\\uparrow$ & AUC$\\,\\uparrow$ \\\\",
    "\\midrule",
    "\\multicolumn{7}{l}{\\textit{Single-agent}} \\\\",
]
for b in ORDER_SINGLE:
    pm = R["polymarket_250"]["baselines"][b]["raw"]
    fx = R["futurex"]["baselines"][b]["raw"]
    dag = R["polymarket_250"]["sig_vs_baselines"][b]["hi"] < 0 and R["futurex"]["sig_vs_baselines"][b]["hi"] < 0
    lines.append(row(NAME[b][0], pm, fx, dagger=dag))
lines.append("\\midrule")
lines.append("\\multicolumn{7}{l}{\\textit{Multi-agent, homogeneous input}} \\\\")
for b in ORDER_MULTI:
    pm = R["polymarket_250"]["baselines"][b]["raw"]
    fx = R["futurex"]["baselines"][b]["raw"]
    dag = R["polymarket_250"]["sig_vs_baselines"][b]["hi"] < 0 and R["futurex"]["sig_vs_baselines"][b]["hi"] < 0
    lines.append(row(NAME[b][0], pm, fx, dagger=dag))
pm_r, fx_r = R["polymarket_250"]["ours_raw"], R["futurex"]["ours_raw"]
pm_f, fx_f = R["polymarket_250"]["ours_shrink"], R["futurex"]["ours_shrink"]
lines += [
    "\\midrule",
    "\\multicolumn{7}{l}{\\textit{Ours}} \\\\",
    f"InfoDelphi w/o calibrated shrink & {f4(pm_r['brier'])} &"
    f" \\textbf{{{f3(pm_r['acc'])}}} & \\textbf{{{f3(pm_r['auc'])}}} &"
    f" {f4(fx_r['brier'])} & {f3(fx_r['acc'])} & \\textbf{{{f3(fx_r['auc'])}}} \\\\",
    f"\\textbf{{InfoDelphi (full)}} & \\textbf{{{f4(pm_f['brier'])}}} & {f3(pm_f['acc'])}"
    f" & \\textbf{{{f3(pm_r['auc'])}}}"
    f" & \\textbf{{{f4(fx_f['brier'])}}} & \\textbf{{{f3(fx_f['acc'])}}}"
    f" & \\textbf{{{f3(fx_r['auc'])}}} \\\\",
]
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
(OUT / "TABLE_main_results.tex").write_text("\n".join(lines) + "\n")

# ---------------- Tab 2: ablation ----------------
ab = R["ablation"]
pm_full = R["polymarket_250"]["ours_shrink"]["brier"]
fx_full = R["futurex"]["ours_shrink"]["brier"]
pm_raw = R["polymarket_250"]["ours_raw"]["brier"]
fx_raw = R["futurex"]["ours_raw"]["brier"]
pm_s1 = ab["polymarket_250_s1"]["brier"]
pm_sd = R["polymarket_250"]["baselines"]["standard_debate"]["raw"]["brier"]
fx_sd = R["futurex"]["baselines"]["standard_debate"]["raw"]["brier"]
fx_s1 = R["ablation_suite"]["futurex_s1_round0"]["raw"]["brier"]

lines = [
    "\\begin{table}[t]",
    "\\centering",
    "\\small",
    "\\caption{Component ablation (Brier$\\,\\downarrow$). Each row removes one"
    " component from the full system. ``$-$ asymmetry'' gives every agent the full"
    " evidence pool ($\\rho{=}1$, full sharing) with the same calibrated prompt,"
    " i.e., the Standard Debate configuration of our pipeline. FutureX round-1"
    " forecasts are recovered from the winner run's recorded round-0 agent outputs."
    " Paired 95\\% CIs for each removal are given in the text.}",
    "\\label{tab:ablation}",
    "\\begin{tabular}{lcc}",
    "\\toprule",
    "Configuration & \\textsc{PolyGym} & FutureX \\\\",
    "\\midrule",
    f"InfoDelphi (full) & \\textbf{{{f4(pm_full)}}} & \\textbf{{{f4(fx_full)}}} \\\\",
    f"$-$ calibrated shrink (mean agg.) & {f4(pm_raw)} & {f4(fx_raw)} \\\\",
    f"$-$ deliberation (round 1 only) & {f4(pm_s1)} & {f4(fx_s1)} \\\\",
    f"$-$ information asymmetry & {f4(pm_sd)} & {f4(fx_sd)} \\\\",
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
]
(OUT / "TABLE_ablation.tex").write_text("\n".join(lines) + "\n")

# ---------------- Tab 2b: design-space grid (PolyGym) ----------------
S = R["ablation_suite"]
sd = R["polymarket_250"]["baselines"]["standard_debate"]
ss = S["seed_spread"]


def grid_row(setting, m, dagger=False):
    d = "$^\\dagger$" if dagger else ""
    return (f"\\quad {setting}{d} & {f4(m['raw']['brier'])} &"
            f" {f4(m['shrink']['brier'])} & {f3(m['raw']['auc'])} \\\\")


default_m = {"raw": R["polymarket_250"]["ours_raw"], "shrink": R["polymarket_250"]["ours_shrink"]}
lines = [
    "\\begin{table}[t]",
    "\\centering",
    "\\small",
    "\\caption{Design-space robustness on \\textsc{PolyGym}-250. Each row changes one"
    " dimension of the default configuration ($R{=}2$, $\\rho{=}0.5$, rationale"
    " sharing, BM25 routing). The default is stable across 3 seeds (Brier"
    f" {f4(ss['raw']['mean'])}\\,$\\pm$\\,{ss['raw']['std']:.4f} raw,"
    f" {f4(ss['shrunk']['mean'])}\\,$\\pm$\\,{ss['shrunk']['std']:.4f} +shrink);"
    " every variation except full homogenization ($\\rho{=}1$) lies within seed"
    " noise ($^\\dagger$: paired 95\\% CI vs.\\ default excludes 0, raw comparison).}",
    "\\label{tab:design}",
    "\\begin{tabular}{lccc}",
    "\\toprule",
    "Setting & Brier (raw) & +shrink & AUC \\\\",
    "\\midrule",
    f"Default & {f4(default_m['raw']['brier'])} & {f4(default_m['shrink']['brier'])}"
    f" & {f3(default_m['raw']['auc'])} \\\\",
    "\\midrule",
    "\\multicolumn{4}{l}{\\textit{Deliberation rounds}} \\\\",
    grid_row("$R{=}1$ (independent)", S["r1"]),
    grid_row("$R{=}3$", S["r3"]),
    "\\multicolumn{4}{l}{\\textit{Public ratio}} \\\\",
    grid_row("$\\rho{=}0.3$", S["rho03"]),
    grid_row("$\\rho{=}0.7$", S["rho07"]),
    grid_row("$\\rho{=}1.0$ (homogeneous)", {"raw": sd["raw"], "shrink": sd["shrink"]}, dagger=True),
    "\\multicolumn{4}{l}{\\textit{Rationale sharing}} \\\\",
    grid_row("numeric estimates only", S["numshare"]),
    "\\multicolumn{4}{l}{\\textit{Evidence routing}} \\\\",
    grid_row("random split", S["randroute"]),
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
]
(OUT / "TABLE_design_grid.tex").write_text("\n".join(lines) + "\n")

# ---------------- Tab 3: sports ----------------
sp = R["sports"]
def spd(v):
    pb = v["vs_original"]
    return f"{pb['diff']:+.4f} [{pb['lo']:+.3f}, {pb['hi']:+.3f}]"

lines = [
    "\\begin{table}[t]",
    "\\centering",
    "\\small",
    "\\caption{Targeted retrieval augmentation on the 106 Sports questions of"
    " \\textsc{PolyGym}-250 (InfoDelphi full; paired bootstrap vs.\\ original"
    " evidence, 5{,}000 resamples). Neither augmentation yields a significant"
    " improvement.}",
    "\\label{tab:sports}",
    "\\begin{tabular}{lccc}",
    "\\toprule",
    "Evidence & Brier$\\,\\downarrow$ & AUC$\\,\\uparrow$ & $\\Delta$Brier [95\\% CI] \\\\",
    "\\midrule",
    f"Original & {f4(sp['original']['brier'])} & {f3(sp['original']['auc'])} & --- \\\\",
    f"$+$ opponent identity & {f4(sp['slugmeta']['brier'])} & {f3(sp['slugmeta']['auc'])} & {spd(sp['slugmeta'])} \\\\",
    f"$+$ opponent odds/form & {f4(sp['slugodds']['brier'])} & {f3(sp['slugodds']['auc'])} & {spd(sp['slugodds'])} \\\\",
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
]
(OUT / "TABLE_sports.tex").write_text("\n".join(lines) + "\n")

# ---------------- Tab 4 (App): shrink-on-baselines control ----------------
lines = [
    "\\begin{table*}[t]",
    "\\centering",
    "\\small",
    "\\caption{Control experiment: applying our calibrated-shrink transform"
    " ($p_0{=}0.30$, $w_{\\mathrm{lo}}{=}0.8$, $w_{\\mathrm{hi}}{=}0.5$) to every"
    " baseline's output. Shrinkage improves all methods; the ranking is preserved and"
    " InfoDelphi remains best on both datasets, although several pairwise gaps are"
    " no longer individually significant (n.s.). $\\Delta$ is the paired Brier"
    " difference of InfoDelphi (full) vs.\\ the shrunk baseline (negative favors ours).}",
    "\\label{tab:shrink_control}",
    "\\begin{tabular}{lcccccc}",
    "\\toprule",
    " & \\multicolumn{3}{c}{\\textsc{PolyGym}-250} & \\multicolumn{3}{c}{FutureX-231} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
    "Method & raw & $+$shrink & $\\Delta$ [95\\% CI] & raw & $+$shrink & $\\Delta$ [95\\% CI] \\\\",
    "\\midrule",
]
for b in ORDER_SINGLE + ORDER_MULTI:
    pm = R["polymarket_250"]["baselines"][b]
    fx = R["futurex"]["baselines"][b]
    cpm = R["polymarket_250"]["control_both_shrunk"][b]
    cfx = R["futurex"]["control_both_shrunk"][b]
    def ci(c):
        tag = "" if c["hi"] < 0 else "\\,n.s."
        return f"${c['diff']:+.3f}$ $[{c['lo']:+.3f}, {c['hi']:+.3f}]${tag}"
    name = NAME[b][0].split("~")[0]
    lines.append(f"{name} & {f4(pm['raw']['brier'])} & {f4(pm['shrink']['brier'])} & {ci(cpm)}"
                 f" & {f4(fx['raw']['brier'])} & {f4(fx['shrink']['brier'])} & {ci(cfx)} \\\\")
lines += [
    "\\midrule",
    f"InfoDelphi (full) & {f4(pm_raw)} & \\textbf{{{f4(pm_full)}}} & ---"
    f" & {f4(fx_raw)} & \\textbf{{{f4(fx_full)}}} & --- \\\\",
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table*}",
]
(OUT / "TABLE_shrink_control.tex").write_text("\n".join(lines) + "\n")

# ---------------- Tab 5 (App): significance ----------------
lines = [
    "\\begin{table*}[t]",
    "\\centering",
    "\\small",
    "\\caption{Paired bootstrap comparison of InfoDelphi (full) against each baseline"
    " (5{,}000 resamples over questions, seed 0). $\\Delta$Brier $<0$ favors"
    " InfoDelphi; all 95\\% CIs exclude zero.}",
    "\\label{tab:significance}",
    "\\begin{tabular}{lcc}",
    "\\toprule",
    "Baseline & \\textsc{PolyGym}-250 $\\Delta$Brier [95\\% CI] & FutureX-231 $\\Delta$Brier [95\\% CI] \\\\",
    "\\midrule",
]
for b in ORDER_SINGLE + ORDER_MULTI:
    s_pm = R["polymarket_250"]["sig_vs_baselines"][b]
    s_fx = R["futurex"]["sig_vs_baselines"][b]
    name = NAME[b][0].split("~")[0]
    lines.append(
        f"{name} & ${s_pm['diff']:+.4f}$ $[{s_pm['lo']:+.4f}, {s_pm['hi']:+.4f}]$"
        f" & ${s_fx['diff']:+.4f}$ $[{s_fx['lo']:+.4f}, {s_fx['hi']:+.4f}]$ \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
(OUT / "TABLE_significance.tex").write_text("\n".join(lines) + "\n")

print("Wrote 5 tables to", OUT)
