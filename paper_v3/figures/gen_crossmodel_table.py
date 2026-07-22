"""Generate TABLE_crossmodel.tex from paper_numbers.json + mechanism_numbers.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "paper_numbers.json")))
M = json.load(open(os.path.join(HERE, "mechanism_numbers.json")))

ours_poly = P["polymarket_250"]["ours_shrink"]
ours_fx = P["futurex"]["ours_shrink"]
deb_poly = P["polymarket_250"]["baselines"]["standard_debate"]["raw"]
deb_fx = P["futurex"]["baselines"]["standard_debate"]["raw"]
gm = M["crossmodel"]["gemini-3.1-flash-lite"]

assert abs(gm["polymarket"]["brier"] - 0.1624) < 5e-4
assert abs(gm["futurex"]["brier"] - 0.2040) < 5e-4

tex = r"""\begin{table}[t]
\centering
\small
\caption{Cross-model generalization: the full InfoDelphi configuration
(same routing, deliberation, and shrinkage parameters, no re-tuning) on a
different vendor's backbone, vs.\ the strongest baseline. The Gemini
Polymarket run covers 248/250 questions.}
\label{tab:crossmodel}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{\textsc{PolyGym}} & \multicolumn{2}{c}{FutureX} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
 & Brier$\,\downarrow$ & Acc$\,\uparrow$ & Brier$\,\downarrow$ & Acc$\,\uparrow$ \\
\midrule
\textit{Strongest baseline (Std.\ Debate)} & %.4f & %.3f & %.4f & %.3f \\
\midrule
InfoDelphi, \texttt{gpt-5.4-mini} & \textbf{%.4f} & %.3f & \textbf{%.4f} & \textbf{%.3f} \\
InfoDelphi, \texttt{gemini-3.1-flash-lite} & %.4f & \textbf{%.3f} & %.4f & %.3f \\
\bottomrule
\end{tabular}
\end{table}
""" % (
    deb_poly["brier"], deb_poly["acc"], deb_fx["brier"], deb_fx["acc"],
    ours_poly["brier"], ours_poly["acc"], ours_fx["brier"], ours_fx["acc"],
    gm["polymarket"]["brier"], gm["polymarket"]["acc"],
    gm["futurex"]["brier"], gm["futurex"]["acc"],
)

with open(os.path.join(HERE, "TABLE_crossmodel.tex"), "w") as f:
    f.write(tex)
print(tex)
