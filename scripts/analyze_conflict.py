"""Cost-carbon conflict: real day versus counter-factual carbon-inverted day.

Both conditions were run with the same methods and budgets (results/extra/days_1.pkl = real day,
shift start 06:00; results/extra/conflict.pkl = carbon series mirrored about its mid-range):
three DP-PreCo-GRL and three LS-GRL training seeds (84-preference sweep), PWG, rule-seeded
NSGA-II (3 seeds).  Per condition the reference front is the pooled non-dominated set of these
runs on each instance (so HV ratios are comparable within, not across, conditions).

Outputs: results/summary/conflict.json, outputs/generated/tab_conflict.tex and macros conf* in
outputs/generated/numbers_rep.tex."""
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.analysis import a_posteriori_selection, feasible, indicators, reference, responsiveness  # noqa: E402
from morl_fjsp.instance import BENCHMARKS  # noqa: E402
from morl_fjsp.latex import merge_macros  # noqa: E402
from morl_fjsp.metrics import eval_preferences  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA = os.path.join(ROOT, "results", "extra")
W = eval_preferences()
PRECO = ["preco_s0", "preco_s1", "preco_s2"]
LS = ["linear_s0", "linear_s1", "linear_s2"]
days = pickle.load(open(os.path.join(EXTRA, "days_1.pkl"), "rb"))
conf = pickle.load(open(os.path.join(EXTRA, "conflict.pkl"), "rb"))
CONDS = {"real": (lambda run, i: days[("real 06:00", run, i)]),
         "inverted": (lambda run, i: conf[(run, i)])}
NSGA = {"real": (lambda i: days[("real 06:00", "NSGA-II", i)]), "inverted": (lambda i: conf[("NSGA-II", i)])}
METHODS = [("DP-PreCo-GRL", PRECO), ("LS-GRL", LS), ("PWG", ["PWG"]), ("NSGA-II", None)]


def analyse(cond):
    get, get_n = CONDS[cond], NSGA[cond]
    hv = {m: [] for m, _ in METHODS}
    rho = {m: [] for m, _ in METHODS}
    corr23, corr14 = [], []
    for inst in BENCHMARKS:
        runs = {}
        for m, names in METHODS:
            if names is not None:
                runs[m] = [feasible(get(n, inst)["F"], get(n, inst)["viol"]) for n in names]
            else:
                runs[m] = [r["F"][r["G"] <= 1e-9] for r in get_n(inst)]
        R, ideal, nadir = reference(runs)
        ind, _ = indicators(runs, R, ideal, nadir)
        for m, names in METHODS:
            hv[m].append(float(np.mean(ind[m]["hv"])))
            if names is not None:
                rho[m].append(np.nanmean([responsiveness(get(n, inst)["F"], W, get(n, inst)["viol"]) for n in names], axis=0))
            else:
                rho[m].append(np.nanmean([responsiveness(a_posteriori_selection(F, W, R, ideal, nadir), W)
                                          for F in runs[m]], axis=0))
        if len(R) > 3:
            corr23.append(spearmanr(R[:, 1], R[:, 2]).statistic)
            corr14.append(spearmanr(R[:, 0], R[:, 3]).statistic)
    return dict(hv={m: float(np.mean(v)) for m, v in hv.items()},
                rho={m: np.nanmean(np.array(v), axis=0).tolist() for m, v in rho.items()},
                corr_cost_co2=float(np.mean(corr23)), corr_mk_risk=float(np.mean(corr14)))


res = {c: analyse(c) for c in CONDS}
res["corr_price_co2"] = dict(real=conf["corr_price_co2_real"], inverted=conf["corr_price_co2"])
json.dump(res, open(os.path.join(ROOT, "results", "summary", "conflict.json"), "w"), indent=1)

macros = {"confCorrPriceCoReal": f"{conf['corr_price_co2_real']:+.2f}", "confCorrPriceCo": f"{conf['corr_price_co2']:+.2f}",
          "confCorrFrontReal": f"{res['real']['corr_cost_co2']:+.2f}", "confCorrFront": f"{res['inverted']['corr_cost_co2']:+.2f}"}
tags = {"DP-PreCo-GRL": "Preco", "LS-GRL": "Ls", "PWG": "Pwg", "NSGA-II": "Nsgaii"}
for c, ct in (("real", "Real"), ("inverted", "")):
    for m, t in tags.items():
        macros[f"confHv{t}{ct}"] = f"{res[c]['hv'][m]:.3f}"
        for k, kn in ((0, "Mk"), (1, "Cost"), (2, "Co"), (3, "Risk")):
            macros[f"confRho{kn}{t}{ct}"] = f"{res[c]['rho'][m][k]:+.2f}"
merge_macros(os.path.join(ROOT, "outputs", "generated", "numbers_rep.tex"), macros)

lines = ["\\begin{table}[htbp]", "\\centering",
         "\\caption{Real day versus carbon-inverted day (price-carbon correlation \\confCorrPriceCoReal{} and \\confCorrPriceCo{}). "
         "HV ratio against the pooled front of each condition and responsiveness $\\rho_k$ of the weights, mean over instances "
         "and seeds. Rank correlation of $f_2$ and $f_3$ on the pooled front: \\confCorrFrontReal{} (real) and \\confCorrFront{} (inverted).}",
         "\\label{tab:conflict}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
         "\\begin{tabular}{llccccc}", "\\toprule",
         "Day & Method & HV ratio & $\\rho_1$ (mk.) & $\\rho_2$ (cost) & $\\rho_3$ (CO$_2$) & $\\rho_4$ (risk) \\\\", "\\midrule"]
for c, cl in (("real", "Real"), ("inverted", "Inverted")):
    for m in tags:
        r = res[c]["rho"][m]
        lines.append(f"{(cl if m == 'DP-PreCo-GRL' else '')} & {m} & {res[c]['hv'][m]:.3f} & " +
                     " & ".join(f"{x:+.2f}" for x in r) + " \\\\")
    if c == "real":
        lines.append("\\midrule")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(ROOT, "outputs", "generated", "tab_conflict.tex"), "w").write("\n".join(lines) + "\n")
print(json.dumps(res, indent=1))
