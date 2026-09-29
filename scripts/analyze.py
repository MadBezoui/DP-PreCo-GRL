"""Compute all indicators, statistics, LaTeX tables and number macros from raw results.

One global reference front per instance pools every feasible solution of every
method and run (main methods, hybrids and ablations), so all HV ratios in all tables
are directly comparable.

Outputs
  results/summary/indicators.json, stats.json
  outputs/generated/tab_main.tex, tab_pref.tex, tab_ablation.tex, tab_runtime.tex
  outputs/generated/numbers.tex
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.analysis import (a_posteriori_metrics, a_posteriori_selection,  # noqa: E402
                                feasible, indicators, load_baselines, load_hybrid,
                                load_policy_evals, method_runs, preference_metrics, reference,
                                responsiveness)
from morl_fjsp.instance import BENCHMARKS  # noqa: E402
from morl_fjsp.metrics import a12, friedman, holm, mannwhitney, wilcoxon  # noqa: E402

from morl_fjsp.metrics import hypervolume as _hvr, nondominated as _ndr, normalise as _nmr  # noqa: E402


def _rule_hv(F, lo, hi):
    """HV in the space normalised by the best / worst dispatching rule (the validation metric)."""
    return _hvr(_nmr(_ndr(F), lo, hi)) if len(F) else 0.0


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "outputs", "generated")
SUM = os.path.join(ROOT, "results", "summary")
os.makedirs(GEN, exist_ok=True)
os.makedirs(SUM, exist_ok=True)

GROUPS = json.load(open(os.path.join(ROOT, "configs", "method_groups.json")))
OURS = GROUPS["ours"]
MAIN = GROUPS["main"]
ABL = GROUPS.get("ablation", {})
POLICY_GROUPS = dict(MAIN, **ABL)

base = load_baselines(ROOT)
long_base = load_baselines(ROOT, "long")     # 2,000-generation NSGA-II runs (reference + time-matched)
unseeded = load_baselines(ROOT, "unseeded")  # NSGA-II with a random initial population
RAND = "NSGA-II (random init.)"
pol = load_policy_evals(ROOT)
pol_s = load_policy_evals(ROOT, "sampled")   # greedy + 8 sampled rollouts per preference
hyb = load_hybrid(ROOT, MAIN)
insts = [n for n in BENCHMARKS if n in base]
NUM = {}


def pfmt(p):
    """p-value for use inside math mode: 3 decimals from 0.01 on, two significant digits below,
    and m\\times10^{e} below 0.001."""
    if p != p:
        return "--"
    if p >= 0.01:
        return f"{p:.3f}"
    if p >= 0.001:
        return f"{p:.2g}"
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def put(k, v, fmt="{:.3f}"):
    if isinstance(v, str):
        NUM[k] = v
    elif fmt == "p":
        NUM[k] = pfmt(v)
    else:
        NUM[k] = fmt.format(v)


def _cnt(x):
    """Integer with LaTeX thousands separators."""
    return f"{int(x):,}".replace(",", "{,}")


# ------------------------------------------------------------------ indicators
table, refs, prefm = {}, {}, {}
for inst in insts:
    runs, walls = method_runs(base, pol, inst, POLICY_GROUPS, hyb)
    for lab, names in MAIN.items():                # sampled inference of the main policies
        rr, ww = [], []
        for n in names:
            if n in pol_s and inst in pol_s[n]["results"]:
                e = pol_s[n]["results"][inst]
                rr.append(np.vstack([feasible(e["F"], e["viol"]),
                                     feasible(e["F_sampled"], e["viol_sampled"])]))
                ww.append(e["wall"] + e["wall_sampled"])
        if rr:
            runs[f"{lab} (sampling)"], walls[f"{lab} (sampling)"] = rr, ww
    if inst in unseeded:
        runs[RAND] = [r["F"][r["G"] <= 1e-9] for r in unseeded[inst]["moea"]["nsga2"]]
        walls[RAND] = [r["wall"] for r in unseeded[inst]["moea"]["nsga2"]]
    pool = dict(runs)
    if inst in long_base:          # long runs enter the reference front only
        pool["_long"] = [r["F"][r["G"] <= 1e-9] for r in long_base[inst]["moea"]["nsga2"]]
    R, ideal, nadir = reference(pool)
    refs[inst] = (R, ideal, nadir)
    ind, hv_ref = indicators(runs, R, ideal, nadir)
    table[inst] = {lab: dict(hv=v["hv"].tolist(), igd=v["igd"].tolist(), nnd=v["nnd"].tolist(),
                             cov=v["cov"].tolist(), wall=list(map(float, walls[lab])))
                   for lab, v in ind.items()}
    lo_r, hi_r = np.asarray(base[inst]["ideal"]), np.asarray(base[inst]["nadir"])
    for lab, rs in runs.items():
        table[inst][lab]["hv_rule"] = [_rule_hv(F, lo_r, hi_r) for F in rs]
    table[inst]["_ref"] = dict(size=int(len(R)), ideal=ideal.tolist(), nadir=nadir.tolist())
    # preference metrics (a priori for policies/PWG, a posteriori for MOEAs)
    pm = {}
    W = None
    for lab, names in POLICY_GROUPS.items():
        vals = []
        for n in names:
            if n in pol and inst in pol[n]["results"]:
                e = pol[n]["results"][inst]
                W = pol[n]["prefs"]
                reg, ang, wsr = preference_metrics(e["F"], W, R, ideal, nadir, e["viol"])
                vals.append([np.nanmean(reg), np.nanmean(ang), np.nanmean(wsr),
                             float(np.mean(e["viol"] > 1e-9)), *responsiveness(e["F"], W, e["viol"])])
        if vals:
            pm[lab] = vals
    if W is not None:
        b = base[inst]
        reg, ang, wsr = preference_metrics(b["pwg"]["F"], W, R, ideal, nadir, b["pwg"]["viol"])
        pm["PWG"] = [[np.nanmean(reg), np.nanmean(ang), np.nanmean(wsr),
                      float(np.mean(b["pwg"]["viol"] > 1e-9)),
                      *responsiveness(b["pwg"]["F"], W, b["pwg"]["viol"])]]
        for algo, lab in (("nsga2", "NSGA-II"), ("nsga3", "NSGA-III")):
            vals = []
            for r in b["moea"][algo]:
                F = r["F"][r["G"] <= 1e-9]
                reg, ang, wsr = a_posteriori_metrics(F, W, R, ideal, nadir)
                vals.append([reg.mean(), ang.mean(), wsr.mean(), 0.0,
                             *responsiveness(a_posteriori_selection(F, W, R, ideal, nadir), W)])
            pm[lab + " (a posteriori)"] = vals
    prefm[inst] = pm

json.dump(dict(indicators=table, preference=prefm), open(os.path.join(SUM, "indicators.json"), "w"),
          indent=1, default=float)
import pickle as _pkl  # noqa: E402
_pkl.dump(refs, open(os.path.join(SUM, "reference_fronts.pkl"), "wb"))   # used by make_figures.py

# ------------------------------------------------------------------ statistics (HV ratio)
def labels_present(cands):
    return [l for l in cands if all(l in table[i] for i in insts)]


main_labels = labels_present(list(MAIN) + [f"{l} + NSGA-II" for l in MAIN] +
                             [f"{l} + NSGA-III" for l in MAIN] + ["NSGA-II", "NSGA-III", "PWG", "Rules"])
means = {l: np.array([np.mean(table[i][l]["hv"]) for i in insts]) for l in main_labels}


def pairwise(ref_lab, labs):
    out = {}
    for lab in labs:
        if lab == ref_lab:
            continue
        pv, a = [], []
        for inst in insts:
            x, y = table[inst][ref_lab]["hv"], table[inst][lab]["hv"]
            pv.append(mannwhitney(x, y) if len(x) > 1 and len(y) > 1 else np.nan)
            a.append(a12(x, y))
        out[lab] = dict(p=pv, a12=a)
    flat = [(lab, k) for lab in out for k in range(len(insts)) if not np.isnan(out[lab]["p"][k])]
    adj = holm([out[lab]["p"][k] for lab, k in flat]) if flat else []
    for lab in out:
        out[lab]["p_holm"] = [np.nan] * len(insts)
    for (lab, k), pa in zip(flat, adj):
        out[lab]["p_holm"][k] = float(pa)
    return out


stats = {}
for ref_lab in [OURS, f"{OURS} + NSGA-II"]:
    if ref_lab in main_labels:
        stats[ref_lab] = pairwise(ref_lab, main_labels)
overall = {}
for ref_lab in stats:
    ov = {}
    for lab in main_labels:
        if lab == ref_lab:
            continue
        ov[lab] = dict(wilcoxon_p=wilcoxon(means[ref_lab], means[lab]),
                       a12=a12(means[ref_lab], means[lab]),
                       wins=int(np.sum(means[ref_lab] > means[lab])))
    ps = holm([ov[l]["wilcoxon_p"] for l in ov])
    for l, p in zip(ov, ps):
        ov[l]["wilcoxon_p_holm"] = float(p)
    overall[ref_lab] = ov
fried = friedman(*[means[l] for l in main_labels]) if len(main_labels) >= 3 else np.nan
json.dump(dict(pairwise=stats, overall=overall, friedman_p=fried,
               mean_hv={l: float(means[l].mean()) for l in main_labels}),
          open(os.path.join(SUM, "stats.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------ LaTeX helpers
SHORT = {OURS: "PreCo", "LS-GRL": "LS", f"{OURS} + NSGA-II": "PreCo+NSGA-II",
         "LS-GRL + NSGA-II": "LS+NSGA-II", f"{OURS} + NSGA-III": "PreCo+NSGA-III",
         "LS-GRL + NSGA-III": "LS+NSGA-III"}


def tex_cell(x, sd=None, bold=False, mark=""):
    s = f"{x:.3f}" if sd is None else f"{x:.3f}\\,$\\pm$\\,{sd:.3f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s + mark


def main_table(metric="hv", fname="tab_main.tex", cols=None, higher=True, caption="", label=""):
    cols = cols or main_labels
    lines = ["\\begin{table*}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Instance & " + " & ".join(SHORT.get(c, c) for c in cols) + " \\\\", "\\midrule"]
    for k, inst in enumerate(insts):
        vals = {c: np.array(table[inst][c][metric], dtype=float) for c in cols}
        mv = {c: np.nanmean(v) for c, v in vals.items()}
        best = max(mv.values()) if higher else min(mv.values())
        row = [f"\\texttt{{{inst}}}"]
        for c in cols:
            v = vals[c]
            sd = np.nanstd(v, ddof=1) if len(v) > 1 else None
            mark = ""
            if metric == "hv" and OURS in stats and c in stats[OURS]:
                ph = stats[OURS][c]["p_holm"][k]
                if not np.isnan(ph) and ph < 0.05:
                    mark = "$^{+}$" if mv[OURS] > mv[c] else "$^{-}$"
            row.append(tex_cell(mv[c], sd, bold=np.isclose(mv[c], best), mark=mark))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\midrule")
    avg = [np.mean([np.nanmean(table[i][c][metric]) for i in insts]) for c in cols]
    lines.append("Mean & " + " & ".join(f"{x:.3f}" for x in avg) + " \\\\")
    if metric == "hv":
        rn = [np.mean([np.mean(table[i][c]["hv_rule"]) for i in insts]) for c in cols]
        lines.append("Mean rule-scaled HV & " + " & ".join(f"{x:.2f}" for x in rn) + " \\\\")
        tt = [np.mean([np.mean(table[i][c]["wall"]) for i in insts]) for i_c, c in enumerate(cols)]
        lines.append("Time (s) & " + " & ".join(f"{x:.1f}" for x in tt) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(os.path.join(GEN, fname), "w").write("\n".join(lines) + "\n")


if main_labels:
    main_table("hv", "tab_main.tex", caption=(
        "Hypervolume ratio per instance (mean\\,$\\pm$\\,standard deviation over runs, best mean in bold). "
        "Rule-scaled HV is measured in the box spanned by the dispatching rules. Times are seconds per instance "
        "from parallel evaluation runs."), label="tab:maininst")
    main_table("igd", "tab_igd.tex", higher=False, caption=(
        "IGD$^+$ per instance (mean\\,$\\pm$\\,standard deviation over runs, lower is better)."),
        label="tab:igd")

# ------------------------------------------------------------------ preference table
# ------------------------------------------------------------------ compact summary table for the manuscript body
SUB_NAMES = {"Rules": "Dispatching rules", "PWG": "PWG", "LS-GRL": "LS-GRL", OURS: "DP-PreCo-GRL", "NSGA-II": "NSGA-II",
             "NSGA-III": "NSGA-III", "LS-GRL + NSGA-II": "LS-GRL + NSGA-II", f"{OURS} + NSGA-II": "DP-PreCo-GRL + NSGA-II",
             "LS-GRL + NSGA-III": "LS-GRL + NSGA-III", f"{OURS} + NSGA-III": "DP-PreCo-GRL + NSGA-III"}
sub_order = [l for l in ["Rules", "PWG", "LS-GRL", OURS, "NSGA-II", "NSGA-III", "LS-GRL + NSGA-II", f"{OURS} + NSGA-II",
                         "LS-GRL + NSGA-III", f"{OURS} + NSGA-III"] if l in main_labels]
if sub_order:
    tl = ["\\begin{table}[htbp]", "\\centering",
          "\\caption{Front quality on the 12 benchmarks with the real test day (mean over instances, standard deviation over training "
          "seeds or runs). Times are seconds per instance on one thread and include the policy sweep for hybrids.}",
          "\\label{tab:main}", "\\small", "\\begin{tabular}{lcccc}", "\\toprule",
          "Method & HV ratio & Rule-scaled HV & IGD$^+$ & Time (s) \\\\", "\\midrule"]
    for l in sub_order:
        nr = min(len(table[i][l]["hv"]) for i in insts)
        rm = np.array([np.mean([table[i][l]["hv"][r] for i in insts]) for r in range(nr)])
        sd = f"\\,$\\pm$\\,{rm.std(ddof=1):.3f}" if nr > 1 else ""
        tl.append(f"{SUB_NAMES[l]} & {rm.mean():.3f}{sd} & "
                  f"{np.mean([np.mean(table[i][l]['hv_rule']) for i in insts]):.2f} & "
                  f"{np.mean([np.nanmean(table[i][l]['igd']) for i in insts]):.2f} & "
                  f"{np.mean([np.mean(table[i][l]['wall']) for i in insts]):.1f} \\\\")
        if l in ("PWG", OURS + "", "NSGA-III") and False:
            pass
    tl += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(GEN, "tab_sub_main.tex"), "w").write("\n".join(tl) + "\n")

pref_labels = [l for l in list(MAIN) + ["PWG", "NSGA-II (a posteriori)", "NSGA-III (a posteriori)"]
               if all(l in prefm[i] for i in insts)]
pref_summary = {}
for l in pref_labels:
    arr = np.array([np.nanmean(np.array(prefm[i][l], dtype=float), axis=0) for i in insts])  # inst x 5
    pref_summary[l] = arr
if pref_labels:
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Preference control on the 84 evaluation preferences (mean over instances). Policies and PWG receive the "
             "preference in advance, the NSGA rows select a posteriori. ASF: achievement function, WS: weighted sum, "
             "$\\rho$: mean responsiveness.}", "\\label{tab:pref}", "\\footnotesize",
             "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{lccccc}", "\\toprule",
             "Method & ASF reg.\\ $\\downarrow$ & Angle ($^\\circ$) $\\downarrow$ & WS reg.\\ $\\downarrow$ & $\\rho$ $\\downarrow$ & Infeas.\\ $\\downarrow$ \\\\",
             "\\midrule"]
    for l in pref_labels:
        m = np.nanmean(pref_summary[l], axis=0)
        lab_ = SHORT.get(l, l).replace("(a posteriori)", "(a post.)")
        lines.append(f"{lab_} & {m[0]:.3f} & {m[1]:.1f} & {m[2]:.3f} & {np.nanmean(m[4:8]):+.2f} & {100 * m[3]:.1f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(GEN, "tab_pref.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ responsiveness per objective
vr_p = os.path.join(SUM, "val_responsiveness.json")
vresp = {k: v for k, v in json.load(open(vr_p)).items() if k != "per_run"} if os.path.exists(vr_p) else {}
if pref_labels:
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Responsiveness of each objective to its weight: Spearman correlation between $w_k$ and $f_k$ over the 84 "
             "preferences, mean over instances and runs ($-1$: raising the weight always improves the objective). "
             "Test: benchmarks. Validation: synthetic scenarios.}",
             "\\label{tab:resp}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{llcccc}", "\\toprule",
             "Method & Data & Makespan & Cost & CO$_2$ & OWA risk \\\\", "\\midrule"]
    for l in pref_labels:
        m = np.nanmean(pref_summary[l], axis=0)[4:8]
        lines.append(f"{SHORT.get(l, l)} & test & " + " & ".join(f"{x:+.2f}" for x in m) + " \\\\")
        if l in vresp:
            lines.append(" & validation & " + " & ".join(f"{x:+.2f}" for x in vresp[l]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(GEN, "tab_resp.tex"), "w").write("\n".join(lines) + "\n")
    for l, a in ((OURS, "Preco"), ("LS-GRL", "Ls"), ("PWG", "Pwg")):
        if l in vresp:
            for k, on in enumerate(("Mk", "Cost", "Co", "Risk")):
                put(f"vrho{a}{on}", vresp[l][k], "{:+.2f}")

# ------------------------------------------------------------------ ablation table
abl_labels = [l for l in [OURS, "LS-GRL"] + list(ABL) if all(l in table[i] for i in insts)]
if len(abl_labels) > 1:
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Ablation of the original run (policy alone, mean over the 12 benchmarks, standard deviation over training "
             "seeds, two seeds per ablated variant). $p$: Wilcoxon test against the full model, Holm-corrected. "
             "Infeas.: share of rollouts ending in a forced-fallback violation.}",
             "\\label{tab:ablation}", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
             "\\begin{tabular}{lccc}", "\\toprule",
             "Variant & HV ratio $\\uparrow$ & $p$ & Infeas. \\\\",
             "\\midrule"]
    lines2 = ["\\begin{table}[htbp]", "\\centering",
              "\\caption{Secondary indicators of the original ablation: IGD$^+$ and regret of the achievement function.}", "\\label{tab:ablation2}", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
              "\\begin{tabular}{lcc}", "\\toprule", "Variant & IGD$^+$ $\\downarrow$ & ASF regret $\\downarrow$ \\\\", "\\midrule"]
    inst_mean = {l: np.array([np.mean(table[i][l]["hv"]) for i in insts]) for l in abl_labels}
    others = [l for l in abl_labels if l != OURS]
    p_abl = dict(zip(others, holm([wilcoxon(inst_mean[OURS], inst_mean[l]) for l in others])))
    for l in abl_labels:
        runs_n = POLICY_GROUPS[l][:len(table[insts[0]][l]["hv"])]
        per_run = np.array([[table[i][l]["hv"][r] for i in insts] for r in range(len(runs_n))]).mean(1)
        igd = np.nanmean([np.nanmean(table[i][l]["igd"]) for i in insts])
        pr = (np.nanmean([np.nanmean(np.array(prefm[i][l], dtype=float), axis=0) for i in insts], axis=0)
              if l in prefm[insts[0]] else [np.nan] * 5)
        name = {OURS: "Full DP-PreCo-GRL", "LS-GRL": "LS-GRL (linear comb.)",
                "BC warm start only (PreCo targets)": "BC only, PreCo targets",
                "BC warm start only (LS targets)": "BC only, LS targets"}.get(l, l)
        key = {OURS: "Full", "LS-GRL": "Ls", "w/o graph attention": "Nogat", "w/o FiLM": "Nofilm",
               "w/o criteria head": "Nocrit", "w/o expert pool": "Noexpert", "w/o elite pool": "Noelite",
               "w/o postponement": "Nodelay", "w/o credibility mask": "Nomask",
               "BC warm start only (PreCo targets)": "BcPreco", "BC warm start only (LS targets)": "BcLs"}.get(l)
        if key:
            put(f"abl{key}", per_run.mean())
            put(f"abl{key}Infeas", 100 * pr[3], "{:.1f}")
            if l != OURS:
                put(f"abl{key}P", p_abl[l], "p")
        pcell = "-" if l == OURS else (f"{p_abl[l]:.3f}" if p_abl[l] >= 0.001 else "$<$0.001")
        lines.append(f"{name} & {per_run.mean():.3f}\\,$\\pm$\\,{per_run.std(ddof=1) if len(per_run) > 1 else 0:.3f} & "
                     f"{pcell} & {100 * pr[3]:.1f}\\% \\\\")
        lines2.append(f"{name} & {igd:.3f} & {pr[0]:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    lines2 += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(GEN, "tab_ablation.tex"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(GEN, "tab_ablation2.tex"), "w").write("\n".join(lines2) + "\n")

# ------------------------------------------------------------------ greedy versus sampled inference
samp_rows = [(l, f"{l} (sampling)") for l in MAIN if all(f"{l} (sampling)" in table[i] for i in insts)]
if samp_rows:
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Greedy inference (84 rollouts) versus greedy plus 8 sampled rollouts per preference (mean over benchmarks "
             "and seeds, time in seconds per instance).}",
             "\\label{tab:sampling}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{llcccr}", "\\toprule",
             "Policy & Inference & HV ratio $\\uparrow$ & IGD$^+$ $\\downarrow$ & \\#ND & Time \\\\", "\\midrule"]
    for g, sl in samp_rows:
        for lab, inf in ((g, "greedy"), (sl, "greedy + sampling")):
            hv = np.mean([np.mean(table[i][lab]["hv"]) for i in insts])
            ig = np.nanmean([np.nanmean(table[i][lab]["igd"]) for i in insts])
            nd = np.mean([np.mean(table[i][lab]["nnd"]) for i in insts])
            tt = np.mean([np.mean(table[i][lab]["wall"]) for i in insts])
            lines.append(f"{SHORT.get(g, g)} & {inf} & {hv:.3f} & {ig:.3f} & {nd:.0f} & {tt:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(GEN, "tab_sampling.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ rule seeding of NSGA-II (appendix)
if all(RAND in table[i] for i in insts):
    labs3 = ("Rules", RAND, "NSGA-II")
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{NSGA-II with a random or a rule-seeded initial population (HV ratio and rule-scaled HV, mean over seeds). "
             "Rules: front of the eight dispatching rules.}", "\\label{tab:seeding}",
             "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lcccccc}", "\\toprule",
             " & \\multicolumn{3}{c}{HV ratio} & \\multicolumn{3}{c}{Rule-scaled HV} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Inst. & Rules & Random & Seeded & Rules & Random & Seeded \\\\", "\\midrule"]
    for inst in insts:
        v = [np.mean(table[inst][l]["hv"]) for l in labs3] + [np.mean(table[inst][l]["hv_rule"]) for l in labs3]
        lines.append(f"\\texttt{{{inst}}} & " + " & ".join(f"{x:.3f}" for x in v[:3]) + " & "
                     + " & ".join(f"{x:.2f}" for x in v[3:]) + " \\\\")
    mv = [np.mean([np.mean(table[i][l]["hv"]) for i in insts]) for l in labs3] + \
         [np.mean([np.mean(table[i][l]["hv_rule"]) for i in insts]) for l in labs3]
    lines += ["\\midrule", "Mean & " + " & ".join(f"{x:.3f}" for x in mv[:3]) + " & "
              + " & ".join(f"{x:.2f}" for x in mv[3:]) + " \\\\", "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(GEN, "tab_seeding.tex"), "w").write("\n".join(lines) + "\n")
    put("hvNsgaiiRand", np.mean([np.mean(table[i][RAND]["hv"]) for i in insts]))
    below = [i for i in insts if np.mean(table[i][RAND]["hv_rule"]) < np.mean(table[i]["Rules"]["hv_rule"])]
    put("nRandBelowRules", str(len(below)))
    put("randBelowList", ", ".join(f"\\texttt{{{i}}}" for i in below[:-1]) +
        (f" and \\texttt{{{below[-1]}}}" if len(below) > 1 else (f"\\texttt{{{below[0]}}}" if below else "--")))
    put("nRandBelowSeeded", str(sum(np.mean(table[i][RAND]["hv"]) < np.mean(table[i]["NSGA-II"]["hv"]) for i in insts)))

# ------------------------------------------------------------------ interpreted vs compiled decoder
dj, dp = os.path.join(SUM, "decoder_speed_jit.json"), os.path.join(SUM, "decoder_speed_python.json")
if os.path.exists(dj) and os.path.exists(dp):
    dj, dp = json.load(open(dj)), json.load(open(dp))
    sp = {i: dp[i] / dj[i] for i in dj if i in dp}
    put("decSpeedupMin", min(sp.values()), "{:.0f}"); put("decSpeedupMax", max(sp.values()), "{:.0f}")
    put("decSpeedupMed", float(np.median(list(sp.values()))), "{:.0f}")
    put("decJitMs", float(np.mean(list(dj.values()))), "{:.3f}")
    put("decPyMs", float(np.mean(list(dp.values()))), "{:.1f}")
    put("decPyNsgaHours", 50000 * max(dp.values()) / 1000 / 3600, "{:.1f}")

# ------------------------------------------------------------------ training cost
train_t = {}
for lab, names in POLICY_GROUPS.items():
    ts = []
    for n in names:
        p = os.path.join(ROOT, "results", "runs", n, "history.json")
        if os.path.exists(p):
            h = json.load(open(p))
            ts.append(max(x.get("time", 0) for x in h) / 3600.0)
    if ts:
        train_t[lab] = ts

# ------------------------------------------------------------------ time-matched hybrid comparison
# The hybrid costs one policy sweep + 50k evaluations.  Using the controlled timing
# (results/summary/timing.json), the sweep time is converted into NSGA-II evaluations
# on the same instance, and the hybrid is compared with long rule-seeded NSGA-II runs
# (tag "long", 2,000 generations) at that evaluation-equivalent budget.
tm = {}
timing_p = os.path.join(ROOT, "results", "summary", "timing.json")
hyb_lab = f"{OURS} + NSGA-II"
if os.path.exists(timing_p) and long_base and hyb_lab in hyb:
    timing = json.load(open(timing_p))
    from morl_fjsp.metrics import hypervolume as _hv, nondominated as _nd, normalise as _nm
    for inst in insts:
        if inst not in long_base or inst not in timing:
            continue
        R, ideal, nadir = refs[inst]
        href = _hv(_nm(R, ideal, nadir))
        rate = timing[inst]["nsga2"]["evals"] / timing[inst]["nsga2"]["wall"]
        sweep_t = timing[inst][GROUPS["main"][OURS][0]]["sweep84"]
        budget = 50000 + sweep_t * rate
        long_hv = []
        for r in long_base[inst]["moea"]["nsga2"]:
            # first snapshot at or after the budget (traces every 2,500 evaluations): the
            # long run gets at least the hybrid's evaluation-equivalent budget
            snaps = [(e, F) for g, e, t, F in r["trace"] if e >= budget]
            F = snaps[0][1] if snaps else r["trace"][-1][3]
            F = F if len(F) else np.zeros((0, 4))
            long_hv.append(_hv(_nm(_nd(F), ideal, nadir)) / href if len(F) else 0.0)
        tm[inst] = dict(budget=float(budget), hybrid=table[inst][hyb_lab]["hv"], long=long_hv,
                        p=mannwhitney(table[inst][hyb_lab]["hv"], long_hv),
                        a12=a12(table[inst][hyb_lab]["hv"], long_hv))
    json.dump(tm, open(os.path.join(SUM, "time_matched.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------ runtime table (controlled timing)
if os.path.exists(timing_p):
    timing = json.load(open(timing_p))
    r0 = GROUPS["main"][OURS][0]
    # eight-rule pre-pass (objective scale sigma of a new scenario): sum of the eight rule runs of the
    # baseline campaign (four instances in parallel), the only place where it was timed on the VM
    prepass = {i: sum(v["time"] for v in base[i]["rules"].values()) for i in insts}
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Wall-clock seconds on one thread. Rules (8): pre-pass that provides the objective scale of a new scenario. "
             "Policy (1): one rollout with the scale cached. Cold (1): sum of both. Policy (84) and PWG (84): sweeps over "
             "84 preferences. NSGA: one run of 50{,}000 evaluations.}",
             "\\label{tab:runtimeinst}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.4pt}",
             "\\begin{tabular}{lrrrrrrr}", "\\toprule",
             "Inst. & Rules (8) & Policy (1) & Cold (1) & Policy (84) & PWG (84) & NSGA-II & NSGA-III \\\\", "\\midrule"]
    for inst in insts:
        if inst not in timing:
            continue
        t = timing[inst]
        lines.append(f"\\texttt{{{inst}}} & {prepass[inst]:.2f} & {t[r0]['single']:.2f} & {prepass[inst] + t[r0]['single']:.2f} & "
                     f"{t[r0]['sweep84']:.1f} & {t['PWG']['sweep84']:.1f} & {t['nsga2']['wall']:.1f} & {t['nsga3']['wall']:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(GEN, "tab_runtime.tex"), "w").write("\n".join(lines) + "\n")
    def _mm(vals):
        k = max(vals, key=vals.get)
        return np.mean(list(vals.values())), vals[k], k
    rows_rt = [("Eight-rule pre-pass", {i: prepass[i] for i in timing}, "{:.2f}"),
               ("Policy, one preference, scale cached", {i: timing[i][r0]["single"] for i in timing}, "{:.2f}"),
               ("Policy, one preference, cold start", {i: prepass[i] + timing[i][r0]["single"] for i in timing}, "{:.2f}"),
               ("Policy, 84 preferences, scale cached", {i: timing[i][r0]["sweep84"] for i in timing}, "{:.1f}"),
               ("PWG, 84 preferences", {i: timing[i]["PWG"]["sweep84"] for i in timing}, "{:.1f}"),
               ("NSGA-II, 50{,}000 evaluations", {i: timing[i]["nsga2"]["wall"] for i in timing}, "{:.1f}"),
               ("NSGA-III, 50{,}000 evaluations", {i: timing[i]["nsga3"]["wall"] for i in timing}, "{:.1f}")]
    tr = ["\\begin{table}[htbp]", "\\centering",
          "\\caption{Wall-clock seconds on one thread and an idle machine, mean and maximum over the 12 benchmarks.}",
          "\\label{tab:runtime}", "\\small", "\\begin{tabular}{lcc}", "\\toprule", "Task & Mean & Maximum (instance) \\\\", "\\midrule"]
    for name_, vals_, f_ in rows_rt:
        m_, x_, k_ = _mm(vals_)
        tr.append(f"{name_} & {f_.format(m_)} & {f_.format(x_)} (\\texttt{{{k_}}}) \\\\")
    tr += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(GEN, "tab_sub_runtime.tex"), "w").write("\n".join(tr) + "\n")
    NUM["timePolicyMean"] = f"{np.mean([timing[i][r0]['sweep84'] for i in timing]):.1f}"
    NUM["timeSingleMean"] = f"{np.mean([timing[i][r0]['single'] for i in timing]):.2f}"
    NUM["timeNsgaiiMean"] = f"{np.mean([timing[i]['nsga2']['wall'] for i in timing]):.1f}"
    NUM["timeNsgaiiiMean"] = f"{np.mean([timing[i]['nsga3']['wall'] for i in timing]):.1f}"
    NUM["timePrepassMean"] = f"{np.mean([prepass[i] for i in timing]):.2f}"
    NUM["timeColdMean"] = f"{np.mean([prepass[i] + timing[i][r0]['single'] for i in timing]):.2f}"
    _worst = max(timing, key=lambda i: prepass[i] + timing[i][r0]["single"])
    NUM["timeColdMax"] = f"{prepass[_worst] + timing[_worst][r0]['single']:.2f}"
    NUM["timeColdMaxInst"] = _worst
    NUM["timeSweepColdMean"] = f"{np.mean([prepass[i] + timing[i][r0]['sweep84'] for i in timing]):.1f}"
    # zero-shot evaluation runs of the five main seeds (four processes in parallel): per-seed mean sweep time
    _ev = [np.mean([pol[n]["results"][i]["wall"] for i in insts]) for n in GROUPS["main"][OURS] if n in pol]
    if _ev:
        NUM["evalWallMin"], NUM["evalWallMax"] = f"{min(_ev):.1f}", f"{max(_ev):.1f}"

# ------------------------------------------------------------------ graph encoder cost (Apple M1 Pro, scripts/timing_coldstart.py)
_tc = os.path.join(SUM, "timing_coldstart.json")
if os.path.exists(_tc):
    tc = json.load(open(_tc))
    ti_ = tc["instances"]
    mods = [("preco_s0", "Full (graph attention)", "Full"), ("nogat_s0", "Flat (no message passing)", "NoGat")]
    st = {r: (np.mean([ti_[i][r]["single_cached"] for i in ti_]), np.mean([ti_[i][r]["sweep84_cached"] for i in ti_])) for r, _, _ in mods}
    pre = np.mean([ti_[i]["rules8"] for i in ti_])
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Size and cost of the graph encoder (mean over the benchmarks, one thread, Apple M1 Pro). Policy (1) and "
             "Policy (84): seconds for one rollout and for the sweep. Ratio: sweep time relative to the full model.}",
             "\\label{tab:graphcost}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lrrrr}", "\\toprule", "Model & Params & Policy (1) & Policy (84) & Ratio \\\\", "\\midrule"]
    for r, lab, tag in mods:
        lines.append(f"{lab} & {_cnt(tc['params'][r])} & {st[r][0]:.3f} & {st[r][1]:.1f} & "
                     f"{st[r][1] / st['preco_s0'][1]:.2f} \\\\")
        put(f"params{tag}", _cnt(tc["params"][r]))
        put(f"macSingle{tag}", st[r][0], "{:.3f}"); put(f"macSweep{tag}", st[r][1], "{:.1f}")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(GEN, "tab_graph_cost.tex"), "w").write("\n".join(lines) + "\n")
    put("macPrepassMean", pre, "{:.3f}")
    put("gatSweepRatio", st["nogat_s0"][1] / st["preco_s0"][1], "{:.2f}")
    put("gatParamRatio", tc["params"]["preco_s0"] / tc["params"]["nogat_s0"], "{:.2f}")

# ------------------------------------------------------------------ numbers for the text
put("numseedsmain", str(len(MAIN[OURS])))
cfg0 = os.path.join(ROOT, "results", "runs", MAIN[OURS][0], "config.json")
put("numitersmain", str(json.load(open(cfg0))["iters"]) if os.path.exists(cfg0) else "--")
alias = {OURS: "Preco", "LS-GRL": "Ls", "NSGA-II": "Nsgaii", "NSGA-III": "Nsgaiii", "PWG": "Pwg",
         "Rules": "Rules", f"{OURS} + NSGA-II": "HybPrecoii", "LS-GRL + NSGA-II": "HybLsii",
         f"{OURS} + NSGA-III": "HybPrecoiii", "LS-GRL + NSGA-III": "HybLsiii"}
for l in main_labels:
    a = alias.get(l)
    if a:
        put(f"hv{a}", means[l].mean())
        put(f"time{a}", np.mean([np.mean(table[i][l]["wall"]) for i in insts]), "{:.1f}")
        put(f"igd{a}", np.mean([np.nanmean(table[i][l]["igd"]) for i in insts]))
        put(f"rhv{a}", np.mean([np.mean(table[i][l]["hv_rule"]) for i in insts]), "{:.2f}")
for ref_lab, ov in overall.items():
    ra = alias[ref_lab]
    for l, d in ov.items():
        if l in alias:
            put(f"wins{ra}vs{alias[l]}", str(d["wins"]))
            put(f"p{ra}vs{alias[l]}", d["wilcoxon_p_holm"], "p")
for g, sl in (samp_rows if samp_rows else []):
    a = alias[g]
    put(f"hv{a}Samp", np.mean([np.mean(table[i][sl]["hv"]) for i in insts]))
    put(f"time{a}Samp", np.mean([np.mean(table[i][sl]["wall"]) for i in insts]), "{:.1f}")
for l in pref_labels:
    a = {OURS: "Preco", "LS-GRL": "Ls", "PWG": "Pwg", "NSGA-II (a posteriori)": "Nsgaii",
         "NSGA-III (a posteriori)": "Nsgaiii"}[l]
    m = np.nanmean(pref_summary[l], axis=0)
    put(f"asf{a}", m[0]); put(f"ang{a}", m[1], "{:.1f}"); put(f"wsr{a}", m[2])
    put(f"infeas{a}", 100 * m[3], "{:.1f}"); put(f"rho{a}", np.nanmean(m[4:8]), "{:+.2f}")
    for k, on in enumerate(("Mk", "Cost", "Co", "Risk")):
        put(f"rho{a}{on}", m[4 + k], "{:+.2f}")
# PreCo vs LS on the preference metrics: paired Wilcoxon over the per-instance means
if OURS in pref_summary and "LS-GRL" in pref_summary:
    A_, B_ = pref_summary[OURS], pref_summary["LS-GRL"]
    for k, nm in ((0, "Asf"), (1, "Ang"), (2, "Wsr")):
        put(f"p{nm}PrecoLs", wilcoxon(A_[:, k], B_[:, k]), "p")
        put(f"wins{nm}PrecoLs", str(int(np.sum(A_[:, k] < B_[:, k]))))
    ra, rb = np.nanmean(A_[:, 4:8], axis=1), np.nanmean(B_[:, 4:8], axis=1)
    put("pRhoPrecoLs", wilcoxon(ra, rb), "p")
for lab, ts in train_t.items():
    if lab in (OURS, "LS-GRL"):
        put(f"train{alias[lab]}", np.mean(ts), "{:.2f}")
# checkpoint selection: how often the warm-start policy (iteration 0) was the best on validation
for lab in (OURS, "LS-GRL"):
    it0, best, last, sel_it = [], [], [], []
    for n in MAIN[lab]:
        hp_ = os.path.join(ROOT, "results", "runs", n, "history.json")
        if not os.path.exists(hp_):
            continue
        h = [x for x in json.load(open(hp_)) if "val_hv" in x]
        b = max(h, key=lambda x: x["val_hv"])
        it0.append(h[0]["val_hv"]); best.append(b["val_hv"]); last.append(h[-1]["val_hv"])
        sel_it.append(b["iter"])
    if sel_it:
        a = alias[lab]
        put(f"sel{a}Zero", str(sum(i == 0 for i in sel_it)))
        put(f"val{a}ItZero", np.mean(it0), "{:.2f}")
        put(f"val{a}Best", np.mean(best), "{:.2f}")
        put(f"val{a}Last", np.mean(last), "{:.2f}")
put("friedmanp", fried, "p")
if tm:
    put("tmHyb", np.mean([np.mean(v["hybrid"]) for v in tm.values()]))
    put("tmLong", np.mean([np.mean(v["long"]) for v in tm.values()]))
    put("tmWins", str(sum(np.mean(v["hybrid"]) > np.mean(v["long"]) for v in tm.values())))
    put("tmN", str(len(tm)))
    put("tmBudget", np.mean([v["budget"] for v in tm.values()]) / 1000, "{:.0f}")
# ------------------------------------------------------------------ generalisation diagnostic
# The validation metric (HV in the space normalised by the best/worst dispatching rule) is
# recomputed on the benchmarks, with NSGA-II at the validation budget (20,000 evaluations,
# from its anytime trace), so that validation and test can be compared on one scale: the share
# of the PWG-to-NSGA-II gap that a policy closes.
gen_rows = {}
for inst in insts:
    b = base[inst]
    lo, hi = np.asarray(b["ideal"]), np.asarray(b["nadir"])
    runs_g, _ = method_runs(base, pol, inst, MAIN, None)
    row = {l: float(np.mean([_rule_hv(F, lo, hi) for F in runs_g[l]])) for l in (OURS, "LS-GRL", "PWG")
           if l in runs_g}
    n20 = []
    for r in b["moea"]["nsga2"]:
        snap = [F for g, e, t_, F in r["trace"] if e <= 20000]
        n20.append(_rule_hv(snap[-1], lo, hi) if snap else 0.0)
    row["NSGA-II (20k)"] = float(np.mean(n20))
    gen_rows[inst] = row
vref_p = os.path.join(SUM, "val_reference.json")
if gen_rows and all(OURS in r for r in gen_rows.values()) and os.path.exists(vref_p):
    vref = json.load(open(vref_p))
    bm = {l: np.mean([gen_rows[i][l] for i in insts]) for l in (OURS, "LS-GRL", "PWG", "NSGA-II (20k)")}
    closure_test = {l: (bm[l] - bm["PWG"]) / (bm["NSGA-II (20k)"] - bm["PWG"]) for l in (OURS, "LS-GRL")}
    val_sel = {}
    for l in (OURS, "LS-GRL"):
        vs = []
        for n in MAIN[l]:
            hp_ = os.path.join(ROOT, "results", "runs", n, "history.json")
            if os.path.exists(hp_):
                vs.append(max(x["val_hv"] for x in json.load(open(hp_)) if "val_hv" in x))
        val_sel[l] = float(np.mean(vs)) if vs else np.nan
    closure_val = {l: (val_sel[l] - vref["pwg"]) / (vref["nsga2"] - vref["pwg"]) for l in val_sel}
    json.dump(dict(benchmark=gen_rows, benchmark_mean=bm, closure_test=closure_test,
                   validation_selected=val_sel, closure_val=closure_val, val_reference=vref),
              open(os.path.join(SUM, "generalisation.json"), "w"), indent=1, default=float)
    put("genValPreco", 100 * closure_val[OURS], "{:.0f}")
    put("genValLs", 100 * closure_val["LS-GRL"], "{:.0f}")
    put("genTestPreco", 100 * closure_test[OURS], "{:.0f}")
    put("genTestLs", 100 * closure_test["LS-GRL"], "{:.0f}")
    put("valPrecoSel", val_sel[OURS], "{:.2f}")
    put("valLsSel", val_sel["LS-GRL"], "{:.2f}")
    put("valPwg", vref["pwg"], "{:.2f}")
    put("valNsga", vref["nsga2"], "{:.2f}")
    if "nsga2_expert" in vref:
        put("valExpert", vref["nsga2_expert"], "{:.2f}")

# ------------------------------------------------------------------ robustness: energy day / shift start
day_files = sorted(glob.glob(os.path.join(ROOT, "results", "extra", "days_*.pkl")))
if day_files:
    import pickle
    D = {}
    for f_ in day_files:
        D.update(pickle.load(open(f_, "rb")))
    cond_info = {c[0]: c for c in D["conditions"]}
    conds = [c for c in cond_info if any(isinstance(k, tuple) and k[0] == c for k in D)]
    dlabs = [OURS, "LS-GRL", "PWG", "NSGA-II"]
    from morl_fjsp.metrics import eval_preferences
    W_eval = eval_preferences()
    dres = {}
    for c in conds:
        per = {l: [] for l in dlabs}
        for inst in insts:
            rd = {}
            for lab in (OURS, "LS-GRL"):
                rr = [feasible(D[(c, n, inst)]["F"], D[(c, n, inst)]["viol"])
                      for n in MAIN[lab] if (c, n, inst) in D]
                if rr:
                    rd[lab] = rr
            if (c, "PWG", inst) in D:
                rd["PWG"] = [feasible(D[(c, "PWG", inst)]["F"], D[(c, "PWG", inst)]["viol"])]
            if (c, "NSGA-II", inst) in D:
                rd["NSGA-II"] = [r["F"][r["G"] <= 1e-9] for r in D[(c, "NSGA-II", inst)]]
            if len(rd) < len(dlabs):
                continue
            Rd, idl, ndr = reference(rd)
            ind_d, _ = indicators(rd, Rd, idl, ndr)
            for l in dlabs:
                per[l].append(float(np.mean(ind_d[l]["hv"])))
            for lab in (OURS, "LS-GRL"):          # cost responsiveness of the policies
                rc = [responsiveness(D[(c, n, inst)]["F"], W_eval, D[(c, n, inst)]["viol"])[1]
                      for n in MAIN[lab] if (c, n, inst) in D]
                per.setdefault(f"rho_cost {lab}", []).append(float(np.nanmean(rc)))
        dres[c] = {l: np.array(v) for l, v in per.items()}
    json.dump({c: {l: v.tolist() for l, v in d.items()} for c, d in dres.items()},
              open(os.path.join(SUM, "days.json"), "w"), indent=1)
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Mean HV ratio under other energy conditions (real day with three shift starts, four unseen synthetic days). "
             "$\\bar c$ and $\\Delta c$: mean and range of the price (EUR/MWh) over 16~h. $\\rho_{\\rm cost}$: cost responsiveness "
             "for PreCo and LS.}",
             "\\label{tab:days}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lrrccccc}", "\\toprule",
             "Condition & $\\bar c$ & $\\Delta c$ & PreCo & LS & PWG & NSGA-II & $\\rho_{\\rm cost}$ \\\\", "\\midrule"]
    for c in conds:
        mv = {l: dres[c][l].mean() for l in dlabs}
        best = max(mv.values())
        cells = [(f"\\textbf{{{mv[l]:.3f}}}" if np.isclose(mv[l], best) else f"{mv[l]:.3f}") for l in dlabs]
        win = (cond_info[c][4] // 15 + np.arange(64)) % 96          # 16-h window from the shift start
        pw_ = np.asarray(cond_info[c][1])[win]
        rcell = f"{np.nanmean(dres[c][f'rho_cost {OURS}']):+.2f} / {np.nanmean(dres[c]['rho_cost LS-GRL']):+.2f}"
        lines.append(f"{c} & {pw_.mean():.0f} & {pw_.max() - pw_.min():.0f} & " + " & ".join(cells) + f" & {rcell} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(GEN, "tab_days.tex"), "w").write("\n".join(lines) + "\n")
    NUM["daysN"] = str(len(conds))
    NUM["daysNsgaBest"] = str(sum(max(dlabs, key=lambda l: dres[c][l].mean()) == "NSGA-II" for c in conds))
    NUM["daysPrecoOverPwg"] = str(sum(dres[c][OURS].mean() > dres[c]["PWG"].mean() for c in conds))
    NUM["daysPrecoOverLs"] = str(sum(dres[c][OURS].mean() > dres[c]["LS-GRL"].mean() for c in conds))
    allv = {l: np.concatenate([dres[c][l] for c in conds]) for l in dlabs}
    for l, a in ((OURS, "Preco"), ("LS-GRL", "Ls"), ("PWG", "Pwg"), ("NSGA-II", "Nsgaii")):
        put(f"days{a}Min", min(dres[c][l].mean() for c in conds))
        put(f"days{a}Max", max(dres[c][l].mean() for c in conds))
    put("daysPrecoVsLsP", wilcoxon(allv[OURS], allv["LS-GRL"]), "p")

# ------------------------------------------------------------------ behaviour of the policies
from morl_fjsp.instance import load_benchmarks as _lb  # noqa: E402

_nops = {i.name: i.n_ops for i in _lb(os.path.join(ROOT, "data", "fjsp"))}
for lab in (OURS, "LS-GRL"):
    lo_p, hi_p, lo_e, hi_e, allp = [], [], [], [], []
    for n in MAIN[lab]:
        if n not in pol:
            continue
        Wp = pol[n]["prefs"]
        x = Wp[:, 1] + Wp[:, 2]
        for inst, e in pol[n]["results"].items():
            if "n_postponed" not in e:
                continue
            post = 100 * e["n_postponed"] / _nops[inst]
            rel = 100 * (e["rel_machine_energy"] - 1)
            lo_p += list(post[x <= 0.2]); hi_p += list(post[x >= 0.8]); allp += list(post)
            lo_e += list(rel[x <= 0.2]); hi_e += list(rel[x >= 0.8])
    if allp:
        a = alias[lab]
        put(f"behPost{a}", np.mean(allp), "{:.2f}")
        put(f"behPost{a}Lo", np.mean(lo_p), "{:.2f}"); put(f"behPost{a}Hi", np.mean(hi_p), "{:.2f}")
        put(f"behEn{a}Lo", np.mean(lo_e), "{:.1f}"); put(f"behEn{a}Hi", np.mean(hi_e), "{:.1f}")
# the schedule example of Fig. schedule (first PreCo run, Mk01)
r0 = MAIN[OURS][0]
if r0 in pol and "Mk01" in pol[r0]["results"]:
    e = pol[r0]["results"]["Mk01"]
    Wp = pol[r0]["prefs"]
    ia, ib = int(np.argmax(Wp[:, 0])), int(np.argmax(Wp[:, 1]))
    for tag, i in (("A", ia), ("B", ib)):
        put(f"ex{tag}Mk", e["F"][i, 0], "{:.1f}"); put(f"ex{tag}Cost", e["F"][i, 1], "{:.1f}")
        put(f"ex{tag}Co", e["F"][i, 2], "{:.0f}"); put(f"ex{tag}Risk", e["F"][i, 3], "{:.2f}")
    if "actions" in e:
        from morl_fjsp.scenarios import benchmark_scenarios as _bs
        from morl_fjsp.sim import evaluate_actions as _ev
        Pm = _bs(os.path.join(ROOT, "data"), names=["Mk01"])[0].P
        for tag, i in (("A", ia), ("B", ib)):
            st = _ev(Pm, e["actions"][i])
            dur = Pm.proc[np.arange(Pm.N), st.op_m]
            e_proc = float(np.sum(dur * Pm.p_proc[st.op_m]) / 60)
            e_idle = 0.0
            for m in range(Pm.M):
                on = st.op_m == m
                if on.any():
                    e_idle += ((st.op_e[on].max() - st.op_s[on].min()) - dur[on].sum()) * Pm.p_idle[m] / 60
            ratio = np.mean([dur[o] * Pm.p_proc[st.op_m[o]] /
                             np.min(np.where(Pm.compat[o], Pm.proc[o] * Pm.p_proc, np.inf)) for o in range(Pm.N)])
            put(f"ex{tag}Eproc", e_proc, "{:.0f}"); put(f"ex{tag}Eidle", e_idle, "{:.0f}")
            put(f"ex{tag}Ratio", 100 * (ratio - 1), "{:.1f}")

# ------------------------------------------------------------------ supplementary experiments
import pickle as _pk  # noqa: E402

cred_p = os.path.join(ROOT, "results", "extra", "credibility.pkl")
if os.path.exists(cred_p):
    cr = _pk.load(open(cred_p, "rb"))
    cruns = sorted({k[0] for k in cr if k[0] != "PWG"})
    cbetas = sorted({k[1] for k in cr})
    cred = {}
    for b in cbetas:
        keys = [(r, b, i) for r in cruns for i in insts if (r, b, i) in cr]
        row = {d: 100 * float(np.mean([np.mean(cr[k]["feas"][d]) for k in keys]))
               for d in ("triangular", "lognormal", "uniform")}
        row["gate_viol"] = 100 * float(np.mean([np.mean(cr[k]["viol"] > 1e-9) for k in keys]))
        row["F"] = np.mean([np.mean(cr[k]["F"], axis=0) / np.mean(cr[(k[0], 0.5, k[2])]["F"], axis=0)
                            for k in keys if (k[0], 0.5, k[2]) in cr], axis=0).tolist()
        pk_ = [("PWG", b, i) for i in insts if ("PWG", b, i) in cr]
        row["pwg_triangular"] = 100 * float(np.mean([np.mean(cr[k]["feas"]["triangular"]) for k in pk_])) if pk_ else np.nan
        cred[b] = row
    json.dump({str(b): v for b, v in cred.items()}, open(os.path.join(SUM, "credibility.json"), "w"), indent=1)
    # smallest beta from which the Monte-Carlo feasibility is >= 99.9 % under every sampling model
    ok_b = [b for b in cbetas if all(cred[bb][d] >= 99.9 for bb in cbetas if bb >= b
                                     for d in ("triangular", "lognormal", "uniform"))]
    put("credBetaSafe", f"{min(ok_b):.1f}" if ok_b else "--")
    if ok_b:
        put("credMinSafe", min(cred[bb][d] for bb in cbetas if bb >= min(ok_b)
                               for d in ("triangular", "lognormal", "uniform")), "{:.2f}")
    for b, tag in ((0.5, "Half"), (0.6, "Six"), (0.7, "Seven"), (0.9, "Nine"), (0.95, "NineFive"), (0.99, "NineNine")):
        if b in cred:
            put(f"credTri{tag}", cred[b]["triangular"], "{:.1f}")
            put(f"credLogn{tag}", cred[b]["lognormal"], "{:.1f}")
            put(f"credUnif{tag}", cred[b]["uniform"], "{:.1f}")
            put(f"credPwgTri{tag}", cred[b]["pwg_triangular"], "{:.1f}")
            put(f"credGateViol{tag}", cred[b]["gate_viol"], "{:.1f}")
            for k, on in enumerate(("Mk", "Cost", "Co", "Risk")):
                put(f"cred{on}{tag}", 100 * (cred[b]["F"][k] - 1), "{:+.1f}")

pv_p = os.path.join(ROOT, "results", "extra", "pv.pkl")
if os.path.exists(pv_p):
    pvr = _pk.load(open(pv_p, "rb"))
    out_pv = {}
    for phi in sorted({k[1] for k in pvr}):
        red_c, red_e, lvl_c = [], [], []
        for (r, ph, inst), v in pvr.items():
            if ph != phi:
                continue
            F = v["F"][v["viol"] <= 1e-9]
            if len(F) == 0:
                continue
            i_mk = int(np.argmin(F[:, 0]))
            red_c.append(100 * (1 - F[:, 1].min() / F[i_mk, 1]))
            red_e.append(100 * (1 - F[:, 2].min() / F[i_mk, 2]))
            lvl_c.append(F[:, 1].min())
        out_pv[phi] = dict(cost_red=float(np.mean(red_c)), co2_red=float(np.mean(red_e)),
                           min_cost=float(np.mean(lvl_c)))
    json.dump({str(k): v for k, v in out_pv.items()}, open(os.path.join(SUM, "pv.json"), "w"), indent=1)
    for phi, tag in ((0.0, "Zero"), (0.25, "Quarter"), (0.5, "Half"), (1.0, "One")):
        if phi in out_pv:
            put(f"pvCostRed{tag}", out_pv[phi]["cost_red"], "{:.1f}")
            put(f"pvCoRed{tag}", out_pv[phi]["co2_red"], "{:.1f}")
    if 0.0 in out_pv and 1.0 in out_pv:
        put("pvMinCostDrop", 100 * (1 - out_pv[1.0]["min_cost"] / out_pv[0.0]["min_cost"]), "{:.1f}")

sc_p = os.path.join(ROOT, "results", "extra", "scale.pkl")
if os.path.exists(sc_p):
    scr = _pk.load(open(sc_p, "rb"))
    big = max(scr, key=lambda r: r["N"])
    put("scaleNmax", str(big["N"]))
    put("scaleSweepMax", big["policy_wall"], "{:.1f}")
    put("scalePerPrefMax", big["per_pref"], "{:.2f}")
    put("scaleDecodeMax", 50000 * big["decode_sec"], "{:.1f}")
    small = min(scr, key=lambda r: r["N"])
    put("scaleNmin", str(small["N"]))
    put("scaleSweepMin", small["policy_wall"], "{:.1f}")
    # log-log slope of the policy sweep time in N
    Ns = np.log([r["N"] for r in scr]); Ts = np.log([r["policy_wall"] for r in scr])
    put("scaleSlope", float(np.polyfit(Ns, Ts, 1)[0]), "{:.2f}")

sv_p = os.path.join(SUM, "seeding_validation.json")
if os.path.exists(sv_p):
    sv = json.load(open(sv_p))
    put("seedk", str(sv["best_k"]))
    put("seedgain", 100 * (sv["mean_val_hv"][str(sv["best_k"])] / sv["mean_val_hv"]["0"] - 1), "{:.1f}")
    put("seedlist", ", ".join(f"{k}: {v:.2f}" for k, v in sv["mean_val_hv"].items()))

# ------------------------------------------------------------------ audit diagnostics (LOMO ref, epsilon+, correlations, seed-level SDs, MC CI)
from scipy.stats import spearmanr as _spearmanr, beta as _beta_dist  # noqa: E402
lomo_hv, ind_hv, eps_ind = {m: [] for m in ["DP-PreCo-GRL", "LS-GRL", "NSGA-II", "NSGA-III", "PWG", "Rules"]}, {m: [] for m in ["DP-PreCo-GRL", "LS-GRL", "NSGA-II", "NSGA-III", "PWG", "Rules"]}, {m: [] for m in ["DP-PreCo-GRL", "LS-GRL", "NSGA-II", "NSGA-III", "PWG", "Rules"]}
corrs = []
for inst in insts:
    runs_i, _ = method_runs(base, pol, inst, MAIN, hyb)
    R_full, id_full, nad_full = refs[inst]
    if len(R_full) > 3:
        C = np.zeros((4, 4))
        for ii in range(4):
            for jj in range(4):
                C[ii, jj] = _spearmanr(R_full[:, ii], R_full[:, jj]).statistic
        corrs.append(C)
    if inst in long_base:
        R_long, id_l, nad_l = reference({"_long": [r["F"][r["G"] <= 1e-9] for r in long_base[inst]["moea"]["nsga2"]]})
        ind_l, _ = indicators(runs_i, R_long, id_l, nad_l)
        for m in ind_hv:
            ind_hv[m].append(float(np.mean(ind_l[m]["hv"])))
    Rn = _nmr(R_full, id_full, nad_full)
    for m in lomo_hv:
        pool_m = {k: v for k, v in runs_i.items() if m not in k}
        Rm, idm, nadm = reference(pool_m)
        ind_m, _ = indicators({m: runs_i[m]}, Rm, idm, nadm)
        lomo_hv[m].append(float(np.mean(ind_m[m]["hv"])))
        eps_vals = []
        for F in runs_i[m]:
            if len(F) == 0:
                eps_vals.append(1.0)
            else:
                Fn = _nmr(_ndr(F), id_full, nad_full)
                diff = Fn[None, :, :] - Rn[:, None, :]
                eps_vals.append(float(np.max(np.min(np.max(diff, axis=2), axis=1))))
        eps_ind[m].append(float(np.mean(eps_vals)))

for m, a in (("DP-PreCo-GRL", "Preco"), ("LS-GRL", "Ls"), ("NSGA-II", "Nsgaii"), ("NSGA-III", "Nsgaiii"), ("PWG", "Pwg"), ("Rules", "Rules")):
    put(f"hvLomo{a}", np.mean(lomo_hv[m]))
    if ind_hv[m]:
        put(f"hvIndRef{a}", np.mean(ind_hv[m]))
    put(f"eps{a}", np.mean(eps_ind[m]))
    n_runs = len(table[insts[0]][m]["hv"])
    if n_runs > 1:
        s_means = [np.mean([table[i][m]["hv"][r] for i in insts]) for r in range(n_runs)]
        put(f"hvSeedSd{a}", np.std(s_means, ddof=1))

if corrs:
    Cm = np.mean(corrs, axis=0)
    put("corrCostCo", Cm[1, 2], "{:+.2f}")
    put("corrMkCost", Cm[0, 1], "{:+.2f}")
    put("corrMkCo", Cm[0, 2], "{:+.2f}")
    put("corrCostRisk", Cm[1, 3], "{:+.2f}")
    put("corrCoRisk", Cm[2, 3], "{:+.2f}")

from morl_fjsp.stats import cluster_bootstrap, mc_summary, paired_bootstrap_diff, rule_of_three_upper  # noqa: E402


if os.path.exists(cred_p):
    # Monte-Carlo feasibility of the schedules of one policy (preco_s0): 12 instances x 84 preferences
    # x 200 independent duration draws.  feas[model][schedule] = k/200 exactly, so counts are exact.
    cr = _pk.load(open(cred_p, "rb"))
    MODELS = (("triangular", "Tri", "Triangular"), ("lognormal", "Logn", "Lognormal"),
              ("uniform", "Unif", "Uniform"))
    ci_out, rows, inst_rows = {}, [], {}
    for beta_, btag in ((0.7, "Seven"), (0.9, "Nine")):
        keys_i = [i for i in insts if ("preco_s0", beta_, i) in cr]
        put(f"credDistinct{btag}", _cnt(sum(len(np.unique(np.round(cr[("preco_s0", beta_, i)]["F"], 9), axis=0))
                                            for i in keys_i)))
        for mname, mtag, mlab in MODELS:
            K = [np.rint(cr[("preco_s0", beta_, i)]["feas"][mname] * 200).astype(int) for i in keys_i]
            s_ = mc_summary(K, 200)
            e_lo, e_hi = cluster_bootstrap(K, 200, "exec", seed=1)
            t_lo, t_hi = cluster_bootstrap(K, 200, "strict", seed=2)
            ci_out[f"{beta_}_{mname}"] = dict(s_, exec_ci_cluster=[e_lo, e_hi], strict_ci_cluster=[t_lo, t_hi])
            t = f"{btag}{mtag}"
            put(f"credFail{t}", _cnt(s_["exec_fail"]))
            put(f"credRate{t}", 100 * s_["exec_rate"], "{:.4f}")
            put(f"credExecLo{t}", 100 * e_lo, "{:.3f}"); put(f"credExecHi{t}", 100 * e_hi, "{:.3f}")
            put(f"credExecCpLo{t}", 100 * s_["exec_ci_iid"][0], "{:.3f}")
            put(f"credExecCpHi{t}", 100 * s_["exec_ci_iid"][1], "{:.3f}")
            put(f"credStrictOk{t}", _cnt(s_["strict_ok"]))
            put(f"credStrictRate{t}", 100 * s_["strict_rate"], "{:.1f}")
            put(f"credStrictCpLo{t}", 100 * s_["strict_ci_iid"][0], "{:.1f}")
            put(f"credStrictCpHi{t}", 100 * s_["strict_ci_iid"][1], "{:.1f}")
            put(f"credStrictLo{t}", 100 * t_lo, "{:.1f}"); put(f"credStrictHi{t}", 100 * t_hi, "{:.1f}")
            put(f"credWorstInst{t}", 100 * s_["worst_instance"], "{:.2f}")
            put(f"credWorstSched{t}", 100 * s_["worst_schedule"], "{:.1f}")
            put(f"credInstFail{t}", str(s_["n_inst_with_failure"]))
            rows.append((beta_, mlab, s_, (e_lo, e_hi), (t_lo, t_hi)))
            inst_rows[(beta_, mname)] = {i: float(k.sum() / (len(k) * 200)) for i, k in zip(keys_i, K)}
    # failed executions for every beta of the sweep (the policy is re-run at each beta)
    for beta_, btag in ((0.5, "Half"), (0.6, "Six"), (0.7, "Seven"), (0.8, "Eight"), (0.9, "Nine"),
                        (0.95, "NineFive"), (0.99, "NineNine")):
        for mname, mtag, mlab in MODELS:
            ks_ = [np.rint(cr[("preco_s0", beta_, i)]["feas"][mname] * 200).astype(int) for i in insts]
            put(f"credFailAll{btag}{mtag}", _cnt(mc_summary(ks_, 200)["exec_fail"]))
    put("credSchedulesPerBeta", _cnt(rows[0][2]["n_sched"]))
    put("credTrialsPerModel", _cnt(rows[0][2]["n_exec"]))
    put("credRuleOfThree", 100 * rule_of_three_upper(len(insts)), "{:.0f}")
    json.dump(ci_out, open(os.path.join(SUM, "credibility_ci.json"), "w"), indent=1)
    tl = ["\\begin{table*}[htbp]", "\\centering",
          "\\caption{Monte-Carlo feasibility of one policy (1{,}008 schedules with 200 draws each, " + "\\credDistinctSeven{} distinct at $\\beta=0.7$). "
          "Execution level: share of feasible executions with a 95\\% cluster-bootstrap interval. Schedule level: schedules feasible in "
          "all draws with Clopper-Pearson (CP) and bootstrap intervals.}",
          "\\label{tab:cred}", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
          "\\begin{tabular}{clccccc}", "\\toprule",
          "$\\beta$ & Model & Failed / executions & Execution level [\\%] (boot.\\ 95\\% CI) & Worst inst. / sched. [\\%] & "
          "All-200-feasible schedules & CP / boot.\\ 95\\% CI [\\%] \\\\", "\\midrule"]
    for beta_, mlab, s_, (e_lo, e_hi), (t_lo, t_hi) in rows:
        tl.append(
            f"{beta_} & {mlab} & {_cnt(s_['exec_fail'])} / {_cnt(s_['n_exec'])} & "
            f"{100 * s_['exec_rate']:.4f} [{100 * e_lo:.3f}, {100 * e_hi:.3f}] & "
            f"{100 * s_['worst_instance']:.2f} / {100 * s_['worst_schedule']:.1f} & "
            f"{_cnt(s_['strict_ok'])} / {_cnt(s_['n_sched'])} ({100 * s_['strict_rate']:.1f}\\%) & "
            f"[{100 * s_['strict_ci_iid'][0]:.1f}, {100 * s_['strict_ci_iid'][1]:.1f}] / [{100 * t_lo:.1f}, {100 * t_hi:.1f}] \\\\")
    tl += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(os.path.join(GEN, "tab_cred.tex"), "w").write("\n".join(tl) + "\n")
    ti = ["\\begin{table}[htbp]", "\\centering",
          "\\caption{Execution-level feasibility per instance (\\%, 16{,}800 executions each).}", "\\label{tab:credinst}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
          "\\begin{tabular}{lcccc}", "\\toprule",
          "Inst. & $\\beta{=}0.7$ tri. & $\\beta{=}0.7$ unif. & $\\beta{=}0.9$ tri. & $\\beta{=}0.9$ unif. \\\\", "\\midrule"]
    for i in insts:
        ti.append(f"\\texttt{{{i}}} & " + " & ".join(f"{100 * inst_rows[(b_, m_)][i]:.2f}"
                  for b_, m_ in ((0.7, "triangular"), (0.7, "uniform"), (0.9, "triangular"), (0.9, "uniform"))) + " \\\\")
    ti += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(GEN, "tab_cred_inst.tex"), "w").write("\n".join(ti) + "\n")

# ------------------------------------------------------------------ Monte-Carlo feasibility for further trained policies
cs_p = os.path.join(ROOT, "results", "extra", "credibility_seeds.pkl")
if os.path.exists(cs_p):
    cs = _pk.load(open(cs_p, "rb"))
    cs_runs = [r for r in dict.fromkeys(k[0] for k in cs)
               if all((r, b_, i) in cs for b_ in (0.7, 0.9) for i in insts)]
    tot = {(b_, m_): 0 for b_ in (0.7, 0.9) for m_, _, _ in MODELS}
    per = {}
    for r in cs_runs:
        for b_ in (0.7, 0.9):
            for m_, mt, _ in MODELS:
                K = [np.rint(cs[(r, b_, i)]["feas"][m_] * 200).astype(int) for i in insts]
                sm = mc_summary(K, 200)
                per[(r, b_, m_)] = sm
                tot[(b_, m_)] += sm["exec_fail"]
    n_pol = len(cs_runs)
    put("credSeedsN", str(n_pol))
    if "preco_s0" in cs_runs:                      # same policy as Table tab:cred, independent Monte-Carlo draws
        put("credSeedsZeroSevenUnif", _cnt(per[("preco_s0", 0.7, "uniform")]["exec_fail"]))
    put("credSeedsExecTotal", _cnt(n_pol * 201600))
    for b_, bt in ((0.7, "Seven"), (0.9, "Nine")):
        for m_, mt, _ in MODELS:
            fails = [per[(r, b_, m_)]["exec_fail"] for r in cs_runs]
            put(f"credSeeds{bt}{mt}Total", _cnt(tot[(b_, m_)]))
            put(f"credSeeds{bt}{mt}Max", _cnt(max(fails)))
            put(f"credSeeds{bt}{mt}Min", _cnt(min(fails)))
            put(f"credSeeds{bt}{mt}Rate", 100 * (1 - tot[(b_, m_)] / (n_pol * 201600)), "{:.4f}")
            put(f"credSeeds{bt}{mt}Round", 100 * (1 - tot[(b_, m_)] / (n_pol * 201600)), "{:.2f}")
            put(f"credSeeds{bt}{mt}PolFail", str(sum(f > 0 for f in fails)))
            put(f"credSeeds{bt}{mt}StrictMin", _cnt(min(per[(r, b_, m_)]["strict_ok"] for r in cs_runs)))
            put(f"credSeeds{bt}{mt}StrictMinRate", 100 * min(per[(r, b_, m_)]["strict_rate"] for r in cs_runs), "{:.1f}")
            put(f"credSeeds{bt}{mt}StrictMaxRate", 100 * max(per[(r, b_, m_)]["strict_rate"] for r in cs_runs), "{:.1f}")
            put(f"credSeeds{bt}{mt}WorstSched", 100 * min(per[(r, b_, m_)]["worst_schedule"] for r in cs_runs), "{:.1f}")
            put(f"credSeeds{bt}{mt}WorstInst", 100 * min(per[(r, b_, m_)]["worst_instance"] for r in cs_runs), "{:.2f}")
    # policy-level cluster bootstrap over the trained policies (policies, then instances) of the pooled execution rate
    _rng = np.random.default_rng(5)
    for m_, mt, _ in MODELS:
        rate = np.array([[per[(r, 0.7, m_)]["inst_rate"][k] for k in range(len(insts))] for r in cs_runs])   # (policy, instance)
        bs = [rate[_rng.integers(0, n_pol, n_pol)][:, _rng.integers(0, len(insts), len(insts))].mean() for _ in range(10000)]
        put(f"credSeedsSeven{mt}Lo", 100 * np.quantile(bs, 0.025), "{:.3f}")
        put(f"credSeedsSeven{mt}Hi", 100 * np.quantile(bs, 0.975), "{:.3f}")
    tl = ["\\begin{table*}[htbp]", "\\centering",
          "\\caption{Failed executions out of 201{,}600 per cell for ten trained policies (protocol of the main text). "
          "Last column: share of schedules feasible in all 200 uniform draws at $\\beta=0.7$.}",
          "\\label{tab:credseeds}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.2pt}",
          "\\begin{tabular}{lcccccccc}", "\\toprule",
          " & \\multicolumn{4}{c}{$\\beta=0.7$} & \\multicolumn{3}{c}{$\\beta=0.9$} \\\\", "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}",
          "Policy & tri. & logn. & unif. & all-feas. [\\%] & tri. & logn. & unif. \\\\", "\\midrule"]
    for r in cs_runs:
        lab = r.replace("preco_s", "PreCo s").replace("linear_s", "LS s")
        tl.append(f"{lab} & " + " & ".join([_cnt(per[(r, 0.7, m_)]["exec_fail"]) for m_, _, _ in MODELS]) +
                  f" & {100 * per[(r, 0.7, 'uniform')]['strict_rate']:.1f} & " +
                  " & ".join(_cnt(per[(r, 0.9, m_)]["exec_fail"]) for m_, _, _ in MODELS) + " \\\\")
    tl += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(os.path.join(GEN, "tab_cred_seeds.tex"), "w").write("\n".join(tl) + "\n")

# ------------------------------------------------------------------ sigma floor and expert replay (audit E4, E8)
_sg = os.path.join(SUM, "sigma_guard.json")
if os.path.exists(_sg):
    _g = json.load(open(_sg))
    put("sigmaPairsBench", _cnt(_g["benchmarks"]["n_pairs"]))
    put("sigmaFloorBench", _cnt(_g["benchmarks"]["floor_active_pairs"]))
    put("sigmaPairsTrain", _cnt(_g["training"]["n_pairs"]))
    put("sigmaFloorTrain", _cnt(_g["training"]["floor_active_pairs"]))
    put("sigmaZeroTrain", _cnt(_g["training"]["zero_range_pairs"]))
_er = os.path.join(SUM, "expert_replay.json")
if os.path.exists(_er):
    _e = json.load(open(_er))
    put("expertReplayN", _cnt(_e["schedules"]))

# ------------------------------------------------------------------ hierarchical bootstrap and seed-level tests
def _spec(lab, kind):
    if kind == "crossed":         # trained seeds evaluated on every instance
        return dict(kind="crossed", M=np.array([[table[i][lab]["hv"][r] for r in range(len(table[i][lab]["hv"]))]
                                                for i in insts]))
    return dict(kind="nested", L=[np.asarray(table[i][lab]["hv"]) for i in insts])


for tag, a_lab, a_kind, b_lab, b_kind in (
        ("NsgaiiPreco", "NSGA-II", "nested", "DP-PreCo-GRL", "crossed"),
        ("NsgaiiiPreco", "NSGA-III", "nested", "DP-PreCo-GRL", "crossed"),
        ("NsgaiiLs", "NSGA-II", "nested", "LS-GRL", "crossed"),
        ("PrecoLs", "DP-PreCo-GRL", "crossed", "LS-GRL", "crossed")):
    d_, lo_, hi_ = paired_bootstrap_diff(_spec(a_lab, a_kind), _spec(b_lab, b_kind), n_boot=10000, seed=7)
    put(f"hbDiff{tag}", d_); put(f"hbLo{tag}", lo_); put(f"hbHi{tag}", hi_)
_sm = {m: [np.mean([table[i][m]["hv"][r] for i in insts]) for r in range(len(table[insts[0]][m]["hv"]))]
       for m in ("DP-PreCo-GRL", "LS-GRL", "NSGA-II", "NSGA-III")}
put("pMwNsgaiiPreco", mannwhitney(_sm["NSGA-II"], _sm["DP-PreCo-GRL"]), "p")
put("pMwNsgaiiiPreco", mannwhitney(_sm["NSGA-III"], _sm["DP-PreCo-GRL"]), "p")
put("pMwPrecoLs", mannwhitney(_sm["DP-PreCo-GRL"], _sm["LS-GRL"]), "p")

with open(os.path.join(GEN, "numbers.tex"), "w") as f:
    for k, v in NUM.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
json.dump(NUM, open(os.path.join(SUM, "numbers.json"), "w"), indent=1)
print(json.dumps(NUM, indent=1))
print("overall:", json.dumps(overall, indent=1, default=float))

