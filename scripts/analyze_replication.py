"""Second-platform replication of the principal ablations (5 seeds per variant).

The runs of results/runs_rep (scripts/train_replication.sh) were trained with a training cache
(scenario pool, validation set, expert archive) regenerated on a second platform and evaluated on
the 12 benchmarks with the real test day (results/policy_eval/replication).  Their HV ratios are
computed against the FROZEN pooled reference front of the main study
(results/summary/reference_fronts.pkl, written by scripts/analyze.py): the replication runs are
not added to the pool, so no number of the main study changes.

Outputs: results/summary/replication.json, outputs/generated/tab_ablation_rep.tex and
outputs/generated/numbers_rep.tex (macros rep*).
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.analysis import feasible, indicators  # noqa: E402
from morl_fjsp.instance import BENCHMARKS  # noqa: E402
from morl_fjsp.latex import merge_macros, pfmt  # noqa: E402
from morl_fjsp.metrics import a12, holm, mannwhitney, wilcoxon  # noqa: E402
from morl_fjsp.stats import paired_bootstrap_diff  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUM = os.path.join(ROOT, "results", "summary")
GEN = os.path.join(ROOT, "outputs", "generated")
refs = pickle.load(open(os.path.join(SUM, "reference_fronts.pkl"), "rb"))

# (run prefix, table label, macro tag, macro of the original two-seed value or None)
VARIANTS = [("rep_preco", "Full DP-PreCo-GRL", "Preco", "ablFull"),
            ("rep_linear", "LS-GRL (linear comb.)", "Ls", "ablLs"),
            ("rep_precoraw", "PreCo-insp., un-normalised weights", "PrecoRaw", None),
            ("rep_nogat", "w/o graph attention (flat encoder)", "Nogat", "ablNogat"),
            ("rep_nofilm", "w/o FiLM", "Nofilm", "ablNofilm"),
            ("rep_nocrit", "w/o criteria head", "Nocrit", "ablNocrit"),
            ("rep_noexpert", "w/o expert pool", "Noexpert", "ablNoexpert")]
insts = [i for i in BENCHMARKS if i in refs]


def evaluate(run_glob, tag_dir):
    """(instances x seeds) matrices of HV ratio and of the infeasible-rollout share."""
    files = sorted(glob.glob(os.path.join(ROOT, "results", "policy_eval", tag_dir, run_glob)))
    HV = np.zeros((len(insts), len(files)))
    IN = np.zeros_like(HV)
    for j, f in enumerate(files):
        res = pickle.load(open(f, "rb"))["results"]
        for i, inst in enumerate(insts):
            R, ideal, nadir = refs[inst]
            e = res[inst]
            ind, _ = indicators({"x": [feasible(e["F"], e["viol"])]}, R, ideal, nadir)
            HV[i, j] = ind["x"]["hv"][0]
            IN[i, j] = float(np.mean(e["viol"] > 1e-9))
    return HV, IN, [os.path.basename(f) for f in files]


hv, infeas, names = {}, {}, {}
for pref, _, _, _ in VARIANTS:
    hv[pref], infeas[pref], names[pref] = evaluate(f"{pref}_s[0-9].pkl", "replication")
orig = {"preco": evaluate("preco_s[0-9].pkl", "main")[0], "linear": evaluate("linear_s[0-9].pkl", "main")[0]}

full = hv["rep_preco"]
others = [v for v in VARIANTS[1:]]
p_inst = holm([wilcoxon(hv[p].mean(1), full.mean(1)) for p, _, _, _ in others])
p_seed = holm([mannwhitney(hv[p].mean(0), full.mean(0)) for p, _, _, _ in others])
res, macros = {}, {}


def put(k, v, fmt="{:.3f}"):
    if not isinstance(v, str) and abs(v) < 5e-4:
        v = 0.0                                   # avoid printing "-0.000"
    macros[k] = v if isinstance(v, str) else fmt.format(v)


for k, (pref, lab, tag, _) in enumerate(VARIANTS):
    sm = hv[pref].mean(0)
    rec = dict(label=lab, n_seeds=int(hv[pref].shape[1]), mean=float(hv[pref].mean()),
               sd_seeds=float(sm.std(ddof=1)), seed_means=sm.tolist(),
               infeas_pct=float(100 * infeas[pref].mean()))
    put(f"rep{tag}", rec["mean"])
    put(f"rep{tag}Sd", rec["sd_seeds"])
    put(f"rep{tag}Infeas", rec["infeas_pct"], "{:.1f}")
    put(f"rep{tag}N", str(rec["n_seeds"]))
    if k > 0:
        d, lo, hi = paired_bootstrap_diff(dict(kind="crossed", M=hv[pref]), dict(kind="crossed", M=full),
                                          n_boot=10000, seed=11)
        rec.update(delta=d, delta_lo=lo, delta_hi=hi, p_inst_holm=float(p_inst[k - 1]),
                   p_seed_holm=float(p_seed[k - 1]), vd=a12(sm, full.mean(0)))
        put(f"rep{tag}Delta", d, "{:+.3f}"); put(f"rep{tag}Lo", lo, "{:+.3f}"); put(f"rep{tag}Hi", hi, "{:+.3f}")
        put(f"rep{tag}Pi", pfmt(rec["p_inst_holm"])); put(f"rep{tag}Ps", pfmt(rec["p_seed_holm"]))
        put(f"rep{tag}Vd", rec["vd"], "{:.2f}")
    res[pref] = rec

# replication of the main comparison across platforms (same frozen reference; 5 vs 5 training seeds)
put("repVsOrigPrecoP", pfmt(mannwhitney(hv["rep_preco"].mean(0), orig["preco"].mean(0))))
put("repVsOrigLsP", pfmt(mannwhitney(hv["rep_linear"].mean(0), orig["linear"].mean(0))))
put("repOrigPreco", float(orig["preco"].mean())); put("repOrigLs", float(orig["linear"].mean()))
# PreCo versus LS-GRL is reported with the LS-GRL row of the table (variant minus full model), one interval only
res["_orig"] = dict(preco=float(orig["preco"].mean()), linear=float(orig["linear"].mean()))
json.dump(res, open(os.path.join(SUM, "replication.json"), "w"), indent=1)

lines = ["\\begin{table*}[htbp]", "\\centering",
         "\\caption{Five-seed replication of the principal ablations on a second platform (HV ratio against the frozen reference "
         "front, mean\\,$\\pm$\\,standard deviation over seeds). Orig.: original two-seed run. Infeas.: share of rollouts with a "
         "forced-fallback violation. Test definitions below the table.}",
         "\\label{tab:ablrep}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{lrrrrrrr}", "\\toprule",
         "Variant & HV ratio $\\uparrow$ & Orig. & $\\Delta$ [95\\% CI] & $A$ & $p_{\\mathrm{s}}$ & $p_{\\mathrm{i}}$ & Infeas. \\\\", "\\midrule"]


def _p(x):
    return f"$\\mathbf{{{pfmt(x)}}}$" if x < 0.05 else f"${pfmt(x)}$"


for k, (pref, lab, tag, om) in enumerate(VARIANTS):
    r = res[pref]
    orig = f"\\{om}{{}}" if om else "-"
    hv_cell = f"{r['mean']:.3f}\\,$\\pm$\\,{r['sd_seeds']:.3f}"
    if k == 0:
        row = f"{lab} & {hv_cell} & {orig} & - & - & - & - & {r['infeas_pct']:.1f}\\% \\\\"
    else:
        row = (f"{lab} & {hv_cell} & {orig} & {r['delta'] + 0.0:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}] "
               f"& {r['vd']:.2f} & {_p(r['p_seed_holm'])} & {_p(r['p_inst_holm'])} & {r['infeas_pct']:.1f}\\% \\\\")
    lines.append(row)
lines += ["\\bottomrule", "\\end{tabular}",
          "\\par\\smallskip",
          "{\\footnotesize Note. $\\Delta$: variant minus full model with a 95\\% percentile-bootstrap interval that resamples instances and seeds. "
          "$A$: Vargha-Delaney effect size across seeds, oriented as variant versus full model ($A<0.5$ favours the full model). "
          "$p_{\\mathrm{s}}$: two-sided Mann-Whitney test over the five seed-level averages. $p_{\\mathrm{i}}$: two-sided paired Wilcoxon test over the "
          "12 per-instance means. Both are Holm-corrected over the six comparisons shown. Bold: $p<0.05$.}",
          "\\end{table*}"]
open(os.path.join(GEN, "tab_ablation_rep.tex"), "w").write("\n".join(lines) + "\n")
merge_macros(os.path.join(GEN, "numbers_rep.tex"), macros)
print(json.dumps({k: {a: b for a, b in v.items() if a != 'seed_means'} for k, v in res.items()}, indent=1))
