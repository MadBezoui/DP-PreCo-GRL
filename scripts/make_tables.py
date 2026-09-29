"""Tables and numeric macros that depend only on data/instances (not on results)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.energy import load_energy_charts_day  # noqa: E402
from morl_fjsp.instance import load_benchmarks  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "outputs", "generated")
os.makedirs(GEN, exist_ok=True)

insts = load_benchmarks(os.path.join(ROOT, "data", "fjsp"))
def _load_meta(path):
    """The metadata file (SchedulingLab/fjsp-instances) was truncated at download time;
    parse every complete record and ignore the rest."""
    import re
    out = {}
    for rec in re.findall(r"\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}", open(path).read()):
        try:
            d = json.loads(rec)
        except json.JSONDecodeError:
            continue
        if "name" in d:
            out[d["name"]] = d
    return out


meta = _load_meta(os.path.join(ROOT, "data", "fjsp", "instances_metadata.json"))
rows = []
for i in insts:
    s = i.summary()
    m = meta.get(i.name.lower(), {})
    bks = m.get("optimum") or (m.get("bounds", {}) or {}).get("upper")
    rows.append(f"\\texttt{{{i.name}}} & {s['jobs']} & {s['machines']} & {s['ops']} & "
                f"{s['flexibility']:.2f} & {s['edges']} & {bks if bks else '-'} & {i.tau} & "
                f"{i.overtime_L / 60:.1f} & {np.mean(i.capacity) / 60:.1f} \\\\")
tab = r"""\begin{table}[htbp]
\centering
\caption{Benchmark instances: jobs $J$, machines $M$, operations $N$, flexibility (compatible machines per operation), graph edges, best known makespan (BKS, time units), time factor $\tau$ (min/unit), overload threshold $L$ and mean capacity $\bar C$ (h).}
\label{tab:instances}
\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{lrrrrrrrrr}
\toprule
Inst. & $J$ & $M$ & $N$ & Flex. & Edges & BKS & $\tau$ & $L$ & $\bar C$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(GEN, "tab_instances.tex"), "w").write(tab)

day = load_energy_charts_day(os.path.join(ROOT, "data", "energy"))
pw = json.load(open(os.path.join(ROOT, "data", "energy", "smard_de_lu_power.json")))
solar = [x for x in pw["production_types"] if x["name"] == "Solar"][0]["data"]
macros = {
    "numpricemin": f"{day.price.min():.2f}",
    "numpricemax": f"{day.price.max():.2f}",
    "numpricemean": f"{day.price.mean():.1f}",
    "numcomin": f"{day.co2.min():.0f}",
    "numcomax": f"{day.co2.max():.0f}",
    "numsolarpeak": f"{max(v for v in solar if v is not None) / 1000:.1f}",
}
# calibration statistics: natural (ungated) load factors of the dispatching rules
from morl_fjsp.fuzzy import credibility_quantile  # noqa: E402
from morl_fjsp.instance import CAPACITY_FACTOR  # noqa: E402
from morl_fjsp.rules import RULES, run_rule  # noqa: E402
from morl_fjsp.sim import Problem  # noqa: E402
med_viol, best_ok, best_max = 0, 0, 0.0
for inst in insts:
    Pn = Problem(inst, day, use_capacity=False)
    facs = []
    for r in RULES:
        st = run_rule(Pn, r)
        q = max(credibility_quantile(st.WA[m], st.WB[m], st.WC[m], 0.9) for m in range(inst.n_machines))
        facs.append(q / inst.meta["lp_max_load_q"])
    med_viol += int(np.median(facs) > CAPACITY_FACTOR)
    best_ok += int(min(facs) <= CAPACITY_FACTOR)
    best_max = max(best_max, min(facs))
macros.update(numcapmedianviol=str(med_viol), numcapbestok=str(best_ok),
              numcapbestmax=f"{best_max:.2f}", numcapfactor=f"{CAPACITY_FACTOR:.1f}")
with open(os.path.join(GEN, "numbers_data.tex"), "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(tab)
print(macros)
