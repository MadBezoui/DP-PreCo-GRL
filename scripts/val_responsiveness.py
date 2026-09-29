"""Preference responsiveness on the 16 synthetic validation scenarios (training distribution):
per objective, Spearman's rho between the weight w_k and the obtained f_k over the 84
evaluation preferences, for the main policies and for PWG.  Compared in the paper with the
same statistic on the benchmarks (real test day).  Output: results/summary/val_responsiveness.json"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.analysis import responsiveness  # noqa: E402
from morl_fjsp.metrics import eval_preferences  # noqa: E402
from morl_fjsp.policy_io import load_policy, sweep  # noqa: E402
from morl_fjsp.rules import run_rule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)
GROUPS = json.load(open(os.path.join(ROOT, "configs", "method_groups.json")))
val, _ = pickle.load(open(os.path.join(ROOT, "results", "cache", "val.pkl"), "rb"))
W = eval_preferences()
out, per_run = {}, {}
for lab, names in GROUPS["main"].items():
    rows = []
    for n in names:
        if not os.path.exists(os.path.join(ROOT, "results", "runs", n, "model.pt")):
            continue
        net, cfg = load_policy(os.path.join(ROOT, "results", "runs", n))
        r = []
        for s in val:
            F, V, _, _ = sweep(net, cfg, s, W)
            r.append(responsiveness(F, W, V))
        per_run[n] = np.nanmean(r, axis=0).tolist()
        rows.append(np.nanmean(r, axis=0))
        print(n, np.round(rows[-1], 2), flush=True)
    if rows:
        out[lab] = np.nanmean(rows, axis=0).tolist()
r = []
for s in val:
    sts = [run_rule(s.P, "PWG", w=w, scale=s.span) for w in W]
    r.append(responsiveness(np.array([st.objectives() for st in sts]), W,
                            np.array([st.violation() for st in sts])))
out["PWG"] = np.nanmean(r, axis=0).tolist()
print("PWG", np.round(out["PWG"], 2))
json.dump(dict(out, per_run=per_run), open(os.path.join(ROOT, "results", "summary",
                                                        "val_responsiveness.json"), "w"), indent=1)
