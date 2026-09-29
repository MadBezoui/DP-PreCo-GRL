"""Reference levels of the validation metric (rule-normalised HV on the 16 validation
scenarios) for PWG, a rule-seeded NSGA-II with 20,000 evaluations, and NSGA-II with the
budget of the expert archive (population 60, 60 generations = 3,600 evaluations), for
Fig. training."""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.metrics import eval_preferences, hypervolume, nondominated, normalise  # noqa: E402
from morl_fjsp.moea import run_moea  # noqa: E402
from morl_fjsp.rules import run_rule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
val, _ = pickle.load(open(os.path.join(ROOT, "results", "cache", "val.pkl"), "rb"))
W = eval_preferences(4, 3, 0.1)
hp, hn, he = [], [], []
for s in val:
    # feasible schedules only, exactly as in the validation metric of the policies
    sts = [run_rule(s.P, "PWG", w=w, scale=s.span) for w in W]
    Fp = np.array([st.objectives() for st in sts if st.violation() <= 1e-9]).reshape(-1, 4)
    hp.append(hypervolume(normalise(nondominated(Fp), s.ideal, s.nadir)) if len(Fp) else 0.0)
    r = run_moea(s.P, "nsga2", pop=100, n_gen=200, seed=0)
    Fn = r["F"][r["G"] <= 1e-9]
    hn.append(hypervolume(normalise(nondominated(Fn), s.ideal, s.nadir)) if len(Fn) else 0.0)
    r = run_moea(s.P, "nsga2", pop=60, n_gen=60, seed=0)          # expert-archive budget
    Fe = r["F"][r["G"] <= 1e-9]
    he.append(hypervolume(normalise(nondominated(Fe), s.ideal, s.nadir)) if len(Fe) else 0.0)
os.makedirs(os.path.join(ROOT, "results", "summary"), exist_ok=True)
json.dump(dict(pwg=float(np.mean(hp)), nsga2=float(np.mean(hn)), nsga2_expert=float(np.mean(he)),
               pwg_all=hp, nsga2_all=hn, nsga2_expert_all=he),
          open(os.path.join(ROOT, "results", "summary", "val_reference.json"), "w"), indent=1)
print(np.mean(hp), np.mean(hn), np.mean(he))
