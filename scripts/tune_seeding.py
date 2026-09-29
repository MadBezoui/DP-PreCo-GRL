"""Select the number of policy schedules injected into the NSGA-II initial population on
the synthetic VALIDATION scenarios (never on the benchmarks).  Output:
results/summary/seeding_validation.json"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.metrics import eval_preferences, hypervolume, nondominated, normalise  # noqa: E402
from morl_fjsp.moea import RandomKeyProblem, encode_actions, run_moea, seeded_population  # noqa: E402
from morl_fjsp.policy_io import load_policy, sweep  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)
run = sys.argv[1] if len(sys.argv) > 1 else "preco_s0"
val, _ = pickle.load(open(os.path.join(ROOT, "results", "cache", "val.pkl"), "rb"))
net, cfg = load_policy(os.path.join(ROOT, "results", "runs", run))
W = eval_preferences()
ks = [0, 10, 20, 40, 60]
res = {k: [] for k in ks}
for s in val:
    F, V, acts, _ = sweep(net, cfg, s, W)
    prob = RandomKeyProblem(s.P)
    keys = np.unique(np.array([encode_actions(prob, a) for a in acts]), axis=0)
    for k in ks:
        hv = []
        for seed in range(2):
            rng = np.random.default_rng(seed)
            X = seeded_population(prob, 100, rng)
            sel = keys[rng.permutation(len(keys))[:k]]
            X[20:20 + len(sel)] = sel
            r = run_moea(s.P, "nsga2", pop=100, n_gen=200, seed=seed, init_X=X)
            Fe = r["F"][r["G"] <= 1e-9]
            hv.append(hypervolume(normalise(nondominated(Fe), s.ideal, s.nadir)) if len(Fe) else 0.0)
        res[k].append(float(np.mean(hv)))
    print(s.name, {k: round(v[-1], 3) for k, v in res.items()}, flush=True)
summary = {k: float(np.mean(v)) for k, v in res.items()}
best = max(summary, key=summary.get)
json.dump(dict(run=run, mean_val_hv=summary, per_scenario=res, best_k=best),
          open(os.path.join(ROOT, "results", "summary", "seeding_validation.json"), "w"), indent=1)
print("mean", summary, "best k", best)
