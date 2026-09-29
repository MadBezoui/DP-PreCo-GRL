"""How often is a rule-portfolio range degenerate, i.e. the floor of sigma_k (Scenario.span) active?

Benchmarks (real day) and the synthetic training / validation scenarios in results/cache.
Output: results/summary/sigma_guard.json"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.scenarios import benchmark_scenarios  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stats(scs):
    raw = np.array([s.nadir - s.ideal for s in scs])
    span = np.array([s.span for s in scs])
    active = span > raw + 1e-15                     # floor of sigma_k binds
    return dict(n_scenarios=len(scs), n_pairs=int(active.size),
                floor_active_per_objective=active.sum(0).tolist(),
                floor_active_pairs=int(active.sum()),
                zero_range_pairs=int((raw <= 0).sum()))


out = dict(benchmarks=stats(benchmark_scenarios(os.path.join(ROOT, "data"))))
pool = pickle.load(open(os.path.join(ROOT, "results/cache/pool.pkl"), "rb"))
val = pickle.load(open(os.path.join(ROOT, "results/cache/val.pkl"), "rb"))[0]
out["training"], out["validation"] = stats(pool), stats(val)
json.dump(out, open(os.path.join(ROOT, "results", "summary", "sigma_guard.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
