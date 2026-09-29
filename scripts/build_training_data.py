"""Build the leak-free training pool, the validation set and the offline expert
archive (NSGA-II fronts of *training* scenarios only).

Outputs (results/cache/):
  pool.pkl       256 synthetic scenarios (instance x synthetic day x start x PV size)
  val.pkl        16 held-out synthetic validation scenarios (checkpoint selection)
  expert.pkl     per-scenario expert archive: list of (F, actions)
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.metrics import nondominated  # noqa: E402
from morl_fjsp.moea import run_moea  # noqa: E402
from morl_fjsp.scenarios import make_training_pool  # noqa: E402
from morl_fjsp.sim import actions_of_state, evaluate_actions  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", type=int, default=256)
ap.add_argument("--val", type=int, default=16)
ap.add_argument("--out", default="results/cache")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

t0 = time.time()
pool = make_training_pool(args.pool, seed=2024)
val = make_training_pool(args.val, seed=777)          # different generator seed
print(f"pool built in {time.time() - t0:.1f}s", flush=True)

expert = {}
for i, s in enumerate(pool):
    res = run_moea(s.P, "nsga2", pop=60, n_gen=60, seed=i)
    prob = res["problem"]
    items = [(F, prob.actions(x)) for F, x in zip(res["F"], res["X"])]
    nd = nondominated(np.array([f for f, _ in items]))
    items = [(F, a) for F, a in items if any(np.allclose(F, g) for g in nd)]
    # canonical replay order: start time (reproduces the schedule exactly)
    items = [(F, actions_of_state(evaluate_actions(s.P, a))) for F, a in items]
    expert[i] = items
    if i % 32 == 0:
        print(f"expert {i}/{len(pool)} |front|={len(items)} t={time.time() - t0:.0f}s", flush=True)

pickle.dump(pool, open(os.path.join(args.out, "pool.pkl"), "wb"))
pickle.dump((val, None), open(os.path.join(args.out, "val.pkl"), "wb"))
pickle.dump(expert, open(os.path.join(args.out, "expert.pkl"), "wb"))
print(f"done in {time.time() - t0:.0f}s", flush=True)
