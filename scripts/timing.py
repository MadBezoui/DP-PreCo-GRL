"""Controlled runtime measurement (run alone on an otherwise idle machine, 1 thread):
policy sweep (84 preferences), a single-preference policy rollout, PWG sweep, and one
NSGA-II / NSGA-III run (50,000 evaluations) per benchmark.  Output:
results/summary/timing.json"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.env import StaticGraph  # noqa: E402
from morl_fjsp.metrics import eval_preferences  # noqa: E402
from morl_fjsp.moea import run_moea  # noqa: E402
from morl_fjsp.policy_io import load_policy, sweep  # noqa: E402
from morl_fjsp.rules import run_rule  # noqa: E402
from morl_fjsp.scenarios import benchmark_scenarios  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)
runs = sys.argv[1:] or ["preco_s0", "linear_s0"]
W = eval_preferences()
out = {}
for sc in benchmark_scenarios(os.path.join(ROOT, "data")):
    G = StaticGraph(sc.P)
    rec = {}
    for run in runs:
        net, cfg = load_policy(os.path.join(ROOT, "results", "runs", run))
        sweep(net, cfg, sc, W[:2], static=G)                    # warm-up
        _, _, _, wall = sweep(net, cfg, sc, W, static=G)
        _, _, _, wall1 = sweep(net, cfg, sc, W[40:41], static=G)
        rec[run] = dict(sweep84=wall, single=wall1)
    t = time.perf_counter()
    for w in W:
        run_rule(sc.P, "PWG", w=w, scale=sc.span)
    rec["PWG"] = dict(sweep84=time.perf_counter() - t)
    for algo in ("nsga2", "nsga3"):
        r = run_moea(sc.P, algo, pop=100, n_gen=500, seed=0)
        rec[algo] = dict(wall=r["wall"], evals=r["n_evals"])
    out[sc.name] = rec
    print(sc.name, json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "summary", "timing.json"), "w"), indent=1)
