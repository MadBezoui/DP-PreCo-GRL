"""Timing of the per-scenario pre-processing and of cold / cached policy latency, for a full and
a graph-free (no message passing) policy.

A *new* scenario needs (i) the eight-rule pre-pass that provides the objective scale sigma
(morl_fjsp.scenarios.rule_front) and (ii) the instance-level graph tensors (StaticGraph) before
the first rollout; both can be cached for further preferences on the same scenario.  Measured
sequentially on an otherwise idle machine, one CPU thread, median of REP repeats.

Usage: python scripts/timing_coldstart.py [run ...]      (default: preco_s0 nogat_s0)
Output: results/summary/timing_coldstart.json (the platform is recorded)."""
import json
import os
import platform
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.env import StaticGraph  # noqa: E402
from morl_fjsp.metrics import eval_preferences  # noqa: E402
from morl_fjsp.policy_io import load_policy, sweep  # noqa: E402
from morl_fjsp.scenarios import benchmark_scenarios, rule_front  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)
RUNS = sys.argv[1:] or ["preco_s0", "nogat_s0"]
REP = 5
W = eval_preferences()
nets = {r: load_policy(os.path.join(ROOT, "results", "runs", r)) for r in RUNS}
params = {r: int(sum(p.numel() for p in nets[r][0].parameters())) for r in RUNS}
out = dict(platform=platform.platform(), machine=platform.machine(), torch=torch.__version__,
           threads=1, repeats=REP, params=params, instances={})


def wall(f, n=REP):
    """Median wall-clock seconds of f() over n calls."""
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def inner(f, n=REP):
    """Median of the wall clock returned by f() (sweep() measures the rollout itself)."""
    return statistics.median(f() for _ in range(n))


for sc in benchmark_scenarios(os.path.join(ROOT, "data")):
    rec = dict(rules8=wall(lambda: rule_front(sc.P)),
               static_graph=wall(lambda: StaticGraph(sc.P)))
    G = StaticGraph(sc.P)
    for r, (net, cfg) in nets.items():
        sweep(net, cfg, sc, W[:2], static=G)                                    # warm-up
        rec[r] = dict(single_cached=inner(lambda: sweep(net, cfg, sc, W[40:41], static=G)[3]),
                      sweep84_cached=inner(lambda: sweep(net, cfg, sc, W, static=G)[3], n=3))
    out["instances"][sc.name] = rec
    print(sc.name, json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "summary", "timing_coldstart.json"), "w"), indent=1)
