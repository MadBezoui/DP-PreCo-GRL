"""Zero-shot evaluation of trained policies on the 12 public benchmarks.

For every run in results/runs (or --runs), roll out the policy greedily for the
84 interior Das-Dennis preferences (one batched pass per instance, 1 CPU thread).
Output: results/policy_eval/<tag>/<run>.pkl
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.env import StaticGraph  # noqa: E402
from morl_fjsp.metrics import eval_preferences  # noqa: E402
from morl_fjsp.model import PolicyNet  # noqa: E402
from morl_fjsp.scenarios import benchmark_scenarios  # noqa: E402
from morl_fjsp.train import TrainConfig, rollout  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="*", default=None)
ap.add_argument("--tag", default="main")
ap.add_argument("--runs-dir", default="results/runs", help="folder with the trained runs")
ap.add_argument("--start", type=int, default=360)
ap.add_argument("--pv", type=float, default=0.5)
ap.add_argument("--beta", type=float, default=0.9)
ap.add_argument("--samples", type=int, default=0, help="extra sampled rollouts per preference")
ap.add_argument("--keep-actions", nargs="*", default=["Mk01", "Mk06", "Mk10"])
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)

runs = args.runs or sorted(os.path.basename(os.path.dirname(p))
                           for p in glob.glob(os.path.join(ROOT, args.runs_dir, "*/model.pt")))
scs = benchmark_scenarios(os.path.join(ROOT, "data"), start_min=args.start, pv_factor=args.pv,
                          beta=args.beta)
statics = [StaticGraph(s.P) for s in scs]
W = eval_preferences()
out_dir = os.path.join(ROOT, "results", "policy_eval", args.tag)
os.makedirs(out_dir, exist_ok=True)

for run in runs:
    rdir = os.path.join(ROOT, args.runs_dir, run)
    cfgd = json.load(open(os.path.join(rdir, "config.json")))
    cfg = TrainConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfgd.items()})
    net = PolicyNet(d=cfg.d_model, gat=cfg.gat, film=cfg.film, crit=getattr(cfg, "crit", True))
    net.load_state_dict(torch.load(os.path.join(rdir, "model.pt")))
    net.eval()
    rng = np.random.default_rng(0)
    res = {}
    for s, G in zip(scs, statics):
        t = time.perf_counter()
        _, acts, finals = rollout(net, [s] * len(W), W, rng, greedy=True, store=False,
                                  statics=[G] * len(W), mask_capacity=cfg.mask_capacity,
                                  allow_delay=cfg.delay)
        wall = time.perf_counter() - t
        F = np.array([f[0] for f in finals])
        V = np.array([f[1] for f in finals])
        forced = np.array([f[2] for f in finals])
        # postponement statistics per preference rollout
        n_post = np.array([sum(1 for a in ac if a[2] > 0) for ac in acts])
        post_min = np.array([sum(int(s.P.delays[a[2]]) for a in ac) for ac in acts])
        # energy intensity of the chosen machines relative to the most efficient option
        rel_e = []
        for ac in acts:
            P = s.P
            nxt = np.zeros(P.J, int); tot = 0.0; best = 0.0
            for j, m, d in ac:
                o = P.job_first[j] + nxt[j]; nxt[j] += 1
                e_all = np.where(P.compat[o], P.proc[o] * P.p_proc, np.inf)
                tot += e_all[m]; best += e_all.min()
            rel_e.append(tot / best)
        entry = dict(F=F, viol=V, forced=forced, wall=wall, n_postponed=n_post,
                     postponed_min=post_min, rel_machine_energy=np.array(rel_e))
        if args.samples:
            t = time.perf_counter()
            Ws = np.repeat(W, args.samples, axis=0)
            _, _, fs = rollout(net, [s] * len(Ws), Ws, rng, greedy=False, store=False,
                               statics=[G] * len(Ws), mask_capacity=cfg.mask_capacity,
                               allow_delay=cfg.delay)
            entry["F_sampled"] = np.array([f[0] for f in fs])
            entry["viol_sampled"] = np.array([f[1] for f in fs])
            entry["wall_sampled"] = time.perf_counter() - t
        if s.name in args.keep_actions:
            entry["actions"] = acts
        res[s.name] = entry
        print(f"{run} {s.name} wall={wall:.2f}s", flush=True)
    pickle.dump(dict(run=run, config=cfgd, prefs=W, results=res),
                open(os.path.join(out_dir, f"{run}.pkl"), "wb"))
