"""Non-learning baselines on the 12 public benchmarks (real DE-LU day, 06:00 start):
dispatching rules, preference-weighted greedy (PWG) over the 84 evaluation
preferences, and rule-seeded NSGA-II / NSGA-III (10 seeds, anytime traces).

Output: results/baselines/<scenario_tag>/<instance>.pkl
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.metrics import eval_preferences  # noqa: E402
from morl_fjsp.moea import run_moea  # noqa: E402
from morl_fjsp.rules import RULES, run_rule  # noqa: E402
from morl_fjsp.scenarios import benchmark_scenarios  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--pop", type=int, default=100)
ap.add_argument("--gens", type=int, default=500)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--start", type=int, default=360)
ap.add_argument("--pv", type=float, default=0.5)
ap.add_argument("--beta", type=float, default=0.9)
ap.add_argument("--tag", default="main")
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--algos", nargs="*", default=["nsga2", "nsga3"])
ap.add_argument("--unseeded", action="store_true",
                help="random initial population (no rule schedules)")
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "baselines", args.tag)
os.makedirs(OUT, exist_ok=True)


def job(name):
    sc = benchmark_scenarios(os.path.join(ROOT, "data"), start_min=args.start,
                             pv_factor=args.pv, beta=args.beta, names=[name])[0]
    P = sc.P
    out = dict(name=name, ideal=sc.ideal, nadir=sc.nadir, rules={}, pwg={}, moea={})
    for r in RULES:
        t = time.perf_counter()
        st = run_rule(P, r)
        out["rules"][r] = dict(F=st.objectives(), viol=st.violation(),
                               time=time.perf_counter() - t)
    W = eval_preferences()
    t = time.perf_counter()
    Fp, Vp = [], []
    for w in W:
        st = run_rule(P, "PWG", w=w, scale=sc.span)
        Fp.append(st.objectives()); Vp.append(st.violation())
    out["pwg"] = dict(F=np.array(Fp), viol=np.array(Vp), time=time.perf_counter() - t)
    for algo in args.algos:
        runs = []
        for seed in range(args.seeds):
            res = run_moea(P, algo, pop=args.pop, n_gen=args.gens, seed=seed, trace_every=25,
                           seeded=not args.unseeded)
            runs.append(dict(F=res["F"], G=res["G"], wall=res["wall"], n_evals=res["n_evals"],
                             trace=[(g, e, t_, F) for g, e, t_, F in res["trace"]]))
        out["moea"][algo] = runs
    pickle.dump(out, open(os.path.join(OUT, f"{name}.pkl"), "wb"))
    return name, {a: np.mean([r["wall"] for r in out["moea"][a]]) for a in args.algos}


if __name__ == "__main__":
    from morl_fjsp.instance import BENCHMARKS
    names = args.only or BENCHMARKS
    # largest instances first for better load balance
    names = sorted(names, key=lambda n: -{"Mk09": 9, "Mk10": 9, "Mk08": 8, "Mk03": 7,
                                          "Mk06": 7, "La21": 6}.get(n, 0))
    t0 = time.time()
    with Pool(args.workers) as pool:
        for name, walls in pool.imap_unordered(job, names):
            print(f"{name} done {walls} elapsed={time.time() - t0:.0f}s", flush=True)
