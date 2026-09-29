"""Supplementary experiments (all on the 12 benchmarks, real DE-LU test day):

  credibility  beta sweep of the credibility gate at inference time + Monte-Carlo
               (frequentist) capacity feasibility under three sampling models
  pv           PV-capacity sensitivity (phi in {0, 0.25, 0.5, 1.0})
  days         robustness to the energy day and shift start (real day at 00:00, 06:00,
               12:00 and four unseen synthetic days) incl. PWG and NSGA-II (3 seeds)
  hybrid       NSGA-II seeded with the policy's schedules vs rule-seeded NSGA-II
  scale        inference time on larger synthetic instances
  conflict     counter-factual carbon-inverted day (genuine cost-carbon conflict)
  credseeds    Monte-Carlo feasibility at beta = 0.7 and 0.9 for further trained policies
Outputs: results/extra/<name>.pkl
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.energy import load_energy_charts_day, synthetic_day  # noqa: E402
from morl_fjsp.env import Scenario  # noqa: E402
from morl_fjsp.instance import generate_instance  # noqa: E402
from morl_fjsp.metrics import eval_preferences  # noqa: E402
from morl_fjsp.moea import RandomKeyProblem, encode_actions, run_moea, seeded_population  # noqa: E402
from morl_fjsp.policy_io import load_policy, sweep  # noqa: E402
from morl_fjsp.robustness import DISTS, capacity_feasibility  # noqa: E402
from morl_fjsp.rules import run_rule  # noqa: E402
from morl_fjsp.scenarios import benchmark_scenarios, bounds_from, rule_front  # noqa: E402
from morl_fjsp.sim import Problem  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "extra")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument("exp", choices=["credibility", "pv", "days", "hybrid", "scale", "conflict", "credseeds"])
ap.add_argument("--runs", nargs="*", default=["preco_s0"])
ap.add_argument("--mc", type=int, default=200)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--cond", type=int, default=None, help="days: run only this condition index")
ap.add_argument("--decode-only", action="store_true",
                help="scale: re-measure only the decoding times of the stored instances")
ap.add_argument("--k", type=int, default=None,
                help="policy schedules injected into the initial population "
                     "(default: best_k of results/summary/seeding_validation.json)")
args = ap.parse_args()
W = eval_preferences()


def credibility():
    betas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    rng = np.random.default_rng(0)
    res = {}
    for run in args.runs:
        net, cfg = load_policy(os.path.join(ROOT, "results", "runs", run))
        for beta in betas:
            scs = benchmark_scenarios(os.path.join(ROOT, "data"), beta=beta)
            for sc in scs:
                F, V, acts, wall = sweep(net, cfg, sc, W)
                feas = {d: [] for d in DISTS}
                for a in acts:
                    st = sc.P.new_state()
                    for j, m, dd in a:
                        st.step(j, m, dd)
                    for d in DISTS:
                        feas[d].append(capacity_feasibility(sc.P, st.op_m, d, args.mc, rng)["feasible"])
                res[(run, beta, sc.name)] = dict(F=F, viol=V, feas={d: np.array(v) for d, v in feas.items()})
                print(run, beta, sc.name, {d: round(float(np.mean(v)), 3) for d, v in feas.items()},
                      flush=True)
    # rule-based reference: PWG with/without the gate at each beta
    for beta in betas:
        scs = benchmark_scenarios(os.path.join(ROOT, "data"), beta=beta)
        for sc in scs:
            Fs, Vs, fe = [], [], []
            for w in W[::4]:
                st = run_rule(sc.P, "PWG", w=w, scale=sc.span)
                Fs.append(st.objectives()); Vs.append(st.violation())
                fe.append(capacity_feasibility(sc.P, st.op_m, "triangular", args.mc, rng)["feasible"])
            res[("PWG", beta, sc.name)] = dict(F=np.array(Fs), viol=np.array(Vs),
                                              feas={"triangular": np.array(fe)})
    pickle.dump(res, open(os.path.join(OUT, "credibility.pkl"), "wb"))


def pv():
    res = {}
    for run in args.runs:
        net, cfg = load_policy(os.path.join(ROOT, "results", "runs", run))
        for phi in [0.0, 0.25, 0.5, 1.0]:
            for sc in benchmark_scenarios(os.path.join(ROOT, "data"), pv_factor=phi):
                F, V, acts, wall = sweep(net, cfg, sc, W)
                res[(run, phi, sc.name)] = dict(F=F, viol=V)
                print(run, phi, sc.name, np.round(F.min(0), 2), flush=True)
    pickle.dump(res, open(os.path.join(OUT, "pv.pkl"), "wb"))


def days():
    """Robustness of the comparison to the energy signals.  Conditions: the real test day
    with shift starts 00:00, 06:00 (default) and 12:00, and four synthetic days drawn with a
    generator seed used nowhere else (never seen in training, validation or model selection),
    shift start 06:00.  Methods: the policies in --runs (84-preference sweep), PWG, and
    rule-seeded NSGA-II (pop 100, 500 generations, 3 seeds)."""
    real = load_energy_charts_day(os.path.join(ROOT, "data", "energy"))
    rng = np.random.default_rng(31337)
    conds = [("real 00:00", real, 0), ("real 06:00", real, 360), ("real 12:00", real, 720)]
    conds += [(f"synthetic {i + 1}", synthetic_day(rng, name=f"testday{i + 1}"), 360)
              for i in range(4)]
    nets = {run: load_policy(os.path.join(ROOT, "results", "runs", run)) for run in args.runs}
    res = dict(conditions=[(c, d.price, d.co2, d.solar, st) for c, d, st in conds])
    todo = range(len(conds)) if args.cond is None else [args.cond]
    for ci in todo:
        cname, day, st_min = conds[ci]
        for sc in benchmark_scenarios(os.path.join(ROOT, "data"), start_min=st_min, day=day):
            for run, (net, cfg) in nets.items():
                F, V, acts, wall = sweep(net, cfg, sc, W)
                res[(cname, run, sc.name)] = dict(F=F, viol=V)
            Fp, Vp = [], []
            for w in W:
                stt = run_rule(sc.P, "PWG", w=w, scale=sc.span)
                Fp.append(stt.objectives()); Vp.append(stt.violation())
            res[(cname, "PWG", sc.name)] = dict(F=np.array(Fp), viol=np.array(Vp))
            runs = []
            for seed in range(3):
                r = run_moea(sc.P, "nsga2", pop=100, n_gen=500, seed=seed)
                runs.append(dict(F=r["F"], G=r["G"]))
            res[(cname, "NSGA-II", sc.name)] = runs
            print(cname, sc.name, flush=True)
    tag = "" if args.cond is None else f"_{args.cond}"
    pickle.dump(res, open(os.path.join(OUT, f"days{tag}.pkl"), "wb"))


def hybrid():
    """NSGA-II / NSGA-III (pop 100, 500 generations) whose initial population is the
    rule-seeded default (8 rule schedules + 12 noisy copies) plus k distinct schedules of
    the trained policy (84-preference sweep), completed with random keys.  k is the value
    selected on the synthetic validation scenarios by scripts/tune_seeding.py (never on
    the benchmarks).  Compared with the rule-seeded runs of run_baselines.py (same seeds
    and budget)."""
    k = args.k
    if k is None:
        k = int(json.load(open(os.path.join(ROOT, "results", "summary",
                                            "seeding_validation.json")))["best_k"])
    print("policy seeds per population:", k, flush=True)
    res = {}
    scs = benchmark_scenarios(os.path.join(ROOT, "data"))
    for run in args.runs:
        net, cfg = load_policy(os.path.join(ROOT, "results", "runs", run))
        for sc in scs:
            F, V, acts, wall = sweep(net, cfg, sc, W)
            prob = RandomKeyProblem(sc.P)
            keys = np.unique(np.array([encode_actions(prob, a) for a in acts]), axis=0)
            for algo in ("nsga2", "nsga3"):
                out = []
                for seed in range(args.seeds):
                    rng = np.random.default_rng(seed)
                    X = seeded_population(prob, 100, rng)            # rules + noisy + random
                    sel = keys[rng.permutation(len(keys))[:k]]
                    X[20:20 + len(sel)] = sel                        # keep the 20 rule seeds
                    r = run_moea(sc.P, algo, pop=100, n_gen=500, seed=seed, trace_every=25,
                                 init_X=X)
                    out.append(dict(F=r["F"], G=r["G"], wall=r["wall"], policy_wall=wall, k=k,
                                    trace=[(g, e, t_ + wall, Fx) for g, e, t_, Fx in r["trace"]]))
                res[(run, algo, sc.name)] = out
            print("hybrid", run, sc.name, flush=True)
    tag = "_".join(args.runs) if len(args.runs) < 3 else f"{args.runs[0]}_etc"
    pickle.dump(res, open(os.path.join(OUT, f"hybrid_{tag}.pkl"), "wb"))


def _decode_time(P, n=200):
    """Seconds per decoding (= one NSGA evaluation) with the compiled decoder, measured on
    random keys after a warm-up call (population construction is not included)."""
    prob = RandomKeyProblem(P)
    X = np.random.default_rng(0).random((n, prob.n_var))
    prob.decode_many(X[:2])
    t0 = time.perf_counter()
    prob.decode_many(X)
    return (time.perf_counter() - t0) / n


def scale():
    """Inference time of one policy on synthetic instances of growing size and the time of
    one decoding.  With --decode-only, the instances are regenerated (same seed) and only the
    decoding times in results/extra/scale.pkl are (re)measured."""
    rng = np.random.default_rng(99)
    day = load_energy_charts_day(os.path.join(ROOT, "data", "energy"))
    sizes = [(10, 5), (20, 10), (30, 10), (40, 15), (50, 20), (80, 20)]
    path = os.path.join(OUT, "scale.pkl")
    if args.decode_only:
        res = pickle.load(open(path, "rb"))
        for r, (J, M) in zip(res, sizes):
            inst = generate_instance(rng, J, M, name=f"scale{J}x{M}", flex_max=3)
            assert inst.n_ops == r["N"]
            r["decode_sec"] = _decode_time(Problem(inst, day))
            print(r, flush=True)
        pickle.dump(res, open(path, "wb"))
        return
    res = []
    run = args.runs[0]
    net, cfg = load_policy(os.path.join(ROOT, "results", "runs", run))
    for J, M in sizes:
        inst = generate_instance(rng, J, M, name=f"scale{J}x{M}", flex_max=3)
        P = Problem(inst, day)
        ideal, nadir = bounds_from(rule_front(P))
        sc = Scenario(P, ideal, nadir, inst.name)
        F, V, acts, wall = sweep(net, cfg, sc, W)
        res.append(dict(J=J, M=M, N=P.N, edges=int(inst.compat.sum() + P.N - J), policy_wall=wall,
                        per_pref=wall / len(W), decode_sec=_decode_time(P)))
        print(res[-1], flush=True)
    pickle.dump(res, open(path, "wb"))


def conflict():
    """Counter-factual stress test: the real day with the carbon series mirrored about its
    mid-range, co2' = min + max - co2 (same range and spectrum; corr(price, co2) flips from +0.75
    to -0.75), so that cheap hours are carbon-intensive and expensive hours are clean: a genuine
    cost-carbon conflict.  Methods as in the robustness experiment: the policies in --runs
    (84-preference sweep), PWG and rule-seeded NSGA-II (pop 100, 500 generations, 3 seeds)."""
    import dataclasses
    real = load_energy_charts_day(os.path.join(ROOT, "data", "energy"))
    inv = dataclasses.replace(real, name="DE-LU carbon-inverted",
                              co2=real.co2.min() + real.co2.max() - real.co2)
    nets = {run: load_policy(os.path.join(ROOT, "results", "runs", run)) for run in args.runs}
    res = dict(corr_price_co2=float(np.corrcoef(inv.price, inv.co2)[0, 1]),
               corr_price_co2_real=float(np.corrcoef(real.price, real.co2)[0, 1]))
    for sc in benchmark_scenarios(os.path.join(ROOT, "data"), start_min=360, day=inv):
        for run, (net, cfg) in nets.items():
            F, V, acts, wall = sweep(net, cfg, sc, W)
            res[(run, sc.name)] = dict(F=F, viol=V)
        Fp, Vp = [], []
        for w in W:
            st = run_rule(sc.P, "PWG", w=w, scale=sc.span)
            Fp.append(st.objectives()); Vp.append(st.violation())
        res[("PWG", sc.name)] = dict(F=np.array(Fp), viol=np.array(Vp))
        res[("NSGA-II", sc.name)] = [dict(F=r["F"], G=r["G"]) for r in
                                     (run_moea(sc.P, "nsga2", pop=100, n_gen=500, seed=s) for s in range(3))]
        print("conflict", sc.name, flush=True)
    pickle.dump(res, open(os.path.join(OUT, "conflict.pkl"), "wb"))


def credseeds():
    """Same protocol as `credibility` (84 preferences x 12 benchmarks x 3 sampling models x --mc draws)
    for the policies in --runs, restricted to beta in {0.7, 0.9}; each policy has its own generator
    (seed 1000 + index in --runs).  Stored per (run, beta, instance): F, forced-violation flags and the
    number of feasible draws (feas = k / mc, exact)."""
    res = {}
    for ri, run in enumerate(args.runs):
        rng = np.random.default_rng(1000 + ri)
        net, cfg = load_policy(os.path.join(ROOT, "results", "runs", run))
        for beta in (0.7, 0.9):
            for sc in benchmark_scenarios(os.path.join(ROOT, "data"), beta=beta):
                F, V, acts, wall = sweep(net, cfg, sc, W)
                feas = {d: [] for d in DISTS}
                for a in acts:
                    st = sc.P.new_state()
                    for j, m, dd in a:
                        st.step(j, m, dd)
                    for d in DISTS:
                        feas[d].append(capacity_feasibility(sc.P, st.op_m, d, args.mc, rng)["feasible"])
                res[(run, beta, sc.name)] = dict(F=F, viol=V, feas={d: np.array(v) for d, v in feas.items()})
                print(run, beta, sc.name, {d: round(float(np.mean(v)), 4) for d, v in feas.items()}, flush=True)
        pickle.dump(res, open(os.path.join(OUT, "credibility_seeds.pkl"), "wb"))


dict(credibility=credibility, pv=pv, days=days, hybrid=hybrid, scale=scale,
     conflict=conflict, credseeds=credseeds)[args.exp]()
