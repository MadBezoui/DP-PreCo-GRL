"""Aggregation of raw results into indicators (all numbers in the paper come from here)."""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np

from .metrics import (angular_error_deg, asf, hypervolume, igd_plus, nondominated, normalise,
                      utilities)

FEAS_TOL = 1e-9


def load_baselines(root, tag="main"):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "results", "baselines", tag, "*.pkl"))):
        d = pickle.load(open(p, "rb"))
        out[d["name"]] = d
    return out


def load_policy_evals(root, tag="main"):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "results", "policy_eval", tag, "*.pkl"))):
        d = pickle.load(open(p, "rb"))
        out[d["run"]] = d
    return out


def feasible(F, viol):
    F, viol = np.asarray(F), np.asarray(viol)
    return F[viol <= FEAS_TOL]


def load_hybrid(root, groups, pattern="hybrid_*.pkl"):
    """{label: {inst: [(F_feasible, wall), ...]}} for policy-seeded NSGA runs.
    `groups` maps a policy label to its run names; labels become
    '<policy label> + NSGA-II' / '+ NSGA-III'."""
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "results", "extra", pattern))):
        d = pickle.load(open(p, "rb"))
        for (run, algo, inst), runs in d.items():
            lab_pol = next((l for l, ns in groups.items() if run in ns), None)
            if lab_pol is None:
                continue
            lab = f"{lab_pol} + {'NSGA-II' if algo == 'nsga2' else 'NSGA-III'}"
            for r in runs:
                out.setdefault(lab, {}).setdefault(inst, []).append(
                    (r["F"][r["G"] <= FEAS_TOL], r["wall"] + r["policy_wall"], r.get("trace")))
    return out


def method_runs(base, pol, inst, groups, hybrid=None):
    """Return {method: [F_run, ...]} (feasible objective vectors per run) and
    {method: [wall_run, ...]} for one instance.  `groups` maps a method label to a
    list of policy run names."""
    runs, walls = {}, {}
    b = base[inst]
    for algo, lab in (("nsga2", "NSGA-II"), ("nsga3", "NSGA-III")):
        if algo in b["moea"]:
            runs[lab] = [r["F"][r["G"] <= FEAS_TOL] for r in b["moea"][algo]]
            walls[lab] = [r["wall"] for r in b["moea"][algo]]
    runs["PWG"] = [feasible(b["pwg"]["F"], b["pwg"]["viol"])]
    walls["PWG"] = [b["pwg"]["time"]]
    Fr = np.array([v["F"] for v in b["rules"].values()])
    Vr = np.array([v["viol"] for v in b["rules"].values()])
    runs["Rules"] = [feasible(Fr, Vr)]
    walls["Rules"] = [sum(v["time"] for v in b["rules"].values())]
    for lab, names in groups.items():
        rr, ww = [], []
        for n in names:
            if n in pol and inst in pol[n]["results"]:
                e = pol[n]["results"][inst]
                rr.append(feasible(e["F"], e["viol"]))
                ww.append(e["wall"])
        if rr:
            runs[lab], walls[lab] = rr, ww
    for lab, per_inst in (hybrid or {}).items():
        if inst in per_inst:
            runs[lab] = [x[0] for x in per_inst[inst]]
            walls[lab] = [x[1] for x in per_inst[inst]]
    return runs, walls


def reference(runs: dict):
    allF = np.vstack([F for rs in runs.values() for F in rs if len(F)])
    R = nondominated(allF)
    ideal, nadir = R.min(0), R.max(0)
    return R, ideal, nadir


def indicators(runs, R, ideal, nadir):
    Rn = normalise(R, ideal, nadir)
    hv_ref = hypervolume(Rn)
    out = {}
    for lab, rs in runs.items():
        hv, igd, nnd, cov = [], [], [], []
        for F in rs:
            if len(F) == 0:
                hv.append(0.0); igd.append(np.nan); nnd.append(0); cov.append(0.0)
                continue
            A = nondominated(F)
            An = normalise(A, ideal, nadir)
            hv.append(hypervolume(An) / hv_ref)
            igd.append(igd_plus(An, Rn))
            nnd.append(len(A))
            # share of reference points contributed by this run
            cov.append(float(np.mean([np.any(np.all(np.isclose(A, r), axis=1)) for r in R])))
        out[lab] = dict(hv=np.array(hv), igd=np.array(igd), nnd=np.array(nnd), cov=np.array(cov))
    return out, hv_ref


def _clipped(F, ideal, nadir):
    """Normalised objectives clipped to [0, 1]: a solution beyond the nadir of the reference
    front in some objective gets utility 0 there.  This keeps the ASF (in [0, 1]), the
    weighted-sum regret and the angle bounded; without clipping a single schedule far outside
    the reference box dominates the mean through the division by small weights."""
    return np.clip(normalise(F, ideal, nadir), 0.0, 1.0)


def preference_metrics(F_w, W, R, ideal, nadir, viol=None):
    """A-priori preference metrics of a policy that returned F_w[i] for W[i]."""
    Rn = _clipped(R, ideal, nadir)
    U_ref = utilities(Rn)
    Fn = _clipped(F_w, ideal, nadir)
    U = utilities(Fn)
    reg, ang, wsr = [], [], []
    for i, w in enumerate(W):
        if viol is not None and viol[i] > FEAS_TOL:
            reg.append(np.nan); ang.append(np.nan); wsr.append(np.nan)
            continue
        reg.append(float(asf(U_ref, w).max() - asf(U[i], w)[0]))
        ang.append(angular_error_deg(U[i], w))
        wsr.append(float(Fn[i] @ w - (Rn @ w).min()))
    return np.array(reg), np.array(ang), np.array(wsr)


def a_posteriori_metrics(F_set, W, R, ideal, nadir):
    """Best achievable preference metrics when the DM picks a posteriori from a set."""
    Rn = _clipped(R, ideal, nadir)
    U_ref = utilities(Rn)
    Fn = _clipped(F_set, ideal, nadir)
    Un = utilities(Fn)
    reg, ang, wsr = [], [], []
    for w in W:
        a = asf(Un, w)
        i = int(np.argmax(a))
        reg.append(float(asf(U_ref, w).max() - a[i]))
        ang.append(angular_error_deg(Un[i], w))
        wsr.append(float((Fn @ w).min() - (Rn @ w).min()))
    return np.array(reg), np.array(ang), np.array(wsr)


def responsiveness(F_w, W, viol=None):
    """Per objective k, Spearman's rho between the weight w_k and the obtained f_k across the
    preferences (-1: raising a weight always improves its objective; 0: no response).
    Infeasible rollouts are ignored; a constant objective gives NaN."""
    from scipy.stats import spearmanr
    F_w, W = np.asarray(F_w, float), np.asarray(W, float)
    ok = np.ones(len(W), bool) if viol is None else np.asarray(viol) <= FEAS_TOL
    rho = []
    for k in range(W.shape[1]):
        x, y = W[ok, k], F_w[ok, k]
        if len(x) < 3 or np.allclose(y, y[0]):
            rho.append(np.nan)
            continue
        rho.append(float(spearmanr(x, y).statistic))
    return np.array(rho)


def a_posteriori_selection(F_set, W, R, ideal, nadir):
    """Objective vectors picked a posteriori (ASF-best member of F_set) for each preference."""
    Fn = _clipped(F_set, ideal, nadir)
    Un = utilities(Fn)
    return np.array([F_set[int(np.argmax(asf(Un, w)))] for w in W])

