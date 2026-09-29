"""Evolutionary baselines (pymoo): NSGA-II, NSGA-III and MOEA/D with a random-key
encoding decoded by the shared simulator.

Encoding (3N keys in [0,1)): sequence keys (operation-based sequence obtained by
sorting), machine keys (index into the compatible-machine list) and delay keys
(keys < 0.5 -> no postponement, otherwise one of the four positive delays).
The capacity constraint is handled by constraint-domination (G = violation <= 0).
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem as PymooProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from .sim import Problem, _decode, compat_tables

DELAY_SPLIT = 0.5


@njit(cache=True)
def _decode_pop(X, joblist, job_first, job_n, cidx, ccnt, proc, A, B, C, p_proc, p_idle,
                price, pv, co2, delays, L, cap, beta, H, split):
    n, _ = X.shape
    N = joblist.shape[0]
    F = np.empty((n, 4))
    G = np.empty(n)
    for i in range(n):
        seq = joblist[np.argsort(X[i, :N], kind="mergesort")]
        f1, f2, f3, f4, v, _, _, _ = _decode(job_first, job_n, cidx, ccnt, proc, A, B, C,
                                             p_proc, p_idle, price, pv, co2, delays, L, cap,
                                             beta, seq, X[i, N:2 * N], X[i, 2 * N:], H, split)
        F[i, 0] = f1
        F[i, 1] = f2
        F[i, 2] = f3
        F[i, 3] = f4
        G[i] = v
    return F, G


class RandomKeyProblem(PymooProblem):
    def __init__(self, P: Problem):
        self.P = P
        self.cidx, self.ccnt = compat_tables(P)
        self.joblist = np.repeat(np.arange(P.J), P.job_n).astype(np.int64)
        self.n_evals = 0
        super().__init__(n_var=3 * P.N, n_obj=4, n_ieq_constr=1, xl=0.0, xu=1.0 - 1e-9)

    def decode_many(self, X):
        P = self.P
        return _decode_pop(np.ascontiguousarray(X, dtype=np.float64), self.joblist,
                           P.job_first, P.job_n, self.cidx, self.ccnt, P.proc, P.A, P.B, P.C,
                           P.p_proc, P.p_idle, P.price, P.pv, P.co2, P.delays, P.L, P.cap,
                           P.beta, P.H, DELAY_SPLIT)

    def _evaluate(self, X, out, *args, **kwargs):
        F, G = self.decode_many(X)
        self.n_evals += X.shape[0]
        out["F"] = F
        out["G"] = G[:, None]

    def actions(self, x):
        """Action sequence (job, machine, delay) encoded by one individual."""
        P = self.P
        N = P.N
        seq = self.joblist[np.argsort(x[:N], kind="mergesort")]
        out = _decode(P.job_first, P.job_n, self.cidx, self.ccnt, P.proc, P.A, P.B, P.C,
                      P.p_proc, P.p_idle, P.price, P.pv, P.co2, P.delays, P.L, P.cap, P.beta,
                      seq, x[N:2 * N], x[2 * N:], P.H, DELAY_SPLIT)
        op_m, op_d = out[5], out[7]
        nxt = np.zeros(P.J, np.int64)
        acts = []
        for j in seq:
            o = P.job_first[j] + nxt[j]
            nxt[j] += 1
            acts.append((int(j), int(op_m[o]), int(op_d[o])))
        return acts


def encode_actions(prob: RandomKeyProblem, actions) -> np.ndarray:
    """Random-key vector whose decoding reproduces the given action sequence."""
    P = prob.P
    N, K = P.N, len(P.delays)
    x = np.empty(3 * N)
    nxt = np.zeros(P.J, np.int64)
    for pos, (j, m, d) in enumerate(actions):
        k = nxt[j]
        nxt[j] += 1
        o = P.job_first[j] + k
        x[P.job_first[j] + k] = (pos + 0.5) / N      # slot of the k-th occurrence of job j
        mi = int(np.flatnonzero(prob.cidx[o, :prob.ccnt[o]] == m)[0])
        x[N + o] = (mi + 0.5) / prob.ccnt[o]
        x[2 * N + o] = 0.5 * DELAY_SPLIT if d == 0 else \
            DELAY_SPLIT + (d - 1 + 0.5) / (K - 1) * (1.0 - DELAY_SPLIT)
    return x


def seeded_population(prob: RandomKeyProblem, pop: int, rng: np.random.Generator,
                      frac: float = 0.2, sigma: float = 0.05) -> np.ndarray:
    """Initial population: the dispatching-rule schedules, noisy copies of them
    (a fraction `frac` of the population in total) and uniform random keys."""
    from .rules import RULES, run_rule
    from .sim import actions_of_state
    seeds = [encode_actions(prob, actions_of_state(run_rule(prob.P, r))) for r in RULES]
    X = rng.random((pop, prob.n_var))
    n_seed = max(len(seeds), int(frac * pop))
    for i in range(n_seed):
        base = seeds[i % len(seeds)]
        X[i] = base if i < len(seeds) else np.clip(
            base + sigma * rng.standard_normal(base.shape), 0.0, 1.0 - 1e-9)
    return X


class PenalizedProblem(RandomKeyProblem):
    """Unconstrained variant for MOEA/D (pymoo's MOEA/D has no constraint handling):
    objectives normalised with rule-based bounds plus a large violation penalty."""

    def __init__(self, P: Problem, lo, span, penalty=100.0):
        super().__init__(P)
        self.n_ieq_constr = 0
        self.lo, self.span, self.penalty = np.asarray(lo), np.asarray(span), penalty

    def _evaluate(self, X, out, *args, **kwargs):
        F, G = self.decode_many(X)
        self.n_evals += X.shape[0]
        out["F"] = (F - self.lo) / self.span + self.penalty * G[:, None]


class _Trace(Callback):
    """Records the non-dominated feasible front every `every` generations."""

    def __init__(self, every=10):
        super().__init__()
        self.every, self.trace, self.t0 = every, [], time.perf_counter()

    def notify(self, algorithm):
        if algorithm.n_gen % self.every == 0:
            opt = algorithm.opt
            F = opt.get("F")
            feas = opt.get("feasible").ravel() if opt.get("feasible") is not None else None
            if feas is not None:
                F = F[feas]
            self.trace.append((algorithm.n_gen, algorithm.evaluator.n_eval,
                               time.perf_counter() - self.t0, np.array(F)))


def run_moea(P: Problem, algo: str = "nsga2", pop: int = 100, n_gen: int = 200, seed: int = 0,
             trace_every: int = 0, time_limit: float | None = None, bounds=None,
             seeded: bool = True, init_X: np.ndarray | None = None):
    if algo == "moead":
        if bounds is None:
            from .rules import reference_bounds
            ideal, _, span, _ = reference_bounds(P)
        else:
            ideal, span = bounds
        prob = PenalizedProblem(P, ideal, span)
    else:
        prob = RandomKeyProblem(P)
    cx, mut = SBX(prob=0.9, eta=15), PM(eta=20)
    ref = get_reference_directions("das-dennis", 4, n_partitions=6)   # 84 directions
    pop_size = max(pop, len(ref)) if algo == "nsga3" else pop
    if init_X is not None:
        sampling = init_X
    elif seeded:
        sampling = seeded_population(prob, pop_size, np.random.default_rng(seed))
    else:
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        sampling = FloatRandomSampling()
    if algo == "nsga2":
        alg = NSGA2(pop_size=pop_size, sampling=sampling, crossover=cx, mutation=mut,
                    eliminate_duplicates=True)
    elif algo == "nsga3":
        alg = NSGA3(ref_dirs=ref, pop_size=pop_size, sampling=sampling, crossover=cx,
                    mutation=mut, eliminate_duplicates=True)
    elif algo == "moead":
        alg = MOEAD(ref_dirs=ref, n_neighbors=15, prob_neighbor_mating=0.7, sampling=sampling,
                    crossover=cx, mutation=mut)
    else:
        raise ValueError(algo)
    termination = ("n_gen", n_gen) if time_limit is None else ("time", time_limit)
    cb = _Trace(trace_every) if trace_every else None
    t0 = time.perf_counter()
    kw = dict(callback=cb) if cb is not None else {}
    res = minimize(prob, alg, termination, seed=seed, verbose=False, save_history=False, **kw)
    wall = time.perf_counter() - t0
    X = res.pop.get("X")
    F, G = prob.decode_many(X)
    feas = G <= 1e-12
    if not feas.any():                      # fall back to least-violating solutions
        feas = G <= G.min() + 1e-12
    return dict(F=F[feas], X=X[feas], G=G[feas], wall=wall, n_evals=prob.n_evals,
                trace=cb.trace if cb else None, problem=prob)
