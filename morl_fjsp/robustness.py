"""Out-of-sample (frequentist) evaluation of schedules built with possibilistic
(credibility) constraints.

Realised processing times are sampled for the machine each operation was assigned
to, from distributions that are *not* implied by the fuzzy model:
  * "triangular": Triangular(a, b, c) (same support/mode as the TFN);
  * "lognormal" : median b, log-sd such that the 97.5% quantile equals c
                  (heavier right tail: can exceed the TFN support);
  * "uniform"   : Uniform(a, c).
The schedule is then executed semi-actively (same machine sequences, same
postponements, all start times recomputed), which yields realised machine
workloads, capacity feasibility, makespan, cost and CO2.
"""
from __future__ import annotations

import copy

import numpy as np

from .sim import Problem, evaluate_actions

DISTS = ("triangular", "lognormal", "uniform")


def sample_times(P: Problem, op_m: np.ndarray, dist: str, rng: np.random.Generator):
    o = np.arange(P.N)
    a, b, c = P.A[o, op_m], P.B[o, op_m], P.C[o, op_m]
    if dist == "triangular":
        x = rng.triangular(a, b, np.maximum(c, b + 1e-9))
    elif dist == "lognormal":
        sigma = np.log(np.maximum(c, b + 1e-9) / b) / 1.959964
        x = b * np.exp(sigma * rng.standard_normal(P.N))
    elif dist == "uniform":
        x = rng.uniform(a, c)
    else:
        raise ValueError(dist)
    return x


def capacity_feasibility(P: Problem, op_m: np.ndarray, dist: str, n: int, rng) -> dict:
    """Monte-Carlo frequency of {realised workload <= cap_m for all m} and the
    realised overtime frequency per machine (workload > L)."""
    W = np.zeros((n, P.M))
    for s in range(n):
        x = sample_times(P, op_m, dist, rng)
        W[s] = np.bincount(op_m, weights=x, minlength=P.M)
    feas = (W <= P.cap[None, :]).all(axis=1)
    over = (W > P.L).mean(axis=0)
    return dict(feasible=float(feas.mean()), overtime_freq=over, n=n)


def stochastic_execution(P: Problem, actions, op_m, dist: str, n: int, rng) -> np.ndarray:
    """Objective vectors of `n` semi-active executions with sampled durations."""
    out = np.empty((n, 4))
    for s in range(n):
        x = sample_times(P, op_m, dist, rng)
        Q = copy.copy(P)
        proc = P.proc.copy()
        proc[np.arange(P.N), op_m] = np.maximum(1, np.rint(x)).astype(np.int64)
        Q.proc = proc
        Q.H = int(P.H + proc.sum())
        Q.price, Q.pv, Q.co2 = P.day.minute_arrays(P.start_min, Q.H, P.pv_kwp)
        out[s] = evaluate_actions(Q, actions).objectives()
    return out
