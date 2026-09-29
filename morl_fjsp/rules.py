"""Constructive heuristics: classical dispatching rules, energy-aware greedy rules and
a preference-weighted greedy (PWG) baseline.  All rules use the same simulator and
the same credibility-gated capacity mask as the learned policy."""
from __future__ import annotations

import numpy as np

from .sim import (C_DCO2, C_DCOST, C_DMK, C_DOWA, C_D, C_E, C_FEAS, C_JOB, C_M, C_S,
                  Problem, State)


def _remaining(P: Problem, st: State):
    """Remaining work (mean modal proc. time over compatible machines) and ops per job."""
    mean_p = np.where(P.compat, P.proc, 0).sum(1) / np.maximum(P.compat.sum(1), 1)
    rem_work = np.zeros(P.J)
    rem_ops = P.job_n - st.nxt
    for j in range(P.J):
        a = P.job_first[j] + st.nxt[j]
        rem_work[j] = mean_p[a:P.job_first[j] + P.job_n[j]].sum()
    return rem_work, rem_ops


def _pick(cand, key_cols, sign=None):
    """Lexicographic argmin over key columns (sign=-1 -> maximise that key)."""
    keys = []
    for c, s in zip(key_cols, sign or [1] * len(key_cols)):
        keys.append(s * cand[:, c] if isinstance(c, int) else s * c)
    order = np.lexsort(tuple(reversed(keys)))
    return order[0]


def rule_choice(name: str, P: Problem, st: State, cand: np.ndarray, w=None, scale=None) -> int:
    feas = cand[:, C_FEAS] > 0.5
    if feas.any():
        idx = np.flatnonzero(feas)
    else:
        idx = np.arange(len(cand))
    c = cand[idx]
    if name in ("EET", "SPT", "MWKR", "MOR", "FIFO", "LB"):
        nd = c[:, C_D] == 0
        idx, c = idx[nd], c[nd]
    if name == "EET":
        k = _pick(c, [C_E, C_S])
    elif name == "SPT":
        k = _pick(c, [c[:, C_E] - c[:, C_S], C_E])
    elif name in ("MWKR", "MOR", "FIFO"):
        rem_work, rem_ops = _remaining(P, st)
        jobs = c[:, C_JOB].astype(int)
        if name == "MWKR":
            k = _pick(c, [-rem_work[jobs], C_E])
        elif name == "MOR":
            k = _pick(c, [-rem_ops[jobs].astype(float), C_E])
        else:
            k = _pick(c, [st.job_ready[jobs].astype(float), C_E])
    elif name == "EA":
        k = _pick(c, [C_DCOST, C_E])
    elif name == "EC":
        k = _pick(c, [C_DCO2, C_E])
    elif name == "LB":
        k = _pick(c, [C_DOWA, C_E])
    elif name == "PWG":
        dm = c[:, C_DMK] / 60.0
        score = (w[0] * dm / scale[0] + w[1] * c[:, C_DCOST] / scale[1]
                 + w[2] * c[:, C_DCO2] / scale[2] + w[3] * c[:, C_DOWA] / scale[3])
        k = _pick(c, [score, C_E])
    else:
        raise ValueError(name)
    return int(idx[k])


def run_rule(P: Problem, name: str, w=None, scale=None) -> State:
    st = P.new_state()
    buf = np.empty((P.max_cand, 16))
    while not st.done:
        cand = st.candidates(buf)
        i = rule_choice(name, P, st, cand, w, scale)
        st.step(int(cand[i, C_JOB]), int(cand[i, C_M]), int(cand[i, C_D]))
    return st


RULES = ["EET", "SPT", "MWKR", "MOR", "FIFO", "EA", "EC", "LB"]


def reference_bounds(P: Problem):
    """Ideal/nadir estimates from the rule portfolio (used for reward scaling and for
    the utilities of the preference-alignment term during training)."""
    F = np.array([run_rule(P, r).objectives() for r in RULES])
    ideal, nadir = F.min(0), F.max(0)
    span = np.maximum(nadir - ideal, 0.05 * np.maximum(np.abs(nadir), 1e-6))
    return ideal, nadir, span, F
