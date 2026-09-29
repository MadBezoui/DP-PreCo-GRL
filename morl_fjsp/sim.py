"""Exact schedule simulator for EA-MOFJSP-FU (shared by the RL environment,
the dispatching rules and the evolutionary baselines).

Schedule semantics (identical for every method):
  * the next unscheduled operation o of job j is appended to machine m and starts at
        S = max(ready_j, ready_m) + delta,   delta in DELAYS_MIN (postponement)
    (delta = 0 gives the classical semi-active, non-delay-at-decision placement);
  * machine m is switched on at the start of its first operation and off after its
    last one; in between it draws processing power while busy and idle power while
    waiting (no intermediate shutdowns).  Modal durations are used for energy.
    On-site PV covers demand first; surplus PV is curtailed (no feed-in).

Objectives (all minimised):
  f1 makespan [h];  f2 net grid electricity cost [EUR];
  f3 grid CO2eq emissions [kg];  f4 front-loaded OWA of machine overtime risks
     r_m = Cr(W_m > L), W_m the TFN workload of machine m.
Constraint: Cr(W_m <= cap_m) >= beta for all m (credibility-gated capacity).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .energy import DayProfile
from .fuzzy import credibility_leq, credibility_quantile
from .instance import Instance

DELAYS_MIN = np.array([0, 15, 30, 60, 120], dtype=np.int64)
N_OBJ = 4
OBJ_NAMES = ["makespan_h", "cost_eur", "co2_kg", "owa_risk"]


class Problem:
    """Instance + energy scenario + constraint level, flattened into arrays."""

    def __init__(self, inst: Instance, day: DayProfile, start_min: int = 360,
                 pv_factor: float = 0.5, beta: float = 0.9, delays=DELAYS_MIN,
                 use_capacity: bool = True, reserve: bool = True):
        self.inst, self.day = inst, day
        self.start_min, self.pv_factor, self.beta = start_min, pv_factor, beta
        self.delays = np.asarray(delays, dtype=np.int64)
        self.J, self.M, self.N = inst.n_jobs, inst.n_machines, inst.n_ops
        self.proc = inst.proc.astype(np.int64)
        self.A, self.B, self.C = (x.astype(np.float64) for x in inst.tfn)
        self.compat = inst.compat.copy()
        self.p_proc = inst.p_proc.astype(np.float64)
        self.p_idle = inst.p_idle.astype(np.float64)
        self.pv_kwp = pv_factor * float(self.p_proc.sum())
        self.L = float(inst.overtime_L)
        self.cap = inst.capacity.astype(np.float64) if use_capacity else np.full(self.M, 1e18)
        # capacity reservation for operations that have a single compatible machine:
        # credibility quantiles of TFN sums are additive, so the gate can reserve
        # exactly the quantile load such operations will add later
        single = self.compat.sum(axis=1) == 1
        m_star = np.where(single, self.compat.argmax(axis=1), -1)
        o = np.arange(self.N)
        qo = np.where(beta <= 0.5,
                      self.A[o, np.maximum(m_star, 0)] + 2 * beta * (self.B - self.A)[o, np.maximum(m_star, 0)],
                      self.B[o, np.maximum(m_star, 0)] + (2 * beta - 1) * (self.C - self.B)[o, np.maximum(m_star, 0)])
        self.forced_m = m_star.astype(np.int64)
        self.forced_q = np.where(single & reserve, qo, 0.0)
        self.job_first = inst.job_first_op.astype(np.int64)
        self.job_n = inst.job_n_ops.astype(np.int64)
        self.op_job = inst.op_job.astype(np.int64)
        pmax = np.where(self.compat, self.proc, 0).max(axis=1)
        self.H = int(pmax.sum() + self.N * self.delays.max() + 1)
        self.price, self.pv, self.co2 = day.minute_arrays(start_min, self.H, self.pv_kwp)
        # bound on candidates per step: sum over jobs of max compat of any op of the job
        per_job = [self.compat[self.job_first[j]:self.job_first[j] + self.job_n[j]].sum(1).max()
                   for j in range(self.J)]
        self.max_cand = int(np.sum(per_job)) * len(self.delays)

    def new_state(self) -> "State":
        return State(self)


class State:
    def __init__(self, P: Problem):
        self.P = P
        self.nxt = np.zeros(P.J, np.int64)
        self.job_ready = np.zeros(P.J, np.int64)
        self.mach_ready = np.zeros(P.M, np.int64)
        self.D = np.zeros(P.H, np.float64)          # shop-floor demand [kW] per minute
        self.mcount = np.zeros(P.M, np.int64)       # operations assigned to each machine
        self.WA = np.zeros(P.M); self.WB = np.zeros(P.M); self.WC = np.zeros(P.M)
        self.op_m = np.full(P.N, -1, np.int64)
        self.op_s = np.full(P.N, -1, np.int64)
        self.op_e = np.full(P.N, -1, np.int64)
        self.op_d = np.full(P.N, -1, np.int64)
        self.R = np.zeros(P.M)                       # reserved quantile load per machine
        np.add.at(self.R, np.maximum(P.forced_m, 0), P.forced_q)
        # scalars kept in a small array so numba can update them in place:
        # [cmax, cost, co2, n_sched, n_forced]
        self.sc = np.zeros(5, np.float64)

    # ------------------------------------------------------------------ API
    def step(self, j: int, m: int, d: int) -> None:
        P = self.P
        o = P.job_first[j] + self.nxt[j]
        if P.forced_q[o] > 0.0:
            self.R[m] -= P.forced_q[o]
        _apply_op(P.job_first, P.proc, P.A, P.B, P.C, P.p_proc, P.p_idle, P.price, P.pv, P.co2,
                  P.delays, self.nxt, self.job_ready, self.mach_ready, self.mcount, self.D,
                  self.WA, self.WB, self.WC, self.op_m, self.op_s, self.op_e, self.op_d,
                  self.sc, j, m, d)

    @property
    def done(self) -> bool:
        return int(self.sc[3]) == self.P.N

    def objectives(self) -> np.ndarray:
        P = self.P
        owa, _ = _owa_risk(self.WA, self.WB, self.WC, P.L)
        return np.array([self.sc[0] / 60.0, self.sc[1], self.sc[2], owa])

    def risks(self) -> np.ndarray:
        return _owa_risk(self.WA, self.WB, self.WC, self.P.L)[1]

    def violation(self) -> float:
        P = self.P
        return float(_violation(self.WA, self.WB, self.WC, P.cap, P.beta))

    def candidates(self, out: np.ndarray | None = None) -> np.ndarray:
        P = self.P
        if out is None:
            out = np.empty((P.max_cand, N_CAND_COLS), np.float64)
        n = _candidates(P.job_first, P.job_n, P.compat, P.proc, P.A, P.B, P.C, P.p_proc,
                        P.p_idle, P.price, P.pv, P.co2, P.delays, P.L, P.cap, P.beta,
                        self.nxt, self.job_ready, self.mach_ready, self.mcount, self.D,
                        self.WA, self.WB, self.WC, self.sc, self.R, P.forced_q, out)
        return out[:n]


# candidate table columns
C_JOB, C_OP, C_M, C_D, C_S, C_E, C_DMK, C_DCOST, C_DCO2, C_DOWA, C_FEAS, C_PRICE, C_PV, \
    C_CO2, C_GAP, C_WAIT = range(16)
N_CAND_COLS = 16


# ---------------------------------------------------------------------------- kernels
@njit(cache=True)
def _owa_risk(WA, WB, WC, L):
    M = WA.shape[0]
    r = np.empty(M)
    for m in range(M):
        r[m] = 1.0 - credibility_leq(WA[m], WB[m], WC[m], L)
    s = np.sort(r)[::-1]
    tot = 0.0
    for h in range(M):
        tot += 2.0 * (M - h) / (M * (M + 1.0)) * s[h]
    return tot, r


@njit(cache=True)
def _violation(WA, WB, WC, cap, beta):
    v = 0.0
    for m in range(WA.shape[0]):
        q = credibility_quantile(WA[m], WB[m], WC[m], beta)
        if q > cap[m]:
            v += (q - cap[m]) / cap[m]
    return v


@njit(cache=True)
def _add_power(D, pv, price, co2, t0, t1, add):
    """Add `add` kW on [t0, t1) to the demand profile; return (d_cost, d_co2) in
    (EUR/MWh * kW * min) units, i.e. before the /60000 conversion."""
    cost = 0.0
    em = 0.0
    for t in range(t0, t1):
        g0 = D[t] - pv[t]
        g1 = g0 + add
        c0 = g0 if g0 > 0.0 else 0.0
        c1 = g1 if g1 > 0.0 else 0.0
        cost += price[t] * (c1 - c0)
        em += co2[t] * (c1 - c0)
        D[t] += add
    return cost, em


@njit(cache=True)
def _probe_power(D, pv, price, co2, t0, t1, add):
    """Same as _add_power without modifying D."""
    cost = 0.0
    em = 0.0
    for t in range(t0, t1):
        g0 = D[t] - pv[t]
        g1 = g0 + add
        c0 = g0 if g0 > 0.0 else 0.0
        c1 = g1 if g1 > 0.0 else 0.0
        cost += price[t] * (c1 - c0)
        em += co2[t] * (c1 - c0)
    return cost, em


@njit(cache=True)
def _apply_op(job_first, proc, A, B, C, p_proc, p_idle, price, pv, co2, delays,
              nxt, job_ready, mach_ready, mcount, D, WA, WB, WC, op_m, op_s, op_e, op_d,
              sc, j, m, d):
    o = job_first[j] + nxt[j]
    S = max(job_ready[j], mach_ready[m]) + delays[d]
    E = S + proc[o, m]
    c1, e1 = 0.0, 0.0
    if mcount[m] > 0:          # machine already on: idle between its last op and S
        c1, e1 = _add_power(D, pv, price, co2, mach_ready[m], S, p_idle[m])
    c2, e2 = _add_power(D, pv, price, co2, S, E, p_proc[m])
    if E > sc[0]:
        sc[0] = E
    sc[1] += (c1 + c2) / 60000.0
    sc[2] += (e1 + e2) / 60000.0
    sc[3] += 1
    WA[m] += A[o, m]
    WB[m] += B[o, m]
    WC[m] += C[o, m]
    op_m[o] = m
    op_s[o] = S
    op_e[o] = E
    op_d[o] = d
    job_ready[j] = E
    mach_ready[m] = E
    mcount[m] += 1
    nxt[j] += 1


@njit(cache=True)
def _candidates(job_first, job_n, compat, proc, A, B, C, p_proc, p_idle, price, pv, co2,
                delays, L, cap, beta, nxt, job_ready, mach_ready, mcount, D, WA, WB, WC,
                sc, R, forced_q, out):
    J = job_first.shape[0]
    M = WA.shape[0]
    K = delays.shape[0]
    cmax = int(sc[0])
    owa0, r0 = _owa_risk(WA, WB, WC, L)
    n = 0
    rr = np.empty(M)
    for j in range(J):
        if nxt[j] >= job_n[j]:
            continue
        o = job_first[j] + nxt[j]
        for m in range(M):
            if not compat[o, m]:
                continue
            a = WA[m] + A[o, m]
            b = WB[m] + B[o, m]
            c = WC[m] + C[o, m]
            feas = 1.0 if credibility_quantile(a, b, c, beta) + R[m] - forced_q[o] <= cap[m] + 1e-9 else 0.0
            for i in range(M):
                rr[i] = r0[i]
            rr[m] = 1.0 - credibility_leq(a, b, c, L)
            s = np.sort(rr)[::-1]
            owa1 = 0.0
            for h in range(M):
                owa1 += 2.0 * (M - h) / (M * (M + 1.0)) * s[h]
            p = proc[o, m]
            for k in range(K):
                S = max(job_ready[j], mach_ready[m]) + delays[k]
                E = S + p
                c1, e1 = 0.0, 0.0
                if mcount[m] > 0:
                    c1, e1 = _probe_power(D, pv, price, co2, mach_ready[m], S, p_idle[m])
                c2, e2 = _probe_power(D, pv, price, co2, S, E, p_proc[m])
                sp = 0.0
                spv = 0.0
                sco = 0.0
                for t in range(S, E):
                    sp += price[t]
                    spv += pv[t]
                    sco += co2[t]
                out[n, 0] = j
                out[n, 1] = o
                out[n, 2] = m
                out[n, 3] = k
                out[n, 4] = S
                out[n, 5] = E
                out[n, 6] = E - cmax if E > cmax else 0.0
                out[n, 7] = (c1 + c2) / 60000.0
                out[n, 8] = (e1 + e2) / 60000.0
                out[n, 9] = owa1 - owa0
                out[n, 10] = feas
                out[n, 11] = sp / p
                out[n, 12] = spv / p
                out[n, 13] = sco / p
                out[n, 14] = S - mach_ready[m]
                out[n, 15] = S - job_ready[j]
                n += 1
    return n


@njit(cache=True)
def _decode(job_first, job_n, compat_idx, compat_cnt, proc, A, B, C, p_proc, p_idle, price,
            pv, co2, delays, L, cap, beta, seq_jobs, mkey, dkey, H, delay_split):
    """Decode a random-key individual.  seq_jobs: job id per position (operation-based
    sequence); mkey/dkey: keys in [0,1) per operation (global op index)."""
    J = job_first.shape[0]
    M = p_proc.shape[0]
    N = seq_jobs.shape[0]
    nxt = np.zeros(J, np.int64)
    job_ready = np.zeros(J, np.int64)
    mach_ready = np.zeros(M, np.int64)
    mcount = np.zeros(M, np.int64)
    D = np.zeros(H, np.float64)
    WA = np.zeros(M)
    WB = np.zeros(M)
    WC = np.zeros(M)
    op_m = np.full(N, -1, np.int64)
    op_s = np.full(N, -1, np.int64)
    op_e = np.full(N, -1, np.int64)
    op_d = np.full(N, -1, np.int64)
    sc = np.zeros(5)
    K = delays.shape[0]
    for i in range(N):
        j = seq_jobs[i]
        o = job_first[j] + nxt[j]
        cnt = compat_cnt[o]
        mi = int(mkey[o] * cnt)
        if mi >= cnt:
            mi = cnt - 1
        m = compat_idx[o, mi]
        kd = dkey[o]
        if kd < delay_split:      # keys below delay_split -> no postponement
            d = 0
        else:
            d = 1 + int((kd - delay_split) / (1.0 - delay_split) * (K - 1))
            if d >= K:
                d = K - 1
        _apply_op(job_first, proc, A, B, C, p_proc, p_idle, price, pv, co2, delays, nxt,
                  job_ready, mach_ready, mcount, D, WA, WB, WC, op_m, op_s, op_e, op_d, sc,
                  j, m, d)
    owa, r = _owa_risk(WA, WB, WC, L)
    viol = _violation(WA, WB, WC, cap, beta)
    return sc[0] / 60.0, sc[1], sc[2], owa, viol, op_m, op_s, op_d


def compat_tables(P: Problem):
    cnt = P.compat.sum(axis=1).astype(np.int64)
    idx = np.full((P.N, P.M), -1, np.int64)
    for o in range(P.N):
        ms = np.flatnonzero(P.compat[o])
        idx[o, :len(ms)] = ms
    return idx, cnt


def evaluate_actions(P: Problem, actions) -> State:
    """Replay a list of (job, machine, delay_idx) actions and return the final state."""
    st = P.new_state()
    for j, m, d in actions:
        st.step(int(j), int(m), int(d))
    return st


def actions_of_state(st: State):
    """Recover the action sequence (ordered by start time, ties by op index) of a
    complete schedule; replaying it reproduces the schedule exactly."""
    P = st.P
    order = np.lexsort((np.arange(P.N), st.op_s))
    # start times increase along every machine and every job, so this order is
    # consistent with both sequences and reproduces all start times exactly
    return [(int(P.op_job[o]), int(st.op_m[o]), int(st.op_d[o])) for o in order
            if st.op_s[o] >= 0]
