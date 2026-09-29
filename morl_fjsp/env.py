"""Batched scheduling environment producing heterogeneous-graph observations.

One environment = one (instance, energy day, start time, preference) episode.  At
each step the agent picks one row of the candidate table (job's next operation,
compatible machine, postponement delta).  Candidates violating the credibility-
gated capacity are masked unless no feasible candidate exists (then all are
allowed and the forced violation is recorded).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import (C_D, C_DCO2, C_DCOST, C_DMK, C_DOWA, C_E, C_FEAS, C_GAP, C_JOB, C_M,
                  C_OP, C_PRICE, C_PV, C_CO2, C_S, C_WAIT, N_CAND_COLS, Problem)

D_OP, D_MACH, D_EDGE, D_CAND, D_GLOB = 10, 9, 3, 20, 34
PRICE_NORM, CO2_NORM = 200.0, 400.0
FORECAST_BUCKETS, BUCKET_MIN = 8, 120


@dataclass
class Scenario:
    """A training/test scenario with its normalisation bounds."""
    P: Problem
    ideal: np.ndarray
    nadir: np.ndarray
    name: str = ""

    @property
    def span(self):
        return np.maximum(self.nadir - self.ideal, 1e-3 * np.maximum(np.abs(self.nadir), 1.0))


def _z(x):
    s = x.std()
    return np.clip((x - x.mean()) / s, -5, 5) if s > 1e-12 else np.zeros_like(x)


class StaticGraph:
    """Instance-level tensors shared by all steps of an episode."""

    def __init__(self, P: Problem):
        self.P = P
        J, M, N = P.J, P.M, P.N
        # precedence edges (both directions) between consecutive ops of a job
        src, dst = [], []
        for j in range(J):
            f, n = P.job_first[j], P.job_n[j]
            for k in range(n - 1):
                src += [f + k, f + k + 1]
                dst += [f + k + 1, f + k]
        self.prec = np.array([src, dst], dtype=np.int64).reshape(2, -1)
        pairs = np.argwhere(P.compat)
        self.cop, self.cm = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
        pmin = np.where(P.compat, P.proc, np.inf).min(1)
        self.p_norm = float(np.where(P.compat, P.proc, 0).max())
        self.power_norm = float(P.p_proc.max())
        p = P.proc[self.cop, self.cm].astype(np.float64)
        self.edge_feat = np.stack([
            p / self.p_norm,
            p * P.p_proc[self.cm] / (self.p_norm * self.power_norm),
            (P.C[self.cop, self.cm] - P.A[self.cop, self.cm]) / np.maximum(p, 1e-9)],
            axis=1).astype(np.float32)
        mean_p = np.where(P.compat, P.proc, 0).sum(1) / P.compat.sum(1)
        self.mean_p = mean_p
        self.mean_p_all = float(mean_p.mean())
        spread = np.where(P.compat, (P.C - P.A) / np.maximum(P.B, 1e-9), 0).sum(1) / P.compat.sum(1)
        self.time_norm = float(P.inst.meta.get("c_ref_units", 1) * P.inst.tau)
        # static op features
        rem_work = np.zeros(N)
        rem_ops = np.zeros(N)
        for j in range(J):
            f, n = P.job_first[j], P.job_n[j]
            rem_work[f:f + n] = np.cumsum(mean_p[f:f + n][::-1])[::-1]
            rem_ops[f:f + n] = np.arange(n, 0, -1)
        self.op_static = np.stack([pmin / self.p_norm, mean_p / self.p_norm,
                                   P.compat.sum(1) / M, spread,
                                   rem_ops / max(P.job_n.max(), 1),
                                   rem_work / self.time_norm], axis=1)
        self.pv_norm = max(P.pv_kwp, 1e-6)
        # edge-padded signals so that every forecast window is a plain slice
        pad = FORECAST_BUCKETS * BUCKET_MIN
        self.fc = [(np.pad(P.price, (0, pad), mode="edge"), PRICE_NORM),
                   (np.pad(P.pv, (0, pad), mode="edge"), self.pv_norm),
                   (np.pad(P.co2, (0, pad), mode="edge"), CO2_NORM)]


class BatchEnv:
    """Runs a batch of independent episodes in lock-step (finished ones idle)."""

    def __init__(self, scenarios: list[Scenario], prefs: np.ndarray,
                 statics: list[StaticGraph] | None = None, mask_capacity: bool = True,
                 allow_delay: bool = True):
        self.sc = scenarios
        self.W = np.asarray(prefs, dtype=np.float64)
        self.B = len(scenarios)
        self.statics = statics or [StaticGraph(s.P) for s in scenarios]
        self.mask_capacity, self.allow_delay = mask_capacity, allow_delay
        self.states = [s.P.new_state() for s in scenarios]
        self.bufs = [np.empty((s.P.max_cand, N_CAND_COLS)) for s in scenarios]
        self.cands = [None] * self.B
        self.prev_F = np.zeros((self.B, 4))
        self.n_forced = np.zeros(self.B, dtype=np.int64)

    @property
    def done(self) -> np.ndarray:
        return np.array([st.done for st in self.states])

    # ------------------------------------------------------------------ observation
    def observe(self, b: int) -> dict:
        s, st, G = self.sc[b], self.states[b], self.statics[b]
        P = s.P
        cand = st.candidates(self.bufs[b])
        if not self.allow_delay:
            cand = cand[cand[:, C_D] == 0]
        if self.mask_capacity:
            feas = cand[:, C_FEAS] > 0.5
            if feas.any():
                cand = cand[feas]
        self.cands[b] = cand.copy()
        tn, pn = G.time_norm, G.p_norm
        cmax = st.sc[0]
        span = s.span
        # ---- operation features (only unscheduled operations are graph nodes)
        keep = st.op_s < 0
        remap = np.cumsum(keep) - 1
        elig = np.zeros(P.N)
        unfinished = st.nxt < P.job_n
        elig[(P.job_first + st.nxt)[unfinished]] = 1.0
        jr = st.job_ready[P.op_job] / tn
        prog = (st.nxt / P.job_n)[P.op_job]
        op = np.column_stack([prog, elig, G.op_static, jr,
                              (st.mach_ready.min() / tn) * np.ones(P.N)])[keep].astype(np.float32)
        ek = keep[G.cop]
        c_op, c_m, c_feat = remap[G.cop[ek]], G.cm[ek], G.edge_feat[ek]
        pk = keep[G.prec[0]] & keep[G.prec[1]]
        prec = remap[G.prec[:, pk]]
        # ---- machine features
        if P.beta <= 0.5:
            q = st.WA + 2.0 * P.beta * (st.WB - st.WA)
        else:
            q = st.WB + (2.0 * P.beta - 1.0) * (st.WC - st.WB)
        r = st.risks()
        can = np.zeros(P.M)
        if len(cand):
            mj = np.unique(cand[:, [C_M, C_JOB]].astype(np.int64), axis=0)
            can = np.bincount(mj[:, 0], minlength=P.M) / P.J
        mach = np.column_stack([st.mach_ready / tn, st.WB / P.L, (st.WC - st.WA) / P.L, r,
                                np.clip((P.cap - q) / P.cap, -1, 1),
                                P.p_proc / G.power_norm, P.p_idle / G.power_norm,
                                (st.mcount > 0).astype(np.float64), can]).astype(np.float32)
        # ---- candidate features
        c = cand
        feats = np.column_stack([
            (c[:, C_S] - cmax) / tn, (c[:, C_E] - cmax) / tn,
            c[:, C_DMK] / 60.0 / span[0], c[:, C_DCOST] / span[1], c[:, C_DCO2] / span[2],
            c[:, C_DOWA] / span[3], c[:, C_PRICE] / PRICE_NORM, c[:, C_PV] / G.pv_norm,
            c[:, C_CO2] / CO2_NORM, c[:, C_GAP] / pn, c[:, C_WAIT] / pn,
            c[:, C_D] / max(len(P.delays) - 1, 1), (c[:, C_D] == 0).astype(np.float64),
            c[:, C_FEAS],
            # within-state relative features (decision-relevant differences are small in
            # absolute units): offsets to the best candidate and within-state z-scores
            np.clip((c[:, C_E] - c[:, C_E].min()) / G.mean_p_all, 0, 20),
            np.clip((c[:, C_S] - c[:, C_S].min()) / G.mean_p_all, 0, 20),
            _z(c[:, C_DMK]), _z(c[:, C_DCOST]), _z(c[:, C_DCO2]), _z(c[:, C_DOWA])]
        ).astype(np.float32)
        # ---- global features
        F = st.objectives()
        Fn = (F - s.ideal) / span
        tmin = int(st.mach_ready.min()) if st.sc[3] > 0 else 0
        tod = (P.start_min + cmax) / 1440.0 * 2 * np.pi
        fc = []
        for arr, nrm in G.fc:
            seg = arr[tmin:tmin + FORECAST_BUCKETS * BUCKET_MIN]
            fc.append(seg.reshape(FORECAST_BUCKETS, BUCKET_MIN).mean(1) / nrm)
        glob = np.concatenate([[st.sc[3] / P.N, cmax / tn, np.sin(tod), np.cos(tod)],
                               np.clip(Fn, -5, 5), fc[0], fc[1], fc[2],
                               [P.pv_kwp / (P.p_proc.sum() + 1e-9), P.M / 15.0]]
                              ).astype(np.float32)
        assert glob.shape[0] == D_GLOB
        cop = c[:, C_OP].astype(np.int64)
        cm = c[:, C_M].astype(np.int64)
        newpair = np.ones(len(c), bool)
        newpair[1:] = (cop[1:] != cop[:-1]) | (cm[1:] != cm[:-1])
        pair_id = np.cumsum(newpair) - 1
        pidx = np.flatnonzero(newpair)
        return dict(op=op, mach=mach, cand=feats, pair_op=remap[cop[pidx]], pair_m=cm[pidx],
                    cand_pair=pair_id, glob=glob, c_op=c_op, c_m=c_m, c_feat=c_feat, prec=prec,
                    w=self.W[b].astype(np.float32))

    def observe_all(self, idx=None):
        idx = range(self.B) if idx is None else idx
        return [self.observe(b) for b in idx]

    # ------------------------------------------------------------------ transition
    def step(self, b: int, k: int) -> np.ndarray:
        """Apply candidate k of env b (as returned by the last observe); returns the
        normalised reward vector r = -(F_t - F_{t-1}) / span."""
        s, st = self.sc[b], self.states[b]
        c = self.cands[b][k]
        if self.mask_capacity and c[C_FEAS] < 0.5:
            self.n_forced[b] += 1
        st.step(int(c[C_JOB]), int(c[C_M]), int(c[C_D]))
        F = st.objectives()
        r = -(F - self.prev_F[b]) / s.span
        self.prev_F[b] = F
        return r

    def final(self, b: int):
        st = self.states[b]
        return st.objectives(), st.violation(), int(self.n_forced[b])
