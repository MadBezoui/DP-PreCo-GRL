"""FJSP instances: benchmark parser, synthetic generator and the energy/fuzzy
annotations used by the EA-MOFJSP-FU model.

All times inside an :class:`Instance` are expressed in *minutes*.  Benchmark
processing times (integer "time units") are converted with an integer factor
``tau`` (minutes per unit) that is chosen per instance so that a reference
non-delay schedule spans a nominal production day (see :func:`calibrate_time_scale`).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import numpy as np

# Default annotation parameters (documented in the paper, Section V-A).
TFN_LO_RANGE = (0.05, 0.20)   # relative optimistic spread  (a = (1-lo) p)
TFN_HI_RANGE = (0.10, 0.40)   # relative pessimistic spread (c = (1+hi) p)
POWER_PROC_RANGE = (10.0, 40.0)   # kW, processing power of a machine
IDLE_RATIO_RANGE = (0.20, 0.40)   # idle power = ratio * processing power
REF_HORIZON_MIN = 960             # 16 h: two-shift production day
OVERTIME_SLACK = 0.10             # L = (1 + slack) x LP-balanced max modal load
CAPACITY_FACTOR = 1.5             # cap = factor x LP-balanced max credibility-quantile load
BETA_REF = 0.90                   # credibility level used to calibrate capacities


@dataclass
class Instance:
    name: str
    n_jobs: int
    n_machines: int
    job_n_ops: np.ndarray          # (J,) number of operations per job
    job_first_op: np.ndarray       # (J,) global index of the first op of each job
    op_job: np.ndarray             # (N,) job of each operation
    op_pos: np.ndarray             # (N,) position of the operation inside its job
    compat: np.ndarray             # (N, M) bool
    proc_units: np.ndarray         # (N, M) nominal processing time in benchmark units (0 if incompatible)
    tfn_lo: np.ndarray             # (N, M) relative optimistic spread
    tfn_hi: np.ndarray             # (N, M) relative pessimistic spread
    p_proc: np.ndarray             # (M,) kW
    p_idle: np.ndarray             # (M,) kW
    tau: int = 1                   # minutes per benchmark time unit
    overtime_L: float = 0.0        # minutes, overtime threshold L of the risk objective
    capacity: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (M,) minutes, hard capacity
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ derived
    @property
    def n_ops(self) -> int:
        return int(self.op_job.shape[0])

    @property
    def proc(self) -> np.ndarray:
        """Modal processing times in minutes (N, M)."""
        return self.proc_units * self.tau

    @property
    def tfn(self):
        """Return (A, B, C) arrays of TFN processing times in minutes."""
        b = self.proc.astype(np.float64)
        return (1.0 - self.tfn_lo) * b, b, (1.0 + self.tfn_hi) * b

    @property
    def n_compat_edges(self) -> int:
        return int(self.compat.sum())

    @property
    def n_prec_edges(self) -> int:
        return int(self.n_ops - self.n_jobs)

    def summary(self) -> dict:
        return dict(name=self.name, jobs=self.n_jobs, machines=self.n_machines,
                    ops=self.n_ops, flexibility=round(self.n_compat_edges / self.n_ops, 2),
                    nodes=self.n_ops + self.n_machines,
                    edges=self.n_compat_edges + self.n_prec_edges, tau=self.tau,
                    mean_p_proc=float(self.p_proc.mean()), mean_p_idle=float(self.p_idle.mean()),
                    overtime_L_h=self.overtime_L / 60.0)


def _stable_seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def _build(name, jobs, n_machines, rng) -> Instance:
    """jobs: list (per job) of list (per op) of dict machine->units."""
    n_jobs = len(jobs)
    job_n_ops = np.array([len(j) for j in jobs], dtype=np.int64)
    job_first_op = np.concatenate([[0], np.cumsum(job_n_ops)[:-1]]).astype(np.int64)
    N = int(job_n_ops.sum())
    compat = np.zeros((N, n_machines), dtype=bool)
    proc = np.zeros((N, n_machines), dtype=np.int64)
    op_job = np.zeros(N, dtype=np.int64)
    op_pos = np.zeros(N, dtype=np.int64)
    o = 0
    for j, ops in enumerate(jobs):
        for k, alts in enumerate(ops):
            op_job[o], op_pos[o] = j, k
            for m, p in alts.items():
                compat[o, m] = True
                proc[o, m] = max(1, int(p))
            o += 1
    lo = np.where(compat, rng.uniform(*TFN_LO_RANGE, size=compat.shape), 0.0)
    hi = np.where(compat, rng.uniform(*TFN_HI_RANGE, size=compat.shape), 0.0)
    p_proc = rng.uniform(*POWER_PROC_RANGE, size=n_machines)
    p_idle = rng.uniform(*IDLE_RATIO_RANGE, size=n_machines) * p_proc
    inst = Instance(name, n_jobs, n_machines, job_n_ops, job_first_op, op_job, op_pos,
                    compat, proc, lo, hi, p_proc, p_idle)
    calibrate_time_scale(inst)
    return inst


def parse_fjs(path: str, name: str | None = None) -> Instance:
    """Parse a Brandimarte/Hurink ``.fjs`` file (0- or 1-based machine ids).

    Energy and fuzzy annotations are drawn from a generator seeded by the
    instance name, so every run of the pipeline sees identical annotations.
    """
    name = name or os.path.splitext(os.path.basename(path))[0]
    rows = [list(map(float, l.split())) for l in open(path) if l.strip()]
    n_jobs, n_machines = int(rows[0][0]), int(rows[0][1])
    raw = []
    max_idx, min_idx = -1, 10 ** 9
    for j in range(1, n_jobs + 1):
        t = rows[j]
        k, ops = 1, []
        for _ in range(int(t[0])):
            c = int(t[k]); k += 1
            alts = []
            for _ in range(c):
                m, p = int(t[k]), int(t[k + 1]); k += 2
                alts.append((m, p))
                max_idx, min_idx = max(max_idx, m), min(min_idx, m)
            ops.append(alts)
        if k != len(t):
            raise ValueError(f"{path}: malformed job line {j}")
        raw.append(ops)
    one_based = min_idx >= 1 and max_idx == n_machines
    jobs = [[{(m - 1 if one_based else m): p for m, p in alts} for alts in ops] for ops in raw]
    rng = np.random.default_rng(_stable_seed(name))
    return _build(name, jobs, n_machines, rng)


def generate_instance(rng: np.random.Generator, n_jobs: int, n_machines: int,
                      name: str = "synthetic", ops_range=None, flex_max=None,
                      p_range=(1, 20)) -> Instance:
    """Brandimarte-style random instance (same scheme as Song et al., IEEE TII 2023):
    ops per job ~ U[0.8M, 1.2M], compatible machines per op ~ U[1, M], times ~ U[1, 20]."""
    lo_ops = ops_range[0] if ops_range else max(1, int(np.floor(0.8 * n_machines)))
    hi_ops = ops_range[1] if ops_range else int(np.ceil(1.2 * n_machines))
    fmax = flex_max or n_machines
    jobs = []
    for _ in range(n_jobs):
        ops = []
        for _ in range(int(rng.integers(lo_ops, hi_ops + 1))):
            n_c = int(rng.integers(1, fmax + 1))
            machines = rng.choice(n_machines, size=n_c, replace=False)
            ops.append({int(m): int(rng.integers(p_range[0], p_range[1] + 1)) for m in machines})
        jobs.append(ops)
    return _build(name, jobs, n_machines, rng)


def reference_makespan_units(inst: Instance) -> int:
    """Makespan (benchmark units) of the non-delay earliest-completion-time rule,
    used only to calibrate the time scale."""
    J, M = inst.n_jobs, inst.n_machines
    nxt = np.zeros(J, dtype=np.int64)
    jr = np.zeros(J, dtype=np.int64)
    mr = np.zeros(M, dtype=np.int64)
    for _ in range(inst.n_ops):
        best = None
        for j in range(J):
            if nxt[j] >= inst.job_n_ops[j]:
                continue
            o = inst.job_first_op[j] + nxt[j]
            for m in np.flatnonzero(inst.compat[o]):
                e = max(jr[j], mr[m]) + inst.proc_units[o, m]
                if best is None or e < best[0]:
                    best = (e, j, m)
        e, j, m = best
        jr[j] = mr[m] = e
        nxt[j] += 1
    return int(mr.max())


def min_max_load(inst: Instance, coef: np.ndarray) -> float:
    """LP relaxation of min_x max_m sum_o coef[o,m] x[o,m] (fractional assignment).
    It is a lower bound on the most-loaded machine of any feasible assignment."""
    from scipy.optimize import linprog
    pairs = np.argwhere(inst.compat)
    n = len(pairs)
    c = np.zeros(n + 1); c[-1] = 1.0
    A_eq = np.zeros((inst.n_ops, n + 1))
    A_eq[pairs[:, 0], np.arange(n)] = 1.0
    A_ub = np.zeros((inst.n_machines, n + 1))
    A_ub[pairs[:, 1], np.arange(n)] = coef[pairs[:, 0], pairs[:, 1]]
    A_ub[:, -1] = -1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(inst.n_machines), A_eq=A_eq,
                  b_eq=np.ones(inst.n_ops), bounds=[(0, None)] * (n + 1), method="highs")
    assert res.status == 0, res.message
    return float(res.x[-1])


def calibrate_time_scale(inst: Instance, horizon_min: int = REF_HORIZON_MIN,
                         beta_ref: float = BETA_REF) -> None:
    """Set tau (min/unit), the overtime threshold L and the hard capacities.

    L     = (1 + OVERTIME_SLACK) x LP-minimal max modal machine load;
    cap_m = CAPACITY_FACTOR x LP-minimal max beta_ref-credibility-quantile load,
            floored at 1.05x the quantile load forced by single-machine operations.
    """
    c_ref = reference_makespan_units(inst)
    inst.tau = max(1, int(round(horizon_min / c_ref)))
    A, B, C = inst.tfn
    q = B + (2.0 * beta_ref - 1.0) * (C - B)
    wb = min_max_load(inst, B)
    wq = min_max_load(inst, q)
    inst.overtime_L = float((1.0 + OVERTIME_SLACK) * wb)
    single = inst.compat.sum(axis=1) == 1
    forced_q = np.where(inst.compat & single[:, None], q, 0.0).sum(axis=0)
    inst.capacity = np.maximum(CAPACITY_FACTOR * wq, 1.05 * forced_q)
    inst.meta.update(c_ref_units=c_ref, lp_max_load_B=wb, lp_max_load_q=wq)


BENCHMARKS = ["Mk01", "Mk02", "Mk03", "Mk04", "Mk05", "Mk06", "Mk07", "Mk08", "Mk09",
              "Mk10", "La16", "La21"]


def load_benchmarks(data_dir: str, names=BENCHMARKS) -> list[Instance]:
    return [parse_fjs(os.path.join(data_dir, f"{n}.fjs"), n) for n in names]
