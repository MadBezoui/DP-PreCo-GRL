"""Statistical helpers for the Monte-Carlo feasibility study and hierarchical bootstraps.

Nesting of the credibility experiment: instance (12) > preference / schedule (84) >
Monte-Carlo execution (200).  Executions of one schedule are conditionally independent
draws, schedules of one instance share the instance, so intervals that treat the 201,600
executions as i.i.d. are anti-conservative; the clustered bootstrap resamples instances
and, within them, schedules.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as _beta


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact two-sided (1-alpha) confidence interval of a binomial proportion."""
    k, n = int(k), int(n)
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def rule_of_three_upper(m: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) upper bound of a cluster-level event rate when 0 of m independent
    clusters showed the event (exact: 1 - alpha**(1/m); ~3/m for alpha=0.05)."""
    return 1.0 - alpha ** (1.0 / m)


def mc_summary(K, n_draws: int) -> dict:
    """K: list with one integer array per instance, entry = number of feasible Monte-Carlo
    draws (out of n_draws) of each schedule of that instance."""
    K = [np.asarray(k, dtype=np.int64) for k in K]
    allk = np.concatenate(K)
    n_sched, n_exec = len(allk), len(allk) * n_draws
    ok = int(allk.sum())
    inst_rate = np.array([k.sum() / (len(k) * n_draws) for k in K])
    strict_ok = int((allk == n_draws).sum())
    return dict(
        n_sched=n_sched, n_exec=n_exec, exec_ok=ok, exec_fail=n_exec - ok, exec_rate=ok / n_exec,
        exec_ci_iid=clopper_pearson(ok, n_exec),
        strict_ok=strict_ok, strict_rate=strict_ok / n_sched,
        strict_ci_iid=clopper_pearson(strict_ok, n_sched),
        worst_schedule=float(allk.min() / n_draws), worst_instance=float(inst_rate.min()),
        inst_rate=inst_rate.tolist(),
        n_inst_with_failure=int(sum((k < n_draws).any() for k in K)), n_inst=len(K))


def cluster_bootstrap(K, n_draws: int, stat: str, n_boot: int = 10000, seed: int = 0,
                      alpha: float = 0.05):
    """Two-stage (instances, then schedules within instance) percentile bootstrap of the
    execution-level feasibility rate (stat='exec') or of the share of schedules whose
    n_draws draws are all feasible (stat='strict')."""
    rng = np.random.default_rng(seed)
    if stat == "exec":
        vals = [np.asarray(k, float) / n_draws for k in K]
    elif stat == "strict":
        vals = [(np.asarray(k) == n_draws).astype(float) for k in K]
    else:
        raise ValueError(stat)
    I = len(vals)
    out = np.empty(n_boot)
    for b in range(n_boot):
        tot, cnt = 0.0, 0
        for i in rng.integers(0, I, I):
            v = vals[i]
            tot += v[rng.integers(0, len(v), len(v))].sum()
            cnt += len(v)
        out[b] = tot / cnt
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _cell_mean(spec, inst_idx, rng) -> float:
    if spec["kind"] == "crossed":            # same trained seeds evaluated on every instance
        M = spec["M"]
        cols = rng.integers(0, M.shape[1], M.shape[1])
        return float(M[np.ix_(inst_idx, cols)].mean())
    L = spec["L"]                            # independent runs within each instance
    return float(np.mean([L[i][rng.integers(0, len(L[i]), len(L[i]))].mean() for i in inst_idx]))


def _point(spec) -> float:
    return float(spec["M"].mean()) if spec["kind"] == "crossed" else float(
        np.mean([np.mean(x) for x in spec["L"]]))


def paired_bootstrap_diff(a, b, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Hierarchical bootstrap of mean(a) - mean(b), paired over instances.
    a, b: dict(kind='crossed', M=(instances, seeds)) or dict(kind='nested', L=[arrays])."""
    rng = np.random.default_rng(seed)
    I = a["M"].shape[0] if a["kind"] == "crossed" else len(a["L"])
    d = np.empty(n_boot)
    for t in range(n_boot):
        rows = rng.integers(0, I, I)
        d[t] = _cell_mean(a, rows, rng) - _cell_mean(b, rows, rng)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return _point(a) - _point(b), float(lo), float(hi)
