"""Quality indicators, preference-alignment metrics and statistical tests."""
from __future__ import annotations

import itertools

import moocore
import numpy as np
from scipy import stats

HV_REF = 1.1


def nondominated(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    if len(F) == 0:
        return F
    F = np.unique(F, axis=0)
    return F[moocore.is_nondominated(F)]


def normalise(F, ideal, nadir):
    span = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
    return (np.asarray(F) - ideal) / span


def hypervolume(Fn: np.ndarray, ref: float = HV_REF) -> float:
    """Exact hypervolume in the normalised space w.r.t. the point (ref,...,ref)."""
    Fn = np.asarray(Fn, dtype=np.float64)
    if len(Fn) == 0:
        return 0.0
    Fn = Fn[(Fn < ref).all(axis=1)]
    if len(Fn) == 0:
        return 0.0
    return float(moocore.hypervolume(Fn, ref=np.full(Fn.shape[1], ref)))


def igd_plus(Fn: np.ndarray, Rn: np.ndarray) -> float:
    """IGD+ (Ishibuchi et al., EMO 2015) of approximation Fn w.r.t. reference set Rn."""
    Fn, Rn = np.asarray(Fn), np.asarray(Rn)
    d = np.maximum(Fn[None, :, :] - Rn[:, None, :], 0.0)   # (|R|, |F|, m)
    return float(np.sqrt((d ** 2).sum(-1)).min(axis=1).mean())


def das_dennis(m: int, p: int) -> np.ndarray:
    pts = []
    for c in itertools.combinations(range(p + m - 1), m - 1):
        prev, w = -1, []
        for x in c:
            w.append(x - prev - 1)
            prev = x
        w.append(p + m - 2 - prev)
        pts.append(np.array(w, dtype=np.float64) / p)
    return np.array(pts)


def eval_preferences(m: int = 4, p: int = 6, shrink: float = 0.1) -> np.ndarray:
    """Interior Das-Dennis preference set: (1-shrink) w_DD + shrink/m (84 vectors)."""
    W = das_dennis(m, p)
    return (1.0 - shrink) * W + shrink / m


def utilities(Fn: np.ndarray) -> np.ndarray:
    """Normalised utilities u = 1 - f_hat (higher is better; 0 = nadir, 1 = ideal)."""
    return 1.0 - np.asarray(Fn)


def asf(U: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Direction-based achievement function (weighted Tchebycheff from the nadir):
    a_w(u) = max_j w_j * min_k u_k / w_k  in [0, 1]; maximal on the Pareto point
    lying on the ray {t w, t >= 0}."""
    U = np.atleast_2d(U)
    return w.max() * (U / w).min(axis=1)


def asf_regret(u: np.ndarray, w: np.ndarray, U_ref: np.ndarray) -> float:
    return float(asf(U_ref, w).max() - asf(u, w)[0])


def angular_error_deg(u: np.ndarray, w: np.ndarray) -> float:
    u = np.maximum(np.asarray(u, dtype=np.float64), 0.0)
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        return 90.0
    c = float(np.dot(u, w) / (nu * np.linalg.norm(w)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def a12(x, y) -> float:
    """Vargha-Delaney A12: probability that a value from x exceeds one from y."""
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    n = len(p)
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mannwhitney(x, y) -> float:
    if np.allclose(np.var(x), 0) and np.allclose(np.var(y), 0) and np.allclose(np.mean(x), np.mean(y)):
        return 1.0
    return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def wilcoxon(x, y) -> float:
    d = np.asarray(x) - np.asarray(y)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)


def friedman(*groups) -> float:
    return float(stats.friedmanchisquare(*groups).pvalue)


def bootstrap_ci(x, n_boot=5000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
