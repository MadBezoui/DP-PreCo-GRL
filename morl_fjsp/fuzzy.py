"""Triangular fuzzy numbers, credibility measure, OWA aggregation and Gini index."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def credibility_leq(a, b, c, t):
    """Cr(xi <= t) for the TFN xi = (a, b, c) (Liu & Liu, IEEE TFS 2002)."""
    if t >= c:
        return 1.0
    if t < a or (t == a and a < b):
        return 0.0
    if t <= b:
        return 0.5 if b <= a else 0.5 * (t - a) / (b - a)
    return 0.5 + 0.5 * (t - b) / (c - b)


@njit(cache=True)
def credibility_quantile(a, b, c, beta):
    """Smallest t with Cr(xi <= t) >= beta (piecewise-linear inverse)."""
    if beta <= 0.5:
        return a + 2.0 * beta * (b - a)
    return b + (2.0 * beta - 1.0) * (c - b)


def owa_weights(M: int, scheme: str = "front") -> np.ndarray:
    h = np.arange(1, M + 1, dtype=np.float64)
    if scheme == "front":        # linearly decreasing: generalized-Gini weights
        w = 2.0 * (M - h + 1) / (M * (M + 1))
    elif scheme == "uniform":
        w = np.full(M, 1.0 / M)
    elif scheme == "back":
        w = 2.0 * h / (M * (M + 1))
    elif scheme == "max":
        w = np.zeros(M); w[0] = 1.0
    else:
        raise ValueError(scheme)
    return w


@njit(cache=True)
def owa_front(r):
    """OWA with front-loaded weights 2(M-h+1)/(M(M+1)) of the descending order stats."""
    M = r.shape[0]
    s = np.sort(r)[::-1]
    tot = 0.0
    for h in range(M):
        tot += 2.0 * (M - h) / (M * (M + 1.0)) * s[h]
    return tot


def owa(r: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(np.sort(r)[::-1], weights))


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    if mu <= 1e-12:
        return 0.0
    return float(np.abs(x[:, None] - x[None, :]).sum() / (2.0 * x.size ** 2 * mu))
