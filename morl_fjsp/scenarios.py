"""Training pool (synthetic instances x synthetic energy days) and test scenarios."""
from __future__ import annotations

import os
import pickle

import numpy as np

from .energy import load_energy_charts_day, synthetic_day
from .env import Scenario
from .instance import generate_instance, load_benchmarks
from .rules import RULES, run_rule
from .sim import Problem

TRAIN_SIZES = dict(jobs=(6, 15), machines=(4, 10))


def rule_front(P: Problem) -> np.ndarray:
    return np.array([run_rule(P, r).objectives() for r in RULES])


def bounds_from(F: np.ndarray):
    """Scenario scale from the dispatching-rule portfolio: per-objective best and worst
    rule values (the same definition is used for training and test scenarios)."""
    F = np.asarray(F)
    return F.min(0), F.max(0)


def make_training_pool(n: int, seed: int, cache: str | None = None) -> list[Scenario]:
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    rng = np.random.default_rng(seed)
    pool = []
    for i in range(n):
        J = int(rng.integers(TRAIN_SIZES["jobs"][0], TRAIN_SIZES["jobs"][1] + 1))
        M = int(rng.integers(TRAIN_SIZES["machines"][0], TRAIN_SIZES["machines"][1] + 1))
        # flexibility cap f ~ U{1..M}: from job-shop-like (f=1) to fully flexible instances,
        # covering the range of the public benchmarks (1.1 to 4.1 machines per operation)
        f = int(rng.integers(1, M + 1))
        inst = generate_instance(rng, J, M, name=f"train{i:04d}", flex_max=f)
        day = synthetic_day(rng, name=f"synday{i:04d}")
        start = int(rng.integers(0, 24)) * 60
        P = Problem(inst, day, start_min=start, pv_factor=float(rng.uniform(0.25, 0.75)))
        ideal, nadir = bounds_from(rule_front(P))
        pool.append(Scenario(P, ideal, nadir, name=inst.name))
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pickle.dump(pool, open(cache, "wb"))
    return pool


def benchmark_scenarios(data_root: str, start_min: int = 360, pv_factor: float = 0.5,
                        beta: float = 0.9, names=None, day=None) -> list[Scenario]:
    day = day or load_energy_charts_day(os.path.join(data_root, "energy"))
    insts = load_benchmarks(os.path.join(data_root, "fjsp"), *( [names] if names else []))
    out = []
    for inst in insts:
        P = Problem(inst, day, start_min=start_min, pv_factor=pv_factor, beta=beta)
        ideal, nadir = bounds_from(rule_front(P))
        out.append(Scenario(P, ideal, nadir, name=inst.name))
    return out
