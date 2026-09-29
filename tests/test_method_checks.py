"""Method checks: floor of the objective scale, PreCo weights (normalised and un-normalised), replay of expert schedules."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from morl_fjsp.energy import load_energy_charts_day  # noqa: E402
from morl_fjsp.env import Scenario  # noqa: E402
from morl_fjsp.instance import generate_instance  # noqa: E402
from morl_fjsp.sim import Problem, actions_of_state, evaluate_actions  # noqa: E402
from test_core import random_rollout  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_sigma_guard_is_positive_and_scaled():
    ideal = np.array([10.0, 0.0, 5.0, 0.0])
    nadir = np.array([10.0, 0.0, 6.0, 0.0])           # zero range in objectives 1, 2 and 4
    span = Scenario(P=None, ideal=ideal, nadir=nadir).span
    assert (span > 0).all()
    assert abs(span[0] - 1e-3 * 10.0) < 1e-12          # 1e-3 * max(|nadir|, 1)
    assert abs(span[1] - 1e-3) < 1e-12                  # nadir = 0 -> 1e-3 * 1
    assert abs(span[2] - 1.0) < 1e-12                   # regular range untouched


def test_replay_order_is_start_time_then_operation_index_and_exact():
    day = load_energy_charts_day(os.path.join(ROOT, "data", "energy"))
    rng = np.random.default_rng(11)
    ties_seen = 0
    for _ in range(20):
        P = Problem(generate_instance(rng, 8, 5), day)
        st = random_rollout(P, rng)
        acts = actions_of_state(st)
        st2 = evaluate_actions(P, acts)
        assert np.array_equal(st.op_s, st2.op_s) and np.array_equal(st.op_e, st2.op_e)
        assert np.array_equal(st.op_m, st2.op_m) and np.array_equal(st.op_d, st2.op_d)
        # integer schedule is identical; the objective vector agrees to floating-point rounding
        # (cost/risk are accumulated incrementally, so the summation order differs)
        assert np.allclose(st.objectives(), st2.objectives(), rtol=1e-12, atol=0.0)
        # order key: (start time, global operation index)
        nxt = np.zeros(P.J, int)
        keys = []
        for j, m, d in acts:
            o = P.job_first[j] + nxt[j]
            nxt[j] += 1
            keys.append((int(st.op_s[o]), int(o)))
        assert keys == sorted(keys)
        ties_seen += len(keys) - len({k[0] for k in keys})
    assert ties_seen > 0            # the test does exercise equal start times


def test_preco_raw_vs_normalised_direction():
    from morl_fjsp.train import preco_weights
    rng = np.random.default_rng(5)
    A = rng.standard_normal((64, 4))
    u = np.array([0.5, 0.25, 0.2, 0.1])
    w = np.array([0.4, 0.3, 0.2, 0.1])
    c_n, alpha, g = preco_weights(A, u, w, lam=3.0)                 # default: normalised
    c_r, alpha_r, g_r = preco_weights(A, u, w, lam=3.0, normalise=False)
    assert np.allclose(alpha, alpha_r) and np.allclose(g, g_r)
    assert np.allclose(c_r, alpha + 3.0 * g)                         # Yang et al. Eq. (6)-(7) scale
    assert abs(c_n.sum() - 1.0) < 1e-12 and np.allclose(c_n, c_r / c_r.sum())
    assert c_r.sum() >= 1.0 - 1e-12                                  # 1^T alpha = 1, g >= 0
