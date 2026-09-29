import os

import numpy as np
import pytest

from morl_fjsp.energy import load_energy_charts_day, synthetic_day
from morl_fjsp.fuzzy import credibility_leq, credibility_quantile, gini, owa, owa_weights
from morl_fjsp.instance import generate_instance, load_benchmarks
from morl_fjsp.sim import (C_DCO2, C_DCOST, C_DMK, C_DOWA, C_FEAS, DELAYS_MIN, Problem,
                           _decode, actions_of_state, compat_tables, evaluate_actions)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def brute_force_objectives(P, st):
    """Recompute f1..f3 from scratch from the final schedule."""
    cmax = int(st.op_e.max())
    D = np.zeros(cmax)
    for m in range(P.M):
        ops = np.flatnonzero(st.op_m == m)
        if len(ops) == 0:
            continue
        D[st.op_s[ops].min():st.op_e[ops].max()] += P.p_idle[m]   # on-window idle power
    for o in range(P.N):
        m = st.op_m[o]
        D[st.op_s[o]:st.op_e[o]] += P.p_proc[m] - P.p_idle[m]
    g = np.maximum(0.0, D - P.pv[:cmax])
    cost = float((P.price[:cmax] * g).sum() / 60000.0)
    co2 = float((P.co2[:cmax] * g).sum() / 60000.0)
    return cmax / 60.0, cost, co2


def random_rollout(P, rng):
    st = P.new_state()
    while not st.done:
        cand = st.candidates()
        i = rng.integers(len(cand))
        st.step(int(cand[i, 0]), int(cand[i, 2]), int(cand[i, 3]))
    return st


@pytest.fixture(scope="module")
def day():
    return load_energy_charts_day(os.path.join(ROOT, "data", "energy"))


def test_benchmarks_parse():
    insts = load_benchmarks(os.path.join(ROOT, "data", "fjsp"))
    ops = {i.name: i.n_ops for i in insts}
    assert ops == dict(Mk01=55, Mk02=58, Mk03=150, Mk04=90, Mk05=106, Mk06=150, Mk07=100,
                       Mk08=225, Mk09=240, Mk10=240, La16=100, La21=150)


def test_real_day(day):
    assert day.price.shape == (96,) and abs(day.price.min() - 114.49) < 1e-9
    assert abs(day.price.max() - 390.60) < 1e-9
    assert day.solar.max() == 1.0 and day.solar[:20].max() == 0.0


def test_credibility_closed_form():
    rng = np.random.default_rng(0)
    xs = np.linspace(-10, 60, 70001)
    for _ in range(30):
        a = rng.uniform(0, 20); b = a + rng.uniform(0.5, 10); c = b + rng.uniform(0.5, 20)
        mu = np.clip(np.minimum((xs - a) / (b - a), (c - xs) / (c - b)), 0, 1)
        for t in rng.uniform(a - 3, c + 3, size=10):
            pos = mu[xs <= t].max() if (xs <= t).any() else 0.0
            nec = 1.0 - (mu[xs > t].max() if (xs > t).any() else 0.0)
            assert abs(0.5 * (pos + nec) - credibility_leq(a, b, c, t)) < 2e-3
        for beta in (0.1, 0.5, 0.75, 0.9, 0.98):
            q = credibility_quantile(a, b, c, beta)
            assert abs(credibility_leq(a, b, c, q) - beta) < 1e-9
    # degenerate TFN cases (a == b and b == c)
    for a, b, c in ((10.0, 10.0, 20.0), (10.0, 20.0, 20.0), (15.0, 15.0, 15.0)):
        for beta in (0.2, 0.5, 0.8):
            q = credibility_quantile(a, b, c, beta)
            assert credibility_leq(a, b, c, q) >= beta - 1e-12


def test_owa_gini_identity():
    # Analytical M=2 check: r=(1, 0) => mean=1/2, G=1/2, M/(M+1)=2/3 => R_OWA = 2/3
    r2 = np.array([1.0, 0.0])
    w2 = owa_weights(2, "front")
    assert abs(owa(r2, w2) - 2.0 / 3.0) < 1e-12
    assert abs(r2.mean() * (1.0 + (2.0 / 3.0) * gini(r2)) - 2.0 / 3.0) < 1e-12
    rng = np.random.default_rng(1)
    for M in (2, 3, 5, 10, 17):
        w = owa_weights(M, "front")
        for _ in range(50):
            r = rng.uniform(0, 1, M) ** 2
            assert abs(owa(r, w) - r.mean() * (1 + M / (M + 1) * gini(r))) < 1e-12


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_incremental_objectives_match_brute_force(day, seed):
    rng = np.random.default_rng(seed)
    inst = generate_instance(rng, 6, 4)
    for dd in (day, synthetic_day(rng)):
        P = Problem(inst, dd, start_min=int(rng.integers(0, 1440)))
        st = random_rollout(P, rng)
        f = st.objectives()
        bf = brute_force_objectives(P, st)
        assert np.allclose(f[:3], bf, rtol=1e-10, atol=1e-9)


def test_candidate_deltas_are_exact(day):
    rng = np.random.default_rng(3)
    inst = generate_instance(rng, 5, 4)
    P = Problem(inst, day)
    st = P.new_state()
    while not st.done:
        cand = st.candidates().copy()
        f0 = st.objectives()
        for i in rng.choice(len(cand), size=min(6, len(cand)), replace=False):
            s2 = evaluate_actions(P, actions_of_state(st)) if st.sc[3] > 0 else P.new_state()
            s2.step(int(cand[i, 0]), int(cand[i, 2]), int(cand[i, 3]))
            f1 = s2.objectives()
            assert abs((f1[0] - f0[0]) * 60 - cand[i, C_DMK]) < 1e-9
            assert abs((f1[1] - f0[1]) - cand[i, C_DCOST]) < 1e-9
            assert abs((f1[2] - f0[2]) - cand[i, C_DCO2]) < 1e-9
            assert abs((f1[3] - f0[3]) - cand[i, C_DOWA]) < 1e-9
        i = rng.integers(len(cand))
        st.step(int(cand[i, 0]), int(cand[i, 2]), int(cand[i, 3]))


def test_replay_and_decoder_consistency(day):
    rng = np.random.default_rng(4)
    inst = generate_instance(rng, 6, 5)
    P = Problem(inst, day)
    st = random_rollout(P, rng)
    st2 = evaluate_actions(P, actions_of_state(st))
    assert np.allclose(st.objectives(), st2.objectives())
    # random-key decoder vs. environment replay
    idx, cnt = compat_tables(P)
    joblist = np.repeat(np.arange(P.J), P.job_n)
    for _ in range(5):
        x = rng.random(3 * P.N)
        seq = joblist[np.argsort(x[:P.N], kind="stable")]
        out = _decode(P.job_first, P.job_n, idx, cnt, P.proc, P.A, P.B, P.C, P.p_proc,
                      P.p_idle, P.price, P.pv, P.co2, P.delays, P.L, P.cap, P.beta, seq,
                      x[P.N:2 * P.N], x[2 * P.N:], P.H, 0.5)
        f_dec = np.array(out[:4])
        nxt = np.zeros(P.J, int)
        acts = []
        for j in seq:
            o = P.job_first[j] + nxt[j]; nxt[j] += 1
            acts.append((j, out[5][o], out[7][o]))
        st3 = evaluate_actions(P, acts)
        assert np.allclose(st3.objectives(), f_dec)
        assert abs(st3.violation() - out[4]) < 1e-12


def test_metrics():
    import moocore
    from morl_fjsp.metrics import (angular_error_deg, asf, das_dennis, eval_preferences,
                                   hypervolume, igd_plus)
    from morl_fjsp.train import preco_weights
    rng = np.random.default_rng(5)
    F = rng.random((12, 4))
    # Monte-Carlo check of the exact hypervolume
    S = rng.random((400000, 4)) * 1.1
    dom = np.zeros(len(S), bool)
    for f in F:
        dom |= (S >= f).all(1)
    assert abs(hypervolume(F) - dom.mean() * 1.1 ** 4) < 0.01
    R = rng.random((20, 4))
    assert abs(igd_plus(F, R) - moocore.igd_plus(F, ref=R)) < 1e-12
    assert das_dennis(4, 6).shape == (84, 4) and np.allclose(das_dennis(4, 6).sum(1), 1)
    W = eval_preferences()
    assert (W > 0).all() and np.allclose(W.sum(1), 1)
    w = np.array([0.4, 0.3, 0.2, 0.1])
    assert abs(angular_error_deg(2 * w, w)) < 1e-5
    assert abs(asf(0.5 * w / w.max(), w)[0] - 0.5) < 1e-12
    # PreCo detached achievement deficit direction gpsi
    A = rng.standard_normal((64, 4))
    u = np.array([0.5, 0.25, 0.2, 0.1])
    c, alpha, gpsi = preco_weights(A, u, w, lam=2.0)
    assert np.all(gpsi >= -1e-12) and abs(gpsi[int(np.argmax(u / w))]) < 1e-12
    assert np.all(c >= 0) and abs(c.sum() - 1.0) < 1e-9


def test_preference_analysis_metrics():
    from morl_fjsp.analysis import preference_metrics, responsiveness
    from morl_fjsp.metrics import eval_preferences
    W = eval_preferences()
    # perfectly responsive outputs: f_k decreases strictly with w_k
    F = 1.0 - W
    assert np.allclose(responsiveness(F, W), -1.0)
    # inverted response and infeasible rollouts ignored
    V = np.zeros(len(W)); V[:5] = 1.0
    assert np.allclose(responsiveness(W.copy(), W, V), 1.0)
    # clipped metrics stay bounded for outputs far outside the reference box
    R = np.array([[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    Fw = np.full((len(W), 4), 0.5); Fw[:, 1] = 50.0
    reg, ang, wsr = preference_metrics(Fw, W, R, R.min(0), R.max(0))
    assert np.all((reg >= -1e-12) & (reg <= 1.0)) and np.all(wsr <= 1.0) and np.all(ang <= 90.0)
