import numpy as np
from scipy.stats import binomtest

from morl_fjsp.stats import (clopper_pearson, cluster_bootstrap, mc_summary,
                             paired_bootstrap_diff, rule_of_three_upper)


def test_clopper_pearson_matches_scipy_exact():
    for k, n in [(970, 1008), (1007, 1008), (0, 10), (10, 10), (201432, 201600)]:
        lo, hi = clopper_pearson(k, n)
        ref = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        assert abs(lo - ref.low) < 1e-9 and abs(hi - ref.high) < 1e-9


def test_reported_interval_was_for_1007_of_1008():
    # the interval printed in the previous manuscript version, [99.45 %, 100 %], is the
    # Clopper-Pearson interval of 1007/1008; the true count at beta=0.7 (uniform) is 970/1008
    lo, _ = clopper_pearson(1007, 1008)
    assert abs(lo - 0.9945) < 5e-4
    lo970, hi970 = clopper_pearson(970, 1008)
    assert 0.948 < lo970 < 0.952 and 0.972 < hi970 < 0.976


def test_rule_of_three_upper():
    assert abs(rule_of_three_upper(12) - (1 - 0.05 ** (1 / 12))) < 1e-12
    assert 0.22 < rule_of_three_upper(12) < 0.23


def test_mc_summary_counts():
    K = [np.array([200, 200, 199]), np.array([200, 200, 200])]
    s = mc_summary(K, 200)
    assert s["n_sched"] == 6 and s["n_exec"] == 1200
    assert s["exec_fail"] == 1 and s["strict_ok"] == 5
    assert s["n_inst_with_failure"] == 1
    assert abs(s["worst_schedule"] - 199 / 200) < 1e-12
    assert abs(s["worst_instance"] - (599 / 600)) < 1e-12


def test_cluster_bootstrap_degenerate_and_bounds():
    K = [np.full(84, 200) for _ in range(12)]
    lo, hi = cluster_bootstrap(K, 200, "exec", n_boot=200, seed=1)
    assert lo == hi == 1.0
    K[3] = np.full(84, 190)
    lo, hi = cluster_bootstrap(K, 200, "exec", n_boot=2000, seed=1)
    point = np.mean([k.mean() / 200 for k in K])
    assert 0.0 <= lo <= point <= hi <= 1.0 and hi > lo


def test_paired_bootstrap_diff():
    rng = np.random.default_rng(0)
    A = dict(kind="nested", L=[0.5 + 0.01 * rng.standard_normal(10) for _ in range(12)])
    B = dict(kind="crossed", M=0.2 + 0.01 * rng.standard_normal((12, 5)))
    point, lo, hi = paired_bootstrap_diff(A, B, n_boot=2000, seed=2)
    assert abs(point - 0.3) < 0.01 and lo < point < hi and lo > 0.25
    same = dict(kind="crossed", M=rng.random((12, 5)))
    p2, lo2, hi2 = paired_bootstrap_diff(same, same, n_boot=500, seed=3)
    assert p2 == 0.0 and lo2 <= 0.0 <= hi2
