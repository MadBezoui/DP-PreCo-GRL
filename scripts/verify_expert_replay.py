"""Replay every schedule of an expert archive (results/cache/expert.pkl) with the append-only
decoder in the order (start time, global operation index) and check that

  * machine, postponement, start and completion times of every operation are reproduced exactly
    (integer schedule), and the objective vector to floating-point precision (relative
    difference below 1e-12: cost and risk are accumulated incrementally, so the summation order
    differs between the original decoder pass and the replay);
  * the objective vector equals the one stored with the archive entry (same tolerance);
  * every action of a random archive member of each scenario is an available candidate in the
    environment used for behaviour cloning (replay_for_bc drops none).

Usage: python scripts/verify_expert_replay.py [cache_dir]      Output: results/summary/expert_replay.json"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.sim import actions_of_state, evaluate_actions  # noqa: E402
from morl_fjsp.train import replay_for_bc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "cache")
pool = pickle.load(open(os.path.join(cache, "pool.pkl"), "rb"))
expert = pickle.load(open(os.path.join(cache, "expert.pkl"), "rb"))
n = bad_int = bad_obj = bad_cand = 0
worst = 0.0
rng = np.random.default_rng(0)
for k, items in expert.items():
    sc = pool[k]
    for F, acts in items:
        n += 1
        st = evaluate_actions(sc.P, acts)                     # archived order
        st2 = evaluate_actions(sc.P, actions_of_state(st))    # (start time, operation index) order
        if not (np.array_equal(st.op_s, st2.op_s) and np.array_equal(st.op_e, st2.op_e)
                and np.array_equal(st.op_m, st2.op_m) and np.array_equal(st.op_d, st2.op_d)):
            bad_int += 1
        f0, f1, fs = st.objectives(), st2.objectives(), np.asarray(F)
        rel = max(np.max(np.abs(f1 - f0) / np.maximum(np.abs(f0), 1e-12)),
                  np.max(np.abs(fs - f0) / np.maximum(np.abs(f0), 1e-12)))
        worst = max(worst, float(rel))
        bad_obj += int(rel > 1e-12)
    F, acts = items[int(rng.integers(len(items)))]
    data = replay_for_bc([sc], np.array([[0.25] * 4]), [acts])
    bad_cand += int(len(data) != len(acts))
res = dict(archive=os.path.relpath(cache, ROOT), scenarios=len(expert), schedules=n,
           integer_schedule_mismatches=bad_int, objective_mismatches_above_1e12=bad_obj,
           max_relative_objective_difference=worst, scenarios_with_unavailable_action=bad_cand)
json.dump(res, open(os.path.join(ROOT, "results", "summary", "expert_replay.json"), "w"), indent=1)
print(res)
