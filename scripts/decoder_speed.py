"""Time of one random-key decoding (= one NSGA evaluation) per benchmark, with the
Numba-compiled simulator and, when run with NUMBA_DISABLE_JIT=1, with the same code
interpreted by CPython.  Output: results/summary/decoder_speed_<jit|python>.json"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.energy import load_energy_charts_day  # noqa: E402
from morl_fjsp.instance import load_benchmarks  # noqa: E402
from morl_fjsp.moea import RandomKeyProblem  # noqa: E402
from morl_fjsp.sim import Problem  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mode = "python" if os.environ.get("NUMBA_DISABLE_JIT") == "1" else "jit"
n = 3 if mode == "python" else 300
day = load_energy_charts_day(os.path.join(ROOT, "data", "energy"))
rng = np.random.default_rng(0)
out = {}
for inst in load_benchmarks(os.path.join(ROOT, "data", "fjsp")):
    prob = RandomKeyProblem(Problem(inst, day))
    X = rng.random((n, prob.n_var))
    prob.decode_many(X[:1])                                   # compilation / warm-up
    t = time.perf_counter()
    prob.decode_many(X)
    out[inst.name] = 1000 * (time.perf_counter() - t) / n     # ms per decoding
    print(mode, inst.name, round(out[inst.name], 3), "ms", flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "summary", f"decoder_speed_{mode}.json"), "w"),
          indent=1)
