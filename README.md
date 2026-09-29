# DP-PreCo-GRL: preference-conditioned graph reinforcement learning for energy-, carbon- and risk-aware flexible job-shop scheduling

Code, data, trained checkpoints and raw results of a critical evaluation of a preference-conditioned
graph policy for a four-objective flexible job-shop problem with triangular fuzzy processing times
(makespan, net grid electricity cost with on-site PV, grid CO2 emissions and an inequality-sensitive
workload-overload risk) under a credibility-gated capacity constraint. The policy is compared with
dispatching rules, a preference-weighted greedy heuristic and rule-seeded NSGA-II and NSGA-III that
share the same compiled decoder.

Every table and figure produced by the analysis is regenerated from the raw results in `results/`
by the scripts in `scripts/`. Nothing is typed in by hand.

## Main results

Means over the 12 public benchmarks with a real German 15-min energy day (5 training seeds per
policy, 10 seeds for NSGA, hypervolume ratio against the pooled reference front).

* The policies reach a hypervolume ratio of 0.078 (DP-PreCo-GRL) and 0.084 (LS-GRL), about twenty
  times the preference-weighted greedy heuristic (0.004) but far below rule-seeded NSGA-II (0.422) and
  NSGA-III (0.438), which are also faster than an 84-preference policy sweep (7.5 s against 12.0 s on
  one thread, plus 0.10 s of rule pre-pass on a new scenario).
* Preferences steer the makespan but not cost, emissions or overload risk (Spearman correlation between
  weight and objective: -0.69 for the makespan, +0.11, +0.08 and +0.11 for the others).
* No significant difference between the PreCo-inspired and the linear combination of advantages was
  detected. A five-seed replication of the ablations (`results/runs_rep`, second platform) shows that the
  expert archive matters most and finds no benefit of graph attention over a flat encoder.
* Ten policy schedules in the initial population raise NSGA-II from 0.422 to 0.512 at equal evaluations,
  but a longer NSGA-II run with the same computing time reaches 0.490.
* The credibility gate is empirically conservative under independent durations at the default level
  0.9 (0 failures in 2,016,000 executions per model for triangular and lognormal durations, 31 for
  uniform, over ten trained policies) but not at 0.7 (99.70% feasible executions for uniform durations).

## Repository layout

| Path | Content |
|---|---|
| `morl_fjsp/` | library: instances, energy signals, fuzzy credibility and OWA, exact Numba simulator, dispatching rules, NSGA-II/III baselines (pymoo), graph policy (PyTorch), PPO trainer, metrics, analysis, statistics, Monte-Carlo robustness |
| `scripts/` | entry points for every experiment, analysis, tables and figures |
| `configs/method_groups.json` | which trained runs form which method or ablation |
| `tests/` | unit tests (simulator against brute force, decoder consistency, credibility closed form, OWA-Gini identity, hypervolume, statistics, replay of expert schedules) |
| `data/fjsp/`, `data/energy/` | benchmark instances and the raw energy snapshot (see `DATA_LICENSES.md`) |
| `results/runs/` | trained policies (`model.pt` is the best validation checkpoint) with `config.json` and `history.json` |
| `results/runs_rep/` | second-platform replication of the principal ablations (35 runs) |
| `results/baselines/`, `results/policy_eval/`, `results/extra/` | raw objective vectors of all methods and supplementary experiments |
| `results/summary/` | indicators, statistics, reference fronts, timing and replication summaries |
| `results/logs/` | logs of all runs |
| `outputs/` | generated LaTeX tables, number macros and figures |

`results/cache/` (training scenarios, validation scenarios and expert archive, about 120 MB) is not
included. `scripts/build_training_data.py` regenerates it on a given platform.

## Installation and tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt     # exact versions used (Python 3.11)
python -m pytest -q tests/               # about 10 s
```

`requirements.txt` lists lower bounds only. The lock file pins the versions (NumPy 2.4, Numba 0.67,
PyTorch 2.14, pymoo 0.6.2, moocore 0.3.2).

## Reproducing the analysis from the stored results

```bash
python scripts/analyze.py              # indicators, statistics, tables and number macros
python scripts/analyze_replication.py  # replication of the ablations (frozen reference fronts)
python scripts/analyze_conflict.py     # real day versus carbon-inverted day
python scripts/make_tables.py
python scripts/make_figures.py
```

Outputs go to `outputs/generated` and `outputs/figures`.

## Reproducing the experiments

```bash
bash scripts/reproduce.sh                   # full pipeline, about 8 h on 4 cores
REPLICATION=1 bash scripts/reproduce.sh     # also the 35 replication runs (2 to 3 h on 6 to 7 cores)
```

The pipeline builds 256 synthetic training scenarios and an expert archive (`build_training_data.py`),
runs the baselines (`run_baselines.py`), trains 24 policies (`train_all.sh`), evaluates them and runs the
supplementary experiments (`post_training.sh`), and finally analyses everything. Wall-clock times of the
reported run on a 4-core Intel Xeon VM without GPU: training data 2.5 min, baselines 17 min, training of
24 policies 2.1 h, post-training pipeline 4.5 h.

## Reproducibility notes

* Evaluating stored checkpoints is exactly reproducible across machines. Re-running the 84-preference
  sweep of a stored policy on an Apple M1 Pro reproduced the stored objective vectors bit for bit.
  `scripts/analyze.py` reproduces the stored tables and number macros byte for byte.
* Training is deterministic within a platform but not across CPU architectures. The NSGA-II runs that
  build the expert archive diverge between an x86 VM and an ARM machine even with identical library
  versions. The archive of the original five main policies is therefore not regenerable bit for bit and
  is not included. The replication in `results/runs_rep` used an archive regenerated on the second
  platform (checksums in `results/summary/replication_cache.sha256`) and is evaluated against the frozen
  pooled reference front of the main study.
* Additional checks: `scripts/verify_expert_replay.py` (every archived expert schedule replays exactly),
  `scripts/count_sigma_guard.py` (floor of the rule-based objective scale) and `scripts/timing_coldstart.py`
  (rule pre-pass and cold versus cached latency).

## Data provenance and limitations

* Only one real energy day (DE-LU, 28 September 2026, operational snapshot retrieved at 22:00) was
  available. The raw API responses are in `data/energy/`, and `morl_fjsp/energy.py:load_energy_charts_day`
  parses any other day downloaded in the same format.
* Machine powers, fuzzy spreads and the time scale of each benchmark are synthetic annotations generated
  deterministically from the instance name (`morl_fjsp/instance.py`).
* `data/fjsp/instances_metadata.json` was truncated at download time. Complete records are parsed and
  missing values are left empty.

## Citation

If you use this software or its results, please cite the associated article (title: "Preference-conditioned
graph reinforcement learning for energy-, carbon- and risk-aware flexible job-shop scheduling with fuzzy
processing times: a critical evaluation", M. Bezoui, submitted to Knowledge-Based Systems) and this
repository (see `CITATION.cff`).

## Licence

MIT for the code and the results (see `LICENSE`). Third-party data keep their own terms
(`DATA_LICENSES.md`).
