#!/usr/bin/env bash
# Full pipeline: every result table and figure.
# Set REPLICATION=1 to also run the second-platform replication of the principal ablations
# (35 additional training runs, about 2-3 h on 6-7 cores); it uses the cache built in step 1.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pytest -q tests/
python3 scripts/build_training_data.py
python3 scripts/verify_expert_replay.py      # every archived expert schedule replays exactly
python3 scripts/count_sigma_guard.py         # how often the floor of the rule-based scale is active
python3 scripts/run_baselines.py --workers "${WORKERS:-4}"
bash scripts/train_all.sh
python3 scripts/tune_seeding.py preco_s0   # validation-selected size of the policy seeding
bash scripts/post_training.sh      # evaluation, hybrids, long NSGA-II, supplementary, timing
if [ "${REPLICATION:-0}" = "1" ]; then
  (cd results/cache && shasum -a 256 *.pkl) > results/summary/replication_cache.sha256
  bash scripts/train_replication.sh
fi
python3 scripts/analyze.py
[ -d results/policy_eval/replication ] && python3 scripts/analyze_replication.py
python3 scripts/analyze_conflict.py
python3 scripts/horizon_audit.py --workers 4   # cyclic continuation of the 24-h profile (regenerates the policy schedules, a few minutes)
python3 scripts/make_tables.py
python3 scripts/make_figures.py
