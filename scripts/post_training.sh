#!/usr/bin/env bash
# Everything that needs the trained policies (run after scripts/train_all.sh).
set -uo pipefail
cd "$(dirname "$0")/.."
P=${PARALLEL:-4}
# 0) warm-start-only policies (behaviour cloning, no PPO) for both target semantics
{ for s in 0 1 2 3 4; do echo "--name bconly_preco_s$s --mode preco --seed $s --iters 0"; echo "--name bconly_linear_s$s --mode linear --seed $s --iters 0"; done; } | \
  xargs -P "$P" -I{} sh -c 'n=$(echo "{}" | cut -d" " -f2); python3 scripts/train_policy.py {} --threads 1 > results/logs/train_$n.log 2>&1'
RUNS=$(ls -d results/runs/*/model.pt | xargs -n1 dirname | xargs -n1 basename | grep -v pilot)
# 1) zero-shot evaluation of every run (parallel)
echo "$RUNS" | xargs -P "$P" -I{} sh -c 'python3 scripts/evaluate_policies.py --runs {} > results/logs/eval_{}.log 2>&1'
# 1b) sampled inference for the main runs: 8 stochastic rollouts per preference in addition
#     to the greedy one (quality/time trade-off of the policy alone)
ls -d results/runs/preco_s* results/runs/linear_s* | xargs -n1 basename | \
  xargs -P "$P" -I{} sh -c 'python3 scripts/evaluate_policies.py --runs {} --tag sampled --samples 8 > results/logs/eval_sampled_{}.log 2>&1'
# 2) hybrids: every main run seeds NSGA-II and NSGA-III (2 seeds each); the number of
#    injected policy schedules is selected on the validation scenarios (not the benchmarks)
[ -f results/summary/seeding_validation.json ] || \
  python3 scripts/tune_seeding.py preco_s0 > results/logs/tune_seeding.log 2>&1
ls -d results/runs/preco_s* results/runs/linear_s* | xargs -n1 basename | \
  xargs -P "$P" -I{} sh -c 'python3 scripts/run_extra_experiments.py hybrid --runs {} --seeds 2 > results/logs/hybrid_{}.log 2>&1'
# 3) long rule-seeded NSGA-II runs (2,000 generations) for time-matched comparisons
python3 scripts/run_baselines.py --tag long --gens 2000 --seeds 5 --algos nsga2 --workers "$P" > results/logs/baselines_long.log 2>&1
#    and NSGA-II with a random initial population (no rule seeding), 3 seeds
python3 scripts/run_baselines.py --tag unseeded --seeds 3 --algos nsga2 --unseeded --workers "$P" > results/logs/baselines_unseeded.log 2>&1
# 4) supplementary experiments (no timing measured), P jobs in parallel: credibility sweep,
#    PV sensitivity, validation reference levels, robustness to energy day / shift start
{
  echo "python3 scripts/run_extra_experiments.py credibility --runs preco_s0 > results/logs/extra_credibility.log 2>&1"
  echo "python3 scripts/run_extra_experiments.py pv --runs preco_s0 > results/logs/extra_pv.log 2>&1"
  echo "python3 scripts/val_reference.py > results/logs/val_reference.log 2>&1"
  echo "python3 scripts/val_responsiveness.py > results/logs/val_responsiveness.log 2>&1"
  echo "python3 scripts/run_extra_experiments.py credseeds --runs preco_s0 preco_s1 preco_s2 preco_s3 preco_s4 linear_s0 linear_s1 linear_s2 linear_s3 linear_s4 > results/logs/extra_credseeds.log 2>&1"
  echo "python3 scripts/run_extra_experiments.py conflict --runs preco_s0 preco_s1 preco_s2 linear_s0 linear_s1 linear_s2 > results/logs/extra_conflict.log 2>&1"
  for c in 0 1 2 3 4 5 6; do
    echo "python3 scripts/run_extra_experiments.py days --cond $c --runs preco_s0 preco_s1 preco_s2 linear_s0 linear_s1 linear_s2 > results/logs/extra_days_$c.log 2>&1"
  done
} | xargs -d '\n' -P "$P" -I{} sh -c '{}'
# 5) timing-sensitive measurements, sequentially on an otherwise idle machine
python3 scripts/run_extra_experiments.py scale --runs preco_s0 > results/logs/extra_scale.log 2>&1
python3 scripts/timing.py preco_s0 linear_s0 > results/logs/timing.log 2>&1
python3 scripts/timing_coldstart.py preco_s0 nogat_s0 > results/logs/timing_coldstart.log 2>&1   # pre-pass, cold vs cached, graph vs flat
python3 scripts/decoder_speed.py > results/logs/decoder_speed.log 2>&1
NUMBA_DISABLE_JIT=1 python3 scripts/decoder_speed.py >> results/logs/decoder_speed.log 2>&1
echo "post-training done"
