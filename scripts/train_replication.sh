#!/usr/bin/env bash
# Second-platform replication of the principal ablations: 7 variants x 5 seeds, each trained with
# the training cache (scenario pool, validation set, expert archive) of the *current* platform
# (results/cache, built by scripts/build_training_data.py).  Runs are written to results/runs_rep
# and evaluated on the benchmarks with tag "replication"; they are never added to the pooled
# reference front of the main study (scripts/analyze_replication.py uses the frozen front).
# Usage: PARALLEL=6 bash scripts/train_replication.sh
cd "$(dirname "$0")/.."
mkdir -p results/logs results/runs_rep
{
for s in 0 1 2 3 4; do
  echo "--name rep_preco_s$s --mode preco --seed $s"
  echo "--name rep_linear_s$s --mode linear --seed $s"
  echo "--name rep_nogat_s$s --mode preco --seed $s --no-gat"
  echo "--name rep_nofilm_s$s --mode preco --seed $s --no-film"
  echo "--name rep_nocrit_s$s --mode preco --seed $s --no-crit"
  echo "--name rep_noexpert_s$s --mode preco --seed $s --no-expert"
  echo "--name rep_precoraw_s$s --mode preco --seed $s --preco-raw"
done
} | xargs -P "${PARALLEL:-6}" -I{} sh -c 'n=$(echo "{}" | cut -d" " -f2); python3 scripts/train_policy.py {} --threads 1 --out results/runs_rep > results/logs/train_$n.log 2>&1'
RUNS=$(ls -d results/runs_rep/*/model.pt | xargs -n1 dirname | xargs -n1 basename)
echo "$RUNS" | xargs -P "${PARALLEL:-6}" -I{} sh -c 'python3 scripts/evaluate_policies.py --runs-dir results/runs_rep --tag replication --runs {} > results/logs/eval_rep_{}.log 2>&1'
echo "replication done"
