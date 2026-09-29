#!/usr/bin/env bash
# Train all policy variants reported in the paper (4 runs in parallel, 1 thread each).
cd "$(dirname "$0")/.."
mkdir -p results/logs
{
for s in 0 1 2 3 4; do echo "--name preco_s$s --mode preco --seed $s"; echo "--name linear_s$s --mode linear --seed $s"; done
for s in 0 1; do
  echo "--name nogat_s$s --mode preco --seed $s --no-gat"
  echo "--name nofilm_s$s --mode preco --seed $s --no-film"
  echo "--name nocrit_s$s --mode preco --seed $s --no-crit"
  echo "--name noexpert_s$s --mode preco --seed $s --no-expert"
  echo "--name noelite_s$s --mode preco --seed $s --no-elite"
  echo "--name nodelay_s$s --mode preco --seed $s --no-delay"
  echo "--name nomask_s$s --mode preco --seed $s --no-mask"
done
} | xargs -P "${PARALLEL:-4}" -I{} sh -c 'n=$(echo "{}" | cut -d" " -f2); python3 scripts/train_policy.py {} --threads 1 > results/logs/train_$n.log 2>&1'
