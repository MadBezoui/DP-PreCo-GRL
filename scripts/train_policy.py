"""Train one policy variant.

Examples
  python scripts/train_policy.py --name preco_s0 --mode preco --seed 0
  python scripts/train_policy.py --name linear_s0 --mode linear --seed 0
  python scripts/train_policy.py --name nogat_s0 --no-gat --seed 0
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morl_fjsp.train import Archive, TrainConfig, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--mode", default="preco", choices=["preco", "linear"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--iters", type=int, default=120)
ap.add_argument("--n-envs", type=int, default=32)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--no-gat", action="store_true")
ap.add_argument("--no-film", action="store_true")
ap.add_argument("--no-crit", action="store_true", help="remove the bilinear criteria head")
ap.add_argument("--no-delay", action="store_true")
ap.add_argument("--no-mask", action="store_true")
ap.add_argument("--no-expert", action="store_true")
ap.add_argument("--no-elite", action="store_true")
ap.add_argument("--preco-raw", action="store_true",
                help="un-normalised PreCo weights c = alpha* + lam*gpsi (Yang et al. scale)")
ap.add_argument("--cache", default="results/cache")
ap.add_argument("--out", default="results/runs")
args = ap.parse_args()

pool = pickle.load(open(os.path.join(args.cache, "pool.pkl"), "rb"))
val, val_ref = pickle.load(open(os.path.join(args.cache, "val.pkl"), "rb"))
expert = None
if not args.no_expert:
    raw = pickle.load(open(os.path.join(args.cache, "expert.pkl"), "rb"))
    expert = Archive(cap=10 ** 6)
    expert.items = {k: list(v) for k, v in raw.items()}

cfg = TrainConfig(name=args.name, mode=args.mode, seed=args.seed, iters=args.iters,
                  n_envs=args.n_envs, threads=args.threads, gat=not args.no_gat,
                  film=not args.no_film, crit=not args.no_crit, delay=not args.no_delay,
                  mask_capacity=not args.no_mask, out_dir=args.out,
                  preco_norm=not args.preco_raw)
if args.no_elite:
    cfg.bc_elite = (0.0, 0.0)
if args.no_expert:
    cfg.bc_exp = (0.0, 0.0)
train(cfg, pool, expert, val, val_ref, log=lambda s: print(s, flush=True))
