"""Loading trained policies and running batched preference sweeps."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .env import Scenario, StaticGraph
from .model import PolicyNet
from .train import TrainConfig, rollout


def load_policy(run_dir: str):
    cfgd = json.load(open(os.path.join(run_dir, "config.json")))
    cfg = TrainConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfgd.items()})
    net = PolicyNet(d=cfg.d_model, gat=cfg.gat, film=cfg.film, crit=getattr(cfg, "crit", True))
    net.load_state_dict(torch.load(os.path.join(run_dir, "model.pt")))
    net.eval()
    return net, cfg


def sweep(net, cfg, sc: Scenario, W: np.ndarray, greedy=True, seed=0, static=None,
          mask_capacity=None):
    """Roll out one scenario for all preferences in W (one batch)."""
    rng = np.random.default_rng(seed)
    G = static or StaticGraph(sc.P)
    t = time.perf_counter()
    _, acts, finals = rollout(net, [sc] * len(W), W, rng, greedy=greedy, store=False,
                              statics=[G] * len(W),
                              mask_capacity=cfg.mask_capacity if mask_capacity is None else mask_capacity,
                              allow_delay=cfg.delay)
    wall = time.perf_counter() - t
    F = np.array([f[0] for f in finals])
    V = np.array([f[1] for f in finals])
    return F, V, acts, wall
