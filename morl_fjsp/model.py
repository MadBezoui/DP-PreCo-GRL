"""Preference-conditioned heterogeneous-graph actor-critic (PyTorch, CPU friendly).

Encoder : relation-specific graph attention over operation/machine nodes
          (op<-machine and machine<-op attention on compatibility edges with edge
          features; op<-op mean aggregation on precedence edges).
Actor   : scores every candidate (operation, machine, delay) with a FiLM-modulated
          MLP; masked softmax over the candidates of each environment.
Critic  : FiLM-modulated MLP with one output per objective (vector value V(s, w)).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .env import D_CAND, D_EDGE, D_GLOB, D_MACH, D_OP

D_EDGE_DYN = D_EDGE


# ---------------------------------------------------------------------------- utils
def seg_softmax(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of x (E, ...) within segments seg (E,)."""
    shape = (n,) + x.shape[1:]
    idx = seg.view(-1, *([1] * (x.dim() - 1))).expand_as(x)
    mx = torch.full(shape, -torch.inf, dtype=x.dtype).scatter_reduce(0, idx, x, "amax",
                                                                     include_self=True)
    ex = torch.exp(x - mx.gather(0, idx))
    s = torch.zeros(shape, dtype=x.dtype).scatter_add(0, idx, ex)
    return ex / (s.gather(0, idx) + 1e-16)


def seg_log_softmax(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    mx = torch.full((n,), -torch.inf, dtype=x.dtype).scatter_reduce(0, seg, x, "amax",
                                                                    include_self=True)
    z = x - mx[seg]
    s = torch.zeros(n, dtype=x.dtype).index_add(0, seg, torch.exp(z))
    return z - torch.log(s)[seg]


def seg_mean(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    s = torch.zeros((n, x.shape[1]), dtype=x.dtype).index_add(0, seg, x)
    c = torch.zeros(n, dtype=x.dtype).index_add(0, seg, torch.ones_like(seg, dtype=x.dtype))
    return s / c.clamp_min(1.0)[:, None]


# ---------------------------------------------------------------------------- layers
class FiLMMLP(nn.Module):
    """MLP whose hidden layers are modulated by FiLM(gamma(w), beta(w)); with
    film=False the preference is only concatenated to the input."""

    def __init__(self, d_in, hidden, d_out, film=True, d_pref=4, d_hyper=64):
        super().__init__()
        self.film = film
        dims = [d_in + d_pref] + list(hidden)
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = nn.Linear(dims[-1], d_out)
        if film:
            self.hyper = nn.Sequential(nn.Linear(d_pref, d_hyper), nn.ReLU(),
                                       nn.Linear(d_hyper, 2 * sum(hidden)))
            nn.init.zeros_(self.hyper[-1].weight)
            nn.init.zeros_(self.hyper[-1].bias)
        self.hidden = list(hidden)

    def forward(self, x, w):
        h = torch.cat([x, w], dim=-1)
        if self.film:
            gb = self.hyper(w)
            off = 0
        for i, lin in enumerate(self.layers):
            h = lin(h)
            if self.film:
                d = self.hidden[i]
                gamma = 1.0 + gb[:, off:off + d]
                beta = gb[:, off + d:off + 2 * d]
                off += 2 * d
                h = gamma * h + beta
            h = Fnn.relu(h)
        return self.out(h)


class HetGATLayer(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q_o = nn.Linear(d, d, bias=False)
        self.k_m = nn.Linear(d + D_EDGE_DYN, d, bias=False)
        self.q_m = nn.Linear(d, d, bias=False)
        self.k_o = nn.Linear(d + D_EDGE_DYN, d, bias=False)
        self.a_mo = nn.Parameter(torch.randn(heads, 2 * self.dh) * 0.1)
        self.a_om = nn.Parameter(torch.randn(heads, 2 * self.dh) * 0.1)
        self.prec = nn.Linear(d, d)
        self.upd_o = nn.Linear(3 * d, d)
        self.upd_m = nn.Linear(2 * d, d)
        self.ln_o = nn.LayerNorm(d)
        self.ln_m = nn.LayerNorm(d)

    def _attend(self, q_dst, k_src, a, dst, n_dst):
        H, dh = self.h, self.dh
        q = q_dst.view(-1, H, dh)
        k = k_src.view(-1, H, dh)
        e = Fnn.leaky_relu((torch.cat([q, k], -1) * a).sum(-1), 0.2)     # (E, H)
        alpha = seg_softmax(e, dst, n_dst)
        msg = (alpha[..., None] * k).reshape(-1, H * dh)
        out = torch.zeros((n_dst, H * dh), dtype=msg.dtype).index_add(0, dst, msg)
        return out

    def forward(self, ho, hm, g):
        eo, em, ef = g["c_op"], g["c_m"], g["c_feat"]
        # machine -> operation
        k_m = self.k_m(torch.cat([hm[em], ef], -1))
        agg_o = self._attend(self.q_o(ho)[eo], k_m, self.a_mo, eo, ho.shape[0])
        # operation -> machine
        k_o = self.k_o(torch.cat([ho[eo], ef], -1))
        agg_m = self._attend(self.q_m(hm)[em], k_o, self.a_om, em, hm.shape[0])
        # precedence (mean over job neighbours)
        ps, pd = g["p_src"], g["p_dst"]
        prec = torch.zeros_like(ho).index_add(0, pd, self.prec(ho)[ps])
        cnt = torch.zeros(ho.shape[0]).index_add(0, pd, torch.ones(pd.shape[0])).clamp_min(1)
        prec = prec / cnt[:, None]
        ho2 = self.ln_o(ho + Fnn.relu(self.upd_o(torch.cat([ho, agg_o, prec], -1))))
        hm2 = self.ln_m(hm + Fnn.relu(self.upd_m(torch.cat([hm, agg_m], -1))))
        return ho2, hm2


class PolicyNet(nn.Module):
    def __init__(self, d=64, layers=2, heads=4, gat=True, film=True, crit=True):
        super().__init__()
        self.gat_enabled = gat
        self.crit_enabled = crit
        self.emb_o = nn.Sequential(nn.Linear(D_OP, d), nn.ReLU(), nn.Linear(d, d))
        self.emb_m = nn.Sequential(nn.Linear(D_MACH, d), nn.ReLU(), nn.Linear(d, d))
        self.gnn = nn.ModuleList([HetGATLayer(d, heads) for _ in range(layers)]) if gat else None
        d_ctx = 2 * d + D_GLOB
        # pair-level (operation, machine) scorer + light delay-specific head
        self.actor = FiLMMLP(2 * d + d_ctx, [128, 64], 64, film=film)
        self.delay_head = nn.Sequential(nn.Linear(64 + D_CAND, 64), nn.ReLU(), nn.Linear(64, 1))
        # bilinear criteria path: v(s, w)^T x_c, a state- and preference-dependent
        # weighting of the candidate criteria (direct gradient path to the logits)
        self.crit = FiLMMLP(d_ctx, [64], D_CAND, film=film)
        nn.init.zeros_(self.crit.out.weight)
        nn.init.zeros_(self.crit.out.bias)
        self.critic = FiLMMLP(d_ctx, [128, 64], 4, film=film)

    def encode(self, g):
        ho = self.emb_o(g["op"])
        hm = self.emb_m(g["mach"])
        if self.gnn is not None:
            for layer in self.gnn:
                ho, hm = layer(ho, hm, g)
        B = g["w"].shape[0]
        ctx = torch.cat([seg_mean(ho, g["op_env"], B), seg_mean(hm, g["mach_env"], B),
                         g["glob"]], -1)
        return ho, hm, ctx

    def forward(self, g):
        ho, hm, ctx = self.encode(g)
        pe = g["pair_env"]
        z = self.actor(torch.cat([ho[g["pair_op"]], hm[g["pair_m"]], ctx[pe]], -1), g["w"][pe])
        z = Fnn.relu(z)
        logits = self.delay_head(torch.cat([z[g["cand_pair"]], g["cand"]], -1)).squeeze(-1)
        if self.crit_enabled:
            v = self.crit(ctx, g["w"])
            logits = logits + (v[g["cand_env"]] * g["cand"]).sum(-1)
        value = self.critic(ctx, g["w"])
        return logits, value


# ---------------------------------------------------------------------------- batching
def collate(obs_list: list[dict]) -> dict:
    """Merge observations of several environments into one disjoint batched graph."""
    op, mach, cand, glob, w = [], [], [], [], []
    c_op, c_m, c_feat, p_src, p_dst = [], [], [], [], []
    pair_op, pair_m, pair_env, cand_pair, cand_env, op_env, mach_env = [], [], [], [], [], [], []
    no = nm = npair = 0
    for b, ob in enumerate(obs_list):
        N, M = ob["op"].shape[0], ob["mach"].shape[0]
        op.append(ob["op"]); mach.append(ob["mach"]); cand.append(ob["cand"])
        glob.append(ob["glob"]); w.append(ob["w"]); c_feat.append(ob["c_feat"])
        c_op.append(ob["c_op"] + no); c_m.append(ob["c_m"] + nm)
        p_src.append(ob["prec"][0] + no); p_dst.append(ob["prec"][1] + no)
        P_ = len(ob["pair_op"])
        pair_op.append(ob["pair_op"] + no); pair_m.append(ob["pair_m"] + nm)
        pair_env.append(np.full(P_, b, np.int64))
        cand_pair.append(ob["cand_pair"] + npair)
        cand_env.append(np.full(len(ob["cand_pair"]), b, np.int64))
        op_env.append(np.full(N, b, np.int64)); mach_env.append(np.full(M, b, np.int64))
        no += N
        nm += M
        npair += P_
    t = lambda xs, dt=torch.float32: torch.from_numpy(np.concatenate(xs)).to(dt)
    return dict(op=t(op), mach=t(mach), cand=t(cand), glob=torch.from_numpy(np.stack(glob)),
                w=torch.from_numpy(np.stack(w)), c_op=t(c_op, torch.long), c_m=t(c_m, torch.long),
                c_feat=t(c_feat), p_src=t(p_src, torch.long), p_dst=t(p_dst, torch.long),
                pair_op=t(pair_op, torch.long), pair_m=t(pair_m, torch.long),
                pair_env=t(pair_env, torch.long), cand_pair=t(cand_pair, torch.long),
                cand_env=t(cand_env, torch.long), op_env=t(op_env, torch.long),
                mach_env=t(mach_env, torch.long))
