"""Training of the preference-conditioned policy (PPO + vector critic) with
  * mode="linear": advantages combined with the preference, A = sum_k w_k A_k;
  * mode="preco" : advantages combined with the PreCo weights c = alpha* + lambda grad Psi
    (Yang et al., ICML 2025), recomputed for every preference group and update;
  * dual-pool guidance: behaviour cloning on an offline expert archive (NSGA-II
    fronts of *training* scenarios) and an online elite archive (best own episodes),
    with linearly annealed weights.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .env import BatchEnv, Scenario, StaticGraph
from .metrics import asf, hypervolume, nondominated, normalise
from .model import PolicyNet, collate, seg_log_softmax
from .sim import C_D, C_JOB, C_M


@dataclass
class TrainConfig:
    name: str = "run"
    mode: str = "preco"              # "preco" | "linear"
    iters: int = 120
    n_envs: int = 32
    n_groups: int = 8
    lr: float = 1e-4
    adv_clip: float = 4.0
    epochs: int = 3
    minibatch: int = 256
    clip: float = 0.2
    ent_coef: tuple = (0.01, 0.001)
    vf_coef: float = 0.5
    max_grad: float = 0.5
    gae_lambda: float = 0.95
    lam: tuple = (1.0, 8.0)          # PreCo homotopy lambda_t (linear schedule)
    preco_norm: bool = True          # False: un-normalised c = alpha* + lam gpsi (Yang et al. scale)
    bc_exp: tuple = (1.0, 0.0)       # expert BC weight schedule
    bc_elite: tuple = (0.5, 0.02)    # elite BC weight schedule
    bc_episodes: int = 8
    bc_samples: int = 32             # BC samples (per pool) added to every minibatch
    max_transitions: int = 1024      # transitions sampled per PPO update
    pretrain_iters: int = 40         # Stage I: behaviour-cloning warm start on the expert pool
    pretrain_lr: float = 1e-3
    pretrain_episodes: int = 16
    elite_cap: int = 24
    gat: bool = True
    film: bool = True
    crit: bool = True
    delay: bool = True
    mask_capacity: bool = True
    d_model: int = 64
    seed: int = 0
    val_every: int = 15
    threads: int = 1
    out_dir: str = "results/runs"
    extra: dict = field(default_factory=dict)


def lin(sched, frac):
    return sched[0] + (sched[1] - sched[0]) * min(max(frac, 0.0), 1.0)


def pick_env_actions(lp, cand_env, n, rng, greedy=False):
    """Sample (or argmax) one candidate per environment from log-probabilities."""
    p = torch.exp(lp).numpy()
    ce = cand_env.numpy()
    starts = np.searchsorted(ce, np.arange(n))
    ends = np.append(starts[1:], len(ce))
    ks = np.empty(n, np.int64)
    for i in range(n):
        pi = p[starts[i]:ends[i]]
        if greedy:
            ks[i] = int(np.argmax(pi))
        else:
            c = np.cumsum(pi)
            ks[i] = min(int(np.searchsorted(c, rng.random() * c[-1])), len(pi) - 1)
    return ks, starts


def utilities_of(F, sc: Scenario):
    return 1.0 - (np.asarray(F) - sc.ideal) / sc.span


def preco_weights(A: np.ndarray, u: np.ndarray, w: np.ndarray, lam: float, iters: int = 300,
                  normalise: bool = True):
    """Detached PreCo-inspired combination: c = alpha* + lam * gpsi, alpha* the min-norm point
    of the simplex for ||G(alpha + lam gpsi)||, G the advantage Jacobian (T x K).

    gpsi = max_i(u_i / w_i) w - u is the direction grad_v Psi(w, u) of Yang et al. (ICML 2025,
    Eq. 44 of their appendix) with the ray scale max_i(u_i / w_i) held fixed (stop-gradient).
    With ``normalise=True`` (default, used for all reported policies) c is rescaled onto the
    simplex (our adaptation); ``normalise=False`` returns the un-normalised c of their Eq. (6)-(7).
    """
    K = len(w)
    w = np.maximum(w, 1e-2)
    w = w / w.sum()
    u = np.maximum(u, 1e-2)
    gpsi = (u / w).max() * w - u                      # >= 0 component-wise
    GtG = A.T @ A / max(len(A), 1)
    Lc = 2.0 * np.linalg.eigvalsh(GtG).max() + 1e-9
    alpha = np.full(K, 1.0 / K)
    for _ in range(iters):
        grad = 2.0 * GtG @ (alpha + lam * gpsi)
        alpha = _proj_simplex(alpha - grad / Lc)
    c = alpha + lam * gpsi
    return (c / c.sum() if normalise else c), alpha, gpsi


def _proj_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


class Archive:
    """Per-scenario archive of complete schedules (objective vector, action list)."""

    def __init__(self, cap):
        self.cap, self.items = cap, {}

    def add(self, key, F, actions, rng):
        lst = self.items.setdefault(key, [])
        F = np.asarray(F)
        if any(np.all(G <= F) for G, _ in lst):
            return
        lst[:] = [(G, a) for G, a in lst if not np.all(F <= G)]
        lst.append((F, actions))
        if len(lst) > self.cap:
            lst.pop(int(rng.integers(len(lst))))

    def keys(self):
        return [k for k, v in self.items.items() if v]

    def select(self, key, w, sc: Scenario, mode):
        lst = self.items[key]
        F = np.array([f for f, _ in lst])
        if mode == "preco":
            i = int(np.argmax(asf(utilities_of(F, sc), w)))
        else:
            i = int(np.argmin(((F - sc.ideal) / sc.span) @ w))
        return lst[i][1]


# ---------------------------------------------------------------------------- rollouts
def rollout(net, scenarios, prefs, rng, greedy=False, store=True, statics=None,
            mask_capacity=True, allow_delay=True):
    env = BatchEnv(scenarios, prefs, statics=statics, mask_capacity=mask_capacity,
                   allow_delay=allow_delay)
    B = env.B
    traj = [[] for _ in range(B)]
    acts = [[] for _ in range(B)]
    while True:
        idx = [b for b in range(B) if not env.states[b].done]
        if not idx:
            break
        obs = env.observe_all(idx)
        g = collate(obs)
        with torch.no_grad():
            logits, values = net(g)
        lp = seg_log_softmax(logits, g["cand_env"], len(idx))
        ks, starts = pick_env_actions(lp, g["cand_env"], len(idx), rng, greedy)
        for i, b in enumerate(idx):
            c = env.cands[b][ks[i]]
            acts[b].append((int(c[C_JOB]), int(c[C_M]), int(c[C_D])))
            r = env.step(b, int(ks[i]))
            if store:
                traj[b].append((obs[i], int(ks[i]), float(lp[starts[i] + ks[i]]),
                                values[i].numpy().copy(), r))
    finals = [env.final(b) for b in range(B)]
    return traj, acts, finals


def replay_for_bc(scenarios, prefs, action_lists, statics=None, mask_capacity=True,
                  allow_delay=True):
    """Replay target action sequences; return (obs, k) pairs where the target action
    is an available candidate."""
    env = BatchEnv(scenarios, prefs, statics=statics, mask_capacity=mask_capacity,
                   allow_delay=allow_delay)
    data = []
    for b, acts in enumerate(action_lists):
        ep = []
        for (j, m, d) in acts:
            ob = env.observe(b)
            c = env.cands[b]
            hit = np.flatnonzero((c[:, C_JOB] == j) & (c[:, C_M] == m) & (c[:, C_D] == d))
            if len(hit):
                r = env.step(b, int(hit[0]))
                ep.append([ob, int(hit[0]), r])
            else:                       # not representable (e.g. masked): apply directly
                F0 = env.states[b].objectives()
                env.states[b].step(j, m, d)
                F1 = env.states[b].objectives()
                env.prev_F[b] = F1
                if ep:
                    ep[-1][2] = ep[-1][2] - (F1 - F0) / env.sc[b].span
        # returns-to-go (undiscounted) for critic warm start
        G = np.zeros(4)
        for t in range(len(ep) - 1, -1, -1):
            G = G + ep[t][2]
            data.append((ep[t][0], ep[t][1], G.copy()))
    return data


def batch_logprob(net, obs, ks):
    g = collate(obs)
    logits, values = net(g)
    n = len(obs)
    lp_all = seg_log_softmax(logits, g["cand_env"], n)
    counts = torch.bincount(g["cand_env"], minlength=n)
    offs = torch.cumsum(counts, 0) - counts
    lp = lp_all[offs + torch.as_tensor(ks)]
    ent = -torch.zeros(n).index_add(0, g["cand_env"], torch.exp(lp_all) * lp_all)
    return lp, ent, values


# ---------------------------------------------------------------------------- evaluation
def evaluate_front(net, scenarios, prefs, cfg, rng, statics=None):
    """Greedy rollouts of all (scenario, preference) pairs; returns F (S, W, 4) and
    violations / forced-assignment counts."""
    S, Wn = len(scenarios), len(prefs)
    scs = [s for s in scenarios for _ in range(Wn)]
    st = [g for g in (statics or [StaticGraph(s.P) for s in scenarios]) for _ in range(Wn)]
    W = np.tile(prefs, (S, 1))
    _, acts, finals = rollout(net, scs, W, rng, greedy=True, store=False, statics=st,
                              mask_capacity=cfg.mask_capacity, allow_delay=cfg.delay)
    F = np.array([f[0] for f in finals]).reshape(S, Wn, 4)
    V = np.array([f[1] for f in finals]).reshape(S, Wn)
    return F, V, acts


# ---------------------------------------------------------------------------- training
def train(cfg: TrainConfig, pool: list[Scenario], expert: Archive | None,
          val: list[Scenario] | None = None, val_ref: list | None = None, log=print):
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    net = PolicyNet(d=cfg.d_model, gat=cfg.gat, film=cfg.film, crit=cfg.crit)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    statics = [StaticGraph(s.P) for s in pool]
    val_statics = [StaticGraph(s.P) for s in val] if val else None
    elite = Archive(cfg.elite_cap)
    run_dir = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(asdict(cfg), open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    history = []
    best_val = -1.0
    val_prefs = _val_prefs()
    t_start = time.time()
    per_group = cfg.n_envs // cfg.n_groups
    # ------------------------------------------------ Stage I: BC warm start (expert pool)
    if expert is not None and cfg.pretrain_iters > 0 and cfg.bc_exp[0] > 0:
        for gparam in opt.param_groups:
            gparam["lr"] = cfg.pretrain_lr
        keys = expert.keys()
        for pit in range(cfg.pretrain_iters):
            ks_ = rng.choice(keys, min(cfg.pretrain_episodes, len(keys)), replace=False)
            wv = rng.dirichlet(np.ones(4), len(ks_))
            tgt = [expert.select(int(k), wv[i], pool[int(k)], cfg.mode) for i, k in enumerate(ks_)]
            data = replay_for_bc([pool[int(k)] for k in ks_], wv, tgt,
                                 statics=[statics[int(k)] for k in ks_],
                                 mask_capacity=cfg.mask_capacity, allow_delay=cfg.delay)
            perm = rng.permutation(len(data))
            losses = []
            for s0 in range(0, len(perm), cfg.minibatch):
                mb = perm[s0:s0 + cfg.minibatch]
                lpb, _, vb = batch_logprob(net, [data[i][0] for i in mb], [data[i][1] for i in mb])
                tgt_v = torch.as_tensor(np.array([data[i][2] for i in mb]), dtype=torch.float32)
                loss = -lpb.mean() + cfg.vf_coef * torch.nn.functional.huber_loss(vb, tgt_v)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad)
                opt.step()
                losses.append(float(loss.detach()))
            if pit % 10 == 0 or pit == cfg.pretrain_iters - 1:
                log(f"[{cfg.name}] pretrain {pit} loss={np.mean(losses):.3f} t={time.time() - t_start:.0f}s")
        history.append(dict(pretrain_done=True, time=time.time() - t_start))
        for gparam in opt.param_groups:
            gparam["lr"] = cfg.lr
    for it in range(cfg.iters + 1):
        frac = it / max(cfg.iters, 1)
        # ------------------------------------------------ validation (real HV curve)
        if val and (it % cfg.val_every == 0 or it == cfg.iters):
            net.eval()
            F, V, _ = evaluate_front(net, val, val_prefs, cfg, rng, val_statics)
            hvr = []
            for s in range(len(val)):
                # hypervolume in the rule-normalised space of the scenario (bounded,
                # reference-free and on the same scale as the policy's utilities)
                ok = V[s] <= 1e-9
                A = nondominated(F[s][ok]) if ok.any() else np.zeros((0, 4))
                hvr.append(hypervolume(normalise(A, val[s].ideal, val[s].nadir)) if len(A) else 0.0)
            history.append(dict(iter=it, time=time.time() - t_start, val_hv=float(np.mean(hvr)),
                                val_hv_all=[float(x) for x in hvr]))
            log(f"[{cfg.name}] it={it} t={time.time() - t_start:.0f}s val_HV={np.mean(hvr):.4f}")
            torch.save(net.state_dict(), os.path.join(run_dir, "model_last.pt"))
            if np.mean(hvr) >= best_val:            # checkpoint selection on validation only
                best_val = float(np.mean(hvr))
                torch.save(net.state_dict(), os.path.join(run_dir, "model.pt"))
                history[-1]["best"] = True
            json.dump(history, open(os.path.join(run_dir, "history.json"), "w"))
        if it == cfg.iters:
            break
        net.train()
        # ------------------------------------------------ rollout
        idx = rng.choice(len(pool), cfg.n_envs, replace=False)
        prefs = np.repeat(rng.dirichlet(np.ones(4), cfg.n_groups), per_group, axis=0)
        scs = [pool[i] for i in idx]
        traj, acts, finals = rollout(net, scs, prefs, rng, statics=[statics[i] for i in idx],
                                     mask_capacity=cfg.mask_capacity, allow_delay=cfg.delay)
        for b, i in enumerate(idx):
            F, viol, forced = finals[b]
            if viol <= 1e-9 and forced == 0:
                elite.add(int(i), F, acts[b], rng)
        # ------------------------------------------------ vector GAE
        obs, ks, lp_old, adv, ret, env_of = [], [], [], [], [], []
        for b, tr in enumerate(traj):
            R = np.array([t[4] for t in tr])
            Vv = np.array([t[3] for t in tr])
            Vn = np.vstack([Vv[1:], np.zeros((1, 4))])
            delta = R + Vn - Vv
            A = np.zeros_like(delta)
            acc = np.zeros(4)
            for t in range(len(tr) - 1, -1, -1):
                acc = delta[t] + cfg.gae_lambda * acc
                A[t] = acc
            for t, x in enumerate(tr):
                obs.append(x[0]); ks.append(x[1]); lp_old.append(x[2]); env_of.append(b)
            adv.append(A); ret.append(A + Vv)
        adv = np.vstack(adv); ret = np.vstack(ret); env_of = np.array(env_of)
        adv_n = np.clip((adv - adv.mean(0)) / (adv.std(0) + 1e-8), -cfg.adv_clip, cfg.adv_clip)
        # ------------------------------------------------ combination weights
        lam_t = lin(cfg.lam, frac)
        C = np.zeros((cfg.n_envs, 4))
        diag = []
        for gi in range(cfg.n_groups):
            envs = np.arange(gi * per_group, (gi + 1) * per_group)
            w = prefs[envs[0]]
            if cfg.mode == "linear":
                C[envs] = w
            else:
                u = np.mean([utilities_of(finals[b][0], scs[b]) for b in envs], axis=0)
                mask = np.isin(env_of, envs)
                c, alpha, gpsi = preco_weights(adv_n[mask], u, w, lam_t, normalise=cfg.preco_norm)
                C[envs] = c
                diag.append(dict(w=w.tolist(), u=u.tolist(), c=c.tolist()))
        a_s = (adv_n * C[env_of]).sum(1)
        a_s = np.clip((a_s - a_s.mean()) / (a_s.std() + 1e-8), -cfg.adv_clip, cfg.adv_clip)
        # ------------------------------------------------ BC datasets (dual pool)
        b_exp, b_eli = lin(cfg.bc_exp, frac), lin(cfg.bc_elite, frac)
        bc_sets = []
        for weight, arch in ((b_exp, expert), (b_eli, elite)):
            if weight <= 0 or arch is None or not arch.keys():
                continue
            keys = rng.choice(arch.keys(), min(cfg.bc_episodes, len(arch.keys())), replace=False)
            wv = rng.dirichlet(np.ones(4), len(keys))
            tgt = [arch.select(int(k), wv[i], pool[int(k)], cfg.mode) for i, k in enumerate(keys)]
            data = replay_for_bc([pool[int(k)] for k in keys], wv, tgt,
                                 statics=[statics[int(k)] for k in keys],
                                 mask_capacity=cfg.mask_capacity, allow_delay=cfg.delay)
            if data:
                bc_sets.append((weight, data))
        # ------------------------------------------------ PPO epochs
        ent_c = lin(cfg.ent_coef, frac)
        n = len(obs)
        subset = rng.permutation(n)[:cfg.max_transitions]
        stats = []
        for _ in range(cfg.epochs):
            perm = rng.permutation(subset)
            for s0 in range(0, len(perm), cfg.minibatch):
                mb = perm[s0:s0 + cfg.minibatch]
                lp, ent, val_pred = batch_logprob(net, [obs[i] for i in mb], [ks[i] for i in mb])
                ratio = torch.exp(lp - torch.as_tensor(np.array(lp_old)[mb], dtype=torch.float32))
                A_t = torch.as_tensor(a_s[mb], dtype=torch.float32)
                pg = -torch.min(ratio * A_t, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * A_t).mean()
                vf = torch.nn.functional.huber_loss(val_pred, torch.as_tensor(ret[mb], dtype=torch.float32))
                loss = pg + cfg.vf_coef * vf - ent_c * ent.mean()
                bc_val = 0.0
                for weight, data in bc_sets:
                    sel = rng.choice(len(data), min(cfg.bc_samples, len(data)), replace=False)
                    lpb, _, _ = batch_logprob(net, [data[i][0] for i in sel], [data[i][1] for i in sel])
                    bc = -lpb.mean()
                    loss = loss + weight * bc
                    bc_val += float(bc.detach())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad)
                opt.step()
                stats.append((float(pg), float(vf), float(ent.mean()), bc_val))
        st = np.mean(stats, axis=0)
        Fm = np.mean([f[0] for f in finals], axis=0)
        rec = dict(iter=it, time=time.time() - t_start, pg=st[0], vf=st[1], ent=st[2], bc=st[3],
                   mean_F=Fm.tolist(), lam=lam_t, b_exp=b_exp, b_elite=b_eli,
                   elite_size=sum(len(v) for v in elite.items.values()))
        if diag:
            rec["preco"] = diag[:2]
        if it % 5 == 0:
            log(f"[{cfg.name}] it={it} t={rec['time']:.0f}s pg={st[0]:.3f} vf={st[1]:.3f} "
                f"ent={st[2]:.2f} bc={st[3]:.2f} F={np.round(Fm, 2)}")
        history.append(rec)
    return net, history


def _val_prefs():
    from .metrics import eval_preferences
    return eval_preferences(4, 3, 0.1)       # 20 interior Das-Dennis vectors
