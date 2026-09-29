"""Generate every figure of the paper from the raw results (no hard-coded numbers).

Usage: python scripts/make_figures.py [--only fig_name ...]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs  # noqa: E402
from figstyle import plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)
fs.setup()


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------- data
def fig_energy_signals():
    from morl_fjsp.energy import load_energy_charts_day, synthetic_day
    day = load_energy_charts_day(os.path.join(ROOT, "data", "energy"))
    rng = np.random.default_rng(2024)
    syn = [synthetic_day(rng) for _ in range(40)]
    h = (np.arange(96) + 0.5) / 4.0
    fig, axes = plt.subplots(3, 1, figsize=(fs.COL_W, 3.6), sharex=True)
    series = [("price", "Day-ahead price\n(EUR/MWh)"), ("solar", "PV shape\n(normalised)"),
              ("co2", "Grid CO$_2$eq\n(g/kWh)")]
    for ax, (key, lab) in zip(axes, series):
        for s in syn:
            ax.plot(h, getattr(s, key), color=fs.GRAY, lw=0.4, alpha=0.35)
        ax.plot(h, getattr(day, key), color=fs.BLUE, lw=1.6)
        ax.set_ylabel(lab)
        ax.axvspan(6, 22, color="#f3f2ef", zorder=0, lw=0)
    axes[0].plot([], [], color=fs.BLUE, lw=1.6, label="Test day (DE-LU, 28 Sep 2026)")
    axes[0].plot([], [], color=fs.GRAY, lw=0.8, label="Synthetic training days (40 of 256)")
    axes[0].legend(loc="upper left", ncol=1)
    axes[-1].set_xlabel("Local time (h); shaded: default 06:00-22:00 production window")
    axes[-1].set_xticks(range(0, 25, 3))
    axes[-1].set_xlim(0, 24)
    fig.align_ylabels(axes)
    save(fig, "fig_energy_signals")



# ---------------------------------------------------------------------------- helpers
import json  # noqa: E402
import pickle  # noqa: E402

from morl_fjsp.analysis import (load_baselines, load_hybrid, load_policy_evals,  # noqa: E402
                                method_runs)
from morl_fjsp.instance import BENCHMARKS  # noqa: E402
from morl_fjsp.metrics import hypervolume, nondominated, normalise  # noqa: E402

GROUPS = json.load(open(os.path.join(ROOT, "configs", "method_groups.json")))
OURS = GROUPS["ours"]
_cache = {}


def data():
    if not _cache:
        pg = dict(GROUPS["main"], **GROUPS.get("ablation", {}))
        _cache["base"] = load_baselines(ROOT)
        _cache["long"] = load_baselines(ROOT, "long")
        _cache["pol"] = load_policy_evals(ROOT)
        _cache["hyb"] = load_hybrid(ROOT, GROUPS["main"])
        _cache["groups"] = pg
        _cache["ind"] = json.load(open(os.path.join(ROOT, "results", "summary", "indicators.json")))
        # pooled reference fronts of analyze.py (all methods, runs, ablations, hybrids, sampling,
        # long NSGA-II runs), so that figures and tables use the same normalisation
        _cache["refs"] = pickle.load(open(os.path.join(ROOT, "results", "summary", "reference_fronts.pkl"), "rb"))
    return _cache


def style(lab):
    base = lab.split(" + ")[0]
    st = dict(fs.METHOD_STYLE.get(base, dict(color=fs.GRAY, marker=".", ls="-")))
    if " + " in lab:
        st = dict(color=st["color"], marker="*", ls="--")
    return st


# ---------------------------------------------------------------------------- training
def fig_training_curves():
    groups = dict(GROUPS["main"])
    fig, ax = plt.subplots(1, 1, figsize=(fs.COL_W, 2.3))
    for lab, names in groups.items():
        curves = []
        for n in names:
            p = os.path.join(ROOT, "results", "runs", n, "history.json")
            if not os.path.exists(p):
                continue
            h = [x for x in json.load(open(p)) if "val_hv" in x]
            curves.append((np.array([x["iter"] for x in h]), np.array([x["val_hv"] for x in h])))
        if not curves:
            continue
        L = min(len(c[1]) for c in curves)
        it = curves[0][0][:L]
        Y = np.array([c[1][:L] for c in curves])
        m, sd = Y.mean(0), Y.std(0, ddof=1) if len(Y) > 1 else np.zeros(L)
        st = style(lab)
        ax.plot(it, m, color=st["color"], marker=st["marker"], ms=3, label=f"{lab} ({len(Y)} seeds)")
        ax.fill_between(it, m - sd, m + sd, color=st["color"], alpha=0.15, lw=0)
    ref = json.load(open(os.path.join(ROOT, "results", "summary", "val_reference.json"))) \
        if os.path.exists(os.path.join(ROOT, "results", "summary", "val_reference.json")) else None
    if ref:
        for k, lab, c in (("pwg", "PWG", fs.GRAY), ("nsga2", "NSGA-II (20k eval.)", fs.AQUA),
                          ("nsga2_expert", "NSGA-II, expert budget (3.6k eval.)", fs.INK)):
            if k not in ref:
                continue
            ax.axhline(ref[k], color=c, lw=1.0, ls=":")
            ax.text(2, ref[k], lab, color=c, fontsize=6.5, va="bottom")
    ax.set_xlabel("PPO iteration (0 = after behaviour-cloning warm start)")
    ax.set_ylabel("Validation HV\n(rule-scaled)")
    ax.legend(loc="center right")
    save(fig, "fig_training_curves")


# ---------------------------------------------------------------------------- main comparison
def fig_hv_by_instance():
    d = data()
    ind = d["ind"]["indicators"]
    insts = [i for i in BENCHMARKS if i in ind]
    labs = [l for l in [OURS, "LS-GRL", f"{OURS} + NSGA-II", "NSGA-II", "NSGA-III", "PWG"]
            if all(l in ind[i] for i in insts)]
    fig, ax = plt.subplots(1, 1, figsize=(fs.PAGE_W, 2.2))
    x = np.arange(len(insts))
    wd = 0.8 / len(labs)
    for k, lab in enumerate(labs):
        m = np.array([np.mean(ind[i][lab]["hv"]) for i in insts])
        sd = np.array([np.std(ind[i][lab]["hv"], ddof=1) if len(ind[i][lab]["hv"]) > 1 else 0 for i in insts])
        st = style(lab)
        ax.errorbar(x - 0.4 + wd * (k + 0.5), m, yerr=sd, fmt=st["marker"], color=st["color"], ms=4,
                    elinewidth=0.8, capsize=0, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(insts)
    ax.set_ylabel("HV ratio")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(ncol=len(labs), loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save(fig, "fig_hv_by_instance")


def fig_fronts(insts=("Mk06", "Mk10")):
    d = data()
    fig, axes = plt.subplots(2, len(insts), figsize=(fs.PAGE_W, 4.2))
    for c, inst in enumerate(insts):
        runs, _ = method_runs(d["base"], d["pol"], inst, d["groups"], d["hyb"])
        R, ideal, nadir = d["refs"][inst]
        show = [("PWG", 0), ("NSGA-II", 0), ("LS-GRL", 0), (OURS, 0), (f"{OURS} + NSGA-II", 0)]
        for r, (i, j, xl, yl) in enumerate([(0, 1, "Makespan (h)", "Net grid cost (EUR)"),
                                             (1, 3, "Net grid cost (EUR)", "OWA overload risk")]):
            ax = axes[r, c]
            ax.scatter(R[:, i], R[:, j], s=10, facecolor="none", edgecolor="#c9c8c3", lw=0.6,
                       label="Pooled reference front", zorder=1)
            for lab, k in show:
                if lab not in runs:
                    continue
                F = nondominated(runs[lab][k])
                st = style(lab)
                ax.scatter(F[:, i], F[:, j], s=12, marker=st["marker"], color=st["color"], lw=0.8,
                           label=lab, zorder=3, alpha=0.9)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"\\texttt{{{inst}}}" if False else inst)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig_fronts")


def fig_anytime(insts=("Mk06", "Mk10", "La21")):
    d = data()
    fig, axes = plt.subplots(1, len(insts), figsize=(fs.PAGE_W, 2.2), sharey=True)
    for ax, inst in zip(axes, insts):
        runs, walls = method_runs(d["base"], d["pol"], inst, d["groups"], d["hyb"])
        R, ideal, nadir = d["refs"][inst]
        href = hypervolume(normalise(R, ideal, nadir))
        hvr = lambda F: hypervolume(normalise(nondominated(F), ideal, nadir)) / href if len(F) else 0.0
        b = d["base"][inst]
        for algo, lab in (("nsga2", "NSGA-II"), ("nsga3", "NSGA-III")):
            curves = []
            for r in b["moea"][algo]:
                curves.append([(t, hvr(F)) for g, e, t, F in r["trace"]])
            T = np.array([[p[0] for p in c] for c in curves]).mean(0)
            Y = np.array([[p[1] for p in c] for c in curves])
            st = style(lab)
            ax.plot(T, np.median(Y, 0), color=st["color"], label=lab + " (rule-seeded)")
        if inst in d["long"]:                  # rule-seeded NSGA-II run 4x longer (200k evaluations)
            curves = [[(t, hvr(F)) for g, e, t, F in r["trace"]] for r in d["long"][inst]["moea"]["nsga2"]]
            L = min(len(c) for c in curves)
            T = np.array([[p[0] for p in c[:L]] for c in curves]).mean(0)
            Y = np.array([[p[1] for p in c[:L]] for c in curves])
            ax.plot(T, np.median(Y, 0), color=style("NSGA-II")["color"], ls=":", lw=1.2,
                    label="NSGA-II (rule-seeded, 200k eval.)")
        for lab in (f"{OURS} + NSGA-II",):
            if lab in d["hyb"] and inst in d["hyb"][lab]:
                curves = [[(t, hvr(F)) for g, e, t, F in x[2]] for x in d["hyb"][lab][inst]]
                T = np.array([[p[0] for p in c] for c in curves]).mean(0)
                Y = np.array([[p[1] for p in c] for c in curves])
                st = style(lab)
                ax.plot(T, np.median(Y, 0), color=st["color"], ls="--", label=lab)
        for lab in (OURS, "LS-GRL"):
            if lab in runs:
                st = style(lab)
                ax.scatter(np.median(walls[lab]), np.median([hvr(F) for F in runs[lab]]),
                           color=st["color"], marker=st["marker"], s=25, zorder=4, label=lab + " (policy only)")
        ax.set_xscale("log")
        ax.set_title(inst)
        ax.set_xlabel("Wall-clock time (s, 1 thread)")
    axes[0].set_ylabel("HV ratio (median)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2))
    save(fig, "fig_anytime")


def fig_preference():
    d = data()
    pm = d["ind"]["preference"]
    insts = [i for i in BENCHMARKS if i in pm]
    labs = [l for l in [OURS, "LS-GRL", "PWG", "NSGA-II (a posteriori)"] if all(l in pm[i] for i in insts)]
    fig, axes = plt.subplots(1, 3, figsize=(fs.PAGE_W, 2.0))
    names = ["ASF regret", "Angle to preference (deg)", "Weighted-sum regret"]
    for m, ax in enumerate(axes):
        for k, lab in enumerate(labs):
            vals = [np.mean(np.array(pm[i][lab])[:, m]) for i in insts]
            st = style(lab.replace(" (a posteriori)", ""))
            ax.scatter(np.full(len(vals), k) + np.linspace(-0.2, 0.2, len(vals)), vals, s=9,
                       color=st["color"], marker=st["marker"])
            ax.hlines(np.mean(vals), k - 0.3, k + 0.3, color=fs.INK, lw=1.2)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels([l.replace(" (a posteriori)", "\n(a post.)").replace("DP-PreCo-GRL", "PreCo")
                            for l in labs], fontsize=6.5)
        ax.set_title(names[m])
    save(fig, "fig_preference")


def fig_credibility():
    p = os.path.join(ROOT, "results", "extra", "credibility.pkl")
    if not os.path.exists(p):
        return
    res = pickle.load(open(p, "rb"))
    runs = sorted({k[0] for k in res if k[0] != "PWG"})
    betas = sorted({k[1] for k in res})
    fig, axes = plt.subplots(1, 2, figsize=(fs.PAGE_W * 0.66, 2.1))
    cols = {"triangular": fs.BLUE, "lognormal": fs.ORANGE, "uniform": fs.AQUA}
    for dist, c in cols.items():
        y = [np.mean([np.mean(res[(r, b, i)]["feas"][dist]) for r in runs
                      for i in BENCHMARKS if (r, b, i) in res]) for b in betas]
        axes[0].plot(betas, np.array(y) * 100, color=c, marker="o", ms=3, label=dist)
    axes[0].plot(betas, [100 * b for b in betas], color=fs.GRAY, ls=":", lw=1, label="$y=100\\beta$")
    axes[0].set_xlabel("Credibility level $\\beta$")
    axes[0].set_ylabel("MC capacity feasibility (%)")
    axes[0].legend(loc="lower right")
    ref = {i: np.mean(res[(runs[0], 0.5, i)]["F"], axis=0) for i in BENCHMARKS if (runs[0], 0.5, i) in res}
    for k, (name, c) in enumerate([("makespan", fs.BLUE), ("cost", fs.ORANGE), ("CO$_2$", fs.AQUA),
                                   ("OWA risk", fs.VIOLET)]):
        y = [np.mean([100 * (np.mean(res[(r, b, i)]["F"][:, k]) / ref[i][k] - 1) for r in runs
                      for i in ref if (r, b, i) in res]) for b in betas]
        axes[1].plot(betas, y, color=c, marker="o", ms=3, label=name)
    axes[1].set_xlabel("Credibility level $\\beta$")
    axes[1].set_ylabel("Change vs. $\\beta=0.5$ (%)")
    axes[1].legend(loc="upper left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_credibility")


def _find_pref(W, k):
    """Index of the evaluation preference that most emphasises objective k."""
    return int(np.argmax(W[:, k] - W.sum(1) * 0))


def fig_schedule(inst="Mk01", run=None):
    from morl_fjsp.scenarios import benchmark_scenarios
    from morl_fjsp.sim import evaluate_actions
    d = data()
    run = run or GROUPS["main"][OURS][0]
    e = d["pol"][run]
    W = e["prefs"]
    acts = e["results"][inst]["actions"]
    sc = benchmark_scenarios(os.path.join(ROOT, "data"), names=[inst])[0]
    P = sc.P
    picks = [(_find_pref(W, 0), "makespan-focused"), (_find_pref(W, 1), "cost-focused")]
    states = [evaluate_actions(P, acts[i]) for i, _ in picks]
    H = int(max(st.sc[0] for st in states)) + 30
    t = np.arange(H)
    clock = (P.start_min + t) / 60.0
    fig, axes = plt.subplots(4, 1, figsize=(fs.PAGE_W, 5.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[0.8, 1.6, 1.6, 1.1]))
    axes[0].plot(clock, P.price[:H], color=fs.INK, lw=1.0)
    axes[0].set_ylabel("Price\n(EUR/MWh)")
    for ax, (i, lab), st in zip(axes[1:3], picks, states):
        for o in range(P.N):
            m, s0, e0 = st.op_m[o], st.op_s[o], st.op_e[o]
            col = fs.ORANGE if st.op_d[o] > 0 else fs.BLUE
            ax.barh(m, (e0 - s0) / 60.0, left=(P.start_min + s0) / 60.0, height=0.7, color=col,
                    edgecolor=fs.SURFACE, linewidth=0.6)
        F = st.objectives()
        ax.set_yticks(range(P.M))
        ax.set_yticklabels([f"M{m + 1}" for m in range(P.M)], fontsize=6)
        ax.set_ylabel("Machine")
        ax.set_title(f"{lab} $\\mathbf{{w}}$=({', '.join(f'{x:.2f}' for x in W[i])}): "
                     f"$C_{{\\max}}$={F[0]:.1f} h, cost={F[1]:.1f} EUR, CO$_2$={F[2]:.0f} kg, "
                     f"$R_{{\\rm OWA}}$={F[3]:.2f}", fontsize=7.5, loc="left")
        ax.grid(axis="y", visible=False)
    for (i, lab), st, c in zip(picks, states, (fs.BLUE, fs.ORANGE)):
        D = st.D[:H]
        axes[3].plot(clock, np.maximum(0, D - P.pv[:H]), color=c, lw=1.0, label=f"grid import, {lab}")
    axes[3].plot(clock, P.pv[:H], color=fs.AQUA, lw=1.0, ls="--", label="on-site PV")
    axes[3].set_ylabel("Power (kW)")
    axes[3].set_xlabel("Local time (h)")
    axes[3].legend(loc="upper right", ncol=3)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=fs.BLUE, label="started without postponement"),
                            Patch(color=fs.ORANGE, label="postponed ($\\delta>0$)")],
                   loc="lower right", ncol=1, fontsize=6.5)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=1.2)
    save(fig, "fig_schedule")


def fig_behavior():
    d = data()
    rows = []
    for run in GROUPS["main"][OURS] + GROUPS["main"]["LS-GRL"]:
        if run not in d["pol"]:
            continue
        e = d["pol"][run]
        W = e["prefs"]
        for inst, r in e["results"].items():
            if "n_postponed" not in r:
                continue
            N = sum(1 for _ in r["actions"][0]) if "actions" in r else None
            for k, w in enumerate(W):
                rows.append((run.startswith("preco"), w[1] + w[2], w[3], r["n_postponed"][k],
                             r["rel_machine_energy"][k], inst))
    if not rows:
        return
    from morl_fjsp.instance import load_benchmarks
    nops = {i.name: i.n_ops for i in load_benchmarks(os.path.join(ROOT, "data", "fjsp"))}
    fig, axes = plt.subplots(1, 2, figsize=(fs.PAGE_W * 0.66, 2.1))
    bins = np.linspace(0, 1, 6)
    for is_preco, lab in ((True, OURS), (False, "LS-GRL")):
        sub = [r for r in rows if r[0] == is_preco]
        x = np.array([r[1] for r in sub])
        post = np.array([100 * r[3] / nops[r[5]] for r in sub])
        rel = np.array([100 * (r[4] - 1) for r in sub])
        idx = np.digitize(x, bins) - 1
        xc = 0.5 * (bins[1:] + bins[:-1])
        st = style(lab)
        for ax, y in zip(axes, (post, rel)):
            m = [y[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(xc))]
            ax.plot(xc, m, color=st["color"], marker=st["marker"], ms=3, label=lab)
    axes[0].set_ylabel("Postponed operations (%)")
    axes[1].set_ylabel("Machine energy above\nmost efficient choice (%)")
    for ax in axes:
        ax.set_xlabel("Weight on cost + CO$_2$ ($w_2+w_3$)")
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_behavior")


def fig_pv():
    p = os.path.join(ROOT, "results", "extra", "pv.pkl")
    if not os.path.exists(p):
        return
    res = pickle.load(open(p, "rb"))
    phis = sorted({k[1] for k in res})
    runs = sorted({k[0] for k in res})
    fig, ax = plt.subplots(1, 1, figsize=(fs.COL_W, 2.0))
    for k, (name, c) in enumerate([("net grid cost", fs.ORANGE), ("grid CO$_2$", fs.AQUA)]):
        y = []
        for phi in phis:
            vals = []
            for (r, ph, inst), v in res.items():
                if ph != phi:
                    continue
                F = v["F"][v["viol"] <= 1e-9]
                if len(F) == 0:
                    continue
                i_mk = int(np.argmin(F[:, 0]))
                vals.append(100 * (1 - F[:, k + 1].min() / F[i_mk, k + 1]))
            y.append(np.mean(vals))
        ax.plot(phis, y, color=c, marker="o", ms=3, label=name)
    ax.set_xlabel("PV sizing factor $\\varphi$")
    ax.set_ylabel("Reduction vs. fastest\nschedule on the front (%)")
    ax.legend()
    save(fig, "fig_pv")


def fig_scalability():
    p = os.path.join(ROOT, "results", "extra", "scale.pkl")
    if not os.path.exists(p):
        return
    res = pickle.load(open(p, "rb"))
    N = [r["N"] for r in res]
    fig, ax = plt.subplots(1, 1, figsize=(fs.COL_W, 2.0))
    ax.plot(N, [r["per_pref"] for r in res], color=fs.BLUE, marker="o", ms=3,
            label="policy: per preference, 84-batch (s)")
    ax.plot(N, [r["decode_sec"] * 50000 for r in res], color=fs.AQUA, marker="^", ms=3,
            label="NSGA: 50,000 decodings (s)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Operations $N$")
    ax.set_ylabel("Seconds (1 thread)")
    ax.legend(loc="upper left")
    save(fig, "fig_scalability")


def fig_credibility_illustration():
    from morl_fjsp.fuzzy import credibility_leq, credibility_quantile
    a, b, c = 20.0, 30.0, 45.0
    t = np.linspace(12, 52, 801)
    mu = np.clip(np.minimum((t - a) / (b - a), (c - t) / (c - b)), 0, 1)
    pos = np.where(t < a, 0, np.where(t <= b, (t - a) / (b - a), 1.0))
    nec = np.where(t <= b, 0, np.where(t < c, (t - b) / (c - b), 1.0))
    cr = np.array([credibility_leq(a, b, c, x) for x in t])
    fig, axes = plt.subplots(1, 2, figsize=(fs.COL_W, 1.75))
    ax = axes[0]
    ax.fill_between(t, 0, mu, color="#eceae6", lw=0)
    ax.plot(t, mu, color=fs.GRAY, lw=1.0)
    ax.plot(t, pos, color=fs.ORANGE, lw=1.2)
    ax.plot(t, nec, color=fs.AQUA, lw=1.2)
    ax.plot(t, cr, color=fs.BLUE, lw=1.6)
    q = credibility_quantile(a, b, c, 0.9)
    ax.plot([q, q], [0, 0.9], color=fs.INK, lw=0.7, ls=":")
    ax.plot([12, q], [0.9, 0.9], color=fs.INK, lw=0.7, ls=":")
    ax.text(q + 0.8, 0.03, "$q_{0.9}$", fontsize=6.5)
    ax.text(21.0, 0.83, "Pos", color=fs.ORANGE, fontsize=6.5, ha="right")
    ax.text(26.5, 0.52, "Cr", color=fs.BLUE, fontsize=6.5, ha="right")
    ax.text(41.5, 0.45, "Nec", color=fs.AQUA, fontsize=6.5)
    ax.text(27.0, 0.08, "$\\mu$", color=fs.GRAY, fontsize=7)
    ax.set_xlabel("$t$ (TFN $\\xi=(20,30,45)$)")
    ax.set_ylim(0, 1.05)
    ax = axes[1]
    L = 100.0
    W = np.linspace(60, 130, 400)
    for (lo, hi), col in (((0.10, 0.25), fs.BLUE), ((0.05, 0.40), fs.ORANGE)):
        r = [1 - credibility_leq((1 - lo) * w, w, (1 + hi) * w, L) for w in W]
        ax.plot(W / L, r, color=col, lw=1.2, label=f"spreads $-{int(100*lo)}\\%/+{int(100*hi)}\\%$")
    ax.set_xlabel("modal workload $W^B_m/L$")
    ax.set_ylabel("risk $r_m$")
    ax.legend(fontsize=5.6, loc="upper left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_credibility_illustration")


FIGS = {"fig_energy_signals": fig_energy_signals,
        "fig_credibility_illustration": fig_credibility_illustration, "fig_training_curves": fig_training_curves,
        "fig_hv_by_instance": fig_hv_by_instance, "fig_fronts": fig_fronts,
        "fig_anytime": fig_anytime, "fig_preference": fig_preference,
        "fig_credibility": fig_credibility,
        "fig_schedule": fig_schedule, "fig_behavior": fig_behavior, "fig_pv": fig_pv,
        "fig_scalability": fig_scalability}
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    for name, fn in FIGS.items():
        if a.only is None or name in a.only:
            fn()
