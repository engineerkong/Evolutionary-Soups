"""
Compare es_summary_0905 populations vs RS chunk and optionally NSGA-III.
Usage:
  python es_summary_0905_gen_vs_rs.py [gen_numbers...]          # single gen per plot
  python es_summary_0905_gen_vs_rs.py --pairs 41,2 42,3        # two gens per plot
Default single: 26 27 28
"""
NUM_CHUNKS = 39   # ceil(39488 / 1024)

import json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE    = "/home/kong/workspace/MOMOE/MOMoE"
RS_DIR  = f"{BASE}/results/ppo_rs/rs_summary_train_0705"
OUT_DIR = f"{BASE}/logs/analysis_diagram"
os.makedirs(OUT_DIR, exist_ok=True)

expert_raw = np.array([
    [ 1.9561145305633545,  -0.4358978271484375, -4.486939907073975],
    [-0.14591821189969778, -0.3547602891921997,  0.9568580240011215],
    [ 0.4194423956796527,  -0.6684675216674805,  3.227126121520996],
])
r_min   = expert_raw.min(axis=0)
r_range = np.maximum(expert_raw.max(axis=0) - r_min, 1e-6)

def normalize(raw): return (raw - r_min) / r_range

experts_norm  = normalize(expert_raw)
expert_labels = ["E0(sum)", "E1(faith)", "E2(deb)"]

rs_files = sorted(f for f in os.listdir(RS_DIR) if f.endswith(".csv"))

def load_rs_chunk(chunk: int) -> np.ndarray:
    lo, hi = chunk * 1024, (chunk + 1) * 1024
    rows = []
    for fname in rs_files:
        df = pd.read_csv(f"{RS_DIR}/{fname}",
                         usecols=["Unnamed: 0", "obtained_score1",
                                  "obtained_score2", "obtained_score3"])
        c = df[(df["Unnamed: 0"] >= lo) & (df["Unnamed: 0"] < hi)]
        if len(c) > 0:
            rows.append(c[["obtained_score1","obtained_score2","obtained_score3"]].mean().values)
    return normalize(np.array(rows))

with open(f"{BASE}/models/ES/es_summary_0905/population_log.json") as f:
    pop_log = json.load(f)

with open(f"{BASE}/models/nsgaii/nsgaiii_summary_1.5_ema_2front_norm_0705/population_log.json") as f:
    nsga_log = json.load(f)

def pareto_mask(pts):
    n, dom = len(pts), np.zeros(len(pts), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                dom[i] = True; break
    return ~dom

def pareto_line(pts2d):
    pf = pts2d[pareto_mask(pts2d)]
    return pf[np.argsort(pf[:, 0])]

projections = [
    (0, 1, "summary",  "faithful"),
    (0, 2, "summary",  "deberta"),
    (1, 2, "faithful", "deberta"),
]

RS_C = "#D62728"

# Palette for ES gens: dark→light navy as gen index increases
ES_PALETTE = [
    ("#B8D4F0", "#4A90D9"),   # gen A: light pop / dark pareto
    ("#1F4E79", "#0A2540"),   # gen B: darker
]
NS_POP_C, NS_PF_C = "#F5A623", "#C0640A"


def plot_one(gen_list: list, out_suffix: str):
    """gen_list: list of int gen numbers to overlay on one figure."""
    chunk_n = gen_list[0] % NUM_CHUNKS

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    rs_norm   = load_rs_chunk(chunk_n)
    legend_handles = []

    for ax, (xi, yi, xlabel, ylabel) in zip(axes, projections):
        # RS
        rs2d  = rs_norm[:, [xi, yi]];  rs_pf = pareto_line(rs2d)
        ax.scatter(rs2d[:,0],  rs2d[:,1],  color=RS_C, marker="D", s=40,  alpha=0.4,  linewidths=0.4, edgecolors="white", zorder=2)
        ax.scatter(rs_pf[:,0], rs_pf[:,1], color=RS_C, marker="D", s=90,  alpha=0.95, linewidths=0.6, edgecolors="white", zorder=3)
        ax.plot(   rs_pf[:,0], rs_pf[:,1], color=RS_C, lw=1.8, alpha=0.85, linestyle="--", zorder=3)

        # ES gens
        for idx, gen_n in enumerate(gen_list):
            gen_key  = f"gen_{gen_n:04d}"
            pop_c, pf_c = ES_PALETTE[idx % len(ES_PALETTE)]
            es_fit   = np.array([v["raw"] for v in pop_log[gen_key].values()])
            es2d     = es_fit[:, [xi, yi]];  es_pf = pareto_line(es2d)
            ax.scatter(es2d[:,0],  es2d[:,1],  color=pop_c, marker="o", s=40,  alpha=0.35, linewidths=0.3, edgecolors="white", zorder=4+idx*2)
            ax.scatter(es_pf[:,0], es_pf[:,1], color=pf_c,  marker="o", s=90,  alpha=0.95, linewidths=0.6, edgecolors="white", zorder=5+idx*2)
            ax.plot(   es_pf[:,0], es_pf[:,1], color=pf_c,  lw=2.0, alpha=0.9,  zorder=5+idx*2)

        # NSGA-III (first gen in list, if available)
        ns_fit = None
        gen_key0 = f"gen_{gen_list[0]:04d}"
        if gen_key0 in nsga_log:
            ns_fit = np.array([v["raw"] for v in nsga_log[gen_key0].values()])
            ns2d   = ns_fit[:, [xi, yi]];  ns_pf = pareto_line(ns2d)
            ax.scatter(ns2d[:,0],  ns2d[:,1],  color=NS_POP_C, marker="s", s=40,  alpha=0.35, linewidths=0.3, edgecolors="white", zorder=8)
            ax.scatter(ns_pf[:,0], ns_pf[:,1], color=NS_PF_C,  marker="s", s=90,  alpha=0.95, linewidths=0.6, edgecolors="white", zorder=9)
            ax.plot(   ns_pf[:,0], ns_pf[:,1], color=NS_PF_C,  lw=2.0, alpha=0.9,  linestyle=":", zorder=9)

        # Expert models
        for ex, label in zip(experts_norm, expert_labels):
            ax.scatter(ex[xi], ex[yi], color="red", marker="*", s=220, zorder=10, edgecolors="darkred", linewidths=0.8)
            ax.annotate(label, (ex[xi], ex[yi]), textcoords="offset points", xytext=(5,3), fontsize=7, color="red", fontweight="bold")

        ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{xlabel} vs {ylabel}", fontsize=13, fontweight="bold")
        ax.set_xlim(-0.05, 1.10); ax.set_ylim(-0.05, 1.10)
        ax.grid(True, alpha=0.25, linestyle="--")

        if xi == 0 and yi == 1:
            handles = []
            for idx, gen_n in enumerate(gen_list):
                pop_c, pf_c = ES_PALETTE[idx % len(ES_PALETTE)]
                gen_key = f"gen_{gen_n:04d}"
                handles += [
                    mpatches.Patch(color=pop_c, label=f"ES pop ({gen_key})",    alpha=0.6),
                    mpatches.Patch(color=pf_c,  label=f"ES Pareto ({gen_key})", alpha=0.95),
                ]
            if ns_fit is not None:
                handles += [
                    mpatches.Patch(color=NS_POP_C, label=f"NSGA-III pop ({gen_key0})",    alpha=0.6),
                    mpatches.Patch(color=NS_PF_C,  label=f"NSGA-III Pareto ({gen_key0})", alpha=0.95),
                ]
            handles.append(mpatches.Patch(color=RS_C, label=f"RS chunk {chunk_n}", alpha=0.85))
            handles.append(mpatches.Patch(color="red", label="Expert models", alpha=1.0))
            ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.85)

    gen_label = " vs ".join(f"gen_{g:04d}" for g in gen_list)
    fig.suptitle(
        f"ES summary 0905 ({gen_label}) vs RS chunk {chunk_n}\n"
        "(normalised scores, 3-objective: summary / faithful / deberta)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = f"{OUT_DIR}/es_summary_0905_{out_suffix}_vs_rs_chunk{chunk_n:02d}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
    plt.close()


# ── Parse args ────────────────────────────────────────────────────────────────
if "--pairs" in sys.argv:
    idx = sys.argv.index("--pairs")
    pairs = [list(map(int, p.split(","))) for p in sys.argv[idx+1:]]
    for pair in pairs:
        suffix = "_".join(f"gen{g:04d}" for g in pair)
        plot_one(pair, suffix)
else:
    gens = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [26, 27, 28]
    for gen_n in gens:
        if f"gen_{gen_n:04d}" not in pop_log:
            print(f"SKIP: gen_{gen_n:04d} not in population_log"); continue
        plot_one([gen_n], f"gen_{gen_n:04d}")
