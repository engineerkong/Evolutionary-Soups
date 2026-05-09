"""
Compare NSGA-IV (HVC) run nsgaiii_summary_0805t_tchebycheff
gen_0001, gen_0005 vs RS baseline (rs_summary_train_0705, first 128 prompts).

Style mirrors 0805t_gen01_10_20_30_pareto_front.png.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE    = "/home/kong/workspace/MOMOE/MOMoE"
RUN_DIR = f"{BASE}/models/nsgaii/nsgaiii_summary_0805t_tchebycheff"
RS_DIR  = f"{BASE}/results/ppo_rs/rs_summary_train_0705"
OUT     = f"{BASE}/logs/nsga4_hvc_gen01_05_vs_rs.png"

os.makedirs(f"{BASE}/logs", exist_ok=True)

# ── Expert bounds (same as nsgaiii_vs_rs_0705.py) ────────────────────────────
expert_raw = np.array([
    [1.9561145305633545,  -0.4358978271484375, -4.486939907073975],
    [-0.14591821189969778, -0.3547602891921997,  0.9568580240011215],
    [0.4194423956796527,  -0.6684675216674805,  3.227126121520996],
])
r_min   = expert_raw.min(axis=0)
r_max   = expert_raw.max(axis=0)
r_range = np.maximum(r_max - r_min, 1e-6)

def normalize(raw):
    return (raw - r_min) / r_range

# ── Load population_log ───────────────────────────────────────────────────────
pop_log = json.load(open(f"{RUN_DIR}/population_log.json"))

GENS = ["gen_0001", "gen_0005", "gen_0010", "gen_0025"]

def load_gen(gen_key):
    return np.array([v["raw"] for v in pop_log[gen_key].values()])

# ── RS baseline: mean over first 128 prompts ──────────────────────────────────
rs_files = sorted(f for f in os.listdir(RS_DIR) if f.endswith(".csv"))
rs_raw = []
for fname in rs_files:
    df = pd.read_csv(f"{RS_DIR}/{fname}",
                     usecols=["Unnamed: 0", "obtained_score1",
                               "obtained_score2", "obtained_score3"])
    df128 = df[df["Unnamed: 0"] < 128]
    rs_raw.append(df128[["obtained_score1","obtained_score2","obtained_score3"]].mean().values)
rs_raw  = np.array(rs_raw)
rs_norm = normalize(rs_raw)

# ── 2-D Pareto front (maximise both) ─────────────────────────────────────────
def pareto_2d(pts):
    """Boolean mask of 2-D non-dominated points (maximise both axes)."""
    n = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and pts[j,0] >= pts[i,0] and pts[j,1] >= pts[i,1] \
                      and (pts[j,0] > pts[i,0] or pts[j,1] > pts[i,1]):
                dominated[i] = True
                break
    return ~dominated

def pareto_front_line(pts2d):
    """Return sorted Pareto-front points for drawing a step line."""
    mask = pareto_2d(pts2d)
    pf   = pts2d[mask]
    return pf[np.argsort(pf[:,0])]

# ── Colour scheme ─────────────────────────────────────────────────────────────
# Navy blue gradient light→dark for gen_0001→gen_0005; red dashed for RS
GEN_COLORS  = ["#7EB6E8", "#4A90D9", "#1F5FA6", "#0D2D5E"]   # light → dark navy
GEN_LABELS  = ["NSGA-IV gen 1", "NSGA-IV gen 5", "NSGA-IV gen 10", "NSGA-IV gen 25"]
RS_COLOR    = "#D62728"

projections = [
    (0, 1, "summary",  "faithful"),
    (0, 2, "summary",  "deberta"),
    (1, 2, "faithful", "deberta"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (xi, yi, xlabel, ylabel) in zip(axes, projections):

    # RS Pareto front
    rs2d = rs_norm[:, [xi, yi]]
    rs_pf = pareto_front_line(rs2d)
    ax.scatter(rs2d[:,0], rs2d[:,1],
               color=RS_COLOR, marker="D", s=40, alpha=0.4,
               linewidths=0.4, edgecolors="white", zorder=2)
    ax.scatter(rs_pf[:,0], rs_pf[:,1],
               color=RS_COLOR, marker="D", s=90, alpha=0.9,
               linewidths=0.6, edgecolors="white", zorder=3)
    ax.plot(rs_pf[:,0], rs_pf[:,1],
            color=RS_COLOR, lw=1.8, alpha=0.85, linestyle="--", zorder=3)

    # NSGA-IV generations
    for gen_key, color, label in zip(GENS, GEN_COLORS, GEN_LABELS):
        pts  = load_gen(gen_key)          # already normalised
        pts2d = pts[:, [xi, yi]]
        pf    = pareto_front_line(pts2d)

        ax.scatter(pts2d[:,0], pts2d[:,1],
                   color=color, marker="o", s=40, alpha=0.35,
                   linewidths=0.3, edgecolors="white", zorder=4)
        ax.scatter(pf[:,0], pf[:,1],
                   color=color, marker="o", s=90, alpha=0.95,
                   linewidths=0.6, edgecolors="white", zorder=5)
        ax.plot(pf[:,0], pf[:,1],
                color=color, lw=2.0, alpha=0.9, zorder=5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{xlabel} vs {ylabel}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")

    if xi == 0 and yi == 1:
        handles = [
            mpatches.Patch(color=GEN_COLORS[0], label=GEN_LABELS[0], alpha=0.8),
            mpatches.Patch(color=GEN_COLORS[1], label=GEN_LABELS[1], alpha=0.9),
            mpatches.Patch(color=GEN_COLORS[2], label=GEN_LABELS[2], alpha=0.95),
            mpatches.Patch(color=GEN_COLORS[3], label=GEN_LABELS[3], alpha=1.0),
            mpatches.Patch(color=RS_COLOR,       label="RS baseline",  alpha=0.85),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.85)

fig.suptitle(
    "NSGA-IV (HVC) gen 1, 5, 10 & 25 vs RS baseline\n"
    "(nsgaiii_summary_0805t_tchebycheff, normalized scores, first 128 prompts)",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
plt.close()
