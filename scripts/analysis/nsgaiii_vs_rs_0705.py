"""
Compare NSGA-III gen_0015 (nsgaiii_summary_1.5_ema_2front_norm_0705)
vs RS baseline (rs_summary_train_0705) across all shared chunks.

Produces a 3-panel scatter plot analogous to gen13_vs_rs_analysis.png:
  panel 1: summary vs faithful
  panel 2: summary vs deberta
  panel 3: faithful vs deberta
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = "/home/kong/workspace/MOMOE/MOMoE"
NSGAII_DIR = f"{BASE}/models/nsgaii/nsgaiii_summary_1.5_ema_2front_norm_0705"
RS_DIR     = f"{BASE}/results/ppo_rs/rs_summary_train_0705"
OUT_DIR    = f"{BASE}/logs/analysis_diagram"
OUT_FILE   = f"{OUT_DIR}/nsgaiii_0705_vs_rs_analysis.png"

# ── Expert bounds (from gen_-001 one-hot evaluations) ────────────────────────
# Expert 0 = summary, Expert 1 = faithful, Expert 2 = deberta
expert_raw = np.array([
    [1.9561145305633545,  -0.4358978271484375, -4.486939907073975],   # E0 sum
    [-0.14591821189969778, -0.3547602891921997,  0.9568580240011215], # E1 faith
    [0.4194423956796527,  -0.6684675216674805,  3.227126121520996],   # E2 deb
])
r_min   = expert_raw.min(axis=0)                         # [-0.146, -0.668, -4.487]
r_max   = expert_raw.max(axis=0)                         # [ 1.956, -0.355,  3.227]
r_range = np.maximum(r_max - r_min, 1e-6)


def normalize(raw: np.ndarray) -> np.ndarray:
    return (raw - r_min) / r_range


# ── Expert points in normalised space ────────────────────────────────────────
experts_norm = normalize(expert_raw)   # shape (3, 3)
expert_labels = ["E0(sum)", "E1(faith)", "E2(deb)"]


# ── NSGA-III gen_0015 population ─────────────────────────────────────────────
state_path = f"{NSGAII_DIR}/gen_0015/nsgaii_state.json"
with open(state_path) as f:
    state = json.load(f)

# fitness is already normalised by expert bounds inside ema.py
nsgaiii_fit = np.array(state["fitness"])   # (80, 3)

# Per-individual fitness from fitness.json files (also normalised)
gen15_dir = f"{NSGAII_DIR}/gen_0015"
ind_dirs  = sorted(d for d in os.listdir(gen15_dir) if d.startswith("ind_"))
ind_fitness = []
for ind in ind_dirs:
    fp = f"{gen15_dir}/{ind}/fitness.json"
    if os.path.exists(fp):
        with open(fp) as f:
            ind_fitness.append(json.load(f)["fitness"])
ind_fitness = np.array(ind_fitness)   # (80, 3) – same order as nsgaii_state


def pareto_front_3d(pts: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated (maximise all) points."""
    n = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                dominated[i] = True
                break
    return ~dominated


nsgaiii_pareto_mask = pareto_front_3d(nsgaiii_fit)
print(f"NSGA-III pop: {len(nsgaiii_fit)} | Pareto front: {nsgaiii_pareto_mask.sum()}")


# ── RS baseline: one point per preference CSV ─────────────────────────────────
rs_files = sorted(f for f in os.listdir(RS_DIR) if f.endswith(".csv"))
rs_raw   = []
for fname in rs_files:
    df = pd.read_csv(f"{RS_DIR}/{fname}", usecols=["obtained_score1",
                                                    "obtained_score2",
                                                    "obtained_score3"])
    rs_raw.append(df.mean().values)

rs_raw  = np.array(rs_raw)             # (21, 3) – raw reward scores
rs_norm = normalize(rs_raw)            # (21, 3) – normalised

rs_pareto_mask = pareto_front_3d(rs_norm)
print(f"RS points: {len(rs_norm)} | Pareto front: {rs_pareto_mask.sum()}")


# ── Population log: full history (all gens, ema values) ──────────────────────
with open(f"{NSGAII_DIR}/population_log.json") as f:
    pop_log = json.load(f)

# Collect ALL individuals across all generations (EMA fitness)
all_ema = []
for gen_key, inds in pop_log.items():
    for ind_key, data in inds.items():
        all_ema.append(data["ema"])
all_ema = np.array(all_ema)   # shape (N_total, 3)

# Use gen_0015 individuals specifically (EMA) for "pop"
gen15_ema = np.array([pop_log["gen_0015"][k]["ema"]
                       for k in sorted(pop_log["gen_0015"].keys())])
gen15_pareto_mask = pareto_front_3d(gen15_ema)


# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

projections = [
    (0, 1, "summary", "faithful"),
    (0, 2, "summary", "deberta"),
    (1, 2, "faithful", "deberta"),
]

COLORS = {
    "rs_all":      ("#5B9BD5", "D", 45, 0.55, "RS (all prefs)"),
    "rs_pareto":   ("#1F4E79", "D", 100, 0.95, "RS Pareto"),
    "nsgaiii_pop": ("#FFC000", "o", 45, 0.55, "MOMoE pop"),
    "nsgaiii_pf":  ("#FF6B00", "o", 100, 0.90, "MOMoE Pareto"),
    "expert":      ("red",     "*", 200, 1.00, "Expert models"),
}

for ax, (xi, yi, xlabel, ylabel) in zip(axes, projections):
    # RS all
    ax.scatter(rs_norm[:, xi], rs_norm[:, yi],
               color=COLORS["rs_all"][0], marker=COLORS["rs_all"][1],
               s=COLORS["rs_all"][2], alpha=COLORS["rs_all"][3],
               label=COLORS["rs_all"][4], zorder=2, linewidths=0.5,
               edgecolors="white")
    # RS Pareto
    ax.scatter(rs_norm[rs_pareto_mask, xi], rs_norm[rs_pareto_mask, yi],
               color=COLORS["rs_pareto"][0], marker=COLORS["rs_pareto"][1],
               s=COLORS["rs_pareto"][2], alpha=COLORS["rs_pareto"][3],
               label=COLORS["rs_pareto"][4], zorder=3, linewidths=0.8,
               edgecolors="white")
    # MOMoE population (EMA)
    ax.scatter(gen15_ema[:, xi], gen15_ema[:, yi],
               color=COLORS["nsgaiii_pop"][0], marker=COLORS["nsgaiii_pop"][1],
               s=COLORS["nsgaiii_pop"][2], alpha=COLORS["nsgaiii_pop"][3],
               label=COLORS["nsgaiii_pop"][4], zorder=4, linewidths=0.3,
               edgecolors="white")
    # MOMoE Pareto front (raw nsgaii_state fitness)
    ax.scatter(nsgaiii_fit[nsgaiii_pareto_mask, xi],
               nsgaiii_fit[nsgaiii_pareto_mask, yi],
               color=COLORS["nsgaiii_pf"][0], marker=COLORS["nsgaiii_pf"][1],
               s=COLORS["nsgaiii_pf"][2], alpha=COLORS["nsgaiii_pf"][3],
               label=COLORS["nsgaiii_pf"][4], zorder=5, linewidths=0.5,
               edgecolors="white")
    # Expert models
    for k, (ex, label) in enumerate(zip(experts_norm, expert_labels)):
        ax.scatter(ex[xi], ex[yi],
                   color="red", marker="*", s=220, zorder=6,
                   edgecolors="darkred", linewidths=0.8)
        ax.annotate(label, (ex[xi], ex[yi]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color="red", fontweight="bold")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{xlabel} vs {ylabel}", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(-0.05, 1.10)
    ax.grid(True, alpha=0.25, linestyle="--")

    # Legend only on first panel
    if xi == 0 and yi == 1:
        handles = [
            mpatches.Patch(color=COLORS["rs_all"][0],    label=COLORS["rs_all"][4],    alpha=0.7),
            mpatches.Patch(color=COLORS["rs_pareto"][0], label=COLORS["rs_pareto"][4], alpha=0.9),
            mpatches.Patch(color=COLORS["nsgaiii_pop"][0], label=COLORS["nsgaiii_pop"][4], alpha=0.6),
            mpatches.Patch(color=COLORS["nsgaiii_pf"][0],  label=COLORS["nsgaiii_pf"][4],  alpha=0.9),
            mpatches.Patch(color="red",                  label="Expert models",         alpha=1.0),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.8)

fig.suptitle(
    "MOMoE NSGA-III gen_0015 vs RS baseline\n"
    "(summary task, nsgaiii_summary_1.5_ema_2front_norm_0705, normalized scores)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT_FILE}")
plt.close()
