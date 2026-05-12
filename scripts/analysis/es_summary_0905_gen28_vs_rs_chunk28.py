"""
Compare es_summary_0905 gen_0028 (population_log) vs RS chunk 28 [28*1024:29*1024].
Style mirrors gen_0016_vs_rs_chunk16.png.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE   = "/home/kong/workspace/MOMOE/MOMoE"
RS_DIR = f"{BASE}/results/ppo_rs/rs_summary_train_0705"
OUT    = f"{BASE}/logs/analysis_diagram/es_summary_0905_gen0028_vs_rs_chunk28.png"

os.makedirs(f"{BASE}/logs/analysis_diagram", exist_ok=True)

# ── gen_0028 population (already normalised) ──────────────────────────────────
with open(f"{BASE}/models/ES/es_summary_0905/population_log.json") as f:
    pop_log = json.load(f)

GEN_KEY = "gen_0028"
CHUNK   = 28
es_fit  = np.array([v["raw"] for v in pop_log[GEN_KEY].values()])  # (40, 3) normalised

# ── RS chunk 28: rows [28*1024 : 29*1024], mean per preference ───────────────
# Expert bounds for normalisation (same as other summary scripts)
expert_raw = np.array([
    [ 1.9561145305633545,  -0.4358978271484375, -4.486939907073975],   # E0 sum
    [-0.14591821189969778, -0.3547602891921997,  0.9568580240011215],  # E1 faith
    [ 0.4194423956796527,  -0.6684675216674805,  3.227126121520996],   # E2 deb
])
r_min   = expert_raw.min(axis=0)
r_max   = expert_raw.max(axis=0)
r_range = np.maximum(r_max - r_min, 1e-6)

def normalize(raw: np.ndarray) -> np.ndarray:
    return (raw - r_min) / r_range

rs_files = sorted(f for f in os.listdir(RS_DIR) if f.endswith(".csv"))
rs_raw = []
for fname in rs_files:
    df = pd.read_csv(f"{RS_DIR}/{fname}",
                     usecols=["Unnamed: 0", "obtained_score1",
                               "obtained_score2", "obtained_score3"])
    chunk = df[(df["Unnamed: 0"] >= 28 * 1024) & (df["Unnamed: 0"] < 29 * 1024)]
    if len(chunk) > 0:
        rs_raw.append(chunk[["obtained_score1", "obtained_score2",
                              "obtained_score3"]].mean().values)
rs_raw  = np.array(rs_raw)    # (n_prefs, 3)
rs_norm = normalize(rs_raw)

experts_norm  = normalize(expert_raw)
expert_labels = ["E0(sum)", "E1(faith)", "E2(deb)"]

# ── Pareto front helpers ──────────────────────────────────────────────────────
def pareto_mask(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    dom = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                dom[i] = True
                break
    return ~dom

def pareto_line(pts2d: np.ndarray) -> np.ndarray:
    pf = pts2d[pareto_mask(pts2d)]
    return pf[np.argsort(pf[:, 0])]

# ── Plot ──────────────────────────────────────────────────────────────────────
projections = [
    (0, 1, "summary",  "faithful"),
    (0, 2, "summary",  "deberta"),
    (1, 2, "faithful", "deberta"),
]

ES_POP_C  = "#4A90D9"
ES_PF_C   = "#1F4E79"
RS_C      = "#D62728"

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (xi, yi, xlabel, ylabel) in zip(axes, projections):
    rs2d  = rs_norm[:, [xi, yi]]
    rs_pf = pareto_line(rs2d)

    # RS
    ax.scatter(rs2d[:, 0], rs2d[:, 1],
               color=RS_C, marker="D", s=40, alpha=0.4,
               linewidths=0.4, edgecolors="white", zorder=2)
    ax.scatter(rs_pf[:, 0], rs_pf[:, 1],
               color=RS_C, marker="D", s=90, alpha=0.95,
               linewidths=0.6, edgecolors="white", zorder=3)
    ax.plot(rs_pf[:, 0], rs_pf[:, 1],
            color=RS_C, lw=1.8, alpha=0.85, linestyle="--", zorder=3)

    # ES population
    es2d  = es_fit[:, [xi, yi]]
    es_pf = pareto_line(es2d)

    ax.scatter(es2d[:, 0], es2d[:, 1],
               color=ES_POP_C, marker="o", s=40, alpha=0.35,
               linewidths=0.3, edgecolors="white", zorder=4)
    ax.scatter(es_pf[:, 0], es_pf[:, 1],
               color=ES_PF_C, marker="o", s=90, alpha=0.95,
               linewidths=0.6, edgecolors="white", zorder=5)
    ax.plot(es_pf[:, 0], es_pf[:, 1],
            color=ES_PF_C, lw=2.0, alpha=0.9, zorder=5)

    # Expert models
    for ex, label in zip(experts_norm, expert_labels):
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

    if xi == 0 and yi == 1:
        handles = [
            mpatches.Patch(color=ES_POP_C, label="ES pop (gen 28)",   alpha=0.6),
            mpatches.Patch(color=ES_PF_C,  label="ES Pareto (gen 28)", alpha=0.95),
            mpatches.Patch(color=RS_C,     label="RS chunk 28",        alpha=0.85),
            mpatches.Patch(color="red",    label="Expert models",      alpha=1.0),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.85)

fig.suptitle(
    "ES summary 0905 gen_0028 vs RS chunk 28 [28×1024 : 29×1024]\n"
    "(normalised scores, 3-objective: summary / faithful / deberta)",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
plt.close()
