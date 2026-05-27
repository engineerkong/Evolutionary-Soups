import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(6, 5))

# ============================================================
# Pareto front points
# (all inside [0,5] x [0,5])
# ============================================================
x = np.array([0.4, 1.0, 1.9, 2.9, 3.6, 4.3, 4.8])
y = np.array([4.6, 4.4, 3.7, 3.2, 2.3, 1.1, 0.45])

# ============================================================
# 1. Draw dominated rectangles
# ============================================================
for xi, yi in zip(x, y):
    rect = Rectangle(
        (0, 0),
        xi,
        yi,
        facecolor="royalblue",
        alpha=0.08,
        edgecolor=None,
        zorder=1
    )
    ax.add_patch(rect)

# ============================================================
# 2. Connect Pareto front points
# ============================================================
ax.plot(
    x,
    y,
    '--',
    color='0.7',
    linewidth=1.8,
    zorder=2
)

# ============================================================
# 3. Pareto front points
# ============================================================
ax.scatter(
    x,
    y,
    s=70,
    facecolors='white',
    edgecolors='0.45',
    linewidths=2,
    zorder=3
)

# # ============================================================
# # 3.1 Highlighted point (star)
# # ============================================================
# ax.scatter(
#     [2.9],
#     [3.2],
#     s=900,                    # star usually needs larger size
#     facecolors='royalblue',
#     edgecolors='navy',
#     linewidths=2.5,
#     zorder=4
# )

# ============================================================
# 4. Inside points
# (must stay INSIDE the dominated region)
# ============================================================
inside = np.array([
    [0.8, 3.8],
    [1.4, 2.9],
    [2.0, 2.5],
    [2.8, 1.6],
    [3.5, 1.2],
    [4.0, 0.6],
    [1.8, 1.9],
    [3.0, 0.9]
])

ax.scatter(
    inside[:, 0],
    inside[:, 1],
    s=90,
    facecolors='lightgray',
    edgecolors='0.4',
    linewidths=1.8,
    zorder=3
)

# # ============================================================
# # 5. Text
# # ============================================================
# ax.text(
#     1.1,
#     1.0,
#     r'$\mathcal{HV}_k$',
#     fontsize=22
# )

# ============================================================
# 6. Axes style
# ============================================================
ax.set_xlim(0, 5.0)
ax.set_ylim(0, 5.0)

ax.set_xticks([])
ax.set_yticks([])

# hide top/right border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# thicker axes
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# axis arrows
ax.plot(5.0, 0, ">k", clip_on=False, markersize=10)
ax.plot(0, 5.0, "^k", clip_on=False, markersize=10)

# # optional labels
# ax.set_xlabel("harmless", fontsize=14)
# ax.set_ylabel("helpful", fontsize=14)

plt.tight_layout()

plt.savefig('./plots/plot.png', dpi=300, bbox_inches='tight', transparent=True)
plt.savefig('./plots/plot.svg', dpi=300, bbox_inches='tight', transparent=True)

plt.show()