"""3D mesh view of methods' Pareto fronts on Assistant and Summary.

Reuses the merged 3D data in _mo_data.py (same source as plot_combined.py).
For each method, computes the 3D Pareto front (maximization on all axes) and
renders it as a triangulated surface via Delaunay on a 2D projection.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import _mo_data as D


def pareto_mask_max(pts):
    n = len(pts)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        p = pts[i]
        for j in range(n):
            if i == j:
                continue
            q = pts[j]
            if np.all(q >= p) and np.any(q > p):
                mask[i] = False
                break
    return mask


def draw_method_surface(ax, pts, color, marker, label):
    pf = pts[pareto_mask_max(pts)]
    rgba_face = to_rgba(color, alpha=0.28)
    rgba_edge = to_rgba(color, alpha=0.85)

    # Triangulate using the projection onto the (x, y) plane — appropriate
    # when the Pareto surface is a graph z = f(x, y) over most of its domain.
    if len(pf) >= 3:
        try:
            tri = Delaunay(pf[:, :2])
            ax.plot_trisurf(
                pf[:, 0], pf[:, 1], pf[:, 2],
                triangles=tri.simplices,
                color=rgba_face,
                edgecolor=rgba_edge,
                linewidth=0.6,
                shade=False,
            )
        except Exception:
            pass

    ax.scatter(
        pf[:, 0], pf[:, 1], pf[:, 2],
        c=[color], marker=marker, s=42,
        edgecolors='white', linewidths=0.5,
        depthshade=False, label=label,
    )


methods_asst = [
    ('RS',     D.asst_rs,   '#75A563', 'P'),
    ('HoE',    D.asst_hoe,  '#C8AC35', 'D'),
    ('MORLHF', D.asst_morl, '#AA7DA8', 'o'),
    ('RiC',    D.asst_ric,  '#D9B390', '^'),
    ('MOD',    D.asst_mod,  '#5E82B2', 'v'),
    ('ES',     D.asst_es,   '#8E1B22', 's'),
]

methods_summ = [
    ('RS',     D.summ_rs,   '#75A563', 'P'),
    ('HoE',    D.summ_hoe,  '#C8AC35', 'D'),
    ('MORLHF', D.summ_morl, '#AA7DA8', 'o'),
    ('RiC',    D.summ_ric,  '#D9B390', '^'),
    ('MOD',    D.summ_mod,  '#5E82B2', 'v'),
    ('ES',     D.summ_es,   '#8E1B22', 's'),
]


fig = plt.figure(figsize=(18, 8))
fig.patch.set_facecolor('white')

# ---------- Assistant ----------
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
for name, pts, color, marker in methods_asst:
    draw_method_surface(ax1, pts, color, marker, name)
ax1.set_xlabel('harmless', fontsize=12, labelpad=8)
ax1.set_ylabel('helpful',  fontsize=12, labelpad=8)
ax1.set_zlabel('humor',    fontsize=12, labelpad=8)
ax1.set_title('Assistant', fontsize=14, fontweight='bold', pad=10)
ax1.view_init(elev=26, azim=-48)
ax1.set_box_aspect((1, 1, 0.75))

# ---------- Summary ----------
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for name, pts, color, marker in methods_summ:
    draw_method_surface(ax2, pts, color, marker, name)
ax2.set_xlabel('summary',  fontsize=12, labelpad=8)
ax2.set_ylabel('faithful', fontsize=12, labelpad=8)
ax2.set_zlabel('deberta',  fontsize=12, labelpad=8)
ax2.set_title('Summary', fontsize=14, fontweight='bold', pad=10)
ax1.view_init(elev=26, azim=-48)
ax1.set_box_aspect((1, 1, 0.75))

# ---------- Shared legend ----------
legend_handles = [
    Line2D([0], [0], marker=mk, color='w',
           markerfacecolor=col, markeredgecolor='white',
           markersize=10, label=name)
    for name, _, col, mk in methods_asst
]
fig.legend(
    handles=legend_handles,
    loc='lower center', ncol=len(methods_asst),
    fontsize=12, frameon=True, facecolor='white',
    edgecolor='#cccccc', bbox_to_anchor=(0.5, -0.02),
)

plt.tight_layout(rect=(0, 0.04, 1, 1))

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/pareto_3d.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('plots/pareto_3d.svg', format='svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Saved: plots/pareto_3d.png and plots/pareto_3d.svg')
