import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Key points
# ============================================================

f11 = np.array([0.8, 0.1])
f12 = np.array([0.65, 0.45])
f21 = np.array([0.4, 0.7])
f22 = np.array([0.15, 0.85])

# ============================================================
# Shared Gating Curve (red curve)
# Use quadratic Bezier curve
# ============================================================

P0 = f22
P1 = np.array([0.45, 0.65])   # control point
P2 = f11

curve = []

for t in np.linspace(0, 1, 250):
    p = (
        (1 - t)**2 * P0
        + 2 * (1 - t) * t * P1
        + t**2 * P2
    )
    curve.append(p)

curve = np.array(curve)

# ============================================================
# Outer boundary:
# f22 -> f21 -> f12 -> f11
# ============================================================

outer_edge = []

for t in np.linspace(0, 1, 250):

    # segment 1: f22 -> f21
    if t < 1/3:
        u = t / (1/3)
        p = (1-u) * f22 + u * f21

    # segment 2: f21 -> f12
    elif t < 2/3:
        u = (t - 1/3) / (1/3)
        p = (1-u) * f21 + u * f12

    # segment 3: f12 -> f11
    else:
        u = (t - 2/3) / (1/3)
        p = (1-u) * f12 + u * f11

    outer_edge.append(p)

outer_edge = np.array(outer_edge)

# ============================================================
# Generate surface using:
# interpolation(curve <-> outer_edge)
# ============================================================

surface_points = []

for i in range(len(curve)):

    c = curve[i]
    e = outer_edge[i]

    # interpolation between curve and edge
    for s in np.linspace(0, 1, 50):

        p = (1 - s) * c + s * e
        surface_points.append(p)

surface_points = np.array(surface_points)

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(6, 4))

# ------------------------------------------------------------
# Dashed polygon edges
# ------------------------------------------------------------

ax.plot(
    [f22[0], f21[0], f12[0], f11[0]],
    [f22[1], f21[1], f12[1], f11[1]],
    '--',
    color='gray',
    linewidth=2,
    alpha=0.9
)

# ------------------------------------------------------------
# Red curve
# ------------------------------------------------------------

ax.plot(
    curve[:, 0],
    curve[:, 1],
    color='#A32D2D',
    linewidth=2.5,
    label='Single gating',
    solid_capstyle='round'
)

# ------------------------------------------------------------
# Surface cloud
# NOW:
# the red curve becomes the bottom boundary
# ------------------------------------------------------------

ax.scatter(
    surface_points[:, 0],
    surface_points[:, 1],
    s=10,
    color='#add8e6',
    alpha=0.28,
    edgecolors='none',
    rasterized=True,
)
ax.plot([], [], color='#add8e6', linewidth=2.5, linestyle='--', label='Per-layer gating (l=2)')

# ------------------------------------------------------------
# Stars
# ------------------------------------------------------------

ax.scatter(*f22, s=260, marker='*', color='green', zorder=5)
ax.scatter(*f21, s=260, marker='*', color='orange', zorder=5)
ax.scatter(*f12, s=260, marker='*', color='orange', zorder=5)
ax.scatter(*f11, s=260, marker='*', color='green', zorder=5)

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.text(f22[0] + 0.015, f22[1] + 0.015, r'$f_{22}$', fontsize=12)
ax.text(f21[0] + 0.015, f21[1] + 0.015, r'$f_{21}$', fontsize=12)
ax.text(f12[0] + 0.015, f12[1] + 0.015, r'$f_{12}$', fontsize=12)
ax.text(f11[0] + 0.015, f11[1] + 0.015, r'$f_{11}$', fontsize=12)

# ------------------------------------------------------------
# Axes / title
# ------------------------------------------------------------

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)

ax.set_xlabel('Helpful', fontsize=12)
ax.set_ylabel('Harmless', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)

ax.grid(False)

# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    fontsize=10,
    loc='lower left',
    frameon=True,
    framealpha=0.9,
    edgecolor='#D3D1C7',
    facecolor='white'
)

plt.tight_layout()
plt.savefig('./plots/reward_space.png', dpi=300, bbox_inches='tight')
plt.savefig('./plots/reward_space.svg', dpi=300, bbox_inches='tight')
print("Saved.")