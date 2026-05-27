import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Endpoints (shared by both curves)
# ============================================================

A = np.array([0.1, 0.88])   # top-left  (harmless-leaning)
B = np.array([0.88, 0.1])   # bottom-right (helpful-leaning)

# ============================================================
# Convex approximation — quadratic Bezier curve
# Control point bows it inward (toward origin),
# representing what scalar RL can reach
# ============================================================

P_ctrl = np.array([0.6, 0.6])

conv_curve = []
for t in np.linspace(0, 1, 300):
    p = (
        (1 - t)**2 * A
        + 2 * (1 - t) * t * P_ctrl
        + t**2 * B
    )
    conv_curve.append(p)

conv_curve = np.array(conv_curve)

# ============================================================
# True non-convex Pareto front
# Same endpoints, but pushed outward (away from origin) with
# non-uniform bumps that break convexity
# ============================================================

true_curve = []
for t in np.linspace(0, 1, 300):
    # Base Bezier (same control point as convex approx)
    p_base = (
        (1 - t)**2 * A
        + 2 * (1 - t) * t * P_ctrl
        + t**2 * B
    )
    # Outward normal direction (away from origin)
    normal = p_base / (np.linalg.norm(p_base) + 1e-8)
    # Offset: zero at both endpoints via sin(pi*t) envelope.
    # Superposition of incommensurate frequencies gives irregular shape.
    envelope = np.sin(np.pi * t)
    wiggle = (
        0.10
        + 0.04 * np.sin(2.3 * np.pi * t + 0.7)
        - 0.025 * np.sin(3.5 * np.pi * t + 1.9)
        + 0.02 * np.sin(6.7 * np.pi * t + 0.4)
        - 0.035 * np.sin(5.2 * np.pi * t + 1.1)
    )
    offset = envelope * wiggle
    p = p_base + offset * normal
    true_curve.append(p)

true_curve = np.array(true_curve)

# ============================================================
# Gap region: fill between convex approx (inner) and
# true front (outer)
# ============================================================

gap_x = np.concatenate([true_curve[:, 0], conv_curve[::-1, 0]])
gap_y = np.concatenate([true_curve[:, 1], conv_curve[::-1, 1]])

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(6, 4))

# Gap fill
ax.fill(gap_x, gap_y,
        color='#FADADD', alpha=0.55, zorder=1,
        rasterized=True)
ax.plot([], [], color='#FADADD', linewidth=8, alpha=0.55,
        label='Unreachable region by scalarization')

# Convex approximation (inner, dashed)
ax.plot(
    conv_curve[:, 0], conv_curve[:, 1],
    color='#A32D2D', linewidth=2.5,
    solid_capstyle='round',
    label='Convex approximation',
    zorder=4,
)

# True non-convex Pareto front (outer, solid)
ax.plot(
    true_curve[:, 0], true_curve[:, 1],
    color='#185FA5', linewidth=2.5,
    linestyle='--', solid_capstyle='round',
    label='True Pareto front',
    zorder=3,
)

# Endpoints
ax.scatter(*A, s=60, color='#2A7A2A', zorder=5)
ax.scatter(*B, s=60, color='#2A7A2A', zorder=5)

# Annotate gap
mid = 150
gap_mid_x = (true_curve[mid, 0] + conv_curve[mid, 0]) / 2
gap_mid_y = (true_curve[mid, 1] + conv_curve[mid, 1]) / 2
ax.annotate(
    'Non-convexity',
    xy=(gap_mid_x, gap_mid_y),
    xytext=(0.25, 0.55),
    fontsize=10, color='#8B0000', style='italic',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#FADADD',
              edgecolor='#A32D2D', linewidth=0.8),
    arrowprops=dict(arrowstyle='->', color='#A32D2D', lw=0.9,
                    connectionstyle='arc3,rad=0.2'),
    ha='center',
)

# ── Axes ─────────────────────────────────────────────────────
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_xlabel('Helpful', fontsize=12)
ax.set_ylabel('Harmless', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)
ax.grid(False)

ax.legend(
    fontsize=10, loc='lower left',
    frameon=True, framealpha=0.9,
    edgecolor='#D3D1C7', facecolor='white',
)

plt.tight_layout()
plt.savefig('./plots/non_convexity.png', dpi=300, bbox_inches='tight')
plt.savefig('./plots/non_convexity.svg', dpi=300, bbox_inches='tight')
print("Saved.")
