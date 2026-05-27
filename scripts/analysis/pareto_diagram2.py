import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
# Don't force ax.set_aspect('equal') — it locks the axes box to a square
# whenever xlim/ylim spans match, which then makes bbox_inches='tight' crop
# the figure back to a square regardless of figsize.

# ── Convex Pareto fronts (both bow outward toward upper-right) ──
# Parametric: angle θ from 90° to 0°, x = r(θ)*cos(θ), y = r(θ)*sin(θ)
# A convex front bowing toward upper-right is a quarter-ellipse arc

theta = np.linspace(np.pi/2, 0, 300)

# Safe front: helpful-leaning ellipse, wider on x-axis
# Semi-axes: a=0.90 (helpful), b=0.65 (harmless)
a_safe, b_safe = 0.90, 0.65
x_safe = a_safe * np.cos(theta)
y_safe = b_safe * np.sin(theta)

# Risky front: harmless-leaning ellipse, taller on y-axis
# Semi-axes: a=0.55 (helpful), b=0.90 (harmless)
a_risky, b_risky = 0.55, 0.92
x_risky = a_risky * np.cos(theta)
y_risky = b_risky * np.sin(theta)

# ── Tangent points: maximize 0.5*x + 0.5*y ─────────────────────
safe_scores  = 0.5 * x_safe  + 0.5 * y_safe
risky_scores = 0.5 * x_risky + 0.5 * y_risky

idx_safe  = np.argmax(safe_scores)
idx_risky = np.argmax(risky_scores)

px_safe,  py_safe  = x_safe[idx_safe],   y_safe[idx_safe]
px_risky, py_risky = x_risky[idx_risky], y_risky[idx_risky]

# ── Tangent iso-lines: 0.5x + 0.5y = C, slope = -1 ─────────────
C_safe  = 0.5 * px_safe  + 0.5 * py_safe
C_risky = 0.5 * px_risky + 0.5 * py_risky

x_line = np.linspace(-0.05, 1.1, 300)

def clip_line(x, y, xmin=0, xmax=1, ymin=0, ymax=1):
    mask = (y >= ymin) & (y <= ymax) & (x >= xmin) & (x <= xmax)
    return x[mask], y[mask]

xl_s, yl_s = clip_line(x_line, 2*C_safe  - x_line)
xl_r, yl_r = clip_line(x_line, 2*C_risky - x_line)

# ── Plot fronts ─────────────────────────────────────────────────
ax.plot(x_safe,  y_safe,  color='#185FA5', linewidth=2.5,
        label='Pareto front (safe prompt)',  zorder=3, solid_capstyle='round')
ax.plot(x_risky, y_risky, color='#A32D2D', linewidth=2.5,
        label='Pareto front (risky prompt)', zorder=3, solid_capstyle='round')

# ── Tangent lines (same slope, parallel) ────────────────────────
ax.plot(xl_s, yl_s, color='#185FA5', linewidth=1.4,
        linestyle='--', alpha=0.75, zorder=2)
ax.plot(xl_r, yl_r, color='#A32D2D', linewidth=1.4,
        linestyle='--', alpha=0.75, zorder=2)

# ── Tangent point dots ──────────────────────────────────────────
ax.scatter([px_safe],  [py_safe],  s=80, color='#185FA5', zorder=5)
ax.scatter([px_risky], [py_risky], s=80, color='#A32D2D', zorder=5)

# ── Crosshairs ──────────────────────────────────────────────────
kw = dict(linewidth=0.8, alpha=0.45, linestyle=':')
ax.plot([0, px_safe],  [py_safe,  py_safe],  color='#185FA5', **kw)
ax.plot([px_safe,  px_safe],  [0, py_safe],  color='#185FA5', **kw)
ax.plot([0, px_risky], [py_risky, py_risky], color='#A32D2D', **kw)
ax.plot([px_risky, px_risky], [0, py_risky], color='#A32D2D', **kw)

# ── Annotation boxes ─────────────────────────────────────────────
bbox_blue = dict(boxstyle='round,pad=0.4', facecolor='#E6F1FB',
                 edgecolor='#185FA5', linewidth=0.8)
bbox_red  = dict(boxstyle='round,pad=0.4', facecolor='#FCEBEB',
                 edgecolor='#A32D2D', linewidth=0.8)

ax.annotate(
    '"Recommend a red wine?"\noptimal $\\lambda$ = [0.8, 0.2]',
    xy=(px_safe, py_safe), xytext=(0.75, 0.68),
    fontsize=8.5, color='#0C447C', bbox=bbox_blue,
    arrowprops=dict(arrowstyle='->', color='#185FA5', lw=0.9,
                    connectionstyle='arc3,rad=0.2'),
    ha='center'
)

ax.annotate(
    '"Make someone drink without consent?"\noptimal $\\lambda$ = [0.2, 0.8]',
    xy=(px_risky, py_risky), xytext=(0.42, 0.90),
    fontsize=8.5, color='#791F1F', bbox=bbox_red,
    arrowprops=dict(arrowstyle='->', color='#A32D2D', lw=0.9,
                    connectionstyle='arc3,rad=-0.2'),
    ha='center'
)

# ── Tangent line labels ──────────────────────────────────────────
ax.text(xl_s[-1] - 0.15, yl_s[-1] + 0.25,
        r'$0.5h + 0.5H = C_1$',
        fontsize=8.5, color='#185FA5', ha='left', va='top')
ax.text(xl_r[-1] - 0.15, yl_r[-1] + 0.25,
        r'$0.5h + 0.5H = C_2$',
        fontsize=8.5, color='#A32D2D', ha='left', va='top')

# ── Key insight ──────────────────────────────────────────────────
ax.text(0.85, 0.78,
        'Same preference [0.5, 0.5]\n$\\Rightarrow$ different optimal $\\lambda$',
        fontsize=8.5, ha='center', va='bottom', color='#444441',
        style='italic', transform=ax.transAxes,
        multialignment='center')

# ── Axes ─────────────────────────────────────────────────────────
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Helpful',   fontsize=12)
ax.set_ylabel('Harmless',  fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)
ax.legend(fontsize=10, loc='lower left', frameon=True,
          framealpha=0.9, edgecolor='#D3D1C7', facecolor='white')

plt.tight_layout()
plt.savefig('./plots/pareto_convex.svg', bbox_inches='tight', dpi=300)
plt.savefig('./plots/pareto_convex.png', bbox_inches='tight', dpi=300)
print("saved")
