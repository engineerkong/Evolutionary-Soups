import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Data for each method (27 points, [summary, faithful, deberta])
data = {
    'RS': np.array([
    [-1.0419, -0.4698, 1.9390],
    [-1.0585, -0.4821, 1.8188],
    [-0.8504, -0.4224, -0.9769],
    [-0.9585, -0.3865, -1.1752],
    [-0.9925, -0.3951, -1.6435],
    [-0.9967, -0.3984, -1.7506],
    [-1.0065, -0.5096, 1.7979],
    [-0.7531, -0.4532, -0.7682],
    [-0.8542, -0.4303, -1.1902],
    [-0.9640, -0.3933, -1.5629],
    [-0.8672, -0.3683, -1.8504],
    [-1.2299, -0.4788, 1.3264],
    [-0.7162, -0.4300, -1.5139],
    [-1.0732, -0.4835, -2.1875],
    [-0.8094, -0.3910, -1.9669],
    [-0.6681, -0.4730, -1.0694],
    [-0.6415, -0.4293, -1.3063],
    [-1.2468, -0.4346, -1.2389],
    [-0.5312, -0.4753, -1.4294],
    [-0.5985, -0.4500, -1.5502],
    [0.1123, -0.4231, -1.8223],
    ]),
    'NSGAII-test': np.array([
    [-0.4261, -0.4545, -1.9595],
    [-1.0604, -0.5098, 1.8002],
    [-0.9085, -0.3503, -1.6742],
    [-1.0269, -0.5182, 1.9423],
    [-1.0188, -0.5054, 1.5057],
    [-0.7846, -0.4554, -0.3642],
    [-0.7224, -0.3654, -1.0226],
    [-0.6600, -0.4091, -1.2928],
    [-0.6677, -0.4294, -1.2361],
    [-0.5541, -0.4433, -1.6188],
    [-1.0313, -0.4787, 1.5461],
    [-0.7694, -0.4648, -0.5553],
    [-0.4995, -0.4532, -1.7922],
    [-0.6934, -0.4437, -0.8394],
    [-1.0389, -0.4973, 1.7675],
    [-0.6097, -0.4581, -1.0657],
    [-0.7404, -0.4380, -0.6441],
    [-0.6463, -0.4301, -1.3467],
    [-0.6509, -0.4698, -0.9247],
    [-0.4944, -0.4574, -1.6481],
    ]),
    'NSGAII': np.array([
    [-0.4419, -0.4556, -1.8144],
    [-1.0702, -0.5143, 1.7179],
    [-1.1053, -0.3856, -2.0880],
    [-1.0455, -0.5023, 1.7123],
    [-0.7207, -0.4265, -1.4216],
    [-0.7514, -0.4795, -1.3024],
    [-1.0479, -0.5061, 1.3793],
    [-0.7098, -0.4340, -1.5107],
    [-0.4450, -0.4638, -1.5340],
    [-1.0607, -0.4064, -1.6102],
    [-0.4624, -0.4561, -1.8208],
    [-0.9842, -0.4538, -2.2050],
    [-0.8719, -0.4610, -2.6450],
    [-0.6701, -0.4286, -0.8614],
    [-0.6796, -0.4293, -1.3754],
    [-0.9954, -0.4438, -2.1904],
    [-0.7439, -0.4625, -0.6974],
    [-0.9622, -0.4137, -1.1973],
    [-0.9631, -0.4116, -1.5736],
    [-1.1860, -0.4219, -1.1188],
    ]),
}

colors = {'NSGAII-test': '#9B59B6', 'NSGAII': '#2ECC71', 'RS': '#E74C3C'}
markers = {'NSGAII-test': 'D', 'NSGAII': 'p', 'RS': 'o'}


def get_pareto_front_3d(points):
    pareto_mask = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i != j and np.all(q >= p) and np.any(q > p):
                pareto_mask[i] = False
                break
    return points[pareto_mask]


def get_pareto_front_2d(points):
    sorted_indices = np.argsort(-points[:, 0])
    sorted_points = points[sorted_indices]
    pareto_front = [sorted_points[0]]
    max_y = sorted_points[0, 1]
    for point in sorted_points[1:]:
        if point[1] > max_y:
            pareto_front.append(point)
            max_y = point[1]
    return np.array(pareto_front)


plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('#1a1a2e')

# ── Subplot layout: 2x2 ──────────────────────────────────────────────
ax3d = fig.add_subplot(2, 2, 1, projection='3d')
ax_sf = fig.add_subplot(2, 2, 2)   # summary vs faithful
ax_sd = fig.add_subplot(2, 2, 3)   # summary vs deberta
ax_fd = fig.add_subplot(2, 2, 4)   # faithful vs deberta

subplots_2d = [
    (ax_sf, 0, 1, 'Summary', 'Faithful'),
    (ax_sd, 0, 2, 'Summary', 'DeBERTa'),
    (ax_fd, 1, 2, 'Faithful', 'DeBERTa'),
]

for ax in [ax_sf, ax_sd, ax_fd]:
    ax.set_facecolor('#16213e')

ax3d.set_facecolor('#16213e')

# ── 3D subplot ───────────────────────────────────────────────────────
for method, points in data.items():
    pareto = get_pareto_front_3d(points)
    ax3d.scatter(points[:, 0], points[:, 1], points[:, 2],
                 c=colors[method], marker=markers[method],
                 s=50, alpha=0.6, edgecolors='white', linewidths=0.3)
    ax3d.scatter(pareto[:, 0], pareto[:, 1], pareto[:, 2],
                 c=colors[method], marker=markers[method],
                 s=130, alpha=1.0, edgecolors='white', linewidths=1.0,
                 label=method)

ax3d.set_xlabel('Summary', fontsize=9, color='#e0e0e0', labelpad=6)
ax3d.set_ylabel('Faithful', fontsize=9, color='#e0e0e0', labelpad=6)
ax3d.set_zlabel('DeBERTa', fontsize=9, color='#e0e0e0', labelpad=6)
ax3d.set_title('3D View', fontsize=12, color='#f0f0f0', pad=10)
ax3d.tick_params(colors='#a0a0a0', labelsize=7)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor('#3a3a5a')
ax3d.yaxis.pane.set_edgecolor('#3a3a5a')
ax3d.zaxis.pane.set_edgecolor('#3a3a5a')
ax3d.grid(True, linestyle='--', alpha=0.3, color='#3a3a5a')
legend3d = ax3d.legend(loc='upper left', fontsize=9, framealpha=0.9,
                        facecolor='#1a1a2e', edgecolor='#3a3a5a')
for text in legend3d.get_texts():
    text.set_color('#e0e0e0')

# ── 2D subplots ──────────────────────────────────────────────────────
for ax, xi, yi, xlabel, ylabel in subplots_2d:
    for method, points in data.items():
        pts2d = points[:, [xi, yi]]
        pareto2d = get_pareto_front_2d(pts2d)
        pareto2d_sorted = pareto2d[np.argsort(pareto2d[:, 0])]

        # All points (faded)
        ax.scatter(pts2d[:, 0], pts2d[:, 1],
                   c=colors[method], marker=markers[method],
                   s=50, alpha=0.5, edgecolors='white', linewidths=0.3)

        # Pareto points (bright)
        ax.scatter(pareto2d[:, 0], pareto2d[:, 1],
                   c=colors[method], marker=markers[method],
                   s=120, alpha=1.0, edgecolors='white', linewidths=1.0,
                   label=method)

        # Pareto front line
        ax.plot(pareto2d_sorted[:, 0], pareto2d_sorted[:, 1],
                c=colors[method], linewidth=2.0, alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=11, color='#e0e0e0')
    ax.set_ylabel(ylabel, fontsize=11, color='#e0e0e0')
    ax.set_title(f'{xlabel} vs {ylabel}', fontsize=12, color='#f0f0f0')
    ax.grid(True, linestyle='--', alpha=0.3, color='#3a3a5a')
    ax.tick_params(colors='#a0a0a0', labelsize=9)
    legend = ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
                       facecolor='#1a1a2e', edgecolor='#3a3a5a')
    for text in legend.get_texts():
        text.set_color('#e0e0e0')

fig.suptitle('Pareto Front Comparison — 3D & 2D Projections',
             fontsize=16, fontweight='bold', color='#f0f0f0', y=1.01)

plt.tight_layout()
plt.savefig('./results/pareto_front_4panel_1504.svg', format='svg',
            facecolor=fig.get_facecolor(), edgecolor='none', dpi=150,
            bbox_inches='tight')
plt.savefig('./results/pareto_front_4panel_1504.png', format='png',
            facecolor=fig.get_facecolor(), edgecolor='none', dpi=150,
            bbox_inches='tight')
plt.show()
print("Saved: pareto_front_4panel_1504.svg and pareto_front_4panel_1504.png")