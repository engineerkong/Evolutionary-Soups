import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Data for each method (27 points, [summary, faithful, deberta])
data = {
    'RS': np.array([
        [0.031620, -0.548953, 2.351010],
        [0.003268, -0.524091, 2.120602],
        [-0.008483, -0.515534, 1.914690],
        [-0.066728, -0.487918, 1.530224],
        [-0.136394, -0.458795, 1.053581],
        [-0.169409, -0.440947, 0.847484],
        [-0.230481, -0.412156, 0.520896],
        [0.056849, -0.565663, 2.245979],
        [0.019721, -0.543592, 1.989607],
        [-0.010320, -0.521851, 1.727506],
        [-0.058759, -0.495336, 1.322659],
        [-0.092618, -0.475321, 1.072126],
        [-0.163239, -0.446713, 0.636577],
        [0.062248, -0.562284, 2.174367],
        [0.029930, -0.545281, 1.851671],
        [-0.004774, -0.520749, 1.465296],
        [-0.044987, -0.498090, 1.227286],
        [-0.113955, -0.462064, 0.753764],
        [0.098201, -0.578553, 1.919170],
        [0.061990, -0.546015, 1.514359],
        [0.039919, -0.530628, 1.276680],
        [-0.020235, -0.501506, 0.849137],
        [0.175432, -0.581932, 1.555270],
        [0.114359, -0.555564, 1.236577],
        [0.144216, -0.541094, 0.861476],
        [0.246272, -0.580940, 1.159493],
        [0.509328, -0.580610, 0.231620],
    ]),
    'NEW': np.array([
        [0.036761, -0.542049, 1.884613],
        [0.039368, -0.540544, 1.907308],
        [0.028131, -0.538450, 1.900331],
        [0.025707, -0.535108, 1.850679],
        [0.029196, -0.535219, 1.839479],
        [0.019684, -0.537606, 1.746236],
        [-0.045061, -0.514396, 1.582152],
        [0.040911, -0.546493, 1.867756],
        [0.033676, -0.546199, 1.802020],
        [0.036394, -0.538524, 1.813698],
        [0.043922, -0.546346, 1.858208],
        [0.028975, -0.535622, 1.730224],
        [0.041829, -0.519280, 1.214176],
        [0.041021, -0.551524, 1.906500],
        [0.034741, -0.548476, 1.775946],
        [0.027323, -0.543996, 1.761880],
        [0.035072, -0.537385, 1.742086],
        [0.095813, -0.532354, 1.171722],
        [0.051855, -0.551818, 1.720162],
        [0.043371, -0.541315, 1.696144],
        [0.070290, -0.541241, 1.642490],
        [0.109401, -0.542086, 1.322255],
        [0.059750, -0.541462, 1.667976],
        [0.091590, -0.548696, 1.586485],
        [0.124495, -0.546640, 1.225597],
        [0.102718, -0.545795, 1.548770],
        [0.156041, -0.552075, 1.078259],
    ]),
}

colors = {'RS': '#9B59B6', 'NEW': '#2ECC71'}
markers = {'RS': 'D', 'NEW': 'p'}


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
plt.savefig('./results/pareto_front_4panel.svg', format='svg',
            facecolor=fig.get_facecolor(), edgecolor='none', dpi=150,
            bbox_inches='tight')
plt.savefig('./results/pareto_front_4panel.png', format='png',
            facecolor=fig.get_facecolor(), edgecolor='none', dpi=150,
            bbox_inches='tight')
plt.show()
print("Saved: pareto_front_4panel.svg and pareto_front_4panel.png")