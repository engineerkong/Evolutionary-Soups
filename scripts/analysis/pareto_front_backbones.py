import os
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data — Beaver Reward × Cost  (both objectives maximised)
# ---------------------------------------------------------------------------
data_llama = {
    'ES': np.array([
        [ 4.6623, 4.9412],[-1.2424,11.2077],[ 8.3281,-1.6938],
        [-4.2135,13.4036],[ 0.9213, 8.8181],[ 6.9424, 1.2662],
        [ 2.8194, 7.2929],[10.0871,-6.4336],[-7.1863,15.1960],
        [ 5.3454, 4.3892],[ 7.7083, 0.1215],[ 9.3570,-5.6764],
        [-0.7004,10.3517],[-6.5917,14.6047],[-3.1079,11.7773],
        [ 8.9281,-3.7415],[ 6.7983, 1.1999],[-5.0579,13.7664],
        [ 9.1755,-4.8581],[ 3.6053, 6.0360],
    ]),
    'NSGAII': np.array([
        [-7.1695,15.1476],[ 8.3895,-2.3916],[ 9.8808,-6.1390],
        [-6.3782,14.4767],[ 9.0813,-4.9587],[ 2.0556, 7.7353],
        [-2.2207,11.6181],[ 7.3749, 0.5963],[-4.9928,13.2135],
        [-3.5819,11.6451],[ 4.8542, 4.3877],[ 5.3768, 3.2777],
        [ 1.2251, 8.9778],[-1.0529,10.7954],[ 0.7854, 8.9942],
        [ 8.0924,-1.6683],[-4.2541,13.2720],[ 3.1579, 6.4340],
        [ 6.2598, 1.8501],[ 7.5734,-0.4488],
    ]),
    'Gradient': np.array([
        [-7.1885,15.1913],[-7.0911,14.9978],[-6.5915,14.1986],
        [-5.2141,12.6767],[-2.0982, 9.0226],[ 3.9133, 1.5367],
        [ 7.8250,-3.3570],[ 9.2185,-5.6796],[ 9.8596,-6.5828],
        [ 9.8658,-6.7351],[ 9.9811,-7.1368],
    ]),
    'Single': np.array([
        [-4.0645,11.3850],[ 5.3606,-0.7907],[-7.1739,15.1924],
        [-6.6784,14.7109],[-5.7546,13.3310],[-6.8383,14.8322],
        [ 5.6701,-1.9381],[-6.7323,14.5745],[ 9.2598,-5.3006],
        [ 3.4240, 2.0303],[-1.5117, 8.7620],[ 6.8686,-2.7059],
        [ 7.5813,-3.6654],[ 7.1295,-2.6782],[ 0.0400, 5.3430],
        [ 2.0742, 2.9763],[-2.7325, 9.7912],[-7.1858,15.1526],
        [ 4.8574,-0.1604],[-4.8890,11.6377],
    ]),
}

data_qwen = {
    'ES': np.array([
        [ 6.9774, 6.0868],[ 5.9221, 8.9555],[ 9.6023,-2.6352],[ 8.1241, 2.4334],
        [10.3366,-4.9426],[ 4.9916,10.5892],[ 3.9795,11.4958],[ 7.4578, 4.2184],
        [ 9.1305,-0.9697],[10.7979,-6.0611],[ 8.4776, 0.4527],[ 5.4985, 9.3274],
        [ 7.6089, 2.8043],[10.2806,-5.4800],[ 9.8437,-3.8885],[ 5.9256, 8.8976],
        [ 6.3233, 7.8567],[ 5.4898, 9.0112],[10.1662,-4.7572],[ 5.2023,10.1001],
    ]),
    'NSGAII': np.array([
        [10.7024,-6.3744],[ 3.9903,11.4583],[ 7.8089, 3.1991],[ 8.4828, 0.6703],
        [ 4.7156,10.3787],[ 9.1174,-1.1327],[ 9.7967,-3.5331],[ 5.9266, 8.2332],
        [ 8.4633, 0.4401],[10.2174,-5.0222],[ 7.5920, 3.6303],[ 4.8720,10.5305],
        [ 6.7809, 5.9271],[ 6.0903, 7.3153],[ 4.2140,11.0581],[ 4.5853,10.8574],
        [ 9.5060,-3.1819],[ 6.5476, 5.2131],[ 7.0478, 4.8541],[ 7.1390, 4.4993],
    ]),
    'Gradient': np.array([
        [ 4.0884,11.2935],[ 4.1075,10.9434],[ 4.2868,10.3167],
        [ 4.8482, 9.5533],[ 6.0046, 6.4329],[ 8.3710, 0.8909],
        [ 9.8431,-3.0601],[10.3677,-4.5544],[10.5262,-5.4395],
        [10.4917,-5.6973],[10.5398,-5.8241],
    ]),
    'Single': np.array([
        [ 5.2415, 8.1429],[ 9.0053,-1.2460],[ 3.7944,11.1680],[ 4.1924,11.0231],
        [ 4.5430, 9.7691],[ 4.2025,11.3946],[ 9.0608,-2.1308],[ 4.2042,10.7769],
        [10.5072,-5.3807],[ 8.2533, 0.7967],[ 6.2015, 6.1034],[ 9.6465,-3.0739],
        [ 9.7671,-3.9020],[ 9.6823,-2.9216],[ 6.8722, 3.6055],[ 7.8435, 1.4756],
        [ 5.6877, 6.9056],[ 4.0397,11.7423],[ 8.7225,-1.1787],[ 4.8060, 8.7698],
    ]),
}

# ---------------------------------------------------------------------------
# Style — LLaMA = warm (purple-pink/red/orange/brown)
#         Qwen = cool (deep purple/blue/teal/cyan)
# ---------------------------------------------------------------------------
colors_llama = {
    'ES':       '#D62728',   # crimson red
    'NSGAII':   '#E67E22',   # warm orange
    'Gradient':   '#F1C40F',   # gold
    'Single': '#8C564B',   # warm brown
}
colors_qwen = {
    'ES':       '#1F4E8C',   # deep navy
    'NSGAII':   '#3498DB',   # bright blue
    'Gradient':   '#17BECF',   # teal/cyan
    'Single': '#5DADE2',   # light blue
}
markers = {
    'ES':       's',
    'NSGAII':   'D',
    'Gradient': 'p',
    'Single':   'X',
}


# ---------------------------------------------------------------------------
# Pareto helpers — both objectives maximised
# ---------------------------------------------------------------------------
def pareto_mask(pts):
    mask = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        for j, q in enumerate(pts):
            if i != j and np.all(q >= p) and np.any(q > p):
                mask[i] = False
                break
    return mask


def pf_sorted(pts):
    pf = pts[pareto_mask(pts)]
    return pf[np.argsort(pf[:, 0])]


# ---------------------------------------------------------------------------
# Plot one backbone onto the given axis
# ---------------------------------------------------------------------------
def plot_backbone(ax, data, colors, backbone_label):
    for method, points in data.items():
        msk   = pareto_mask(points)
        pf    = pf_sorted(points)
        is_es = (method == 'ES')
        ls    = '-' if is_es else '--'
        lw    = 2.6 if is_es else 1.8
        ms    = 12  if is_es else 10
        if (~msk).any():
            ax.scatter(points[~msk, 0], points[~msk, 1],
                       c=colors[method], marker=markers[method],
                       s=ms ** 2,
                       alpha=0.22, edgecolors='none', zorder=3)
        ax.plot(pf[:, 0], pf[:, 1],
                color=colors[method], marker=markers[method], linestyle=ls,
                linewidth=lw, markersize=ms, alpha=0.93,
                label=f'{backbone_label} {method}',
                zorder=6 if is_es else 5,
                markeredgecolor='white', markeredgewidth=0.4)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor('white')

plot_backbone(ax, data_llama, colors_llama, 'LLaMA')
plot_backbone(ax, data_qwen,  colors_qwen,  'Qwen')

ax.set_facecolor('#f0f0f0')
ax.grid(True, color='white', lw=0.8, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=14)

ax.set_xlabel('Beaver Reward', fontsize=16)
ax.set_ylabel('Beaver Cost',   fontsize=16)
# ax.set_title('LLaMA (warm) vs QWEN2 (cool) — Pareto Front Comparison',
#              fontsize=16, fontweight='bold', pad=6)

ax.set_xlim(-8.5, 11)
ax.set_ylim(-7.5, 16)

ax.legend(loc='upper right', fontsize=11, framealpha=0.92,
          facecolor='white', edgecolor='#cccccc', ncol=2,
          columnspacing=1.2, handletextpad=0.5)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/backbones_pareto_front.svg', format='svg',
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('plots/backbones_pareto_front.png', format='png', dpi=150,
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved: plots/backbones_pareto_front.svg and plots/backbones_pareto_front.png")