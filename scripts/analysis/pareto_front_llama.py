import os
import matplotlib.pyplot as plt
import numpy as np

# LLaMA ablation data — Beaver Reward × Cost
data = {
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
    'Single': np.array([
        [-4.0645,11.3850],[ 5.3606,-0.7907],[-7.1739,15.1924],
        [-6.6784,14.7109],[-5.7546,13.3310],[-6.8383,14.8322],
        [ 5.6701,-1.9381],[-6.7323,14.5745],[ 9.2598,-5.3006],
        [ 3.4240, 2.0303],[-1.5117, 8.7620],[ 6.8686,-2.7059],
        [ 7.5813,-3.6654],[ 7.1295,-2.6782],[ 0.0400, 5.3430],
        [ 2.0742, 2.9763],[-2.7325, 9.7912],[-7.1858,15.1526],
        [ 4.8574,-0.1604],[-4.8890,11.6377],
    ]),
    'Gradient': np.array([
        [-7.1885, 15.1913],[-7.0911, 14.9978],[-6.5915, 14.1986],
        [-5.2141, 12.6767],[-2.0982, 9.0226],[ 3.9133, 1.5367],
        [ 7.8250, -3.3570],[ 9.2185, -5.6796],[ 9.8596, -6.5828],
        [ 9.8658, -6.7351],[ 9.9811, -7.1368],
    ]),
    # 'Dummy': np.array([
    #     [-0.5168, 1.7410],[ 8.1433,-3.3751],[ 4.3083,-1.5570],
    #     [ 5.7317,-0.6402],[ 8.0751,-4.5039],[ 1.0973, 4.2088],
    #     [ 4.9298,-6.1335],[ 8.0494,-4.2645],[-0.3511, 7.2226],
    #     [ 4.9277, 3.2252],[ 6.9088,-4.3708],[ 3.3281,-2.7879],
    #     [-3.2808,10.7439],[ 6.2267,-2.4878],[ 3.4861, 4.8517],
    #     [ 9.6268,-5.7835],[ 4.1983,-0.3507],[ 2.0011, 3.4314],
    #     [ 5.7105,-2.2532],[-3.3657,11.1196],
    # ]),
}

colors = {
    'ES':     '#9400D3',
    'NSGAII': '#9B59B6',
    'Single': '#2ECC71',
    'Gradient':  '#F39C12',
}
markers = {
    'ES':     's',
    'NSGAII': 'D',
    'Single': 'p',
    'Gradient':  'X',
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
# Figure — same style as plot_combined.py
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('white')

for method, points in data.items():
    msk = pareto_mask(points)
    pf  = pf_sorted(points)
    is_es = (method == 'ES')
    ls    = '-' if is_es else '--'
    lw    = 2.6 if is_es else 1.8
    ms    = 12  if is_es else 10
    if (~msk).any():
        ax.scatter(points[~msk, 0], points[~msk, 1],
                   c=colors[method], marker=markers[method],
                   s=ms ** 2,
                   alpha=0.25, edgecolors='none', zorder=3)
    ax.plot(pf[:, 0], pf[:, 1],
            color=colors[method], marker=markers[method], linestyle=ls,
            linewidth=lw, markersize=ms, alpha=0.92,
            label=method, zorder=6 if is_es else 5,
            markeredgecolor='white', markeredgewidth=0.4)

ax.set_facecolor('#f0f0f0')
ax.grid(True, color='white', lw=0.8, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=14)

ax.set_xlabel('Beaver Reward', fontsize=16)
ax.set_ylabel('Beaver Cost',   fontsize=16)
ax.set_title('LLaMA Pareto Front Comparison',
             fontsize=16, fontweight='bold', pad=6)

ax.set_xlim(-8.5, 11)
ax.set_ylim(-7.5, 16)

ax.legend(loc='upper right', fontsize=14, framealpha=0.9,
          facecolor='white', edgecolor='#cccccc')

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/llama_pareto_front.svg', format='svg',
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('plots/llama_pareto_front.png', format='png', dpi=150,
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved: plots/llama_pareto_front.svg and plots/llama_pareto_front.png")
