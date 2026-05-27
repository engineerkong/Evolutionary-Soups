import os
import matplotlib.pyplot as plt
import numpy as np

# QWEN2 ablation data — Beaver Reward × Cost
data = {
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
    'Single': np.array([
        [ 5.2415, 8.1429],[ 9.0053,-1.2460],[ 3.7944,11.1680],[ 4.1924,11.0231],
        [ 4.5430, 9.7691],[ 4.2025,11.3946],[ 9.0608,-2.1308],[ 4.2042,10.7769],
        [10.5072,-5.3807],[ 8.2533, 0.7967],[ 6.2015, 6.1034],[ 9.6465,-3.0739],
        [ 9.7671,-3.9020],[ 9.6823,-2.9216],[ 6.8722, 3.6055],[ 7.8435, 1.4756],
        [ 5.6877, 6.9056],[ 4.0397,11.7423],[ 8.7225,-1.1787],[ 4.8060, 8.7698],
    ]),
    'Gradient': np.array([
    [  4.0884, 11.2935],[  4.1075, 10.9434],[  4.2868, 10.3167],
    [  4.8482,  9.5533],[  6.0046,  6.4329],[  8.3710,  0.8909],
    [  9.8431, -3.0601],[ 10.3677, -4.5544],[ 10.5262, -5.4395],
    [ 10.4917, -5.6973],[ 10.5398, -5.8241],
    ]),
    # 'Dummy': np.array([
    #     [ 5.7255, 2.7863],[ 3.7277, 3.4876],[ 0.9069, 8.0598],[ 4.9290, 0.2774],
    #     [ 1.3617, 8.2128],[ 4.5043, 3.6113],[ 3.2128, 5.3654],[ 0.6619, 8.0260],
    #     [ 1.2962, 8.0734],[ 2.2238, 8.4797],[ 1.2168, 7.5639],[ 8.6241,-4.6390],
    #     [ 3.4903, 5.7579],[ 9.3048,-5.7079],[ 5.3332, 6.5108],[ 8.2226,-2.2317],
    #     [ 7.4922,-3.3865],[ 1.1596, 8.3691],[ 2.2447, 6.5897],[10.6781,-6.2217],
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
ax.set_title('QWEN2 Pareto Front Comparison',
             fontsize=16, fontweight='bold', pad=6)

ax.set_xlim(-8.5, 11)
ax.set_ylim(-7.5, 16)

ax.legend(loc='upper right', fontsize=14, framealpha=0.9,
          facecolor='white', edgecolor='#cccccc')

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/qwen2_pareto_front.svg', format='svg',
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('plots/qwen2_pareto_front.png', format='png', dpi=150,
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved: plots/qwen2_pareto_front.svg and plots/qwen2_pareto_front.png")