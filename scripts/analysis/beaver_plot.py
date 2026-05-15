import numpy as np
import matplotlib.pyplot as plt

pts_es = np.array([
    [ 3.5330, 5.6654],[-7.1543,15.1099],[ 9.5043,-5.8946],
    [-5.5297,13.9638],[-4.1724,13.3572],[ 6.6279, 0.1980],
    [ 0.0204, 9.9109],[ 9.0214,-4.9041],[ 1.3716, 7.9398],
    [ 0.2717, 9.2015],[-6.2048,14.3795],[-2.2724,12.2354],
    [ 6.4069, 1.3843],[ 6.2497, 1.5923],[ 7.7322,-1.9463],
    [ 8.6468,-3.5858],[ 3.9704, 4.6084],[-1.0725,10.1024],
    [ 4.9277, 3.2252],[ 5.5187, 3.2357],
])
pts_rs = np.array([
    [-7.1930,15.2041],[-7.1879,15.1966],[-6.8505,14.6612],
    [-5.4134,12.9381],[-1.2938, 8.5680],[ 5.2107, 0.1055],
    [ 8.4943,-4.2479],[ 9.5672,-5.7985],[ 9.9322,-5.9731],
    [10.1871,-6.0086],[10.2115,-6.6296],
])
pts_hoe = np.array([
    [-7.1885,15.1913],[-7.0911,14.9978],[-6.5915,14.1986],
    [-5.2141,12.6767],[-2.0982, 9.0226],[ 3.9133, 1.5367],
    [ 7.8250,-3.3570],[ 9.2185,-5.6796],[ 9.8596,-6.5828],
    [ 9.8658,-6.7351],[ 9.9811,-7.1368],
])
pts_morlhf = np.array([
    [-8.3799, 7.9883],[-8.0939, 5.3451],[-4.6696,11.0221],
    [-4.9177,13.6377],[-1.9487,16.1428],[ 4.1160, 4.9249],
    [ 8.9564,-4.4613],[ 9.3564,-4.8653],[10.1809,-5.9394],
    [10.3124,-6.4066],[10.4486,-7.2391],
])


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
    return pf[np.argsort(pf[:, 0])]   # sort by reward (col0)


# (name, pts, color, marker, markersize)
# order: RS, HoE, MORLHF, ES  — ES last
datasets = [
    ('RS',     pts_rs,     '#FF69B4', 'P',  7),
    ('HoE',    pts_hoe,    '#808000', 'D',  7),
    ('MORLHF', pts_morlhf, '#B22222', 'o',  8),
    ('ES',     pts_es,     '#9400D3', 's',  7),
]

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f0f0f0')
ax.grid(True, color='white', lw=0.8, zorder=0)

for name, pts, col, mk, ms in datasets:
    msk = pareto_mask(pts)
    pf  = pf_sorted(pts)   # x=cost (col1), y=reward (col0)

    # dominated points: faded scatter, no line
    if (~msk).any():
        ax.scatter(pts[~msk, 1], pts[~msk, 0],
                   c=col, marker=mk, s=(ms * 0.8) ** 2,
                   alpha=0.25, edgecolors='none', zorder=3)

    # Pareto-optimal points + connecting line
    ax.plot(pf[:, 1], pf[:, 0],
            color=col, marker=mk, linestyle='-',
            linewidth=1.8, markersize=ms, alpha=0.92,
            label=name, zorder=5,
            markeredgecolor='white', markeredgewidth=0.4)

ax.set_xlabel('cost',   fontsize=12)
ax.set_ylabel('reward', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)
ax.legend(fontsize=9, framealpha=0.9, loc='lower left',
          facecolor='white', edgecolor='#cccccc')

plt.tight_layout()
plt.savefig('beaver_pareto.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('beaver_pareto.svg', format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved: beaver_pareto.png and beaver_pareto.svg")