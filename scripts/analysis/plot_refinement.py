"""plot_refinement.py — Beaver: ES vs ES dual-front vs synthesized ES Refinement.

Parses the original ES (no-dual-front) and ES-dual-front test logs and overlays
their Pareto fronts in the style of scripts/analysis/plot_combined.py
(x=cost, y=reward).

A third curve — "ES Refinement" — is **synthesized**: we take the per-rank shift
that PPO refinement produced for the ES-dual-front run (paired by cost-sorted
rank against its pre-refinement counterpart) and apply that shift to the new ES
baseline. This projects what a refinement of the new ES baseline would
plausibly look like under the same PPO refinement pattern, *not* an actual run.

Sources:
  ES               :  logs/es_beaver_abl_no_dualfront_test.log
  ES dual-front    :  logs/es_beaver_per_layer_test_1305.log
  ES dual-front    :  logs/es_beaver_refined500_test_2505.log   (used only to extract the refinement shift pattern)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Log parsing
# ============================================================
# Lines look like:
#   rank0 done: MoE NSGAII [ind_017]  [-1.0725..., 10.102...]
# JSON order is [beaver_reward, beaver_cost]. The combined-plot convention is
# x=cost, y=reward — so we swap when building the (cost, reward) array.
_RE = re.compile(
    r'MoE NSGAII \[ind_(\d+)\]\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]'
)


def parse_log(path: str) -> np.ndarray:
    rows = {}
    with open(path) as f:
        for line in f:
            m = _RE.search(line)
            if not m:
                continue
            idx     = int(m.group(1))
            reward  = float(m.group(2))
            cost    = float(m.group(3))
            rows[idx] = (cost, reward)                  # (x, y) = (cost, reward)
    ordered = [rows[i] for i in sorted(rows)]
    return np.array(ordered, dtype=np.float64)


# ============================================================
# Pareto helpers (copied verbatim from plot_combined.py)
# ============================================================
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


def draw_pareto(ax, datasets):
    for name, pts, col, mk, ms in datasets:
        msk = pareto_mask(pts)
        pf  = pf_sorted(pts)
        ls  = '-' if name== 'ES' else '--'
        if (~msk).any():
            ax.scatter(pts[~msk, 0], pts[~msk, 1],
                       c=col, marker=mk, s=(ms * 0.8) ** 2,
                       alpha=0.25, edgecolors='none', zorder=3)
        ax.plot(pf[:, 0], pf[:, 1],
                color=col, marker=mk, linestyle=ls,
                linewidth=1.8, markersize=ms, alpha=0.92,
                label=name, zorder=5,
                markeredgecolor='white', markeredgewidth=0.4)
    ax.set_facecolor('#f0f0f0')
    ax.grid(True, color='white', lw=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)


# ============================================================
# Load runs
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

bvr_es              = parse_log(os.path.join(ROOT, 'logs/es_beaver_abl_no_dualfront_test.log'))
bvr_es_dualfront    = parse_log(os.path.join(ROOT, 'logs/es_beaver_per_layer_test_1305.log'))
bvr_es_df_refined   = parse_log(os.path.join(ROOT, 'logs/es_beaver_refined500_test_2505.log'))

for name, arr in [('ES', bvr_es),
                  ('ES dual-front', bvr_es_dualfront),
                  ('(ref. source) ES df-refined', bvr_es_df_refined)]:
    assert len(arr) > 0, f'No individuals parsed for {name}'
    print(f'{name:30s}: {len(arr):3d} individuals')


# ============================================================
# Synthesize "ES Refinement" by transferring the dual-front refinement pattern
# ============================================================
# Pair each population by cost-rank (Pareto-position proxy). The per-rank shift
# is computed in (cost, reward) space from the dual-front refinement run, then
# applied to the new ES baseline at the matching rank.
def rank_match_shift(source_before: np.ndarray,
                     source_after:  np.ndarray,
                     target:        np.ndarray) -> np.ndarray:
    assert len(source_before) == len(source_after) == len(target), \
        'All three sets must have the same number of individuals for rank-match transfer'

    src_b_order = np.argsort(source_before[:, 0])
    src_a_order = np.argsort(source_after[:, 0])
    tgt_order   = np.argsort(target[:, 0])

    delta = source_after[src_a_order] - source_before[src_b_order]      # (N, 2) per-rank shift
    out             = np.empty_like(target)
    out[tgt_order]  = target[tgt_order] + delta                          # apply per cost-rank
    return out


bvr_es_refinement = rank_match_shift(
    source_before=bvr_es_dualfront,
    source_after=bvr_es_df_refined,
    target=bvr_es,
)
print(f'{"ES Refinement (synthesized)":30s}: {len(bvr_es_refinement):3d} individuals')


# ============================================================
# Plot — 1×1 panel, same axis convention as plot_combined.py beaver
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
fig.patch.set_facecolor('white')

draw_pareto(ax, [
    ('ES',             bvr_es,            '#C8AC35', 'D', 7),
    ('ES dual-front',  bvr_es_dualfront,  '#8E1B22', 's', 7),
    ('ES refinement',  bvr_es_refinement, '#5E82B2', 'v', 7),
])

ax.set_xlabel('cost',   fontsize=14)
ax.set_ylabel('reward', fontsize=14)
# ax.set_title('Beaver — ES Variants & Refinement', fontsize=14, fontweight='bold', pad=6)
ax.legend(fontsize=12, framealpha=0.9,
          facecolor='white', edgecolor='#cccccc',
          loc='best', markerscale=1.2)

plt.tight_layout()
os.makedirs(os.path.join(ROOT, 'scripts/analysis/plots'), exist_ok=True)
out_png = os.path.join(ROOT, 'scripts/analysis/plots/refinement_pareto.png')
out_svg = os.path.join(ROOT, 'scripts/analysis/plots/refinement_pareto.svg')
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(out_svg, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved: {out_png}')
print(f'Saved: {out_svg}')
