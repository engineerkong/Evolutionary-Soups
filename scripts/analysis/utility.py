"""
1 x 3 horizontal bar charts for the overview results:
  panel 1: mean linear utility   (higher is better)
  panel 2: mean Tchebyshev utility (lower is better)
  panel 3: hypervolume HV         (higher is better)

Each panel: 3 task groups (Beaver, Summary, Assistant), each with up to 6
method bars (RS, HoE, MORLHF, RiC, MOD, ES). MOD has no value on Summary /
Assistant -> left blank. Per task, the best value is bold; ES is highlighted.

Output: overview_results.pdf / .png
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# DATA  (Table: linear utility, Tchebyshev utility, HV)
#   None  -> bar omitted (MOD on Summary / Assistant)
# ----------------------------------------------------------------------
TASKS = ['Beaver', 'Summary', 'Assistant']
METHODS = ['RS', 'HoE', 'MORLHF', 'RiC', 'MOD', 'ES']

# metric -> task -> {method: value}
# All numbers come from cal_linear.py / cal_tchebyshev.py / cal_hv.py with MOD
# included.  Including MOD shifts every method's normalised value slightly,
# because the shared min-max bounds now span MOD's extremes too — so the
# Summary / Assistant columns differ from a MOD-omitted baseline.
DATA = {
    'lin': {  # mean linear utility, higher is better
        'Beaver':    {'RS': 0.7513, 'HoE': 0.7430, 'MORLHF': 0.7088,
                      'RiC': 0.4906, 'MOD': 0.7513, 'ES': 0.7757},
        'Summary':   {'RS': 0.6648, 'HoE': 0.6402, 'MORLHF': 0.6594,
                      'RiC': 0.5850, 'MOD': 0.6697, 'ES': 0.7319},
        'Assistant': {'RS': 0.7017, 'HoE': 0.6901, 'MORLHF': 0.7462,
                      'RiC': 0.6709, 'MOD': 0.7004, 'ES': 0.7337},
    },
    'tch': {  # mean Tchebyshev utility, lower is better
        'Beaver':    {'RS': 0.1900, 'HoE': 0.1885, 'MORLHF': 0.2359,
                      'RiC': 0.4671, 'MOD': 0.2016, 'ES': 0.1351},
        'Summary':   {'RS': 0.2001, 'HoE': 0.2193, 'MORLHF': 0.2680,
                      'RiC': 0.3183, 'MOD': 0.1991, 'ES': 0.1558},
        'Assistant': {'RS': 0.1695, 'HoE': 0.1789, 'MORLHF': 0.1646,
                      'RiC': 0.2154, 'MOD': 0.1685, 'ES': 0.1476},
    },
    'hv': {   # hypervolume, higher is better
        'Beaver':    {'RS': 0.4275, 'HoE': 0.4276, 'MORLHF': 0.5445,
                      'RiC': 0.0988, 'MOD': 0.2982, 'ES': 0.5830},
        'Summary':   {'RS': 0.2760, 'HoE': 0.2720, 'MORLHF': 0.1994,
                      'RiC': 0.1125, 'MOD': 0.2770, 'ES': 0.3133},
        'Assistant': {'RS': 0.3630, 'HoE': 0.3587, 'MORLHF': 0.4287,
                      'RiC': 0.2665, 'MOD': 0.3679, 'ES': 0.3932},
    },
}

# panel metadata: (data key, title, higher-is-better, x-axis label)
PANELS = [
    ('lin', r'Mean linear utility', True,  r'$\bar{u}^{\mathrm{lin}}$$\uparrow$'),
    ('tch', r'Mean Tchebyshev utility', False, r'$\bar{u}^{\mathrm{tch}}$$\downarrow$'),
    ('hv',  r'Hypervolume', True, r'$\mathcal{HV}$$\uparrow$'),
]

# ----------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.linewidth': 0.8})
COLORS = {
    'ES':     '#8E1B22',   # deep red  -> ours
    'RS':     '#9BBF8A',   # muted green
    'HoE':    '#D8C24A',   # gold
    'MORLHF': '#C6A0C4',   # mauve
    'RiC':    '#F0DAC8',   # pale tan
    'MOD':    '#7E9CC4',   # muted blue
}
# bar stacking order within a task group (top -> bottom)
BAR_ORDER = ['ES', 'RiC', 'MORLHF', 'HoE', 'RS', 'MOD']

# ----------------------------------------------------------------------
# DRAW ONE PANEL
# ----------------------------------------------------------------------
def draw_panel(ax, metric_key, title, higher_better, xlabel, show_ylabels):
    data = DATA[metric_key]
    n_task = len(TASKS)
    bh = 0.80 / len(BAR_ORDER)            # bar height
    group_gap = 0.12                      # vertical spacing between task groups

    all_vals = [v for t in TASKS for v in data[t].values() if v is not None]
    xmax = max(all_vals) * 1.18

    yticks, yticklabels = [], []
    for ti, task in enumerate(TASKS):
        # determine the best method for this task (bold its label)
        present = {m: v for m, v in data[task].items() if v is not None}
        best = (max(present, key=present.get) if higher_better
                else min(present, key=present.get))
        base = ti * (group_gap + len(BAR_ORDER) * bh)
        centers = []
        for bi, method in enumerate(BAR_ORDER):
            val = data[task].get(method)
            y = base + bi * bh
            centers.append(y)
            if val is None:
                continue                  # blank slot (e.g. MOD)
            is_es = (method == 'ES')
            ax.barh(y, val, height=bh * 0.86, color=COLORS[method],
                    edgecolor='white', linewidth=0.6,
                    zorder=3 if is_es else 2)
            is_best = (method == best)
            # Bold reserved for the per-task winner only.  ES still uses
            # its signature red colour so it's visually anchored, but it
            # only goes bold when it's actually the best.
            ax.text(val + xmax * 0.02, y, f'{val:.2f}',
                    va='center', ha='left', fontsize=5.0,
                    fontweight='bold' if is_best else 'normal',
                    color=COLORS['ES'] if is_es else '#444444')
        yticks.append(np.mean(centers))
        yticklabels.append(task)

    ax.set_yticks(yticks)
    if show_ylabels:
        ax.set_yticklabels(yticklabels, fontsize=8.5, fontweight='bold')
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()                     # Beaver on top
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.set_title(f'{title}', fontsize=8.5, fontweight='bold',
                 color='#8E1B22', pad=6)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.grid(axis='x', linewidth=0.5, alpha=0.35, zorder=0)
    ax.tick_params(axis='x', labelsize=6.5)
    ax.tick_params(axis='y', length=0)

# ----------------------------------------------------------------------
# ASSEMBLE
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(6.0, 4.0))
for i, (ax, (key, title, hb, xlab)) in enumerate(zip(axes, PANELS)):
    draw_panel(ax, key, title, hb, xlab, show_ylabels=(i == 0))

# shared legend
handles = [Patch(facecolor=COLORS[m], edgecolor='white',
                 label=('ES (ours)' if m == 'ES' else m))
           for m in METHODS]
fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
           fontsize=6.8, bbox_to_anchor=(0.5, -0.02),
           columnspacing=1.0, handlelength=1.2, handletextpad=0.4)

fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.subplots_adjust(wspace=0.08)
fig.savefig('./plots/overview_results.svg', bbox_inches='tight')
fig.savefig('./plots/overview_results.png', dpi=220,
            bbox_inches='tight')
print('saved')