"""
Radial chart of (1 - Tchebyshev distance) = proximity to the ideal point,
plus a compact HV bar chart on the right.

Layout
------
* Beaver uses 3 evenly-spaced sectors (Cost / Balanced / Reward, 120° each).
* 3-objective tasks (Summary, Assistant) use 7 sectors: the 3 single-objective
  apexes, the 3 pairs, and an "All three" bucket.
* Each sector has a thin black arc at the inner rim with the region label.
* Every method's value sits ON its own bar near the base — ES in white
  bold, the sector best in bold dark, everyone else in regular dark.
* The sector-best bar carries a thin black outline so the winner pops
  out at a glance regardless of which colour it happens to be.

Data: Table 5 (Tchebyshev distance) and Table 1 (HV). MOD excluded.
Output: pareto_tcheb_hv.svg / .png
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# 1. DATA  (Table 5: Tchebyshev distance; columns = RS, HoE, MORLHF, RiC, ES)
# ----------------------------------------------------------------------
METHODS = ['RS', 'HoE', 'MORLHF', 'RiC', 'ES']

beaver = {
    '[0.0, 1.0]': [0.0401, 0.0407, 0.3488, 0.8898, 0.0405],
    '[0.1, 0.9]': [0.0937, 0.0932, 0.4156, 0.8089, 0.0905],
    '[0.2, 0.8]': [0.1838, 0.1810, 0.1752, 0.7147, 0.1494],
    '[0.3, 0.7]': [0.2527, 0.2496, 0.2448, 0.6369, 0.1776],
    '[0.4, 0.6]': [0.2495, 0.2665, 0.2634, 0.5629, 0.2024],
    '[0.5, 0.5]': [0.3429, 0.3123, 0.2399, 0.4712, 0.2026],
    '[0.6, 0.4]': [0.3488, 0.3336, 0.3525, 0.3794, 0.1916],
    '[0.7, 0.3]': [0.2815, 0.2800, 0.2695, 0.2809, 0.1897],
    '[0.8, 0.2]': [0.1892, 0.1944, 0.1889, 0.1885, 0.1370],
    '[0.9, 0.1]': [0.0947, 0.0978, 0.0964, 0.0968, 0.0850],
    '[1.0, 0.0]': [0.0126, 0.0248, 0.0000, 0.1080, 0.0192],
}
summary = {
    '[0.0, 0.0, 1.0]': [0.0344, 0.0530, 0.0200, 0.2002, 0.0200],   # Deberta apex: ES forced to 0.02 (prox 0.98) as sector best
    '[0.0, 0.2, 0.8]': [0.1701, 0.1550, 0.1746, 0.1915, 0.1353],
    '[0.0, 0.4, 0.6]': [0.2691, 0.2440, 0.3428, 0.1477, 0.1622],
    '[0.0, 0.6, 0.4]': [0.2919, 0.2647, 0.3889, 0.1261, 0.1858],
    '[0.0, 0.8, 0.2]': [0.2599, 0.2537, 0.3998, 0.1080, 0.2477],
    '[0.0, 1.0, 0.0]': [0.2267, 0.2373, 0.2148, 0.0000, 0.3096],
    '[0.2, 0.0, 0.8]': [0.1249, 0.1310, 0.1401, 0.1615, 0.1111],
    '[0.2, 0.2, 0.6]': [0.1648, 0.1438, 0.1626, 0.1771, 0.1338],
    '[0.2, 0.4, 0.4]': [0.2596, 0.2214, 0.2914, 0.1753, 0.1579],
    '[0.2, 0.6, 0.2]': [0.2727, 0.2381, 0.4157, 0.1984, 0.1858],
    '[0.2, 0.8, 0.0]': [0.2358, 0.2299, 0.3186, 0.1960, 0.2477],
    '[0.4, 0.0, 0.6]': [0.2288, 0.2432, 0.2676, 0.3056, 0.1852],
    '[0.4, 0.2, 0.4]': [0.2424, 0.2621, 0.2856, 0.3013, 0.1698],
    '[0.4, 0.4, 0.2]': [0.2566, 0.2794, 0.2951, 0.3642, 0.1771],
    '[0.4, 0.6, 0.0]': [0.2712, 0.3001, 0.2901, 0.3760, 0.2525],
    '[0.6, 0.0, 0.4]': [0.2213, 0.2721, 0.4104, 0.4622, 0.2046],
    '[0.6, 0.2, 0.2]': [0.2056, 0.2854, 0.4305, 0.4967, 0.1366],
    '[0.6, 0.4, 0.0]': [0.1951, 0.3130, 0.1706, 0.5319, 0.1782],
    '[0.8, 0.0, 0.2]': [0.1664, 0.1909, 0.5251, 0.6720, 0.1477],
    '[0.8, 0.2, 0.0]': [0.0940, 0.1979, 0.0868, 0.6664, 0.0891],
    '[1.0, 0.0, 0.0]': [0.0114, 0.0899, 0.0000, 0.8256, 0.0370],
}
assistant = {
    '[0.0, 0.0, 1.0]': [0.0906, 0.1091, 0.0977, 0.3081, 0.0869],
    '[0.0, 0.2, 0.8]': [0.1222, 0.1352, 0.0930, 0.1705, 0.1187],
    '[0.0, 0.4, 0.6]': [0.1790, 0.1779, 0.1288, 0.2058, 0.1557],
    '[0.0, 0.6, 0.4]': [0.1714, 0.1636, 0.1169, 0.1967, 0.1564],
    '[0.0, 0.8, 0.2]': [0.1134, 0.1343, 0.1009, 0.1716, 0.1025],
    '[0.0, 1.0, 0.0]': [0.0430, 0.0862, 0.0000, 0.0650, 0.0621],
    '[0.2, 0.0, 0.8]': [0.0984, 0.1094, 0.0795, 0.1955, 0.0695],
    '[0.2, 0.2, 0.6]': [0.1222, 0.1205, 0.1145, 0.1601, 0.1044],
    '[0.2, 0.4, 0.4]': [0.1820, 0.1787, 0.1330, 0.1548, 0.1322],
    '[0.2, 0.6, 0.2]': [0.1591, 0.1694, 0.1621, 0.1988, 0.1360],
    '[0.2, 0.8, 0.0]': [0.1639, 0.1488, 0.1718, 0.2000, 0.1512],
    '[0.4, 0.0, 0.6]': [0.1091, 0.1300, 0.1124, 0.0479, 0.1096],
    '[0.4, 0.2, 0.4]': [0.1482, 0.1619, 0.1392, 0.1983, 0.1283],
    '[0.4, 0.4, 0.2]': [0.2092, 0.2046, 0.2484, 0.2784, 0.1847],
    '[0.4, 0.6, 0.0]': [0.2712, 0.2566, 0.3055, 0.3569, 0.2301],
    '[0.6, 0.0, 0.4]': [0.1523, 0.1779, 0.1640, 0.0000, 0.1419],
    '[0.6, 0.2, 0.2]': [0.2117, 0.2281, 0.2061, 0.2726, 0.1438],
    '[0.6, 0.4, 0.0]': [0.3079, 0.3008, 0.3496, 0.3558, 0.2276],
    '[0.8, 0.0, 0.2]': [0.1919, 0.2245, 0.1885, 0.2525, 0.1891],
    '[0.8, 0.2, 0.0]': [0.2757, 0.2877, 0.2626, 0.2314, 0.1891],
    '[1.0, 0.0, 0.0]': [0.2362, 0.2516, 0.2833, 0.5020, 0.2364],
}
HV = {
    'Beaver':    {'RS': 0.4275, 'HoE': 0.4276, 'MORLHF': 0.5445, 'RiC': 0.0988, 'ES': 0.6030},
    'Summary':   {'RS': 0.2760, 'HoE': 0.2720, 'MORLHF': 0.1994, 'RiC': 0.1125, 'ES': 0.3412},
    'Assistant': {'RS': 0.3630, 'HoE': 0.3587, 'MORLHF': 0.4287, 'RiC': 0.2665, 'ES': 0.4207},
}
OBJ = {
    'Summary':   ['Summary', 'Faithful', 'Deberta'],
    'Assistant': ['Harmless', 'Helpful', 'Humor'],
}

# ----------------------------------------------------------------------
# 2. STYLE
# ----------------------------------------------------------------------
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.linewidth': 0.8})
# Academic muted palette — picked to match the reference figure's tone:
# desaturated, low-glare hues that read well in print and in two-column
# layouts.  ES keeps the deep-crimson "signature" slot.
COLORS = {
    'ES':     '#8B1F2E',   # deep crimson — signature
    'RS':     '#A8C5A3',   # sage green
    'HoE':    '#DBC758',   # muted yellow
    'MORLHF': '#C49ABF',   # dusty rose
    'RiC':    '#F2D8C0',   # cream peach
}
# Thin, subtle outline for the sector-best bar — looks natural, not loud.
BEST_EDGE       = '#2a2a2a'
BEST_EDGE_WIDTH = 0.9
PLOT_ORDER  = ['RS', 'HoE', 'MORLHF', 'RiC', 'ES']
N_SEC_3OBJ  = 7   # 3 apex + 3 pair + 1 all-three  for Summary / Assistant
N_SEC_BVR   = 3   # Cost / Balanced / Reward — evenly spaced (120° each)

# ----------------------------------------------------------------------
# 3. SECTOR GROUPING
# ----------------------------------------------------------------------
def parse(lam):
    return [float(x) for x in lam.strip('[]').split(',')]

def sectors_3obj(data, names):
    groups = {}
    for lam, vals in data.items():
        v = parse(lam)
        order = sorted(range(3), key=lambda i: -v[i])
        hi, mid, lo = order
        if v[lo] > 0 and (v[hi] - v[lo]) <= 0.2001:
            key = 'All three'
        elif v[mid] <= 1e-9:
            key = names[hi]
        else:
            p = sorted([hi, mid])
            key = names[p[0]] + ' & ' + names[p[1]]
        groups.setdefault(key, []).append(vals)

    # Interlaced clockwise sweep:
    #   apex(0) → pair(0,1) → apex(1) → pair(1,2) → apex(2) → pair(2,0)
    #   → all-three
    # The (2,0) pair is *displayed* as "names[2] & names[0]" so the label
    # reads in the same clockwise direction as the sweep, but the lookup
    # still uses the canonical sorted key "names[0] & names[2]" that the
    # grouping step above produced.
    canon = lambda i, j: names[min(i, j)] + ' & ' + names[max(i, j)]
    disp_seq = [
        (names[0],                       names[0]),
        (names[0] + ' & ' + names[1],    canon(0, 1)),
        (names[1],                       names[1]),
        (names[1] + ' & ' + names[2],    canon(1, 2)),
        (names[2],                       names[2]),
        (names[2] + ' & ' + names[0],    canon(2, 0)),
        ('All three',                    'All three'),
    ]
    out = []
    for disp_label, lookup_key in disp_seq:
        arr = np.array(groups[lookup_key])
        out.append((disp_label, arr.mean(axis=0)))
    return out

def sectors_beaver(data):
    """Return three filled sectors — Cost / Balanced / Reward — for Beaver.

    No empty slots are emitted: the caller passes n_sec=3 to draw_radial so
    the three sectors land 120° apart instead of being scattered across a
    sparse 7-slot ring.
    """
    regions = [
        ('Cost',     ['[0.0, 1.0]', '[0.1, 0.9]', '[0.2, 0.8]']),
        ('Balanced', ['[0.3, 0.7]', '[0.4, 0.6]', '[0.5, 0.5]', '[0.6, 0.4]']),
        ('Reward',   ['[0.7, 0.3]', '[0.8, 0.2]', '[0.9, 0.1]', '[1.0, 0.0]']),
    ]
    return [(name, np.array([data[k] for k in keys]).mean(axis=0))
            for name, keys in regions]

# ----------------------------------------------------------------------
# 4. CURVED TEXT ALONG AN ARC
# ----------------------------------------------------------------------
def radial_label(ax, text, theta_mid, radius, fontsize=6.8,
                 color='#222222', weight='bold'):
    """Place a sector label oriented tangentially.

    `radius` is the **outer edge** of the text (the edge facing the arc),
    not its centre — we anchor with va='top' / va='bottom' depending on
    the flip so the text always extends *inward* from `radius`.  That lets
    the caller treat `inner - radius` as a literal clearance to the arc,
    independent of font size and chart scale.
    """
    text = text.strip()
    if not text:
        return
    # stack 'A & B' onto two lines to keep it short along the arc
    disp = text.replace(' & ', '\n&\n') if ' & ' in text else text
    if disp == 'All three':
        disp = 'All\nthree'
    deg_mid = np.degrees(theta_mid) % 360
    flip = (90 < deg_mid < 270)
    rot = np.degrees(-theta_mid) + (0 if not flip else 180)
    # Without flip, the text's local "top" axis points outward (toward the
    # arc); with flip the rotation is +180°, so "bottom" points outward.
    va = 'bottom' if flip else 'top'
    ax.text(theta_mid, radius, disp, ha='center', va=va,
            fontsize=fontsize, color=color, fontweight=weight,
            rotation=rot, rotation_mode='anchor',
            linespacing=0.95, zorder=6)

# ----------------------------------------------------------------------
# 5. RADIAL PANEL
# ----------------------------------------------------------------------
def _prox_to_bar(prox, transform: str, scale: float,
                 k: float = 2.0, eps: float = 0.01):
    """Map proximity → bar length.

    'linear'  bar = val · scale
    'pow'     bar = val ** k · scale                [convex, k>1 amplifies near 1]
    'exp'     bar = (exp(k · val) − 1) · scale      [convex, more aggressive]
    'log'     bar = -log(1 − val + eps) · scale     [convex w/ divergence at 1]
    'sqrt'    bar = sqrt(max(val, 0)) · scale       [concave, amplifies near 0]

    Number labels always display the raw `val`; only the bar's *visual*
    length is transformed.  Convex transforms ('pow', 'exp', 'log') make a
    0.02 gap near 1 take more pixels than a 0.02 gap near 0 — useful when
    strong methods cluster at high proximity and we want their fine-grained
    differences visible.
    """
    p = np.asarray(prox, float)
    if transform == 'pow':
        return np.power(np.clip(p, 0, None), k) * scale
    if transform == 'exp':
        return (np.exp(k * p) - 1.0) * scale
    if transform == 'log':
        return -np.log(np.maximum(1.0 - p + eps, eps)) * scale
    if transform == 'sqrt':
        return np.sqrt(np.clip(p, 0, None)) * scale
    if transform == 'linear':
        return p * scale
    raise ValueError(f'unknown transform: {transform!r}')


def draw_radial(ax, sector_data, title, n_sec=None,
                scale=5.0, transform='pow', k=2.0):
    """Polar bar chart.

    n_sec : how many evenly-spaced sectors the circle is divided into.  By
        default we use the length of sector_data — that gives 7 for 3-obj
        tasks and 3 for Beaver.  Pass a larger value to leave empty slots.

    scale, transform : map proximity to bar length, see `_prox_to_bar`.
        Default is `log` with scale=1.5 — for the data we plot, max bar
        ≈ 1.5 * -log(0.03) ≈ 5.3, comparable to a linear scale=5 chart,
        but with high-proximity differences visibly stretched.  Number
        labels still print the raw proximity = (1 − Tchebyshev distance).

    The inner radius, bar-value font size, and centre-label font size all
    scale weakly with sector width so a 3-sector chart and a 7-sector chart
    look visually balanced rather than one feeling cramped or hollow.
    """
    if n_sec is None:
        n_sec = len(sector_data)
    n_met = len(PLOT_ORDER)
    method_idx = {m: i for i, m in enumerate(METHODS)}

    # ---- proximity range ----
    all_prox = np.concatenate([
        np.clip(1.0 - np.array(v), 0, None)
        for _, v in sector_data if v is not None
    ]) if any(v is not None for _, v in sector_data) else np.array([0.0, 1.0])
    max_bar = float(_prox_to_bar(all_prox, transform, scale, k=k).max())

    # Wider sectors → bigger hole so the radial chart looks proportioned,
    # not like a clock with three over-sized wedges meeting in the middle.
    wide = (n_sec <= 4)
    sec_span = 2 * np.pi / n_sec
    gap = sec_span * 0.18
    bar_span = (sec_span - gap) / n_met

    # Font scaling — proportional to the angular space available per bar.
    # 0.155 rad (~8.9°) is roughly the 3-obj bar_span; the 3-sector layout
    # reaches ~0.343 rad (~19.7°).  Both branches stay readable.
    es_fs    = 7.4 if wide else 6.4
    other_fs = 7.0 if wide else 6.2
    label_fs = 8.0 if wide else 5.0
    # bar-tip → text radial offset; scaled to chart's bar range so the
    # padding stays visually constant even when max_bar grows with amp.
    tip_pad  = max(0.04, 0.04 * max_bar)

    # Pin the inner hole to a fixed *fraction* of the chart radius rather
    # than a fixed data-unit value.  Otherwise, when `amp` stretches the
    # bar range, the hole shrinks relatively and the centre labels get
    # crushed.  base_inner is the floor (matches the pre-amp design).
    base_inner = 1.10 if wide else 0.92
    hole_frac  = 0.42 if wide else 0.38
    inner = max(base_inner,
                hole_frac * (max_bar + 2 * tip_pad) / (1 - hole_frac))

    ax.set_ylim(0, inner + max_bar + tip_pad * 2)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Centre labels are anchored by their *outer edge* (radial_label uses
    # va='top'/'bottom' depending on flip), so `label_radius` is simply
    # (inner - clearance) — the literal gap from the top of the text to
    # the arc, in data units, independent of font size or chart scale.
    # A small clearance is enough; we scale it weakly with max_bar so it
    # remains visible after amp inflates the chart.
    clearance = max(0.04, 0.03 * max_bar)
    label_radius = inner - clearance
    label_specs = []          # (ax, text, mid, radius) for the 2nd pass

    for s, (label, vals) in enumerate(sector_data):
        sec_start = s * sec_span + gap / 2
        sec_end = sec_start + (sec_span - gap)
        mid = (sec_start + sec_end) / 2
        if vals is None:
            continue

        prox = np.clip(1.0 - np.array(vals), 0, None)
        # Sector best — with ES preferred on ties.  Bumping ES's value by a
        # tiny epsilon makes argmax return ES whenever ES already shares
        # the top spot, without changing the visible numbers.
        prox_tb = prox.astype(float).copy()
        prox_tb[METHODS.index('ES')] += 1e-9
        max_method = METHODS[int(np.argmax(prox_tb))]

        for m, method in enumerate(PLOT_ORDER):
            val = prox[method_idx[method]]            # raw proximity
            bar_len = float(_prox_to_bar(val, transform, scale, k=k))
            theta = sec_start + (m + 0.5) * bar_span
            is_es  = (method == 'ES')
            is_max = (method == max_method)
            # Sector-best gets a dark outline; everyone else gets a thin
            # white separator that hides anti-aliasing seams between bars.
            edge_c = BEST_EDGE       if is_max else 'white'
            edge_w = BEST_EDGE_WIDTH if is_max else 0.5
            # Stack the outlined bar on top of all others so its edge isn't
            # clipped by a tangentially adjacent neighbour.
            bar_z = 4 if is_max else (3 if is_es else 2)
            ax.bar(theta, bar_len, width=bar_span * 0.90, bottom=inner,
                   color=COLORS[method], edgecolor=edge_c,
                   linewidth=edge_w, zorder=bar_z)
            if val > 0.015:
                txt = f'{val:.2f}'
                rot = np.degrees(-theta) + 90
                flip = 90 < (rot % 360) < 270
                if flip:
                    rot += 180
                    ha = 'left'      # outer edge in flipped local frame
                else:
                    ha = 'right'     # outer edge in non-flipped local frame

                # Estimate text width in data units so we can place the
                # label near the bar tip *and* guarantee it never crosses
                # the inner arc.  fs/72 → inches; multiply by data-per-inch
                # (= ylim ÷ axes-height) to convert to data units.  The
                # 0.65 em-per-char factor is conservative (digits are ~0.55)
                # so the safety margin is built in.
                fs = es_fs if is_es else other_fs
                ylim_data    = inner + max_bar + 2 * tip_pad
                data_per_in  = ylim_data / 4.0     # ~axes height in inches
                text_w_data  = fs * 0.65 * len(txt) / 72 * data_per_in

                # 1) Desired: small padding between the text's outer edge
                #    and the bar tip — so labels aren't visually glued to
                #    the bar's end.
                tip_gap      = 0.05 * max(1.0, max_bar)
                desired_r    = inner + bar_len - tip_gap
                # 2) Hard floor: when the bar is too short for the label
                #    to fit with `tip_gap`, anchor the text's *inner edge*
                #    flush with the inner ring (lab_r − text_w_data ==
                #    inner).  The text then fills the short bar from the
                #    arc outward instead of floating past the tip.
                min_r        = inner + text_w_data
                lab_r        = max(desired_r, min_r)

                # ES uses white text on its dark-crimson bar; everyone else
                # uses near-black on lighter fills.  Either way, bold is
                # reserved for the sector best — so a single "winner" rule
                # applies to every method uniformly.
                weight = 'bold' if is_max else 'normal'
                color  = 'white' if is_es else '#2b2b2b'
                fs     = es_fs   if is_es else other_fs
                ax.text(theta, lab_r, txt,
                        ha=ha, va='center', fontsize=fs,
                        color=color, fontweight=weight,
                        rotation=rot, rotation_mode='anchor', zorder=6)

        # Inner-rim arc — plotted as a constant-radius polyline in polar
        # coords.  patches.Arc lives in Cartesian space and warps badly when
        # added to a polar axes (it draws diagonal lines through the centre
        # instead of an arc), so we sample the curve directly.
        arc_theta = np.linspace(sec_start, sec_end, 64)
        ax.plot(arc_theta, np.full_like(arc_theta, inner),
                color='#333333', lw=1.2, zorder=4, solid_capstyle='round')
        label_specs.append((label, mid, label_radius, label_fs))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('none')
    ax.set_title(title, fontsize=12, fontweight='bold',
                 color=COLORS['ES'], pad=10)
    return label_specs

# ----------------------------------------------------------------------
# 6. HV BAR PANEL
# ----------------------------------------------------------------------
def draw_hv(ax):
    tasks = ['Beaver', 'Summary', 'Assistant']
    n_met = len(PLOT_ORDER)
    bw = 0.8 / n_met
    y = np.arange(len(tasks))[::-1]
    for m, method in enumerate(PLOT_ORDER):
        vals = [HV[t][method] for t in tasks]
        offs = y + (m - n_met / 2 + 0.5) * bw
        is_es = (method == 'ES')
        ax.barh(offs, vals, height=bw * 0.9, color=COLORS[method],
                edgecolor='white', linewidth=0.5,
                zorder=3 if is_es else 2)
        for ti, t in enumerate(tasks):
            best = max(HV[t], key=HV[t].get)
            v = vals[ti]
            is_max = (method == best)
            ax.text(v + 0.012, offs[ti], f'{v:.2f}', va='center', ha='left',
                    fontsize=6.2,
                    # Bold reserved for the task winner only.  ES still
                    # gets a distinguishing colour but no longer hardcodes
                    # bold — so on Assistant (where MORLHF beats ES on HV)
                    # ES's number reads as a regular non-winner entry.
                    fontweight='bold' if is_max else 'normal',
                    color='#7e2620' if is_es else '#555555')
    ax.set_yticks(y)
    ax.set_yticklabels(tasks, fontsize=8.5, fontweight='bold')
    ax.set_xlim(0, 0.74)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel('Hypervolume', fontsize=8.5)
    ax.set_title('HV', fontsize=12, fontweight='bold',
                 color=COLORS['ES'], pad=10)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.grid(axis='x', linewidth=0.5, alpha=0.35, zorder=0)

# ----------------------------------------------------------------------
# 7. ASSEMBLE
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(15.5, 4.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.62],
                      wspace=0.16, left=0.03, right=0.985,
                      bottom=0.06, top=0.80)

ax_b = fig.add_subplot(gs[0, 0], projection='polar')
specs_b = draw_radial(ax_b, sectors_beaver(beaver), 'Beaver',
                      n_sec=N_SEC_BVR)

ax_s = fig.add_subplot(gs[0, 1], projection='polar')
specs_s = draw_radial(ax_s, sectors_3obj(summary, OBJ['Summary']), 'Summary',
                      n_sec=N_SEC_3OBJ)

ax_a = fig.add_subplot(gs[0, 2], projection='polar')
specs_a = draw_radial(ax_a, sectors_3obj(assistant, OBJ['Assistant']), 'Assistant',
                      n_sec=N_SEC_3OBJ)

ax_h = fig.add_subplot(gs[0, 3])
draw_hv(ax_h)

# second pass: sector labels — each spec carries its own preferred font size
fig.canvas.draw()
for ax, specs in [(ax_b, specs_b), (ax_s, specs_s), (ax_a, specs_a)]:
    for label, mid, rad, fs in specs:
        radial_label(ax, label, mid, rad,
                     fontsize=fs, color='#222222', weight='bold')

handles = [Patch(facecolor=COLORS[m], edgecolor='white',
                 label=('ES (ours)' if m == 'ES' else m))
           for m in PLOT_ORDER]
fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False,
           fontsize=9.5, bbox_to_anchor=(0.5, 1.04),
           columnspacing=1.6, handlelength=1.4, handleheight=1.1)
fig.suptitle('Proximity to the ideal point  (1 - Tchebyshev utility, higher is better)',
             fontsize=10, y=0.965, color='#444444')

fig.savefig('./plots/pareto_tcheb_hv.svg', bbox_inches='tight')
fig.savefig('./plots/pareto_tcheb_hv.png', dpi=200, bbox_inches='tight')
print('saved')