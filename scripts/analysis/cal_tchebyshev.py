"""Tchebyshev (weighted Chebyshev) scalarisation utility.

For each (task, method) we report the Tchebyshev distance to the ideal
    T(w, f) = max_j  w_j · (z*_j - f_norm_j)
under a preference grid w on the unit simplex.  **Lower is better** — it is
the worst-coordinate weighted gap from the ideal point.

Normalisation
-------------
Identical to cal_linear.py: per objective, pool every method's individuals
and compute (lo, hi); then
    f_norm = (f - lo) / (hi - lo)
so every objective lies in [0, 1].  In this normalised space the ideal
point is fixed as z* = (1, …, 1), making the formula above directly
applicable without separately tracking z*.

Method semantics
----------------
* Non-ES methods (RS / HoE / MORLHF / RiC / MOD): the i-th individual was
  trained for the i-th preference (canonical simplex order).  Distance is
      d_i = max_j  w_ij · (1 - f_norm_ij)
* ES: a population, not a per-preference checkpoint.  For each preference w
  we pick the individual that minimises the Tchebyshev distance,
      d(w) = min_i  max_j  w_j · (1 - f_norm_ij)

Outputs
-------
Per task: pretty-printed table with one row per preference and one column
per method, plus the per-method mean distance and #wins (argmin) counts.
CSVs are written to plots/utility_tchebyshev_<task>.csv.
"""
import os
from itertools import product

import numpy as np

from _mo_data import (
    asst_es, asst_rs, asst_hoe, asst_morl, asst_ric, asst_mod,
    summ_es, summ_rs, summ_hoe, summ_morl, summ_ric, summ_mod,
    bvr_es,  bvr_rs,  bvr_hoe,  bvr_morl,  bvr_ric,  bvr_mod,
)


# ---------------------------------------------------------------------------
# Preference grid (matches nsgaii_utils.get_simplex_samples)
# ---------------------------------------------------------------------------
def simplex_grid(n_objectives: int, step: float) -> np.ndarray:
    n_steps = round(1.0 / step)
    vals    = [round(i * step, 8) for i in range(n_steps + 1)]
    pts     = [list(c) for c in product(vals, repeat=n_objectives)
               if abs(sum(c) - 1.0) < 1e-6]
    return np.array(pts)


# ---------------------------------------------------------------------------
# Normalisation (shared lo/hi across all methods in a task)
# ---------------------------------------------------------------------------
def minmax_bounds(arrays):
    stacked = np.vstack(arrays)
    return stacked.min(axis=0), stacked.max(axis=0)


def normalise(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.where(hi > lo, hi - lo, 1.0)
    return (arr - lo) / span


# ---------------------------------------------------------------------------
# Tchebyshev distance
# ---------------------------------------------------------------------------
def tchebyshev_distance(arr: np.ndarray, prefs: np.ndarray,
                        is_es: bool) -> np.ndarray:
    """Return per-preference Tchebyshev distance to the ideal, shape (P,).

    Inputs are already min-max normalised to [0, 1] so z* = 1 for every
    objective and the gap is (1 - f_norm).

    is_es : True  → d(w) = min_i  max_j  w_j · (1 - f_norm_ij)
            False → d_i  = max_j  w_ij · (1 - f_norm_ij)        (row i ↔ pref i)
    """
    gaps = 1.0 - arr                              # (N, k)

    if is_es:
        # broadcast prefs (P,1,k) * gaps (1,N,k) → (P,N,k), max over k → (P,N),
        # then min over N → (P,)
        weighted = prefs[:, None, :] * gaps[None, :, :]
        return weighted.max(axis=2).min(axis=1)

    if len(arr) != len(prefs):
        raise ValueError(
            f'non-ES method has {len(arr)} rows but preference grid has '
            f'{len(prefs)} entries — row/preference order cannot align.')
    weighted = prefs * gaps                       # (P, k), per-row pairing
    return weighted.max(axis=1)                   # (P,)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
TASKS = [
    ('assistant', 3, 0.2, [
        ('ES',     asst_es),
        ('RS',     asst_rs),
        ('HoE',    asst_hoe),
        ('MORLHF', asst_morl),
        ('RiC',    asst_ric),
        ('MOD',    asst_mod),
    ]),
    ('summary',   3, 0.2, [
        ('ES',     summ_es),
        ('RS',     summ_rs),
        ('HoE',    summ_hoe),
        ('MORLHF', summ_morl),
        ('RiC',    summ_ric),
        ('MOD',    summ_mod),
    ]),
    ('beaver',    2, 0.1, [
        ('ES',     bvr_es),
        ('RS',     bvr_rs),
        ('HoE',    bvr_hoe),
        ('MORLHF', bvr_morl),
        ('RiC',    bvr_ric),
        ('MOD',    bvr_mod),
    ]),
]


def _fmt_pref(w):
    return '(' + ', '.join(f'{v:.1f}' for v in w) + ')'


def run(save_csv: bool = True):
    if save_csv:
        os.makedirs('plots', exist_ok=True)

    for task, n_obj, step, methods in TASKS:
        prefs = simplex_grid(n_obj, step)

        # ---- shared min-max bounds across every method's pool ----
        lo, hi = minmax_bounds([arr for _, arr in methods])
        normed = [(name, normalise(arr, lo, hi)) for name, arr in methods]

        results = {}
        for name, arr in normed:
            results[name] = tchebyshev_distance(arr, prefs,
                                                is_es=(name == 'ES'))

        # ---- pretty table ----
        print(f'\n=== Tchebyshev distance — {task} '
              f'(n_obj={n_obj}, step={step}, P={len(prefs)}, lower is better) ===')
        print(f'  lo = {np.array2string(lo, precision=4)}'
              f'   hi = {np.array2string(hi, precision=4)}')
        header = f'{"w":>{n_obj * 6 + 2}s}  ' + '  '.join(
            f'{n:>8s}' for n, _ in methods)
        print(header)
        print('-' * len(header))
        for i, w in enumerate(prefs):
            row_vals = '  '.join(f'{results[n][i]:>8.4f}' for n, _ in methods)
            print(f'{_fmt_pref(w):>{n_obj * 6 + 2}s}  {row_vals}')

        # ---- summary statistics ----
        print(f'{"mean":>{n_obj * 6 + 2}s}  ' + '  '.join(
            f'{results[n].mean():>8.4f}' for n, _ in methods))
        winners = np.array([results[n] for n, _ in methods]).argmin(axis=0)
        win_counts = {n: int((winners == i).sum())
                      for i, (n, _) in enumerate(methods)}
        print(f'{"#wins":>{n_obj * 6 + 2}s}  ' + '  '.join(
            f'{win_counts[n]:>8d}' for n, _ in methods))

        # ---- CSV ----
        if save_csv:
            path = f'plots/utility_tchebyshev_{task}.csv'
            with open(path, 'w') as f:
                f.write('preference,' + ','.join(n for n, _ in methods) + '\n')
                for i, w in enumerate(prefs):
                    f.write(_fmt_pref(w).replace(',', ';') + ',' +
                            ','.join(f'{results[n][i]:.6f}'
                                     for n, _ in methods) + '\n')
            print(f'  → saved {path}')


if __name__ == '__main__':
    run()
