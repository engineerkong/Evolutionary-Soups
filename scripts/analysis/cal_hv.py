"""Hypervolume analysis — one HV value per (task, method).

Hypervolume in the **maximisation** convention is the Lebesgue measure of
the dominated region between every solution and a lower-bound reference
point ``ref`` (an "anti-ideal" point).  We pool every method's individuals
for shared min-max normalisation (same recipe as cal_linear.py /
cal_tchebyshev.py), so all objectives live in [0, 1] and ``ref`` is the
origin ``(0, …, 0)`` — every normalised solution dominates it, and the
maximum attainable HV is 1 (the unit cube / square).

Methods are evaluated on their *entire* pool, not a per-preference pick:
HV measures how much of the objective space the whole population covers.

Outputs
-------
Per task: HV per method (one column), the ES vs best-baseline gap, and a
CSV at plots/hypervolume_<task>.csv.
"""
import os

import numpy as np
from pymoo.indicators.hv import HV as _PymooHV

from _mo_data import (
    asst_es, asst_rs, asst_hoe, asst_morl, asst_ric, asst_mod,
    summ_es, summ_rs, summ_hoe, summ_morl, summ_ric, summ_mod,
    bvr_es,  bvr_rs,  bvr_hoe,  bvr_morl,  bvr_ric,  bvr_mod,
)


# ---------------------------------------------------------------------------
# Hypervolume (maximisation; pymoo wants minimisation, so we flip signs)
# ---------------------------------------------------------------------------
def hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    """HV under maximisation.  ``ref`` must be dominated by every point."""
    indicator = _PymooHV(ref_point=-ref)
    return float(indicator(-points))


# ---------------------------------------------------------------------------
# Min-max normalisation (shared lo/hi across every method in a task)
# ---------------------------------------------------------------------------
def minmax_bounds(arrays):
    stacked = np.vstack(arrays)
    return stacked.min(axis=0), stacked.max(axis=0)


def normalise(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.where(hi > lo, hi - lo, 1.0)
    return (arr - lo) / span


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
TASKS = [
    ('assistant', [
        ('ES',     asst_es),
        ('RS',     asst_rs),
        ('HoE',    asst_hoe),
        ('MORLHF', asst_morl),
        ('RiC',    asst_ric),
        ('MOD',    asst_mod),
    ]),
    ('summary',   [
        ('ES',     summ_es),
        ('RS',     summ_rs),
        ('HoE',    summ_hoe),
        ('MORLHF', summ_morl),
        ('RiC',    summ_ric),
        ('MOD',    summ_mod),
    ]),
    ('beaver',    [
        ('ES',     bvr_es),
        ('RS',     bvr_rs),
        ('HoE',    bvr_hoe),
        ('MORLHF', bvr_morl),
        ('RiC',    bvr_ric),
        ('MOD',    bvr_mod),
    ]),
]


def run(save_csv: bool = True):
    if save_csv:
        os.makedirs('plots', exist_ok=True)

    for task, methods in TASKS:
        lo, hi = minmax_bounds([arr for _, arr in methods])
        n_obj  = len(lo)
        ref    = np.zeros(n_obj)   # anti-ideal in normalised space

        results = {}
        for name, arr in methods:
            normed        = normalise(arr, lo, hi)
            results[name] = hypervolume(normed, ref)

        # ---- pretty table ----
        print(f'\n=== Hypervolume — {task} '
              f'(n_obj={n_obj}, normalised, ref={tuple(ref)}, '
              f'max attainable=1.0) ===')
        print(f'  lo = {np.array2string(lo, precision=4)}'
              f'   hi = {np.array2string(hi, precision=4)}')
        print(f'  {"method":>8s}    {"N":>4s}    {"HV":>8s}')
        print('  ' + '-' * 30)
        best_name = max(results, key=results.get)
        for name, arr in methods:
            mark = '  ←' if name == best_name else ''
            print(f'  {name:>8s}    {len(arr):>4d}    '
                  f'{results[name]:>8.4f}{mark}')

        # ---- ES vs best baseline ----
        baselines = {n: v for n, v in results.items() if n != 'ES'}
        best_base = max(baselines, key=baselines.get)
        gap       = results['ES'] - baselines[best_base]
        rel       = gap / baselines[best_base] * 100 if baselines[best_base] else float('inf')
        print(f'  ES − best-baseline ({best_base}): {gap:+.4f}  ({rel:+.1f}%)')

        # ---- CSV ----
        if save_csv:
            path = f'plots/hypervolume_{task}.csv'
            with open(path, 'w') as f:
                f.write('method,N,HV\n')
                for name, arr in methods:
                    f.write(f'{name},{len(arr)},{results[name]:.6f}\n')
            print(f'  → saved {path}')


if __name__ == '__main__':
    run()
