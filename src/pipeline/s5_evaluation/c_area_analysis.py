"""
Scripts for analysis of reflectance anomaly pixel area size
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from pipeline.config import REFL_DIR_EVAL, GROUND_TRUTH_DIR


_COORD_TOL = 0.6


def plot_centroid_area_analysis() -> None:
    """
    Two-panel figure:
      (a) Distribution of selected B2 / B4 centroid areas for TP vs FP.
      (b) Precision / recall / F1 as a function of minimum area threshold.

    Parameters
    ----------
    det_results     : detection_results DataFrame (output of run_metrics).
                      Required columns: image_id, candidate_idx, gt_matched,
                      c_b2_col, c_b2_row, c_b4_col, c_b4_row
    all_reflectance : all_reflectance DataFrame.
                      Required columns: image_id, candidate_idx, band, col, row, area
    """
    det_results = pd.read_csv(Path(GROUND_TRUTH_DIR) / 'detection_results.csv')
    all_reflectance = _load_all_reflectance()

    matched = _match_centroid_areas()
    _plot_area_distribution(matched)
    _plot_filter_simulation(matched)


# helpers

def _load_all_reflectance() -> pd.DataFrame:
    """
    Concatenates all *_all_reflectance.csv files in refl_dir into one DataFrame.
    """
    files = sorted(Path(REFL_DIR_EVAL).glob("*_all_reflectance.csv"))
    if not files:
        raise FileNotFoundError(f"No *_all_reflectance.csv files found in {REFL_DIR_EVAL}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def _normalise_image_id(s: pd.Series) -> pd.Series:
    """Strip T prefix from tile portion of image_id (e.g. 20250429_T11SPT → 20250429_11SPT)."""
    return s.str.replace(r'_T([A-Z0-9]+)$', r'_\1', regex=True)


def _match_centroid_areas() -> pd.DataFrame:
    """
    For each confirmed detection, look up the pixel area of the B2 and B4
    centroid that was selected for displacement calculation.

    Matching key: image_id + candidate_idx + band + (col, row) within _COORD_TOL.

    Returns det_results with two new columns added:
        b2_area  : area (px) of the matched B2 centroid, NaN if unmatched
        b4_area  : area (px) of the matched B4 centroid, NaN if unmatched
    """
    det_results = pd.read_csv(Path(GROUND_TRUTH_DIR) / 'detection_results.csv')
    all_reflectance = _load_all_reflectance()

    refl = all_reflectance.copy()
    refl['image_id'] = _normalise_image_id(refl['image_id'].astype(str))

    det = det_results.copy()
    det['image_id'] = _normalise_image_id(det['image_id'].astype(str))

    refl_b2 = refl[refl['band'] == 'B2'][['image_id', 'candidate_idx', 'col', 'row', 'area']]
    refl_b4 = refl[refl['band'] == 'B4'][['image_id', 'candidate_idx', 'col', 'row', 'area']]

    det['b2_area'] = det.apply(
        lambda r: _lookup_area(r['image_id'], r['candidate_idx'],
                               r['c_b2_col'], r['c_b2_row'], refl_b2),
        axis=1,
    )
    det['b4_area'] = det.apply(
        lambda r: _lookup_area(r['image_id'], r['candidate_idx'],
                               r['c_b4_col'], r['c_b4_row'], refl_b4),
        axis=1,
    )
    return det


def _lookup_area(image_id: str, candidate_idx: int,
                 col: float, row: float,
                 refl_band: pd.DataFrame) -> float:
    """
    Return the area of the reflectance centroid closest to (col, row)
    for a given image_id + candidate_idx, within _COORD_TOL pixels.
    Returns NaN if no match found.
    """
    subset = refl_band[
        (refl_band['image_id'] == image_id) &
        (refl_band['candidate_idx'] == candidate_idx)
    ]
    if subset.empty:
        return np.nan
    dist = np.sqrt((subset['col'] - col) ** 2 + (subset['row'] - row) ** 2)
    idx_min = dist.idxmin()
    if dist[idx_min] <= _COORD_TOL:
        return float(subset.loc[idx_min, 'area'])
    return np.nan


#plots

_C_TP = '#2E6FA3'
_C_FP = '#C44B30'


def _plot_area_distribution(matched: pd.DataFrame) -> None:
    """
    Panel (a): stacked bar showing fraction of TP vs FP detections
    whose selected B2 / B4 centroid has area == 1px vs > 1px.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    fig.suptitle('(a) Selected centroid area: TP vs FP', fontsize=10, fontweight='bold')

    for ax, band, col in zip(axes, ['B2', 'B4'], ['b2_area', 'b4_area']):
        sub = matched.dropna(subset=[col])
        tp  = sub[sub['gt_matched'] == True][col]
        fp  = sub[sub['gt_matched'] == False][col]

        max_area = int(min(sub[col].max(), 20))   # cap display at 20px
        bins = np.arange(0.5, max_area + 1.5, 1)

        ax.hist(tp, bins=bins, color=_C_TP, alpha=0.65, label='TP', density=True)
        ax.hist(fp, bins=bins, color=_C_FP, alpha=0.65, label='FP', density=True)
        ax.axvline(1.5, color='#333', lw=1.0, ls='--', label='1px threshold')
        ax.set_xlabel(f'{band} centroid area (px)')
        ax.set_ylabel('Density')
        ax.set_title(f'{band}: n_TP={len(tp)}, n_FP={len(fp)}')
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, ls='--', alpha=0.5)

        # print fraction of 1px centroids per class
        tp_1px = (tp == 1).mean() * 100
        fp_1px = (fp == 1).mean() * 100
        print(f'{band}  →  TP with 1px centroid: {tp_1px:.1f}%  |  '
              f'FP with 1px centroid: {fp_1px:.1f}%')

    plt.tight_layout()
    plt.show()

