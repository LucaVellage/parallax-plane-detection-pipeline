"""
Diagnostics plots of evaluation results
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde
from pathlib import Path

from pipeline.config import (
    GROUND_TRUTH_DIR,
    REFL_EVAL_DIR,
    CONF_EVAL_DIR,
    DISPLACEMENT_RATIO,
)


BG_LABELS = {
    1: 'Water',
    2: 'Vegetation',
    3: 'Bare land',
    4: 'Urban',
    5: 'Snow',
}
BG_ORDER  = [1, 2, 3, 4, 5]
BG_COLORS = {
    1: '#4878CF',  
    2: '#6ACC65',   
    3: '#C4A35A',  
    4: '#B47CC7', 
    5: '#77BEDB',  
}

FP_LABELS = {
    'FP1' : 'Water reflection',
    'FP2' : 'Seamline',
    'FP3' : 'Vehicle',
    'FP4' : 'Building reflection',
    'FP5' : 'Reflective object',
    'FP6' : 'Airport runway',
    'FP7' : 'Boat',
    'FP8' : 'Indiscernible',
    'FP9' : 'Snow reflection',
    'FP10': 'Reason unclear',
}

FN_LABELS = {
    'FN1': 'Blue anomaly barely visible',
    'FN2': 'Aircraft visible',
    'FN3': 'Parallax barely visible',
    'FN4': 'Red anomaly barely visible',
    'FN5': 'Colour distortion',
    'FN6': 'Green anomaly barely visible',
}

_C_TP  = '#2E6FA3'
_C_FP  = '#C44B30'
_C_FN  = '#E08C2A'
_C_TP_L = '#A8C8E8'
_C_FP_L = '#F0A898'
_C_FN_L = '#F5D4A0'

_C_TP   = '#4878CF'  
_C_FP   = '#C44E52' 
_C_FN   = '#DD8452'  

_COORD_TOL = 0.6

def _set_style():
    plt.rcParams.update({
        'font.family'       : 'DejaVu Sans',
        'font.size'         : 8,
        'axes.titlesize'    : 8,
        'axes.labelsize'    : 8,
        'xtick.labelsize'   : 7.5,
        'ytick.labelsize'   : 7.5,
        'legend.fontsize'   : 7.5,
        'axes.linewidth'    : 0.6,
        'axes.spines.top'   : False,
        'axes.spines.right' : False,
        #'axes.grid'         : True,
        'grid.linewidth'    : 0.4,
        'grid.alpha'        : 0.4,
        'grid.color'        : '#cccccc',
        'legend.frameon'    : False,
        'figure.dpi'        : 300,
        'savefig.dpi'       : 300,
        'savefig.bbox'      : 'tight',
    })

#helper functions

def load_eval_results():
    """
    Loads all evaluation result tables from GROUND_TRUTH_DIR.
    Returns (det_results, gt_results, gt_visual).
    """
    gt_dir = Path(GROUND_TRUTH_DIR)
    det_results = pd.read_csv(gt_dir / 'detection_results.csv')
    gt_results  = pd.read_csv(gt_dir / 'gt_match_results.csv')
    gt_visual   = pd.read_csv(gt_dir / 'ground_truth_visual.csv')
    return det_results, gt_results, gt_visual


def _tp(det):  return det[det['detection_type'] == 'TP']
def _fp(det):  return det[det['detection_type'] == 'FP']
def _tp1(det): return det[det['detection_type'] == 'TP1']
def _fn(gt):   return gt[gt['match_type'] == 'FN']
def _gt_tp(gt):return gt[gt['match_type'] == 'TP']

def _bg_rows(df, bg):
    """Filter to a single background_type, or all non-zero if bg is None."""
    if bg is None:
        return df[df['background_type'].notna() & (df['background_type'] != 0)]
    return df[df['background_type'] == bg]



def _kde(ax, vals, color, light, label, bw='scott'):
    v = np.array(vals.dropna())
    if len(v) < 3:
        return
    xs  = np.linspace(v.min() - (v.max()-v.min())*0.1,
                      v.max() + (v.max()-v.min())*0.1, 400)
    ys  = gaussian_kde(v, bw_method=bw)(xs)
    ax.fill_between(xs, ys, alpha=0.20, color=light)
    ax.plot(xs, ys, color=color, lw=1.6, label=label)
    ax.plot(v, np.zeros_like(v), '|', color=color, alpha=0.35, ms=5)



def _fp_reason_table(det_results):
    """
    Returns a DataFrame: rows = background, cols = FP note codes.
    """
    fp = _fp(det_results).copy()
    fp = fp[fp['background_type'].notna() & (fp['background_type'] != 0)]
    fp['bg_label']   = fp['background_type'].map(BG_LABELS)
    fp['code_label'] = fp['note_code'].map(FP_LABELS).fillna(fp['note_code'].fillna('Unlabelled'))

    tbl = (
        fp.groupby(['bg_label', 'code_label'])
          .size()
          .unstack(fill_value=0)
          .reindex([BG_LABELS[b] for b in BG_ORDER if b in fp['background_type'].unique()])
    )
    # add totals
    tbl['Total'] = tbl.sum(axis=1)
    tbl.loc['Total'] = tbl.sum()
    return tbl


def _fn_reason_table(gt_visual):
    """
    Returns a DataFrame: rows = background, cols = FN note codes.
    FN note codes come from manual rows in ground_truth_visual.
    """
    fn = gt_visual[
        (gt_visual['source'] == 'manual') &
        (gt_visual['is_visible'] == True)
    ].copy()
    fn = fn[fn['background_type'].notna() & (fn['background_type'] != 0)]
    fn['bg_label']   = fn['background_type'].map(BG_LABELS)
    fn['code_label'] = fn['note_code'].map(FN_LABELS).fillna(fn['note_code'].fillna('Unlabelled'))

    tbl = (
        fn.groupby(['bg_label', 'code_label'])
          .size()
          .unstack(fill_value=0)
          .reindex([BG_LABELS[b] for b in BG_ORDER if b in fn['background_type'].unique()])
    )
    tbl['Total'] = tbl.sum(axis=1)
    tbl.loc['Total'] = tbl.sum()
    return tbl


def _rate_table(det_results, gt_results):
    """
    Returns a DataFrame with TP, FP, FN, TP1, precision, recall per background.
    """
    rows = []
    for bg in BG_ORDER:
        d = _bg_rows(det_results, bg)
        g = _bg_rows(gt_results,  bg)
        tp  = (d['detection_type'] == 'TP').sum()
        fp  = (d['detection_type'] == 'FP').sum()
        tp1 = (d['detection_type'] == 'TP1').sum()
        fn  = (g['match_type']     == 'FN').sum()
        prec = tp/(tp+fp)   if (tp+fp)   > 0 else float('nan')
        rec  = tp/(tp+fn)   if (tp+fn)   > 0 else float('nan')
        rows.append({
            'Background' : BG_LABELS[bg],
            'TP' : tp, 'FP' : fp, 'FN' : fn, 'TP1': tp1,
            'Precision'  : round(prec, 3) if not np.isnan(prec) else '-',
            'Recall'     : round(rec,  3) if not np.isnan(rec)  else '-',
            'FP rate'    : f"{fp/(tp+fp)*100:.1f}%" if (tp+fp) > 0 else '-',
            'FN rate'    : f"{fn/(tp+fn)*100:.1f}%" if (tp+fn) > 0 else '-',
        })
    return pd.DataFrame(rows).set_index('Background')


def _print_table(df, title):
    print(f'\n{"─"*60}')
    print(f'  {title}')
    print(f'{"─"*60}')
    print(df.to_string())
    print()


# analysis 

def _adsb_detectable_table(adsb_gt):
    """
    Table: total ADS-B, detectable ADS-B, and detectable fraction
    per background type.
    """
    rows = []
    for bg in BG_ORDER:
        subset    = adsb_gt[adsb_gt['background_type'] == bg]
        n_total   = len(subset)
        n_visible = (subset['is_visible'] == True).sum()
        frac      = n_visible / n_total if n_total > 0 else float('nan')
        rows.append({
            'Background'         : BG_LABELS[bg],
            'Total ADS-B'        : n_total,
            'Detectable ADS-B'   : n_visible,
            'Detectable fraction': round(frac, 3) if not np.isnan(frac) else '-',
        })

    # Overall row
    ag       = adsb_gt[adsb_gt['background_type'] != 0]
    n_total  = len(ag)
    n_vis    = (ag['is_visible'] == True).sum()
    rows.append({
        'Background'         : 'Overall',
        'Total ADS-B'        : n_total,
        'Detectable ADS-B'   : n_vis,
        'Detectable fraction': round(n_vis / n_total, 3) if n_total > 0 else '-',
    })

    return pd.DataFrame(rows).set_index('Background')

# FP analysis

def _plot_fp_rate_by_background(det_results, gt_results):
    """FP rate (FP / (TP+FP)) and absolute FP count per background."""
    _set_style()
    bgs, rates, counts = [], [], []
    for bg in BG_ORDER:
        d   = _bg_rows(det_results, bg)
        tp  = (d['detection_type'] == 'TP').sum()
        fp  = (d['detection_type'] == 'FP').sum()
        if (tp + fp) == 0:
            continue
        bgs.append(BG_LABELS[bg])
        rates.append(fp / (tp + fp) * 100)
        counts.append(fp)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = np.arange(len(bgs))
    bars = ax1.bar(x, rates, color=[BG_COLORS[b] for b in BG_ORDER if BG_LABELS[b] in bgs],
                   alpha=0.80, edgecolor='none', lw=0.6)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1, f'n={c}',
                 ha='center', va='bottom', fontsize=7)
    ax1.set_xticks(x); ax1.set_xticklabels(bgs)
    ax1.set_ylabel('FP rate (%)')
    ax1.set_ylim(0, max(rates) * 1.25 if rates else 1)
    ax1.set_title('False positive rate by background type')
    ax1.grid(True, ls='--', axis='y')
    ax1.set_axisbelow(True)
    fig.tight_layout()
    plt.show()


def _plot_fp_reasons_by_background(det_results):
    """Stacked bar: FP reason codes per background."""
    _set_style()
    fp = _fp(det_results).copy()
    fp = fp[fp['background_type'].notna() & (fp['background_type'] != 0)]
    fp['bg_label']   = fp['background_type'].map(BG_LABELS)
    fp['code_label'] = fp['note_code'].map(FP_LABELS).fillna('Unlabelled')

    pivot = (
        fp.groupby(['bg_label', 'code_label']).size()
          .unstack(fill_value=0)
          .reindex([BG_LABELS[b] for b in BG_ORDER], fill_value=0)
          .dropna(how='all')
    )

    pivot = pivot[pivot.sum().sort_values(ascending=False).index]

    cmap   = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(pivot.columns))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom  = np.zeros(len(pivot))
    for i, col in enumerate(pivot.columns):
        vals = pivot[col].values.astype(float)
        ax.bar(pivot.index, vals, bottom=bottom,
               label=col, color=colors[i], edgecolor='none', lw=0.4)
        # label non-zero segments
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(j, b + v/2, str(int(v)),
                        ha='center', va='center', fontsize=6.5, color='white', fontweight='bold')
        bottom += vals

    ax.set_ylabel('Number of false positives')
    ax.set_title('FP reason codes by background type')
    ax.legend(loc='upper right', frameon=False, fontsize=7,
              bbox_to_anchor=(1.18, 1.0))
    ax.grid(True, ls='--', axis='y')
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.show()


def _plot_fp_reasons_heatmap(det_results):
    """Heatmap: FP reason × background, absolute counts."""
    _set_style()
    tbl = _fp_reason_table(det_results).drop('Total', errors='ignore').drop(columns='Total', errors='ignore')
    if tbl.empty:
        return

    fig, ax = plt.subplots(figsize=(max(7, len(tbl.columns)*1.2), max(4, len(tbl)*0.7)))
    data = tbl.values.astype(float)
    im   = ax.imshow(data, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='Count')

    ax.set_xticks(range(len(tbl.columns)))
    ax.set_xticklabels(tbl.columns, rotation=35, ha='right', fontsize=7.5)
    ax.set_yticks(range(len(tbl.index)))
    ax.set_yticklabels(tbl.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > 0:
                ax.text(j, i, str(int(data[i, j])),
                        ha='center', va='center', fontsize=7,
                        color='white' if data[i, j] > data.max()*0.6 else '#333')
    ax.set_title('FP reason × background (counts)')
    fig.tight_layout()
    plt.show()

def _plot_precision_recall_by_background(det_results, gt_results):
    """Paired grouped bar: precision, recall and F1 per background type."""
    _set_style()
    bgs, precisions, recalls, f1s, n_labels = [], [], [], [], []

    for bg in BG_ORDER:
        d  = _bg_rows(det_results, bg)
        g  = _bg_rows(gt_results,  bg)
        tp = (d['detection_type'] == 'TP').sum()
        fp = (d['detection_type'] == 'FP').sum()
        fn = (g['match_type']     == 'FN').sum()
        if tp + fp + fn == 0:
            continue
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        bgs.append(BG_LABELS[bg])
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        n_labels.append(f'n={tp+fp}')

    x = np.arange(len(bgs))
    w = 0.25  

    fig, ax = plt.subplots(figsize=(9, 4))
    bars_p  = ax.bar(x - w, precisions, w,
                     label='Precision', color='#4C8BB5',
                     alpha=1.0, edgecolor='none')
    bars_r  = ax.bar(x,     recalls,    w,
                     label='Recall',    color='#D77F26',
                     alpha=1.0, edgecolor='none')
    bars_f1 = ax.bar(x + w, f1s,        w,
                     label='F1',        color='#5AA55E',
                     alpha=1.0, edgecolor='none')

    for bars in [bars_p, bars_r, bars_f1]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f'{bar.get_height():.2f}',
                    ha='center', va='bottom', fontsize=7, color='#333')
            

    # n= label below each background group
    for i, label in enumerate(n_labels):
        ax.text(x[i], -0.07, label,
                ha='center', va='top', fontsize=6.5,
                color='#666', transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(bgs)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall and F1 by Background Type')
    ax.legend(frameon=False)
    ax.grid(True, axis='y', ls='--')
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.show()

#FN analysis

def _plot_fn_rate_by_background(det_results, gt_results):
    """FN rate (FN / (TP+FN)) and absolute FN count per background."""
    _set_style()
    bgs, rates, counts = [], [], []
    for bg in BG_ORDER:
        d  = _bg_rows(det_results, bg)
        g  = _bg_rows(gt_results,  bg)
        tp = (d['detection_type'] == 'TP').sum()
        fn = (g['match_type']     == 'FN').sum()
        if (tp + fn) == 0:
            continue
        bgs.append(BG_LABELS[bg])
        rates.append(fn / (tp + fn) * 100)
        counts.append(fn)

    fig, ax = plt.subplots(figsize=(7, 4))
    x    = np.arange(len(bgs))
    bars = ax.bar(x, rates,
                  color=[BG_COLORS[b] for b in BG_ORDER if BG_LABELS[b] in bgs],
                  alpha=1.0, edgecolor='none', lw=0.6)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1, f'n={c}',
                ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(bgs)
    ax.set_ylabel('FN rate (%)')
    ax.set_ylim(0, max(rates)*1.25 if rates else 1)
    ax.set_title('False negative rate by background type')
    ax.grid(True, ls='--', axis='y')
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.show()


def _plot_fn_reasons_by_background(gt_visual):
    """Stacked bar: FN reason codes per background (from manual rows)."""
    _set_style()
    fn = gt_visual[
        (gt_visual['source']     == 'manual') &
        (gt_visual['is_visible'] == True)
    ].copy()
    fn = fn[fn['background_type'].notna() & (fn['background_type'] != 0)]
    fn['bg_label']   = fn['background_type'].map(BG_LABELS)
    fn['code_label'] = fn['note_code'].map(FN_LABELS).fillna('Unlabelled')

    pivot = (
        fn.groupby(['bg_label', 'code_label']).size()
          .unstack(fill_value=0)
          .reindex([BG_LABELS[b] for b in BG_ORDER], fill_value=0)
          .dropna(how='all')
    )
    pivot = pivot[pivot.sum().sort_values(ascending=False).index]

    cmap   = plt.get_cmap('Set2')
    colors = [cmap(i) for i in range(len(pivot.columns))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom  = np.zeros(len(pivot))
    for i, col in enumerate(pivot.columns):
        vals = pivot[col].values.astype(float)
        ax.bar(pivot.index, vals, bottom=bottom,
               label=col, color=colors[i], edgecolor='none', lw=0.4)
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(j, b + v/2, str(int(v)),
                        ha='center', va='center', fontsize=6.5,
                        color='white', fontweight='bold')
        bottom += vals

    ax.set_ylabel('Number of false negatives')
    ax.set_title('FN reason codes by background type')
    ax.legend(loc='upper right', frameon=False, fontsize=7,
              bbox_to_anchor=(1.22, 1.0))
    ax.grid(True, ls='--', axis='y')
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.show()


def _plot_fn_altitude(gt_results):
    """KDE: ADS-B barometric altitude for TP vs FN records."""
    _set_style()
    tp_alt = _gt_tp(gt_results)['baroaltitude'].dropna()
    fn_alt = _fn(gt_results)['baroaltitude'].dropna()
    if tp_alt.empty and fn_alt.empty:
        print('No baroaltitude data available.')
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    _kde(ax, tp_alt, _C_TP, _C_TP_L, f'Detected (n={len(tp_alt)})')
    _kde(ax, fn_alt, _C_FN, _C_FN_L, f'Missed / FN (n={len(fn_alt)})')
    ax.set_xlabel('Barometric altitude (m)')
    ax.set_ylabel('Density')
    ax.set_title('ADS-B altitude: detected vs missed aircraft')
    ax.legend(frameon=False)
    ax.grid(True, ls='--')
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    plt.show()


def _plot_fn_adsb_speed(gt_results):
    """KDE: ADS-B transponder speed for TP vs FN records."""
    _set_style()
    tp_spd = _gt_tp(gt_results)['speed_kmh'].dropna()
    fn_spd = _fn(gt_results)['speed_kmh'].dropna()
    if tp_spd.empty and fn_spd.empty:
        print('No speed data available in gt_results.')
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    _kde(ax, tp_spd, _C_TP, _C_TP_L, f'Detected (n={len(tp_spd)})')
    _kde(ax, fn_spd, _C_FN, _C_FN_L, f'Missed / FN (n={len(fn_spd)})')
    ax.set_xlabel('ADS-B speed (km h⁻¹)')
    ax.set_ylabel('Density')
    ax.set_title('ADS-B speed: detected vs missed aircraft')
    ax.legend(frameon=False)
    ax.grid(True, ls='--')
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    plt.show()


#Displacement/speed analysis

def _plot_speed_dist(det_results):
    """Pipeline-Estimated Speed for True Positives vs. False Positives"""
    _set_style()
    tp = _tp(det_results)
    fp = _fp(det_results)

    tp_speeds = tp['speed_kmh'].dropna().values
    fp_speeds = fp['speed_kmh'].dropna().values

    x_min = 0
    x_max = min(max(tp_speeds.max(), fp_speeds.max()) * 1.05, 4000)
    xs    = np.linspace(x_min, x_max, 500)

    tp_kde = gaussian_kde(tp_speeds, bw_method='scott')(xs)
    fp_kde = gaussian_kde(fp_speeds, bw_method='scott')(xs)

    tp_kde[xs < tp_speeds.min()] = 0
    tp_kde[xs > tp_speeds.max()] = 0
    fp_kde[xs < fp_speeds.min()] = 0
    fp_kde[xs > fp_speeds.max()] = 0

    # TP range — raw values
    tp_raw_min = tp_speeds.min()
    tp_raw_max = tp_speeds.max()

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.fill_between(xs, tp_kde, alpha=0.15, color=_C_TP)
    ax.plot(xs, tp_kde, color=_C_TP, lw=1.4,
            label=f'TP (n={len(tp_speeds)})')

    ax.fill_between(xs, fp_kde, alpha=0.15, color=_C_FP)
    ax.plot(xs, fp_kde, color=_C_FP, lw=1.4,
            label=f'FP (n={len(fp_speeds)})')

    # TP range dashed lines at raw values
    ax.axvline(tp_raw_min, color=_C_TP, lw=0.9, ls='--', alpha=0.8)
    ax.axvline(tp_raw_max, color=_C_TP, lw=0.9, ls='--', alpha=0.8)

    # raw value annotations just above x-axis
    y_annot = ax.get_ylim()[1] * 0.04
    ax.annotate(f'{tp_raw_min:.0f}',
                xy=(tp_raw_min, y_annot),
                xytext=(4, 0), textcoords='offset points',
                ha='left', va='bottom', fontsize=7, color=_C_TP)
    ax.annotate(f'{tp_raw_max:.0f}',
                xy=(tp_raw_max, y_annot),
                xytext=(4, 0), textcoords='offset points',
                ha='left', va='bottom', fontsize=7, color=_C_TP)

    # phantom line for legend entry
    ax.plot([], [], color=_C_TP, lw=0.9, ls='--',
            label=f'TP range ({tp_raw_min:.0f}–{tp_raw_max:.0f} km/h)')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Estimated speed (km/h)')
    ax.set_ylabel('Density')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(200))
    ax.set_title('Speed Estimation: True Positives vs. False Positives')
    ax.legend(frameon=False)
    ax.grid(True, axis='y', ls='--')
    fig.tight_layout()
    plt.show()


def _plot_angle_dist(det_results):
    """KDE: displacement direction for TP vs FP."""
    _set_style()
    tp = _tp(det_results)
    fp = _fp(det_results)

    tp_ang = tp['angle_deg'].dropna().values
    fp_ang = fp['angle_deg'].dropna().values

    x_min = 160
    x_max = 180
    xs    = np.linspace(x_min, x_max, 500)

    tp_kde = gaussian_kde(tp_ang, bw_method='scott')(xs)
    fp_kde = gaussian_kde(fp_ang, bw_method='scott')(xs)

    tp_kde[xs < tp_ang.min()] = 0
    tp_kde[xs > tp_ang.max()] = 0
    fp_kde[xs < fp_ang.min()] = 0
    fp_kde[xs > fp_ang.max()] = 0

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.fill_between(xs, tp_kde, alpha=0.15, color=_C_TP)
    ax.plot(xs, tp_kde, color=_C_TP, lw=1.4,
            label=f'TP (n={len(tp_ang)})')

    ax.fill_between(xs, fp_kde, alpha=0.15, color=_C_FP)
    ax.plot(xs, fp_kde, color=_C_FP, lw=1.4,
            label=f'FP (n={len(fp_ang)})')

    # TP boundary lines only
    ax.axvline(tp_ang.min(), color=_C_TP, lw=0.9, ls='--', alpha=0.8)
    ax.axvline(tp_ang.max(), color=_C_TP, lw=0.9, ls='--', alpha=0.8)

    for val, side in [
        (tp_ang.min(), 'right'),
        (tp_ang.max(), 'left'),
    ]:
        offset = -4 if side == 'right' else 4
        ax.annotate(f'{val:.0f}°',
                    xy=(val, 0), xycoords=('data', 'axes fraction'),
                    xytext=(offset, 6), textcoords='offset points',
                    ha=side, va='bottom', fontsize=7, color=_C_TP)

    ax.plot([], [], color=_C_TP, lw=0.9, ls='--',
            label=f'TP range ({tp_ang.min():.0f}°–{tp_ang.max():.0f}°)')

    ax.set_xlim(160, 180)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Displacement angle (°)')
    ax.set_ylabel('Density')
    ax.set_title('Displacement direction: True Positives vs. False Positives')
    ax.legend(frameon=False)
    ax.grid(True, axis='y', ls='--')
    fig.tight_layout()
    plt.show()

def _plot_residual_dist(det_results):
    """KDE: proportional displacement residual for TP vs FP."""
    _set_style()
    tp = _tp(det_results)
    fp = _fp(det_results)

    tp_res = tp['residual_m'].dropna().values
    fp_res = fp['residual_m'].dropna().values

    x_min = 0
    x_max = max(tp_res.max(), fp_res.max()) * 1.05
    xs    = np.linspace(x_min, x_max, 500)

    tp_kde = gaussian_kde(tp_res, bw_method='scott')(xs)
    fp_kde = gaussian_kde(fp_res, bw_method='scott')(xs)

    tp_kde[xs < tp_res.min()] = 0
    tp_kde[xs > tp_res.max()] = 0
    fp_kde[xs < fp_res.min()] = 0
    fp_kde[xs > fp_res.max()] = 0

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.fill_between(xs, tp_kde, alpha=0.15, color=_C_TP)
    ax.plot(xs, tp_kde, color=_C_TP, lw=1.4,
            label=f'TP (n={len(tp_res)})')

    ax.fill_between(xs, fp_kde, alpha=0.15, color=_C_FP)
    ax.plot(xs, fp_kde, color=_C_FP, lw=1.4,
            label=f'FP (n={len(fp_res)})')

    # TP boundary lines only
    ax.axvline(tp_res.min(), color=_C_TP, lw=0.9, ls='--', alpha=0.8)
    ax.axvline(tp_res.max(), color=_C_TP, lw=0.9, ls='--', alpha=0.8)

    # raw value annotations
    for val, side in [
        (tp_res.min(), 'right'),
        (tp_res.max(), 'left'),
    ]:
        offset = -4 if side == 'right' else 4
        ax.annotate(f'{val:.0f}',
                    xy=(val, 0), xycoords=('data', 'axes fraction'),
                    xytext=(offset, 6), textcoords='offset points',
                    ha=side, va='bottom', fontsize=7, color=_C_TP)

    # phantom line for legend
    ax.plot([], [], color=_C_TP, lw=0.9, ls='--',
            label=f'TP range ({tp_res.min():.0f} - {tp_res.max():.0f} m)')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Displacement residual (m)')
    ax.set_ylabel('Density')
    ax.set_title('Proportional displacement residual: True Positives vs. False Positives')
    ax.legend(frameon=False)
    ax.grid(True, axis='y', ls='--')
    fig.tight_layout()
    plt.show()


def _plot_displacement_speed_combined(det_results):
    """Two-panel figure: displacement residual and speed KDE side by side."""
    _set_style()
    tp = _tp(det_results)
    fp = _fp(det_results)

    tp_res = tp['residual_m'].dropna().values
    fp_res = fp['residual_m'].dropna().values
    tp_spd = tp['speed_kmh'].dropna().values
    fp_spd = fp['speed_kmh'].dropna().values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    xs = np.linspace(0, max(tp_res.max(), fp_res.max()) * 1.05, 500)

    tp_kde = gaussian_kde(tp_res, bw_method='scott')(xs)
    fp_kde = gaussian_kde(fp_res, bw_method='scott')(xs)
    tp_kde[xs < tp_res.min()] = 0
    tp_kde[xs > tp_res.max()] = 0
    fp_kde[xs < fp_res.min()] = 0
    fp_kde[xs > fp_res.max()] = 0

    ax.fill_between(xs, tp_kde, alpha=0.15, color=_C_TP)
    ax.plot(xs, tp_kde, color=_C_TP, lw=1.4, label=f'TP (n={len(tp_res)})')
    ax.fill_between(xs, fp_kde, alpha=0.15, color=_C_FP)
    ax.plot(xs, fp_kde, color=_C_FP, lw=1.4, label=f'FP (n={len(fp_res)})')

    ax.axvline(tp_res.min(), color=_C_TP, lw=0.9, ls='--', alpha=0.8)
    ax.axvline(tp_res.max(), color=_C_TP, lw=0.9, ls='--', alpha=0.8)
    for val, side in [(tp_res.min(), 'right'), (tp_res.max(), 'left')]:
        offset = -4 if side == 'right' else 4
        ax.annotate(f'{val:.0f}',
                    xy=(val, 0), xycoords=('data', 'axes fraction'),
                    xytext=(offset, 6), textcoords='offset points',
                    ha=side, va='bottom', fontsize=7, color=_C_TP)
    ax.plot([], [], color=_C_TP, lw=0.9, ls='--',
            label=f'TP range ({tp_res.min():.0f}–{tp_res.max():.0f} m)')

    ax.set_xlim(0, max(tp_res.max(), fp_res.max()) * 1.05)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Displacement residual (m)')
    ax.set_ylabel('Density')
    ax.set_title('Proportional displacement residual')
    ax.grid(True, axis='y', ls='--')

    ax = axes[1]
    x_max = min(max(tp_spd.max(), fp_spd.max()) * 1.05, 4000)
    xs = np.linspace(0, x_max, 500)

    tp_kde = gaussian_kde(tp_spd, bw_method='scott')(xs)
    fp_kde = gaussian_kde(fp_spd, bw_method='scott')(xs)
    tp_kde[xs < tp_spd.min()] = 0
    tp_kde[xs > tp_spd.max()] = 0
    fp_kde[xs < fp_spd.min()] = 0
    fp_kde[xs > fp_spd.max()] = 0

    ax.fill_between(xs, tp_kde, alpha=0.15, color=_C_TP)
    ax.plot(xs, tp_kde, color=_C_TP, lw=1.4, label=f'TP (n={len(tp_spd)})')
    ax.fill_between(xs, fp_kde, alpha=0.15, color=_C_FP)
    ax.plot(xs, fp_kde, color=_C_FP, lw=1.4, label=f'FP (n={len(fp_spd)})')

    ax.axvline(tp_spd.min(), color=_C_TP, lw=0.9, ls='--', alpha=0.8)
    ax.axvline(tp_spd.max(), color=_C_TP, lw=0.9, ls='--', alpha=0.8)
    for val, side in [(tp_spd.min(), 'right'), (tp_spd.max(), 'left')]:
        offset = -4 if side == 'right' else 4
        ax.annotate(f'{val:.0f}',
                    xy=(val, 0), xycoords=('data', 'axes fraction'),
                    xytext=(offset, 6), textcoords='offset points',
                    ha=side, va='bottom', fontsize=7, color=_C_TP)
    ax.plot([], [], color=_C_TP, lw=0.9, ls='--',
            label=f'TP range ({tp_spd.min():.0f}–{tp_spd.max():.0f} km/h)')

    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Estimated speed (km/h)')
    ax.set_ylabel('Density')
    ax.set_title('Pipeline speed estimate')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(200))
    ax.grid(True, axis='y', ls='--')


    axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].legend(frameon=False, fontsize=7.5)

    fig.tight_layout()
    plt.show()



def _plot_band_scatter(det_results):
    """Scatter: B2→B3 vs B3→B4 displacement, TP and FP coloured by background."""
    _set_style()
    disp_cols = ['D_Band2_3_m', 'D_Band3_4_m']
    if not all(c in det_results.columns for c in disp_cols):
        print('Band scatter skipped — D_Band2_3_m / D_Band3_4_m not in detection_results.')
        return
    tp = _tp(det_results).dropna(subset=disp_cols)
    fp = _fp(det_results).dropna(subset=disp_cols)
    if tp.empty and fp.empty:
        print('No inter-band displacement data available.')
        return

    all_vals = pd.concat([tp['D_Band2_3_m'], fp['D_Band2_3_m']]).dropna()
    x_max    = all_vals.max() * 1.05 if not all_vals.empty else 1000
    xs       = np.linspace(0, x_max, 200)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xs, xs * DISPLACEMENT_RATIO, 'k--', lw=1.1,
            label=f'Expected ratio ({DISPLACEMENT_RATIO})', zorder=1)

    # TP coloured by background
    for bg in BG_ORDER:
        sub = tp[tp['background_type'] == bg]
        if sub.empty:
            continue
        ax.scatter(sub['D_Band2_3_m'], sub['D_Band3_4_m'],
                   c=BG_COLORS[bg], s=28, alpha=0.75,
                   edgecolors='none', lw=0.3, zorder=3,
                   label=f'TP {BG_LABELS[bg]}')

    # FP as triangles, also by background
    for bg in BG_ORDER:
        sub = fp[fp['background_type'] == bg]
        if sub.empty:
            continue
        ax.scatter(sub['D_Band2_3_m'], sub['D_Band3_4_m'],
                   c=BG_COLORS[bg], s=32, alpha=1.0, marker='^',
                   edgecolors='none', lw=0.3, zorder=4,
                   label=f'FP {BG_LABELS[bg]}')

    ax.set_xlabel('B2→B3 displacement (m)')
    ax.set_ylabel('B3→B4 displacement (m)')
    ax.set_title('Inter-band displacements: TP (circles) vs FP (triangles)')
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.grid(True, ls='--')
    fig.tight_layout()
    plt.show()


def _plot_bland_altman(det_results, gt_visual, gt_results):
    """Bland-Altman: pipeline speed vs ADS-B speed for matched TPs."""
    _set_style()

    # Join: ADS-B speed → pipeline speed via matched_callsign + candidate_idx
    gt_adsb = pd.read_csv(Path(GROUND_TRUTH_DIR) / 'ground_truth_adsb.csv')
    adsb = (
        gt_adsb[gt_adsb['in_pipeline'] == True][
            ['image_id', 'matched_callsign', 'speed_kmh']
        ]
        .dropna(subset=['speed_kmh', 'matched_callsign'])
        .rename(columns={'speed_kmh': 'adsb_speed'})
    )
    bridge = (
        gt_visual[gt_visual['matched_callsign'].notna()][
            ['image_id', 'matched_callsign', 'candidate_idx']
        ].drop_duplicates()
    )
    pipeline = (
        _tp(det_results)[['image_id', 'candidate_idx', 'speed_kmh']]
        .rename(columns={'speed_kmh': 'pipeline_speed'})
    )
    paired = (
        adsb.merge(bridge,   on=['image_id', 'matched_callsign'], how='inner')
            .merge(pipeline, on=['image_id', 'candidate_idx'],    how='inner')
    )
    if len(paired) < 3:
        print('Insufficient paired speed data for Bland-Altman.')
        return

    p    = paired['pipeline_speed'].values
    a    = paired['adsb_speed'].values
    mean = (p + a) / 2
    diff = p - a
    bias = np.mean(diff)
    sd   = np.std(diff, ddof=1)
    lo, hi = bias - 1.96*sd, bias + 1.96*sd

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(mean, diff, c=_C_TP, s=32, alpha=0.8, edgecolors='white', lw=0.4)
    xlim = (mean.min()*0.9, mean.max()*1.1)
    ax.axhline(bias, color='#333', lw=1.3, ls='-',  label=f'Bias: {bias:+.1f} km h⁻¹')
    ax.axhline(hi,   color=_C_FP,  lw=1.0, ls='--', label=f'+1.96 SD: {hi:+.1f}')
    ax.axhline(lo,   color=_C_FP,  lw=1.0, ls='--', label=f'−1.96 SD: {lo:+.1f}')
    ax.fill_between(xlim, lo, hi, alpha=0.07, color=_C_FP)
    ax.axhline(0, color='#999', lw=0.8, ls=':')
    ax.set_xlim(xlim)
    ax.set_xlabel('Mean speed, pipeline & ADS-B (km h⁻¹)')
    ax.set_ylabel('Pipeline − ADS-B (km h⁻¹)')
    ax.set_title(f'Speed agreement: Bland-Altman (n={len(paired)})')
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, ls='--')
    fig.tight_layout()
    plt.show()


#pixel size analysis

def _load_all_reflectance():
    files = sorted(Path(REFL_EVAL_DIR).glob('*_all_reflectance.csv'))
    if not files:
        raise FileNotFoundError(f'No *_all_reflectance.csv in {REFL_EVAL_DIR}')
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def _load_all_confirmed():
    files = sorted(Path(CONF_EVAL_DIR).glob('*_all_confirmed.csv'))
    if not files:
        raise FileNotFoundError(f'No *_all_confirmed.csv in {CONF_EVAL_DIR}')
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def _normalise_image_id(s):
    return s.str.replace(r'_T([A-Z0-9]+)$', r'_\1', regex=True)


def _lookup_area(image_id, candidate_idx, col, row, refl_band):
    subset = refl_band[
        (refl_band['image_id']      == image_id) &
        (refl_band['candidate_idx'] == candidate_idx)
    ]
    if subset.empty:
        return np.nan
    dist    = np.sqrt((subset['col']-col)**2 + (subset['row']-row)**2)
    idx_min = dist.idxmin()
    return float(subset.loc[idx_min, 'area']) if dist[idx_min] <= _COORD_TOL else np.nan


def _match_centroid_areas(det_results):
    refl = _load_all_reflectance()
    refl['image_id'] = _normalise_image_id(refl['image_id'].astype(str))

    # Centroid coordinates come from all_confirmed, not detection_results
    confirmed = _load_all_confirmed()
    confirmed['image_id'] = _normalise_image_id(confirmed['image_id'].astype(str))

    # Join detection_results with confirmed to get c_b2/b4 col/row
    det = det_results.copy()
    det['image_id'] = _normalise_image_id(det['image_id'].astype(str))
    det = det.merge(
        confirmed[['image_id', 'candidate_idx',
                   'c_b2_col', 'c_b2_row', 'c_b4_col', 'c_b4_row']],
        on=['image_id', 'candidate_idx'],
        how='left',
    )

    refl_b2 = refl[refl['band'] == 'B2'][['image_id','candidate_idx','col','row','area']]
    refl_b4 = refl[refl['band'] == 'B4'][['image_id','candidate_idx','col','row','area']]

    det['b2_area'] = det.apply(
        lambda r: _lookup_area(r['image_id'], r['candidate_idx'],
                               r['c_b2_col'], r['c_b2_row'], refl_b2), axis=1)
    det['b4_area'] = det.apply(
        lambda r: _lookup_area(r['image_id'], r['candidate_idx'],
                               r['c_b4_col'], r['c_b4_row'], refl_b4), axis=1)
    return det


def _plot_centroid_area_dist(matched):
    """Side-by-side histogram: B2 and B4 centroid area for TP vs FP."""
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

    for ax, band, col in zip(axes, ['B2', 'B4'], ['b2_area', 'b4_area']):
        sub = matched.dropna(subset=[col])
        tp  = sub[sub['detection_type'] == 'TP'][col]
        fp  = sub[sub['detection_type'] == 'FP'][col]
        max_area = int(min(sub[col].max(), 20))
        bins = np.arange(0.5, max_area + 1.5, 1)

        ax.hist(tp, bins=bins, color=_C_TP, alpha=1.0,
                label=f'TP (n={len(tp)})', density=True, edgecolor='none')
        ax.hist(fp, bins=bins, color=_C_FP, alpha=0.7,
                label=f'FP (n={len(fp)})', density=True, edgecolor='none')
        ax.axvline(1.5, color='#333', lw=0.8, ls='--', label='1-pixel threshold')

        tp_1px = (tp == 1).mean() * 100
        fp_1px = (fp == 1).mean() * 100
        fig.suptitle('Reflectance Anomaly Area: True Positives vs. False Positives', fontsize=10)
        ax.set_xlabel(f'{band} centroid area (pixel)')
        ax.set_ylabel('Density')
        ax.set_title(f'{band} share of 1 px anomalies: TP {tp_1px:.0f}% | FP {fp_1px:.0f}%')
        ax.legend(frameon=False)
        ax.grid(True, axis='y', ls='--')

    fig.tight_layout()
    plt.show()

def _plot_centroid_filter_simulation(matched):
    """P/R/F1 vs minimum centroid area threshold."""
    _set_style()
    thresholds = np.arange(1, 11)
    total_gt   = (matched['detection_type'] == 'TP').sum()
    precisions, recalls, f1s, n_kept = [], [], [], []

    for t in thresholds:
        keep = (matched['b2_area'].fillna(0) >= t) & \
               (matched['b4_area'].fillna(0) >= t)
        sub = matched[keep]
        tp  = (sub['detection_type'] == 'TP').sum()
        fp  = (sub['detection_type'] == 'FP').sum()
        fn  = total_gt - tp
        p   = tp/(tp+fp) if (tp+fp) > 0 else np.nan
        r   = tp/(tp+fn) if (tp+fn) > 0 else np.nan
        f1  = 2*p*r/(p+r) if (not np.isnan(p) and not np.isnan(r) and (p+r)>0) else np.nan
        precisions.append(p); recalls.append(r)
        f1s.append(f1);       n_kept.append(keep.sum())

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(thresholds, precisions, 'o-', color='#4C8BB5', label='Precision')
    ax1.plot(thresholds, recalls,    's-', color='#5AA55E', label='Recall')
    ax1.plot(thresholds, f1s,        '^-', color='#D4A843', label='F1')
    ax1.set_xlabel('Minimum centroid area (px) — both B2 and B4')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(thresholds)
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend(frameon=False, loc='lower right')

    ax2 = ax1.twinx()
    ax2.bar(thresholds, n_kept, alpha=0.15, color='#888', width=0.6)
    ax2.set_ylabel('Detections surviving filter', color='#888')
    ax2.tick_params(axis='y', labelcolor='#888')
    ax1.set_title('Filter simulation: minimum centroid area threshold')
    fig.tight_layout()
    plt.show()


#public function
def run_pr_analysis():
    det_results, gt_results, _ = load_eval_results()

    _plot_precision_recall_by_background(det_results, gt_results)


def run_fp_analysis():
    """
    FP diagnostic analysis.
    Outputs:
        Table  — FP reason codes × background
        Table  — TP/FP/FN rates × background
        Figure — FP rate by background
        Figure — FP reason codes by background (stacked bar)
        Figure — FP reason × background heatmap
    """
    det_results, gt_results, gt_visual = load_eval_results()

    tbl_reasons = _fp_reason_table(det_results)
    tbl_rates   = _rate_table(det_results, gt_results)

    _print_table(tbl_reasons, 'FP reason codes by background')
    _print_table(tbl_rates[['TP','FP','FP rate','Precision']], 'FP summary by background')

    _plot_fp_rate_by_background(det_results, gt_results)
    _plot_fp_reasons_by_background(det_results)
    _plot_fp_reasons_heatmap(det_results)

    return tbl_reasons, tbl_rates


def run_fn_analysis():
    """
    FN diagnostic analysis.
    Outputs:
        Table  — FN reason codes × background
        Table  — TP/FP/FN rates × background
        Figure — FN rate by background
        Figure — FN reason codes by background (stacked bar)
        Figure — ADS-B altitude: detected vs missed
        Figure — ADS-B speed: detected vs missed
    """
    det_results, gt_results, gt_visual = load_eval_results()

    tbl_reasons = _fn_reason_table(gt_visual)
    tbl_rates   = _rate_table(det_results, gt_results)

    _print_table(tbl_reasons, 'FN reason codes by background')
    _print_table(tbl_rates[['TP','FN','FN rate','Recall']], 'FN summary by background')

    _plot_fn_rate_by_background(det_results, gt_results)
    _plot_fn_reasons_by_background(gt_visual)
    _plot_fn_altitude(gt_results)
    _plot_fn_adsb_speed(gt_results)
    

    return tbl_reasons, tbl_rates


def run_displacement_analysis():
    """
    Speed, direction, displacement, and residual analysis.
    Outputs:
        Figure — Pipeline speed KDE: TP vs FP (overall)
        Figure — Pipeline speed KDE: FP by background
        Figure — Displacement angle KDE: TP vs FP
        Figure — Displacement residual KDE: TP vs FP
        Figure — Inter-band scatter: TP vs FP coloured by background
    """
    det_results, gt_results, gt_visual = load_eval_results()

    _plot_speed_dist(det_results)
    _plot_angle_dist(det_results)
    _plot_residual_dist(det_results)
    #_plot_band_scatter(det_results)
    #_plot_displacement_speed_combined(det_results)


def run_pixel_size_analysis():
    """
    Centroid area (pixel size) analysis.
    Outputs:
        Figure — B2 centroid area: TP vs FP
        Figure — B4 centroid area: TP vs FP
        Figure — P/R/F1 vs minimum area threshold
    """
    det_results, _, _ = load_eval_results()
    matched = _match_centroid_areas(det_results)
    _plot_centroid_area_dist(matched)
    #_plot_centroid_filter_simulation(matched)


def run_all_analysis():
    """Runs all four analysis sections in order."""
    print('─── FP Analysis ────────────────────────────')
    run_fp_analysis()
    print('─── FN Analysis ────────────────────────────')
    run_fn_analysis()
    print('─── Displacement / Speed Analysis ──────────')
    run_displacement_analysis()
    print('─── Pixel Size Analysis ────────────────────')
    run_pixel_size_analysis()


def run_adsb_detectable_table():
    """Prints and returns ADS-B detectable fraction table by background."""
    gt_adsb = pd.read_csv(Path(GROUND_TRUTH_DIR) / 'ground_truth_adsb.csv')
    tbl = _adsb_detectable_table(gt_adsb)
    _print_table(tbl, 'ADS-B detectable fraction by background type')
    return tbl


