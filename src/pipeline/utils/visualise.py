"""
Script holds functions to visualise + inspect intermediary pipeline results
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import math
import geopandas as gpd
from shapely.geometry import Point

from pipeline.config import CHIP_CENTRE_PX, CHIPS_DIR, COL_DIR, CONFIRMED_DIR, DISPLACEMENT_OFFSET, DISPLACEMENT_TOLERANCE, DISPLACEMENT_RATIO, COLINEAR_ANGLE_MIN, REFL_DIR
from pipeline.utils.io import get_bands, load_chip_by_idx, _load_col_rows, _load_conf_row, _pixel_dist_m
from pipeline.s1_gee_candidates.aoi_selection import get_bbox
from pipeline.s3_download.chip_download import _normalize_chip
from pipeline.s4_confirmation.reflectance_check import run_segmentation
from pipeline.s4_confirmation.displacement_check import _pixel_dist



#Step 4: Confirmation

def _stretch(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """
    Percentile contrast stretch to [0, 1] to display chips in true color.
    """
    p_lo, p_hi = np.percentile(arr, lo), np.percentile(arr, hi)
    return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-9), 0, 1)
 
 
def _make_rgb(chip: np.ndarray) -> np.ndarray:
    """
    Get true colour RGB (R=B4, G=B3, B=B2) from (3, H, W) chip
    """
    return np.stack([_stretch(chip[2]), _stretch(chip[1]), _stretch(chip[0])], axis=-1)
 
 
def _crosshair(ax, colour='yellow', alpha=0.5, lw=0.5):
    """
    Draw centre crosshair on an axes to spot centroid more easily
    """
    ax.axhline(CHIP_CENTRE_PX, color=colour, linewidth=lw, alpha=alpha)
    ax.axvline(CHIP_CENTRE_PX, color=colour, linewidth=lw, alpha=alpha)


#plots
def plot_chip_by_idx(idx, percentile=2):
    """Displays single chip, callable be index"""
    _, chip_path = load_chip_by_idx(idx)
    rgb = _normalize_chip(str(chip_path), percentile)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(chip_path.stem)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_4B_chip(chip: np.ndarray, chip_path: None = None) -> None:
    """
    4-panels showing true colour + individual bands B2, B3, B4.
    """
    B2, B3, B4 = get_bands(chip)
    rgb = _make_rgb(chip)

    print(f"B2 mean: {B2.mean():.1f}  max: {B2.max():.1f}")
    print(f"B3 mean: {B3.mean():.1f}  max: {B3.max():.1f}")
    print(f"B4 mean: {B4.mean():.1f}  max: {B4.max():.1f}")
 
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
 
    axes[0].imshow(rgb)
    axes[0].set_title("True colour (R4G3B2)")
 
    for ax, band, title in zip(axes[1:], [B2, B3, B4], ["B2 (blue)", "B3 (green)", "B4 (red)"]):
        ax.imshow(band, cmap="gray")
        ax.set_title(title)
 
    for ax in axes:
        _crosshair(ax)
        ax.set_xticks([])
        ax.set_yticks([])
 
    if chip_path is not None:
        fig.suptitle(f"Band Inspection of {Path(chip_path).name}", fontsize=9)
 
    plt.tight_layout()
    plt.show()



def plot_difference_images(chip: np.ndarray) -> None:
    """
    6 panels showing all positive difference images used in segmentation.
    """
    B2, B3, B4 = get_bands(chip)
 
    pairs = [
        (B2 - B3, "B2 - B3"),
        (B2 - B4, "B2 - B4"),
        (B3 - B2, "B3 - B2"),
        (B3 - B4, "B3 - B4"),
        (B4 - B2, "B4 - B2"),
        (B4 - B3, "B4 - B3"),
    ]

    for diff, title in pairs:
        pos = np.clip(diff, 0, None)
        idx = np.unravel_index(pos.argmax(), pos.shape)
        print(f"{title}: max={pos.max():.1f} at row={idx[0]}, col={idx[1]}")
 
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
 
    for ax, (diff, title) in zip(axes.flat, pairs):
        pos = np.clip(diff, 0, None)
        ax.imshow(pos, cmap="hot")
        ax.set_title(title)
        _crosshair(ax, colour='cyan')
        ax.set_xticks([])
        ax.set_yticks([])
        # brightest pixel annotated
        idx = np.unravel_index(pos.argmax(), pos.shape)
        ax.plot(idx[1], idx[0], 'c+', markersize=8, markeredgewidth=1.5)
 
    plt.suptitle("(Positive) Difference Images of all Bands", y=1.01)
    #plt.title("Brightest pixel per Difference Image Highlighted")
    plt.tight_layout()
    plt.show()



def plot_anomaly_masks(chip) -> None:
    """
    3 panels showing anomaly masks (after AND of difference images) with the CCL
    objects highlighted.
 
    Args:
        masks         : dict 'B2','B3','B4' → uint8 binary mask
        all_centroids : dict 'B2','B3','B4' → list of (col, row, area)
    """

    C_B2, C_B3, C_B4, masks = run_segmentation(chip)

    all_centroids = {
        'B2': C_B2,
        'B3': [C_B3] if C_B3 is not None else [],
        'B4': C_B4,
    }

    print("Anomaly pixels:")
    for band in ['B2', 'B3', 'B4']:
        print(f"  {band}: {masks[band].sum()} px  / {len(all_centroids[band])} components")
    

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
 
    for ax, band in zip(axes, ['B2', 'B3', 'B4']):
        mask = masks[band]
        centroids = all_centroids[band]
 
        ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
        _crosshair(ax, colour='cyan')
 
        for col, row, area in centroids:
            color = 'lime' if area > 1 else 'red'
            ax.plot(col, row, '+', color=color, markersize=8, markeredgewidth=1.5)
            ax.annotate(f"{area}", (col, row), color='yellow', fontsize=6,
                        xytext=(3, 3), textcoords='offset points')
 
        ax.set_title(f"{band} anomaly mask: {len(centroids)} components\n"
                     f"({mask.sum()} anomaly pixels)")
        ax.set_xticks([])
        ax.set_yticks([])
 

    # Legend
    lime_patch = mpatches.Patch(color='lime', label='area > 1px')
    red_patch  = mpatches.Patch(color='red',  label='area = 1px')
    axes[-1].legend(handles=[lime_patch, red_patch], fontsize=7, loc='lower right')

    plt.tight_layout()
    plt.show()



def plot_collinearity(idx: int) -> None:
    """
    Shows selected image chips with collinear passing triplets
    """
    chip, chip_path = load_chip_by_idx(idx)
    B2, B3, B4      = get_bands(chip)
    rgb             = _make_rgb(chip)

    df_col = pd.read_csv(Path(COL_DIR) / "all_collinearity.csv")

    #getting passing combinations
    chip_combos = df_col[
        (df_col['candidate_idx'] == idx) &
        (df_col['colinear_found'] == True)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    #left
    axes[0].imshow(rgb)
    _crosshair(axes[0])
    for i, (_, combo) in enumerate(chip_combos.iterrows()):
        axes[0].plot(
            [combo['c_b2_col'], combo['c_b3_col'], combo['c_b4_col']],
            [combo['c_b2_row'], combo['c_b3_row'], combo['c_b4_row']],
            'w--', linewidth=1, alpha=0.6
        )
        axes[0].plot(combo['c_b2_col'], combo['c_b2_row'], 'o',
                color='dodgerblue', markersize=8,
                markeredgecolor='white', markeredgewidth=0.8)
        axes[0].plot(combo['c_b3_col'], combo['c_b3_row'], 'o',
                color='limegreen', markersize=8,
                markeredgecolor='white', markeredgewidth=0.8)
        axes[0].plot(combo['c_b4_col'], combo['c_b4_row'], 'o',
                color='tomato', markersize=8,
                markeredgecolor='white', markeredgewidth=0.8)
        axes[0].annotate(f"{combo['angle_deg']:.1f}°",
                    (combo['c_b3_col'], combo['c_b3_row']),
                    color='white', fontsize=7,
                    xytext=(5, 5+ i * 14), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black',
                              alpha=0.6, ec='none'))

    axes[0].set_title(f"idx={idx:04d}  {chip_path.name}\n"
                 f"{len(chip_combos)} passing combinations", fontsize=8)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    #right
    axes[1].imshow(rgb)
    _crosshair(axes[0])
    axes[1].set_title(f"idx={idx:04d}: true colour", fontsize=8)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    #chip summary
    print(f"\nidx={idx:04d}  {chip_path.name}")
    print(f"  Passing combinations : {len(chip_combos)}")
    for i, (_, combo) in enumerate(chip_combos.iterrows()):
        print(f"  combo {i+1}: "
              f"angle={combo['angle_deg']:.1f}°  "
              f"B2=({combo['c_b2_col']:.1f}, {combo['c_b2_row']:.1f})  "
              f"B3=({combo['c_b3_col']:.1f}, {combo['c_b3_row']:.1f})  "
              f"B4=({combo['c_b4_col']:.1f}, {combo['c_b4_row']:.1f})")

    fig.suptitle(chip_path.name, fontsize=9)
    plt.tight_layout()
    plt.show()


def passing_angle_scatter():
    df_col = pd.read_csv(Path(COL_DIR) / "all_collinearity.csv")
    df_refl = pd.read_csv(Path(REFL_DIR) / "all_reflectance.csv")

    all_angles     = df_col[df_col['angle_deg'].notna()]['angle_deg']
    passing_angles = df_col[df_col['colinear_found'] == True]['angle_deg']
    failing_angles = df_col[
        (df_col['colinear_found'] == False) &
        (df_col['angle_deg'].notna())#excludes missing band anomalies
    ]['angle_deg']

    print(f"Total combinations      : {len(all_angles)}")
    print(f"Passing (≥170°)         : {len(passing_angles)}")
    print(f"Failing (<170°)         : {len(failing_angles)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # left
    axes[0].hist(failing_angles, bins=36, color='tomato',
                edgecolor='white', linewidth=0.5, alpha=0.7,
                label=f'Failed (n={len(failing_angles)})')
    axes[0].hist(passing_angles, bins=36, color='limegreen',
                edgecolor='white', linewidth=0.5, alpha=0.7,
                label=f'Passed (n={len(passing_angles)})')
    axes[0].axvline(170, color='gold', linestyle='--', linewidth=1.5,
                    label='170° threshold')
    axes[0].set_xlabel("Angle at C_B3 (°)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"All computed angles (n={len(all_angles)})")
    axes[0].set_xlim(0, 180)
    axes[0].legend()

    # right: panel with zoom
    axes[1].hist(failing_angles[failing_angles >= 160], bins=20,
                color='tomato', edgecolor='white', linewidth=0.5, alpha=0.7,
                label=f'Failed (n={len(failing_angles)})')
    axes[1].hist(passing_angles, bins=20, color='limegreen',
                edgecolor='white', linewidth=0.5, alpha=0.7,
                label=f'Passed (n={len(passing_angles)})')
    axes[1].axvline(170, color='gold', linestyle='--', linewidth=1.5,
                    label='170° threshold')
    axes[1].set_xlabel("Angle at C_B3 (°)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Zoomed: around threshold")
    axes[1].set_xlim(160, 182)
    axes[1].legend()

    plt.tight_layout()
    plt.show()



def displacement_scatter():
    df_conf = pd.read_csv(Path(CONFIRMED_DIR) / "all_confirmed.csv")
    df_conf['confirmed'] = df_conf['confirmed'].astype(str).map(
        {'True': True, 'False': False}
    )
    confirmed = df_conf[df_conf['confirmed'] == True]
    rejected  = df_conf[
        (df_conf['confirmed'] == False) &
        (df_conf['residual_m'].notna())   # exclude no-colinear chips (NaN)
    ]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Expected line
    d23_max   = df_conf['D_Band2_3_m'].max(skipna=True) * 1.1 + 10
    d23_range = np.linspace(0, d23_max, 100)
    d34_expected = DISPLACEMENT_RATIO * d23_range + DISPLACEMENT_OFFSET
    ax.plot(d23_range, d34_expected, color='gold', linewidth=1.5,
            label=f'Expected D2_4')
            #label=f'Expected: D_34 = {DISPLACEMENT_RATIO}×D_23 + {DISPLACEMENT_OFFSET}')
    ax.fill_between(d23_range,
                    d34_expected - DISPLACEMENT_TOLERANCE,
                    d34_expected + DISPLACEMENT_TOLERANCE,
                    alpha=0.15, color='gold',
                    label=f'±{DISPLACEMENT_TOLERANCE}m tolerance')

    # Rejected detections
    ax.scatter(rejected['D_Band2_3_m'], rejected['D_Band3_4_m'],
               color='tomato', s=50, zorder=4,
               edgecolors='white', linewidths=0.5,
               label=f'Rejected (n={len(rejected)})')

    # Confirmed detections
    ax.scatter(confirmed['D_Band2_3_m'], confirmed['D_Band3_4_m'],
               color='limegreen', s=50, zorder=5,
               edgecolors='white', linewidths=0.5,
               label=f'Confirmed (n={len(confirmed)})')

    ax.set_xlabel("D_Band2_3 (m)")
    ax.set_ylabel("D_Band3_4 (m)")
    ax.set_title("Proportional displacement: confirmed vs rejected")
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_displacement(idx: int) -> None:
    """
    Two panels:
      Left  — true colour chip with confirmed B2/B3/B4 centroids
      Right — proportional displacement scatter for all colinear
              combinations of this chip, confirmed one highlighted
    """
    chip, chip_path = load_chip_by_idx(idx)
    rgb             = _make_rgb(chip)

    df_conf = pd.read_csv(Path(CONFIRMED_DIR) / "all_confirmed.csv")
    df_col = pd.read_csv(Path(COL_DIR) / "all_collinearity.csv")

    # Confirmed result for this chip
    row = df_conf[df_conf['candidate_idx'] == idx].iloc[0]

    # All colinear combinations for this chip (for scatter)
    chip_combos = df_col[
        (df_col['candidate_idx'] == idx) &
        (df_col['colinear_found'] == True)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- Left: true colour with confirmed triplet ---
    axes[0].imshow(rgb)
    _crosshair(axes[0])
    if row['confirmed']:
        axes[0].plot(row['c_b2_col'], row['c_b2_row'], 'o',
                     color='dodgerblue', markersize=10,
                     markeredgecolor='white', markeredgewidth=0.8,
                     label='B2')
        axes[0].plot(row['c_b3_col'], row['c_b3_row'], 'o',
                     color='limegreen', markersize=10,
                     markeredgecolor='white', markeredgewidth=0.8,
                     label='B3')
        axes[0].plot(row['c_b4_col'], row['c_b4_row'], 'o',
                     color='tomato', markersize=10,
                     markeredgecolor='white', markeredgewidth=0.8,
                     label='B4')
        axes[0].plot(
            [row['c_b2_col'], row['c_b3_col'], row['c_b4_col']],
            [row['c_b2_row'], row['c_b3_row'], row['c_b4_row']],
            'w--', linewidth=1.5
        )
        axes[0].set_title(
            f"✓ CONFIRMED\n"
            f"{row['speed_kmh']:.0f} km/h  |  "
            f"angle={row['angle_deg']:.1f}°  |  "
            f"residual={row['residual_m']:.1f}m",
            fontsize=8, color='limegreen'
        )
    else:
        axes[0].set_title("✗ REJECTED", fontsize=8, color='tomato')
    axes[0].legend(fontsize=7, loc='lower right')
    axes[0].set_xticks([]); axes[0].set_yticks([])

     #right
    axes[1].imshow(rgb)
    _crosshair(axes[0])
    axes[1].set_title(f"idx={idx:04d}: true colour", fontsize=8)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    fig.suptitle(f"idx={idx:04d}  {chip_path.name}", fontsize=9)

    #summary
    print(f"\nidx={idx:04d}  confirmed={row['confirmed']}")
    if row['confirmed']:
        print(f"  D_Band2_3 : {row['D_Band2_3_m']:.1f} m")
        print(f"  D_Band3_4 : {row['D_Band3_4_m']:.1f} m")
        print(f"  residual  : {row['residual_m']:.1f} m")
        print(f"  angle     : {row['angle_deg']:.1f}°")
        print(f"  speed     : {row['speed_kmh']:.0f} km/h")


    plt.tight_layout()
    plt.show()


#Summary plot 
#----------------
# 1: Reflectance
#----------------
def plot_chip_reflectance(idx) -> None:
    """
    4-panel figure showing true colour chip + B2/B3/B4 anomaly masks.
    Failure cases shown explicitly:
      - Empty mask labelled 'no anomaly found'
      - Overall status (confirmed / rejected) shown in suptitle
    Recomputes run_segmentation on the fly — masks are not stored on disk.
    """
    chip, chip_path = load_chip_by_idx(idx)
    C_B2, C_B3, C_B4, masks = run_segmentation(chip)
    rgb = _make_rgb(chip)
 
    result    = _load_conf_row(idx, CONFIRMED_DIR)
    confirmed = result['confirmed'] if result else False
    status    = '✓ CONFIRMED' if confirmed else '✗ REJECTED'
    s_colour  = 'limegreen'   if confirmed else 'tomato'
 
    all_centroids = {
        'B2': C_B2,
        'B3': [C_B3] if C_B3 is not None else [],
        'B4': C_B4,
    }
    colours = {'B2': 'dodgerblue', 'B3': 'limegreen', 'B4': 'tomato'}
 
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
 
    # Panel 1 — true colour
    axes[0].imshow(rgb)
    _crosshair(axes[0])
    axes[0].set_title("True colour (RGB)", fontsize=12)
    axes[0].set_xticks([]); axes[0].set_yticks([])
 
    # Panels 2-4 — anomaly masks
    for ax, band in zip(axes[1:], ['B2', 'B3', 'B4']):
        mask      = masks[band]
        centroids = all_centroids[band]
        n_px      = int(mask.sum())
 
        ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
        _crosshair(ax, colour='cyan')
 
        if n_px == 0:
            # Failure: no anomaly found in this band
            ax.text(0.5, 0.5, 'no anomaly\nfound',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='tomato',
                    bbox=dict(fc='black', alpha=0.6, ec='none',
                              boxstyle='round,pad=0.4'))
        else:
            for c in centroids:
                color = 'lime' if c[2] > 1 else 'red'
                ax.plot(c[0], c[1], '+', color=color,
                        markersize=8, markeredgewidth=1.5)
                ax.annotate(f"{c[2]}px", (c[0], c[1]),
                            color='yellow', fontsize=6,
                            xytext=(3, 3), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      fc='black', alpha=0.5, ec='none'))
 
        n_label = f"{len(centroids)} selected" if n_px > 0 else "—"
        ax.set_title(f"{band} anomaly mask\n{n_px} candidates  /  {n_label}",
                     fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
 
    """
    fig.suptitle(
        f"idx={idx:04d}  {Path(chip_path).name}: "
        f"[Step 1: Reflectance]  {status}",
        fontsize=9, fontweight='bold', color=s_colour
    )
    """
    plt.tight_layout()
    plt.show()


#------------
#2: Collinearity
#------------

def plot_chip_collinearity(
        idx:       int
    ) -> None:
        """
        2-panel figure showing collinearity analysis for one chip.
        Panel 1 : true colour with all tested triplets (green=pass, red=fail)
        Panel 2 : angle per combination as scatter, threshold line shown
    
        Failure cases:
        - Missing band  → single panel 'did not reach collinearity check'
        - All fail      → all triplets shown in red
        Reads entirely from all_collinearity.csv — no recomputation needed.
        """
        from pipeline.utils.io import load_chip_by_idx
    
        chip, chip_path = load_chip_by_idx(idx)
        rgb = _make_rgb(chip)
    
        result    = _load_conf_row(idx, CONFIRMED_DIR)
        confirmed = result['confirmed'] if result else False
        status    = '✓ CONFIRMED' if confirmed else '✗ REJECTED'
        s_colour  = 'limegreen'   if confirmed else 'tomato'
    
        chip_df     = _load_col_rows(idx, COL_DIR)
        has_angles  = chip_df['angle_deg'].notna().any()
        combos_all  = chip_df[chip_df['angle_deg'].notna()]
        combos_pass = chip_df[chip_df['colinear_found'] == True]
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
        # --- Panel 1: true colour with triplets ---
        axes[0].imshow(rgb)
        _crosshair(axes[0])
    
        if not has_angles:
            # Missing band — never reached collinearity check
            axes[0].text(0.5, 0.5,
                        'did not reach\ncollinearity check\n(missing band)',
                        transform=axes[0].transAxes,
                        ha='center', va='center', fontsize=12, color='tomato',
                        bbox=dict(fc='black', alpha=0.6, ec='none',
                                boxstyle='round,pad=0.4'))
            axes[0].set_title("[Collinearity] missing band", fontsize=8,
                            color='tomato')
        else:
            for i, (_, combo) in enumerate(combos_all.iterrows()):
                passed = combo['colinear_found']
                lc     = 'limegreen' if passed else 'tomato'
                axes[0].plot(
                    [combo['c_b2_col'], combo['c_b3_col'], combo['c_b4_col']],
                    [combo['c_b2_row'], combo['c_b3_row'], combo['c_b4_row']],
                    '--', color=lc, linewidth=1, alpha=0.7
                )
                axes[0].plot(combo['c_b2_col'], combo['c_b2_row'], 'o',
                            color='dodgerblue', markersize=6,
                            markeredgecolor='white', markeredgewidth=0.5)
                axes[0].plot(combo['c_b3_col'], combo['c_b3_row'], 'o',
                            color='limegreen', markersize=6,
                            markeredgecolor='white', markeredgewidth=0.5,
                            zorder=5)
                axes[0].plot(combo['c_b4_col'], combo['c_b4_row'], 'o',
                            color='tomato', markersize=6,
                            markeredgecolor='white', markeredgewidth=0.5)
                axes[0].annotate(f"{combo['angle_deg']:.1f}°",
                                (combo['c_b3_col'], combo['c_b3_row']),
                                color='white', fontsize=6,
                                xytext=(5, 5 + i * 14),
                                textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.2',
                                        fc=lc, alpha=0.6, ec='none'))
    
            n_pass = int(combos_all['colinear_found'].sum())
            n_fail = len(combos_all) - n_pass
            axes[0].set_title(
                f"Collinearity: {n_pass} passing combinations",
                #f"{n_pass} pass ≥{COLINEAR_ANGLE_MIN}°",
                #f"{n_fail} fail (red)",
                fontsize=12
            )
    
            # Legend
            axes[0].legend(
                handles=[
                    mpatches.Patch(color='limegreen', label='passes ≥170°'),
                    mpatches.Patch(color='tomato',    label='fails <170°'),
                ],
                fontsize=10, loc='lower right'
            )
    
        axes[0].set_xticks([]); axes[0].set_yticks([])
    
        # --- Panel 2: angle scatter ---
        if not has_angles:
            axes[1].text(0.5, 0.5, 'no combinations\nto show',
                        transform=axes[1].transAxes,
                        ha='center', va='center', fontsize=10, color='grey')
            axes[1].set_title("[Collinearity] no combinations", fontsize=8)
        else:
            passing_a = combos_all[combos_all['colinear_found'] == True]['angle_deg']
            failing_a = combos_all[combos_all['colinear_found'] == False]['angle_deg']
    
            x = 0
            for angle in sorted(failing_a):
                axes[1].scatter(x, angle, color='tomato', s=60, zorder=5,
                                edgecolors='white', linewidths=0.5)
                axes[1].annotate(f"{angle:.1f}°", (x, angle),
                                fontsize=10, color='tomato',
                                xytext=(5, 0), textcoords='offset points')
                x += 1
            for angle in sorted(passing_a):
                axes[1].scatter(x, angle, color='limegreen', s=60, zorder=5,
                                edgecolors='white', linewidths=0.5)
                axes[1].annotate(f"{angle:.1f}°", (x, angle),
                                fontsize=10, color='limegreen',
                                xytext=(5, 0), textcoords='offset points')
                x += 1
    
            axes[1].axhline(COLINEAR_ANGLE_MIN, color='gold', linestyle='--',
                            linewidth=1.5,
                            label=f'{COLINEAR_ANGLE_MIN}° threshold')
            axes[1].set_ylabel("Angle at Vertex (Centroid B3)", fontsize=10)
            axes[1].set_xlabel("Combination index", fontsize=10)
            axes[1].set_ylim(
                max(0, combos_all['angle_deg'].min() - 5),
                min(182, combos_all['angle_deg'].max() + 5)
            )
            axes[1].legend(fontsize=10)
            axes[1].set_title("Collinearity: Angles per combination", fontsize=12)
        """
        fig.suptitle(
            f"idx={idx:04d}  {Path(chip_path).name}:"
            f"[Step 2: Collinearity]  {status}",
            fontsize=9, fontweight='bold', color=s_colour
        )
        """
        plt.tight_layout()
        plt.show()


#-----------
#3: Displacement
#-----------

def plot_chip_displacement(
    idx:          int
) -> None:
    """
    2-panel figure showing proportional displacement check for one chip.
    Panel 1 : true colour with confirmed triplet (or failure message)
    Panel 2 : D_23 vs D_34 scatter with expected line and tolerance band
 
    Failure cases:
      - NaN residual  → 'did not reach displacement check'
      - All fail      → all dots outside tolerance band, no star
    Reads from all_confirmed.csv and all_collinearity.csv.
    """
    from pipeline.utils.io import load_chip_by_idx
 
    chip, chip_path = load_chip_by_idx(idx)
    rgb = _make_rgb(chip)
 
    result    = _load_conf_row(idx, CONFIRMED_DIR)
    confirmed = result['confirmed'] if result else False
    status    = '✓ CONFIRMED' if confirmed else '✗ REJECTED'
    s_colour  = 'limegreen'   if confirmed else 'tomato'
 
    # Derive rejection reason
    if result is None:
        reason = 'not found in results'
    elif pd.isna(result.get('residual_m')):
        reason = 'did not reach displacement check\n(no colinear combinations)'
    elif not confirmed:
        reason = (f"best residual = {result['residual_m']:.1f}m  "
                  f"> {DISPLACEMENT_TOLERANCE}m tolerance")
    else:
        reason = None
 
    # Colinear combinations with computed displacements
    chip_combos = _load_col_rows(idx, COL_DIR)
    colinear    = chip_combos[chip_combos['colinear_found'] == True].copy()
 
    if not colinear.empty:
        colinear['D_23'] = colinear.apply(
            lambda r: _pixel_dist_m(r['c_b2_col'], r['c_b2_row'],
                                     r['c_b3_col'], r['c_b3_row']), axis=1
        )
        colinear['D_34'] = colinear.apply(
            lambda r: _pixel_dist_m(r['c_b3_col'], r['c_b3_row'],
                                     r['c_b4_col'], r['c_b4_row']), axis=1
        )
        colinear['residual'] = abs(
            DISPLACEMENT_RATIO * colinear['D_23'] +
            DISPLACEMENT_OFFSET - colinear['D_34']
        )
        colinear['passes'] = colinear['residual'] <= DISPLACEMENT_TOLERANCE
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
 
    # --- Panel 1: true colour with confirmed triplet ---
    axes[0].imshow(rgb)
    _crosshair(axes[0])
 
    if confirmed and not pd.isna(result.get('c_b2_col', float('nan'))):
        bx2, by2 = result['c_b2_col'], result['c_b2_row']
        bx3, by3 = result['c_b3_col'], result['c_b3_row']
        bx4, by4 = result['c_b4_col'], result['c_b4_row']
        axes[0].plot(bx2, by2, 'o', color='dodgerblue', markersize=12,
             markerfacecolor='none', markeredgecolor='dodgerblue', markeredgewidth=1.5, label='B2')
        axes[0].plot(bx3, by3, 'o', color='limegreen', markersize=12,
                    markerfacecolor='none', markeredgecolor='limegreen', markeredgewidth=1.5, label='B3')
        axes[0].plot(bx4, by4, 'o', color='tomato', markersize=12,
                    markerfacecolor='none', markeredgecolor='tomato', markeredgewidth=1.5, label='B4')
        axes[0].plot([bx2, bx3, bx4], [by2, by3, by4],
                     'w--', linewidth=1.5)
        axes[0].set_title(
        f"{result['speed_kmh']:.0f} km/h | "
        f"angle={result['angle_deg']:.1f}° | "
        f"residual={result['residual_m']:.1f}m",
        fontsize=12, color='black'
        )
        axes[0].text(0.5, 1.08, "✓ CONFIRMED", transform=axes[0].transAxes,
                    ha='center', fontsize=12, color='limegreen', fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, f"✗ REJECTED\n{reason}",
                     transform=axes[0].transAxes,
                     ha='center', va='center', fontsize=9, color='tomato',
                     bbox=dict(fc='black', alpha=0.6, ec='none',
                               boxstyle='round,pad=0.4'))
        axes[0].set_title("[Displacement] ✗ REJECTED", fontsize=8,
                          color='tomato')
 
    axes[0].set_xticks([]); axes[0].set_yticks([])
 
    # --- Panel 2: D_23 vs D_34 scatter ---
    if colinear.empty:
        axes[1].text(0.5, 0.5, 'no colinear combinations\nreached this check',
                     transform=axes[1].transAxes,
                     ha='center', va='center', fontsize=12, color='grey')
        axes[1].set_title("[Displacement] no combinations", fontsize=8)
    else:
        d23_max   = colinear['D_23'].max() * 1.2 + 10
        d23_range = np.linspace(0, d23_max, 100)
        d34_exp   = DISPLACEMENT_RATIO * d23_range + DISPLACEMENT_OFFSET
 
        # Expected line + tolerance band
        axes[1].plot(d23_range, d34_exp, color='gold', linewidth=1.5,
                     label=(f'Expected Displacement Ratio'))
                     #label=(f'Expected D_34 = {DISPLACEMENT_RATIO}'
                     #       f'×D_23 + {DISPLACEMENT_OFFSET}'))
        axes[1].fill_between(d23_range,
                             d34_exp - DISPLACEMENT_TOLERANCE,
                             d34_exp + DISPLACEMENT_TOLERANCE,
                             alpha=0.15, color='gold',
                             label=f'±{DISPLACEMENT_TOLERANCE}m tolerance')
 
        # All combinations coloured by pass/fail
        for _, row in colinear.iterrows():
            c = 'limegreen' if row['passes'] else 'tomato'
            axes[1].scatter(row['D_23'], row['D_34'], color=c,
                            s=60, zorder=5,
                            edgecolors='white', linewidths=0.5)
            axes[1].annotate(f"{row['residual']:.1f}m",
                             (row['D_23'], row['D_34']),
                             fontsize=6, color=c,
                             xytext=(4, 4), textcoords='offset points')
 
        # Highlight selected combination
        if confirmed and not pd.isna(result.get('D_Band2_3_m')):
            axes[1].scatter(result['D_Band2_3_m'], result['D_Band3_4_m'],
                color='gold', s=80, zorder=6, 
                marker='D', edgecolors='black', linewidths=0.8,
                label='Selected (best residual)')
 
        n_pass = int(colinear['passes'].sum())
        n_fail = len(colinear) - n_pass
        axes[1].set_xlabel("Displacement B2 to B3 (m)", fontsize=10)
        axes[1].set_ylabel("Displacement B3 to B4 (m)", fontsize=10)
        axes[1].legend(fontsize=10)
        axes[1].set_title(
            f"Displacement: Residual per Collinear Combination",
        #    f"{n_pass} pass (green)  {n_fail} fail (red)  ★=selected",
            fontsize=12
        )
 
    """
    fig.suptitle(
        f"idx={idx:04d}  {Path(chip_path).name}:"
        f"[Step 3: Displacement]  {status}",
        fontsize=9, fontweight='bold', color=s_colour
    )

    """
    plt.tight_layout()
    plt.show()

#------------
#4:Visual Inspection with marker
#-----------
def plot_chip_confirmation(
    idx:          int, 
    zoom_radius:   int = 30
) -> None:
    """
    Single-panel true colour chip with subtle markers showing
    confirmed aircraft position. Designed for visual confirmation
    that the detected object is visible in the image.

    Shows:
      - Hollow rings at C_B2, C_B3, C_B4 positions
      - Tiny centroid dot inside each ring
      - Arrow from C_B4 to C_B2 showing direction of travel
      - Minimal annotation — speed and residual only
    """
    chip, chip_path = load_chip_by_idx(idx)
    rgb = _make_rgb(chip)

    result = _load_conf_row(idx, CONFIRMED_DIR)
    if result is None or not result['confirmed']:
        print(f"idx={idx:04d} is not a confirmed detection — skipping")
        return

    bx2, by2 = result['c_b2_col'], result['c_b2_row']
    bx3, by3 = result['c_b3_col'], result['c_b3_row']
    bx4, by4 = result['c_b4_col'], result['c_b4_row']

    # Zoom window centred on C_B3
    cx  = int(round(bx3))
    cy  = int(round(by3))
    x0  = max(0,   cx - zoom_radius)
    x1  = min(300, cx + zoom_radius)
    y0  = max(0,   cy - zoom_radius)
    y1  = min(300, cy + zoom_radius)

    colours = {'B2': 'dodgerblue', 'B3': 'limegreen', 'B4': 'tomato'}
    points  = {'B2': (bx2, by2), 'B3': (bx3, by3), 'B4': (bx4, by4)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    #left plot
    axes[0].imshow(rgb)
    _crosshair(axes[0])

    # Dashed box showing zoom region
    rect = plt.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=1.5, edgecolor='white',
        facecolor='none', linestyle='--'
    )
    axes[0].add_patch(rect)
    axes[0].set_title("True colour — full chip", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # right zoomed in plots
    axes[1].imshow(rgb[y0:y1, x0:x1])

    for band, colour in colours.items():
        bx, by = points[band]
        # Remap to zoomed coordinate space
        zx = bx - x0
        zy = by - y0
        # Hollow ring
        axes[1].plot(zx, zy, 'o',
                     markerfacecolor='none',
                     markeredgecolor=colour,
                     markeredgewidth=1.5,
                     markersize=10,
                     label=band)
        # Tiny centroid dot
        axes[1].plot(zx, zy, '.', color=colour, markersize=3)

    # Direction arrow C_B4 → C_B2 in zoomed space
    axes[1].annotate('',
                     xy=(bx2 - x0, by2 - y0),
                     xytext=(bx4 - x0, by4 - y0),
                     arrowprops=dict(arrowstyle='->',
                                     color='white',
                                     lw=1.0, alpha=0.8))

    axes[1].legend(fontsize=7, loc='lower right',
                   framealpha=0.4, edgecolor='none')
    axes[1].set_title(
        f"Zoomed — {zoom_radius*2}×{zoom_radius*2} px\n"
        f"{result['speed_kmh']:.0f} km/h  |  "
        f"residual={result['residual_m']:.1f}m",
        fontsize=8
    )
    axes[1].set_xticks([]); axes[1].set_yticks([])
    """
    fig.suptitle(
        f"idx={idx:04d}  {Path(chip_path).name}  —  ✓ CONFIRMED",
        fontsize=9, fontweight='bold', color='limegreen'
    )

    """
    plt.tight_layout()
    plt.show()

#------------
#5: Pipeline Summary
#------------


def plot_pipeline_summary() -> None:
    """
    3-panel summary figure for the full Step 4 pipeline:
      Panel 1 : candidate funnel — counts at each stage
      Panel 2 : speed distribution of confirmed detections
      Panel 3 : spatial map of confirmed detections coloured by speed
    Reads entirely from CSVs — no chip images needed.
    """
    # Load
    df_refl = pd.read_csv(Path(REFL_DIR)      / "all_reflectance.csv")
    df_col  = pd.read_csv(Path(COL_DIR)        / "all_collinearity.csv")
    df_conf = pd.read_csv(Path(CONFIRMED_DIR)  / "all_confirmed.csv")
    df_conf['confirmed'] = df_conf['confirmed'].astype(str).map(
        {'True': True, 'False': False}
    )
    confirmed = df_conf[df_conf['confirmed'] == True]
 
    # Stage counts
    n_total     = len(df_conf)
    n_b3        = (df_refl[df_refl['band'] == 'B3']['candidate_idx']
                   .nunique())
    n_colinear  = (df_col[df_col['colinear_found'] == True]['candidate_idx']
                   .nunique())
    n_confirmed = int(df_conf['confirmed'].sum())
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
    # --- Panel 1: funnel ---
    stages  = ['Candidates\n(after seam filter)',
               'B3 anomaly\nfound',
               'Colinear combo\nfound',
               'Confirmed\naircraft']
    counts  = [n_total, n_b3, n_colinear, n_confirmed]
    colours = ['steelblue', 'steelblue', 'steelblue', 'limegreen']
 
    bars = axes[0].bar(stages, counts, color=colours,
                       edgecolor='white', linewidth=0.5)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     str(count), ha='center', va='bottom', fontsize=9)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Step 4: candidate pipeline steps", fontsize=9)
 
    # --- Panel 2: speed distribution ---
    if not confirmed.empty:
        speeds = confirmed['speed_kmh'].dropna()
        axes[1].hist(speeds, bins=20, color='steelblue',
                     edgecolor='white', linewidth=0.5)
        axes[1].axvline(speeds.median(), color='gold', linestyle='--',
                        linewidth=1.5,
                        label=f"median {speeds.median():.0f} km/h")
        axes[1].set_xlabel("Speed (km/h)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(
            f"Speed distribution: confirmed aircraft (n={len(confirmed)})",
            fontsize=9
        )
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, 'no confirmed detections',
                     transform=axes[1].transAxes,
                     ha='center', va='center', fontsize=10, color='grey')
 
    # --- Panel 3: spatial map ---
    if not confirmed.empty:
        sc = axes[2].scatter(
            confirmed['lon'], confirmed['lat'],
            c=confirmed['speed_kmh'], cmap='RdYlGn_r',
            s=40, alpha=0.85,
            edgecolors='white', linewidths=0.3
        )
        plt.colorbar(sc, ax=axes[2], label='Speed (km/h)')
        axes[2].set_xlabel("Longitude")
        axes[2].set_ylabel("Latitude")
        axes[2].set_title(
            f"Confirmed aircraft locations across all scenes (n={len(confirmed)})",
            fontsize=9
        )
    else:
        axes[2].text(0.5, 0.5, 'no confirmed detections',
                     transform=axes[2].transAxes,
                     ha='center', va='center', fontsize=10, color='grey')
 
    fig.suptitle("Step 4: Aircraft confirmation summary",
                 fontsize=10, fontweight='bold')
 
    # Summary print
    print(f"\n{'=' * 50}")
    print(f"STEP 4 PIPELINE SUMMARY")
    print(f"{'=' * 50}")
    print(f"Candidates entering Step 4 : {n_total}")
    print(f"B3 anomaly found           : {n_b3}  "
          f"({100*n_b3/n_total:.1f}%)")
    print(f"Colinear combination found : {n_colinear}  "
          f"({100*n_colinear/n_total:.1f}%)")
    print(f"Confirmed aircraft         : {n_confirmed}  "
          f"({100*n_confirmed/n_total:.1f}%)")
    if not confirmed.empty:
        speeds = confirmed['speed_kmh'].dropna()
        print(f"\nSpeed stats:")
        print(f"  mean   : {speeds.mean():.0f} km/h")
        print(f"  median : {speeds.median():.0f} km/h")
        print(f"  min    : {speeds.min():.0f} km/h")
        print(f"  max    : {speeds.max():.0f} km/h")

    plt.tight_layout()
    plt.show()








#-------------
#ARCHIVE
#-------------


def inspect_chip_full(idx: int) -> None:
    """
    Consolidated 3-row diagnostic figure for one chip showing all
    three confirmation steps:

      Row 1 — Reflectance segmentation
        Panel 1: true colour chip
        Panel 2: B2 anomaly mask
        Panel 3: B3 anomaly mask
        Panel 4: B4 anomaly mask

      Row 2 — Collinearity
        Panel 1: true colour with all tested triplets overlaid
        Panel 2: angle distribution for this chip

      Row 3 — Proportional displacement
        Panel 1: true colour with confirmed triplet highlighted
        Panel 2: D_23 vs D_34 scatter with tolerance band
    """
    # --- Load data ---
    df_conf = pd.read_csv(Path(CONFIRMED_DIR) / "all_confirmed.csv")
    df_col = pd.read_csv(Path(COL_DIR) / "all_collinearity.csv")
    chip, chip_path = load_chip_by_idx(idx)
    C_B2, C_B3, C_B4, masks = run_segmentation(chip)
    rgb = _make_rgb(chip)

    conf_rows = df_conf[df_conf['candidate_idx'] == idx]
    if conf_rows.empty:
        print(f"idx={idx:04d} not found in df_conf")
        return
    result = conf_rows.iloc[0].to_dict()
    confirmed = result.get('confirmed', False)

    chip_combos = df_col[df_col['candidate_idx'] == idx].copy()
    chip_combos_all = chip_combos[chip_combos['angle_deg'].notna()]
    chip_combos_pass = chip_combos[chip_combos['colinear_found'] == True]

    # --- Compute displacements for all colinear combos ---
    if not chip_combos_pass.empty:
        chip_combos_pass = chip_combos_pass.copy()
        chip_combos_pass['D_23'] = chip_combos_pass.apply(
            lambda r: math.sqrt(((r['c_b2_col'] - r['c_b3_col']) * 10) ** 2 +
                                 ((r['c_b2_row'] - r['c_b3_row']) * 10) ** 2), axis=1
        )
        chip_combos_pass['D_34'] = chip_combos_pass.apply(
            lambda r: math.sqrt(((r['c_b3_col'] - r['c_b4_col']) * 10) ** 2 +
                                 ((r['c_b3_row'] - r['c_b4_row']) * 10) ** 2), axis=1
        )
        chip_combos_pass['residual'] = abs(
            DISPLACEMENT_RATIO * chip_combos_pass['D_23'] +
            DISPLACEMENT_OFFSET - chip_combos_pass['D_34']
        )
        chip_combos_pass['passes_disp'] = (
            chip_combos_pass['residual'] <= DISPLACEMENT_TOLERANCE
        )

    colours = {'B2': 'dodgerblue', 'B3': 'limegreen', 'B4': 'tomato'}

    # --- Figure layout ---
    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    # ── Row 1: Reflectance segmentation ──────────────────────────────────

    # Panel 1: true colour
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(rgb)
    _crosshair(ax_rgb)
    #ax_rgb.set_title("True colour", fontsize=8)
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])

    # Panels 2-4: anomaly masks
    for col_idx, band in enumerate(['B2', 'B3', 'B4'], start=1):
        ax = fig.add_subplot(gs[0, col_idx])
        mask      = masks[band]
        centroids = (C_B2 if band == 'B2' else
                     ([C_B3] if C_B3 is not None else []) if band == 'B3'
                     else C_B4)
        ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
        _crosshair(ax, colour='cyan')
        for c in centroids:
            color = 'lime' if c[2] > 1 else 'red'
            ax.plot(c[0], c[1], '+', color=color,
                    markersize=7, markeredgewidth=1.2)
            ax.annotate(f"{c[2]}", (c[0], c[1]), color='yellow', fontsize=6,
                        xytext=(3, 3), textcoords='offset points')
        ax.set_title(f"[Reflectance] {band}\n"
                     f"{mask.sum()} px — {len(centroids)} selected", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # ── Row 2: Collinearity ───────────────────────────────────────────────

    # Panel 1: true colour with all triplets
    ax_col = fig.add_subplot(gs[1, :2])
    ax_col.imshow(rgb)
    _crosshair(ax_col)

    for i, (_, combo) in enumerate(chip_combos_all.iterrows()):
        passed = combo['colinear_found']
        lc     = 'limegreen' if passed else 'tomato'
        ax_col.plot(
            [combo['c_b2_col'], combo['c_b3_col'], combo['c_b4_col']],
            [combo['c_b2_row'], combo['c_b3_row'], combo['c_b4_row']],
            '--', color=lc, linewidth=0.8, alpha=0.7
        )
        ax_col.annotate(f"{combo['angle_deg']:.1f}°",
                        (combo['c_b3_col'], combo['c_b3_row']),
                        color='white', fontsize=6,
                        xytext=(5, 5 + i * 14),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', fc=lc,
                                  alpha=0.6, ec='none'))

    if C_B3:
        ax_col.plot(C_B3[0], C_B3[1], 'o', color='limegreen',
                    markersize=8, markeredgecolor='white',
                    markeredgewidth=0.8, zorder=5)
    for c in C_B2:
        ax_col.plot(c[0], c[1], 'o', color='dodgerblue', markersize=6,
                    markeredgecolor='white', markeredgewidth=0.6)
    for c in C_B4:
        ax_col.plot(c[0], c[1], 'o', color='tomato', markersize=6,
                    markeredgecolor='white', markeredgewidth=0.6)

    n_pass = chip_combos_all['colinear_found'].sum()
    n_fail = (~chip_combos_all['colinear_found']).sum()
    ax_col.set_title(f"[Collinearity] {len(chip_combos_all)} combinations — "
                     f"{n_pass} pass (green), {n_fail} fail (red)", fontsize=8)
    ax_col.set_xticks([]); ax_col.set_yticks([])

    # Panel 2: angle distribution
    ax_ang = fig.add_subplot(gs[1, 2:])
    if not chip_combos_all.empty:
        passing_a = chip_combos_all[chip_combos_all['colinear_found'] == True]['angle_deg']
        failing_a = chip_combos_all[chip_combos_all['colinear_found'] == False]['angle_deg']
        if len(failing_a) > 0:
            ax_ang.scatter(range(len(failing_a)), sorted(failing_a),
                           color='tomato', s=40, label='Failed',
                           edgecolors='white', linewidths=0.3)
        if len(passing_a) > 0:
            ax_ang.scatter(range(len(passing_a)), sorted(passing_a),
                           color='limegreen', s=40, label='Passed',
                           edgecolors='white', linewidths=0.3)
        ax_ang.axhline(COLINEAR_ANGLE_MIN, color='gold', linestyle='--',
                       linewidth=1.5, label=f'{COLINEAR_ANGLE_MIN}° threshold')
        ax_ang.set_ylabel("Angle (°)")
        ax_ang.set_xlabel("Combination index")
        ax_ang.legend(fontsize=7)
    ax_ang.set_title("[Collinearity] Angles per combination", fontsize=8)

    # ── Row 3: Proportional displacement ─────────────────────────────────

    # Panel 1: true colour with confirmed triplet
    ax_disp = fig.add_subplot(gs[2, :2])
    ax_disp.imshow(rgb)
    _crosshair(ax_disp)

    if confirmed and not pd.isna(result.get('c_b2_col', float('nan'))):
        bx2, by2 = result['c_b2_col'], result['c_b2_row']
        bx3, by3 = result['c_b3_col'], result['c_b3_row']
        bx4, by4 = result['c_b4_col'], result['c_b4_row']
        ax_disp.plot(bx2, by2, 'o', color='dodgerblue', markersize=12,
                     markeredgecolor='white', markeredgewidth=1, label='B2')
        ax_disp.plot(bx3, by3, 'o', color='limegreen', markersize=12,
                     markeredgecolor='white', markeredgewidth=1, label='B3')
        ax_disp.plot(bx4, by4, 'o', color='tomato', markersize=12,
                     markeredgecolor='white', markeredgewidth=1, label='B4')
        ax_disp.plot([bx2, bx3, bx4], [by2, by3, by4], 'w--', linewidth=1.5)
        ax_disp.set_title(
            f"[Displacement] ✓ CONFIRMED — "
            f"{result['speed_kmh']:.0f} km/h  "
            f"angle={result['angle_deg']:.1f}°  "
            f"residual={result['residual_m']:.1f}m",
            fontsize=8, color='limegreen'
        )
    else:
        reason = ('no colinear combinations' if pd.isna(result.get('residual_m'))
                  else f"residual={result['residual_m']:.1f}m > {DISPLACEMENT_TOLERANCE}m")
        ax_disp.set_title(f"[Displacement] ✗ REJECTED — {reason}",
                          fontsize=8, color='tomato')

    ax_disp.legend(fontsize=7, loc='lower right')
    ax_disp.set_xticks([]); ax_disp.set_yticks([])

    # Panel 2: D_23 vs D_34 scatter
    ax_scat = fig.add_subplot(gs[2, 2:])

    if not chip_combos_pass.empty:
        d23_max   = chip_combos_pass['D_23'].max() * 1.2 + 10
        d23_range = np.linspace(0, d23_max, 100)
        d34_exp   = DISPLACEMENT_RATIO * d23_range + DISPLACEMENT_OFFSET
        ax_scat.plot(d23_range, d34_exp, color='gold', linewidth=1.5,
                     label='Expected')
        ax_scat.fill_between(d23_range,
                             d34_exp - DISPLACEMENT_TOLERANCE,
                             d34_exp + DISPLACEMENT_TOLERANCE,
                             alpha=0.15, color='gold',
                             label=f'±{DISPLACEMENT_TOLERANCE}m')

        colors_scat = ['limegreen' if p else 'tomato'
                       for p in chip_combos_pass['passes_disp']]
        ax_scat.scatter(chip_combos_pass['D_23'], chip_combos_pass['D_34'],
                        c=colors_scat, s=60, zorder=5,
                        edgecolors='white', linewidths=0.5)

        if confirmed and not pd.isna(result.get('D_Band2_3_m')):
            ax_scat.scatter(result['D_Band2_3_m'], result['D_Band3_4_m'],
                            color='white', s=150, zorder=6,
                            marker='*', label='Selected')

        ax_scat.set_xlabel("D_Band2_3 (m)")
        ax_scat.set_ylabel("D_Band3_4 (m)")
        ax_scat.legend(fontsize=7)

    ax_scat.set_title("[Displacement] Colinear combinations — "
                      "green=passes, red=fails, ★=selected", fontsize=8)

    # --- Figure title ---
    status = '✓ CONFIRMED' if confirmed else '✗ REJECTED'
    fig.suptitle(f"idx={idx:04d}  {Path(chip_path).name}  —  {status}",
                 fontsize=10, fontweight='bold')

    plt.show()




#--------------------
#simple map visual 
#--------------------

def plot_detection_map(aoi) -> None:
    """
    Plot confirmed aircraft detections within AOI bbox.
    No tile background — clean scatter on blank axis.
    Annotates each detection with its candidate_idx.
    """
    bbox = get_bbox(aoi)

    lon_min = bbox["aoi_lon_min"]
    lon_max = bbox["aoi_lon_max"]
    lat_min = bbox["aoi_lat_min"]
    lat_max = bbox["aoi_lat_max"]

    df_conf = pd.read_csv(Path(CONFIRMED_DIR) / "all_confirmed.csv")
    df_conf['confirmed'] = df_conf['confirmed'].astype(str).map(
        {'True': True, 'False': False}
    )
    confirmed = df_conf[df_conf['confirmed'] == True].copy()

    if confirmed.empty:
        print("No confirmed detections to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(
        confirmed['lon'], confirmed['lat'],
        c=confirmed['speed_kmh'], cmap='RdYlGn_r',
        s=60, zorder=5,
        edgecolors='white', linewidths=0.4
    )
    plt.colorbar(sc, ax=ax, label='Speed (km/h)')

    # Annotate each detection with candidate_idx
    for _, row in confirmed.iterrows():
        ax.annotate(
            f"{int(row['candidate_idx'])}",
            (row['lon'], row['lat']),
            fontsize=6, color='white',
            xytext=(4, 4), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.15', fc='black',
                      alpha=0.6, ec='none')
        )

    # Set extent from AOI bbox if provided
    if all(v is not None for v in [lon_min, lon_max, lat_min, lat_max]):
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

    # Draw AOI bounding box
    if all(v is not None for v in [lon_min, lon_max, lat_min, lat_max]):
        rect = plt.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=1.5, edgecolor='white',
            facecolor='none', linestyle='--',
            zorder=4
        )
        ax.add_patch(rect)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Confirmed aircraft detections (n={len(confirmed)})")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()