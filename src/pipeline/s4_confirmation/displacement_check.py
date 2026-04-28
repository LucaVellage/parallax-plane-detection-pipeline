import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipeline.config import (DISPLACEMENT_RATIO, 
                             DISPLACEMENT_OFFSET, 
                             DISPLACEMENT_TOLERANCE, 
                             METRES_PER_PIXEL, 
                             T_BAND2_3, 
                             COL_DIR, 
                             CONFIRMED_DIR, 
                             T_BAND2_4)


#placeholder until DB is built
REJECTED = {
    'confirmed':   False,
    'D_Band2_3_m': float('nan'),
    'D_Band3_4_m': float('nan'),
    'residual_m':  float('nan'),
    'angle_deg':   float('nan'),
    'speed_kmh':   float('nan'),
    'c_b2_col':    float('nan'), 'c_b2_row': float('nan'),
    'c_b3_col':    float('nan'), 'c_b3_row': float('nan'),
    'c_b4_col':    float('nan'), 'c_b4_row': float('nan'),
}


def _pixel_dist(col_a, row_a, col_b, row_b):
    """
    Euclidian distance in metres between pixel points
    TODO check other implementation
    """
    dx = (col_a - col_b) * METRES_PER_PIXEL
    dy = (row_a - row_b) * METRES_PER_PIXEL
    return math.sqrt(dx * dx + dy * dy)


def check_displacement(triplets):
    best = None
    best_residual = float('inf')
    best_overall = None

    for triplet in triplets:
        D_23     = _pixel_dist(triplet['c_b2_col'], triplet['c_b2_row'],
                                     triplet['c_b3_col'], triplet['c_b3_row'])
        D_34     = _pixel_dist(triplet['c_b3_col'], triplet['c_b3_row'],
                                     triplet['c_b4_col'], triplet['c_b4_row'])
        D_24 = _pixel_dist(triplet['c_b2_col'], triplet['c_b2_row'],
                           triplet['c_b4_col'], triplet['c_b4_row'])
        residual = abs(DISPLACEMENT_RATIO * D_23 + DISPLACEMENT_OFFSET - D_34)

        #tracking all best residuals even if rejected
        if residual < best_residual:
            best_residual = residual
            best_overall  = {
                'D_Band2_3_m': D_23,
                'D_Band3_4_m': D_34,
                'residual_m':  residual,
                'angle_deg':   triplet['angle_deg'],
                'speed_kmh':   (D_24 / T_BAND2_4) * 3.6,
                'c_b2_col':    triplet['c_b2_col'], 'c_b2_row': triplet['c_b2_row'],
                'c_b3_col':    triplet['c_b3_col'], 'c_b3_row': triplet['c_b3_row'],
                'c_b4_col':    triplet['c_b4_col'], 'c_b4_row': triplet['c_b4_row'],
            }

        #tracking best passing combination
        if residual <= DISPLACEMENT_TOLERANCE and residual < float('inf'):
            if best is None or residual < best['residual_m']:
                best = {'confirmed': True, **best_overall}

    if best is not None:
        return best
    
    if best_overall is not None:
        return {'confirmed': False, **best_overall}

    return dict(REJECTED)


#batch entry point 
def run_displacement_check():

    Path(CONFIRMED_DIR).mkdir(parents=True, exist_ok=True)

    in_path = Path(COL_DIR) / "all_collinearity.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Collinearity CSV not found: {in_path}")

    df   = pd.read_csv(in_path)
    n_chips_total          = df.groupby(['image_id', 'candidate_idx']).ngroups
    n_chips_with_colinear  = (df[df['colinear_found'] == True].groupby(['image_id', 'candidate_idx']).ngroups)

    rows = []

    for (image_id, candidate_idx), chip_df in df.groupby(['image_id', 'candidate_idx']):
            meta = chip_df.iloc[0][['tile', 'date', 'lon', 'lat', 'epsg']].to_dict()

            # only passed collinear triplet are passed 
            colinear = chip_df[chip_df['colinear_found'] == True].to_dict('records')

            result = check_displacement(colinear)

            rows.append({
                'image_id':      image_id,
                'candidate_idx': candidate_idx,
                **meta,
                **result,
            })

    result_df = pd.DataFrame(rows)
    n_confirmed_chips = int(result_df['confirmed'].sum())

    out_path = Path(CONFIRMED_DIR) / "all_confirmed.csv"
    result_df.to_csv(out_path, index=False)

    #summary
    print(f"\n{'=' * 40}")
    print(f"Displacement Check Summary")
    print(f"{'=' * 40}")
    print(f"Downloaded chips            : {n_chips_total}")
    print(f"Chips with collinear combos : {n_chips_with_colinear}")
    print(f"→ Confirmed                 : {n_confirmed_chips}")
    print(f"→ Rejected by displacement  : {n_chips_with_colinear - n_confirmed_chips}")
    print(f"Confirmed / downloaded chips: {100 * n_confirmed_chips / n_chips_total:.1f}%")

    print(f"\nPer-tile breakdown:")
    tile_summary = (
        result_df.groupby('tile')
        .agg(
            candidates=('confirmed', 'count'),
            confirmed =('confirmed', 'sum'),
        )
        .assign(rate_pct=lambda x: (
            x['confirmed'] / x['candidates'] * 100
        ).round(1))
    )
    print(tile_summary.to_string())

    confirmed_df = result_df[result_df['confirmed'] == True]
    if not confirmed_df.empty:
        speeds = confirmed_df['speed_kmh'].dropna()
        print(f"\nSpeed (confirmed aircraft):")
        print(f"  mean   : {speeds.mean():.0f} km/h")
        print(f"  median : {speeds.median():.0f} km/h")
        print(f"  min    : {speeds.min():.0f} km/h")
        print(f"  max    : {speeds.max():.0f} km/h")

    print(f"\nOutput written to : {out_path}")

    #return result_df



#relative tolerance: 

def compute_rel_error(confirmed_dir: str = CONFIRMED_DIR) -> pd.DataFrame:
    """
    Compute relative displacement error for all chips that reached
    the displacement check (i.e. have non-NaN residual).
    Adds D_34_expected and relative_error columns.
    Returns DataFrame of all chips with valid displacement metrics.
    """
    df_conf = pd.read_csv(Path(confirmed_dir) / "all_confirmed.csv")
    df_conf['confirmed'] = df_conf['confirmed'].astype(str).map(
        {'True': True, 'False': False}
    )
    # Only chips that reached the displacement check have non-NaN residuals
    df = df_conf[df_conf['residual_m'].notna()].copy()
    df['D_34_expected']  = DISPLACEMENT_RATIO * df['D_Band2_3_m']
    df['relative_error'] = (
        abs(df['D_Band3_4_m'] - df['D_34_expected'])
        / df['D_34_expected']
    )
    return df


def plot_rel_error(
    df:                  pd.DataFrame,
    candidate_threshold,
) -> None:
    """
    Plot relative error distribution for all chips that reached
    the displacement check, coloured by confirmed/rejected.

    Args:
        df                   : DataFrame from compute_rel_error()
        candidate_threshold  : vertical line showing candidate threshold
    """
    confirmed = df[df['confirmed'] == True]
    rejected  = df[df['confirmed'] == False]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # --- Left: full distribution coloured by confirmed/rejected ---
    axes[0].hist(rejected['relative_error'], bins=20,
                 color='tomato', edgecolor='white', linewidth=0.5,
                 alpha=0.7, label=f'Rejected (n={len(rejected)})')
    axes[0].hist(confirmed['relative_error'], bins=20,
                 color='limegreen', edgecolor='white', linewidth=0.5,
                 alpha=0.7, label=f'Confirmed (n={len(confirmed)})')
    axes[0].axvline(candidate_threshold, color='gold', linestyle='--',
                    linewidth=1.5,
                    label=f'candidate threshold {candidate_threshold:.0%}')
    axes[0].set_xlabel("Relative error")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Relative error — all chips reaching displacement check "
                      f"(n={len(df)})")
    axes[0].legend()

    # --- Right: relative error vs D_23 coloured by confirmed/rejected ---
    axes[1].scatter(rejected['D_Band2_3_m'], rejected['relative_error'],
                    color='tomato', s=30, alpha=0.7,
                    edgecolors='white', linewidths=0.3,
                    label=f'Rejected (n={len(rejected)})')
    axes[1].scatter(confirmed['D_Band2_3_m'], confirmed['relative_error'],
                    color='limegreen', s=30, alpha=0.7,
                    edgecolors='white', linewidths=0.3,
                    label=f'Confirmed (n={len(confirmed)})')
    axes[1].axhline(candidate_threshold, color='gold', linestyle='--',
                    linewidth=1.5,
                    label=f'candidate threshold {candidate_threshold:.0%}')
    axes[1].set_xlabel("D_Band2_3 (m)")
    axes[1].set_ylabel("Relative error")
    axes[1].set_title("Relative error vs displacement magnitude")
    axes[1].legend()

    #summary
    n_total  = len(confirmed)
    n_pass   = (confirmed['relative_error'] <= candidate_threshold).sum()
    n_reject = n_total - n_pass
    print(f"\nAmong confirmed detections at threshold {candidate_threshold:.0%}:")
    print(f"  Would keep   : {n_pass}   ({100*n_pass/n_total:.1f}%)")
    print(f"  Would reject : {n_reject} ({100*n_reject/n_total:.1f}%)")
    print(f"\nRelative error stats (confirmed):")
    print(f"  mean   : {confirmed['relative_error'].mean():.3f}")
    print(f"  median : {confirmed['relative_error'].median():.3f}")
    print(f"  max    : {confirmed['relative_error'].max():.3f}")
    
    plt.tight_layout()
    plt.show()


    
    








#____________debugging_______________
#inspect if calcs are correct 
def inspect_displacement_calc(idx: int) -> None:
    """
    Walk through the displacement calculation for one chip showing all
    intermediary results for every colinear combination.
    
    Args:
        idx : global candidate index
    """
    df_col = pd.read_csv(Path(COL_DIR) / "all_collinearity.csv")
    df_conf = pd.read_csv(Path(CONFIRMED_DIR) / "all_confirmed.csv")

    # --- Load confirmed result ---
    row = df_conf[df_conf['candidate_idx'] == idx].iloc[0]
    
    # --- Load all colinear combinations for this chip ---
    chip_combos = df_col[
        (df_col['candidate_idx'] == idx) &
        (df_col['colinear_found'] == True)
    ]
    
    print(f"{'=' * 60}")
    print(f"DISPLACEMENT CALCULATION — idx={idx:04d}")
    print(f"image_id : {row['image_id']}")
    print(f"{'=' * 60}")
    print(f"Colinear combinations to test: {len(chip_combos)}")
    print()

    best_residual = float('inf')
    best_combo_idx = None
    results = []

    for i, (_, combo) in enumerate(chip_combos.iterrows()):
        
        # --- Raw pixel positions ---
        b2 = (combo['c_b2_col'], combo['c_b2_row'])
        b3 = (combo['c_b3_col'], combo['c_b3_row'])
        b4 = (combo['c_b4_col'], combo['c_b4_row'])

        # --- Pixel distances ---
        dx_23 = b2[0] - b3[0]
        dy_23 = b2[1] - b3[1]
        dx_34 = b3[0] - b4[0]
        dy_34 = b3[1] - b4[1]

        d_23_px = math.sqrt(dx_23**2 + dy_23**2)
        d_34_px = math.sqrt(dx_34**2 + dy_34**2)

        # --- Metric distances ---
        D_23 = d_23_px * METRES_PER_PIXEL
        D_34 = d_34_px * METRES_PER_PIXEL

        # --- Expected D_34 ---
        D_34_expected = DISPLACEMENT_RATIO * D_23 + DISPLACEMENT_OFFSET

        # --- Absolute residual ---
        residual = abs(D_34_expected - D_34)

        # --- Relative error ---
        rel_error = abs(D_34 - DISPLACEMENT_RATIO * D_23) / (DISPLACEMENT_RATIO * D_23)

        # --- Speed ---
        speed_kmh = (D_23 / T_BAND2_3) * 3.6

        # --- Pass/fail ---
        passes_abs = residual <= DISPLACEMENT_TOLERANCE
        passes_rel = rel_error <= 0.15   # adjust threshold as needed

        results.append({
            'combo_idx':    i,
            'D_23':         D_23,
            'D_34':         D_34,
            'D_34_expected':D_34_expected,
            'residual':     residual,
            'rel_error':    rel_error,
            'speed_kmh':    speed_kmh,
            'passes_abs':   passes_abs,
            'passes_rel':   passes_rel,
            'passes_both':  passes_abs and passes_rel,
        })

        # Track best residual
        if passes_abs and residual < best_residual:
            best_residual  = residual
            best_combo_idx = i

        # --- Print intermediary results ---
        print(f"  Combo {i+1}  (angle={combo['angle_deg']:.1f}°)")
        print(f"  ├─ B2 pixel : ({b2[0]:.1f}, {b2[1]:.1f})")
        print(f"  ├─ B3 pixel : ({b3[0]:.1f}, {b3[1]:.1f})")
        print(f"  ├─ B4 pixel : ({b4[0]:.1f}, {b4[1]:.1f})")
        print(f"  ├─ dx_23={dx_23:.1f}px  dy_23={dy_23:.1f}px  "
              f"→ d_23={d_23_px:.2f}px = {D_23:.1f}m")
        print(f"  ├─ dx_34={dx_34:.1f}px  dy_34={dy_34:.1f}px  "
              f"→ d_34={d_34_px:.2f}px = {D_34:.1f}m")
        print(f"  ├─ D_34_expected = {DISPLACEMENT_RATIO} × {D_23:.1f} "
              f"+ {DISPLACEMENT_OFFSET} = {D_34_expected:.1f}m")
        print(f"  ├─ residual  = |{D_34_expected:.1f} - {D_34:.1f}| "
              f"= {residual:.1f}m  "
              f"[tol={DISPLACEMENT_TOLERANCE}m] "
              f"→ {'✓' if passes_abs else '✗'}")
        print(f"  ├─ rel_error = |{D_34:.1f} - {DISPLACEMENT_RATIO}×{D_23:.1f}| "
              f"/ {DISPLACEMENT_RATIO:.4f}×{D_23:.1f} "
              f"= {rel_error:.3f} ({rel_error*100:.1f}%)  "
              f"[tol=15%] "
              f"→ {'✓' if passes_rel else '✗'}")
        print(f"  ├─ speed     = {D_23:.1f} / {T_BAND2_3}s × 3.6 = {speed_kmh:.0f} km/h")
        print(f"  └─ PASSES (abs AND rel): {'✓ YES' if passes_abs and passes_rel else '✗ NO'}")
        if best_combo_idx == i:
            print(f"     ★ current best residual")
        print()

    # --- Final selection ---
    print(f"{'─' * 60}")
    print(f"FINAL RESULT (current implementation — abs residual only):")
    print(f"  confirmed      : {row['confirmed']}")
    if row['confirmed']:
        print(f"  selected combo : {best_combo_idx + 1 if best_combo_idx is not None else 'none'}")
        print(f"  D_Band2_3      : {row['D_Band2_3_m']:.1f}m")
        print(f"  D_Band3_4      : {row['D_Band3_4_m']:.1f}m")
        print(f"  residual       : {row['residual_m']:.1f}m")
        print(f"  speed          : {row['speed_kmh']:.0f} km/h")

    # --- What would change with relative error filter ---
    n_pass_both = sum(1 for r in results if r['passes_both'])
    print(f"\nIF relative error filter (15%) also applied:")
    print(f"  combinations passing both : {n_pass_both} / {len(results)}")
    print(f"  would be confirmed        : {n_pass_both > 0}")
    if n_pass_both > 0:
        best_both = min(
            [r for r in results if r['passes_both']],
            key=lambda r: r['residual']
        )
        print(f"  best combo speed          : {best_both['speed_kmh']:.0f} km/h")
        print(f"  best combo residual       : {best_both['residual']:.1f}m")
        print(f"  best combo rel_error      : {best_both['rel_error']*100:.1f}%")