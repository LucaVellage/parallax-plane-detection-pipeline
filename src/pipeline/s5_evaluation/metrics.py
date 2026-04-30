"""
Script computes performance metrics based on ground truth and detection results

Primary ground truth: ground_truth_adsb.csv, is_visible=True rows.
Predictions: all confirmed pipeline detections from ground_truth_visual.csv
  — TP  : pipeline row, is_visible=True, in_adsb=True
  — FP  : pipeline row, is_visible=False (FPx note code)
  — TP1 : pipeline row, is_visible=True, in_adsb=False (untracked aircraft)
  — FN  : ADS-B visible row, in_pipeline=False (missed by pipeline)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.config import (
    GROUND_TRUTH_DIR,
    ADSB_MATCH_THRESHOLD_M,
)

BG_TYPE_LABELS = {
    1: 'Water',
    2: 'Vegetation',
    3: 'Bare land',
    4: 'Urban',
    5: 'Snow',
}

BG_ORDER = [1, 2, 3, 4, 5]



def _load_ground_truth():

    gt_dir = Path(GROUND_TRUTH_DIR)

    visual_path = gt_dir / 'ground_truth_visual.csv'
    adsb_path = gt_dir / 'ground_truth_adsb.csv'

    for p in [visual_path, adsb_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p.name} not found in {gt_dir}. "
                "Run ground_truth.run_build_ground_truth() first."
            )

    visual_gt = pd.read_csv(visual_path)
    adsb_gt   = pd.read_csv(adsb_path)

    missing_bg = visual_gt['background_type'].isna().sum()
    if missing_bg:
        print(f"WARNING: {missing_bg} visual GT row(s) have no background_type "
              f"they will be excluded from background-stratified metrics.")

    missing_pos = visual_gt['lat'].isna() | visual_gt['lon'].isna()
    if missing_pos.any():
        print(f"WARNING: {missing_pos.sum()} visual GT point(s) have no lat/lon — "
              f"excluded. Check manual points in annotations.")
        visual_gt = visual_gt[~missing_pos].copy()

    #print(f"\nGround truth loaded:")
    #print(f"  Visual GT rows         : {len(visual_gt)}")
    #pipeline_rows = visual_gt[visual_gt['source'] == 'pipeline']
    #manual_rows   = visual_gt[visual_gt['source'] == 'manual']
    #print(f"    pipeline (TP+FP+TP1) : {len(pipeline_rows)}")
    #print(f"      is_visible=True    : {(pipeline_rows['is_visible'] == True).sum()}")
    #print(f"      is_visible=False   : {(pipeline_rows['is_visible'] == False).sum()}")
    #print(f"    manual (FN)          : {len(manual_rows)}")
    #print(f"  ADS-B GT records       : {len(adsb_gt)}")
    #n_vis = (adsb_gt['is_visible'] == True).sum()
    #print(f"    is_visible=True      : {n_vis}  ← recall denominator")
    #print(f"    is_visible=False     : {(adsb_gt['is_visible'] == False).sum()}")

    return visual_gt, adsb_gt


def _compute_counts(visual_gt, adsb_gt, bg_filter=None):
    """
    Computes TP, FP, FN, TP1 counts from pre-annotated ground truth tables.

    No spatial re-matching is performed — the in_pipeline and is_visible
    flags set during QGIS annotation are authoritative.

    Args:
        visual_gt  : ground_truth_visual DataFrame
        adsb_gt    : ground_truth_adsb DataFrame
        bg_filter  : int or None — if set, restrict to this background_type

    Returns dict with TP, FP, FN, TP1, n_adsb_total, n_adsb_visible
    """
    vg = visual_gt.copy()
    ag = adsb_gt.copy()

    if bg_filter is not None:
        vg = vg[vg['background_type'] == bg_filter]
        ag = ag[ag['background_type'] == bg_filter]
    else:
        # outside bounds excluded
        vg = vg[vg['background_type'] != 0]
        ag = ag[ag['background_type'] != 0]

    pipeline_rows = vg[vg['source'] == 'pipeline']

    # TP: pipeline detected, visible + matched to ADS-B
    tp  = (
        (pipeline_rows['is_visible'] == True) &
        (pipeline_rows['in_adsb']    == True)
    ).sum()

    # FP: pipeline detected, not visible 
    fp  = (pipeline_rows['is_visible'] == False).sum()

    # TP1: pipeline detected, visible, no ADS-B signal (untracked))
    tp1 = (
        (pipeline_rows['is_visible'] == True) &
        (pipeline_rows['in_adsb']    == False)
    ).sum()

    # FN: visible ADS-B records missed by pipeline
    adsb_visible = ag[ag['is_visible'] == True]
    fn = (adsb_visible['in_pipeline'] == False).sum()

    n_adsb_total   = len(ag)
    n_adsb_visible = len(adsb_visible)

    return dict(
        TP=int(tp), FP=int(fp), FN=int(fn), TP1=int(tp1),
        n_adsb_total=n_adsb_total,
        n_adsb_visible=n_adsb_visible,
    )


def _derive_metrics(counts):
    tp, fp, fn, tp1 = counts['TP'], counts['FP'], counts['FN'], counts['TP1']

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1        = (2 * precision * recall / (precision + recall)
                 if not np.isnan(precision) and not np.isnan(recall)
                 and (precision + recall) > 0 else np.nan)
    fpr       = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    fnr       = fn / (tp + fn) if (tp + fn) > 0 else np.nan
    det_frac  = (counts['n_adsb_visible'] / counts['n_adsb_total']
                 if counts['n_adsb_total'] > 0 else np.nan)
    tp1_pct   = tp1 / (tp + tp1) if (tp + tp1) > 0 else np.nan

    return dict(
        precision=_r(precision), recall=_r(recall), f1=_r(f1),
        fpr=_r(fpr), fnr=_r(fnr),
        detectable_fraction=_r(det_frac),
        tp1_pct=_r(tp1_pct),
    )

def _r(v):
    return round(float(v), 4) if not (v is None or (isinstance(v, float) and np.isnan(v))) else None


def _build_metrics_table(visual_gt, adsb_gt):

    rows = []

    for bg_int in BG_ORDER:
        counts = _compute_counts(visual_gt, adsb_gt, bg_filter=bg_int)
        if counts['n_adsb_visible'] == 0 and counts['TP'] + counts['FP'] == 0:
            # skip when background has no data
            continue
        metrics = _derive_metrics(counts)
        rows.append({
            'background_type'     : bg_int,
            'background_label'    : BG_TYPE_LABELS.get(bg_int, str(bg_int)),
            **counts,
            **metrics,
        })

    agg_counts = _compute_counts(visual_gt, adsb_gt, bg_filter=None)
    agg_metrics = _derive_metrics(agg_counts)
    rows.append({
        'background_type'  : -1,
        'background_label' : 'Overall',
        **agg_counts,
        **agg_metrics,
    })

    return pd.DataFrame(rows)


def _build_scene_metrics(visual_gt, adsb_gt):
    """
    Builds per-scene metrics table 

    """
    rows = []
    all_image_ids = sorted(set(
        visual_gt['image_id'].dropna().unique().tolist() +
        adsb_gt['image_id'].dropna().unique().tolist()
    ))

    for image_id in all_image_ids:
        vg_scene = visual_gt[visual_gt['image_id'] == image_id]
        ag_scene = adsb_gt[adsb_gt['image_id']   == image_id]

        counts  = _compute_counts(vg_scene, ag_scene, bg_filter=None)
        metrics = _derive_metrics(counts)

        pip_scene = vg_scene[
            (vg_scene['source'] == 'pipeline') &
            (vg_scene['background_type'] != 0)
        ]
        modal_bg = (
            pip_scene['background_type'].mode().iloc[0]
            if not pip_scene.empty and not pip_scene['background_type'].isna().all()
            else None
        )

        cloud_pct = (
            vg_scene['cloud_pct'].iloc[0]
            if 'cloud_pct' in vg_scene.columns and not vg_scene.empty
            else None
        )

        rows.append({
            'image_id'        : image_id,
            'tile'            : image_id.split('_')[1] if '_' in image_id else None,
            'date'            : image_id.split('_')[0] if '_' in image_id else None,
            'cloud_pct'       : cloud_pct,
            'modal_background': modal_bg,
            **counts,
            **metrics,
        })

    return pd.DataFrame(rows)


def _build_detection_results(visual_gt, adsb_gt):
    """
    Builds detection results 
    """
    pipeline = visual_gt[
        visual_gt['source'] == 'pipeline'
    ].copy()

    pipeline['detection_type'] = np.where(
        pipeline['is_visible'] == False, 'FP',
        np.where(
            (pipeline['is_visible'] == True) & (pipeline['in_adsb'] == True),
            'TP',
            'TP1'
        )
    )

    return pipeline.reset_index(drop=True)


def _build_gt_match_results(adsb_gt):
    """
    gt match results 
    """
    visible = adsb_gt[adsb_gt['is_visible'] == True].copy()
    visible['match_type'] = np.where(
        visible['in_pipeline'] == True, 'TP', 'FN'
    )
    return visible.reset_index(drop=True)


def _print_summary(metrics_df, scene_metrics_df):
  
    print(f"{'='*40}\n")
    print(f"Metrics Summary (primary GT: visible ADS-B records)")
    print(f"{'='*40}\n")

    print(f"{'Background':<20} {'TP':>5} {'FP':>5} {'FN':>5} {'TP1':>5} "
      f"{'TP1%':>6} {'Precision':>10} {'Recall':>8} {'F1':>8} {'DetFrac':>8}")
    print("─" * 85)

    for _, row in metrics_df.iterrows():
        print(f"{row['background_label']:<20} "
            f"{int(row['TP']):>5} "
            f"{int(row['FP']):>5} "
            f"{int(row['FN']):>5} "
            f"{int(row['TP1']):>5} "
            f"{_fmt(row['tp1_pct']):>6} "
            f"{_fmt(row['precision']):>10} "
            f"{_fmt(row['recall']):>8} "
            f"{_fmt(row['f1']):>8} "
            f"{_fmt(row['detectable_fraction']):>8}")

    #print(f"\nPer-scene breakdown ({len(scene_metrics_df)} scenes):")
    #print(f"{'image_id':<25} {'BG':>3} {'TP':>5} {'FP':>5} {'FN':>5} "
    #      f"{'TP1':>5} {'P':>7} {'R':>7} {'F1':>7}")
    #print("=" * 40)

    #for _, row in scene_metrics_df.iterrows():
    #    print(f"{str(row['image_id']):<25} "
    #          f"{str(row['modal_background'] or '?'):>3} "
    #          f"{int(row['TP']):>5} "
    #          f"{int(row['FP']):>5} "
    #          f"{int(row['FN']):>5} "
    #          f"{int(row['TP1']):>5} "
    #          f"{_fmt(row['precision']):>7} "
    #          f"{_fmt(row['recall']):>7} "
    #          f"{_fmt(row['f1']):>7}")


def _fmt(v):
    """Format a metric value for display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '   -'
    return f"{v:.3f}"


# ── Public entry point ────────────────────────────────────────────────────────

def run_metrics():
    """
    Computes detection performance metrics using visible ADS-B records as gt denominator

    Metric definitions:
        TP: pipeline detection, visible, ADS-B matched
        FP: pipeline detection, not visible (annotated FPx)
        FN: visible ADS-B record, not detected by pipeline
        TP1: pipeline detection, visible, no ADS-B signal (untracked)
              reported separately; excluded from FP count

        metrics_df       : df by-background metrics + overall
        scene_metrics_df : df per-scene metrics
        det_results_df   : df per-detection classification (TP/FP/TP1)
        gt_results_df    : dfper-ADS-B-visible match result (TP/FN)
    """
    print(f"\n{'='*40}")
    print(f"Step 5: Metrics")
    print(f"{'='*40}\n")

    visual_gt, adsb_gt = _load_ground_truth()

    annotated_scenes = set(visual_gt['image_id'].dropna().unique())
    adsb_gt = adsb_gt[adsb_gt['image_id'].isin(annotated_scenes)].copy()

    #print(f"\nBuilding metrics tables...\n")

    metrics_df = _build_metrics_table(visual_gt, adsb_gt)
    scene_metrics_df = _build_scene_metrics(visual_gt, adsb_gt)
    det_results_df = _build_detection_results(visual_gt, adsb_gt)
    gt_results_df = _build_gt_match_results(adsb_gt)

    # Save
    out_dir = Path(GROUND_TRUTH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(out_dir / 'metrics_by_background.csv', index=False)
    scene_metrics_df.to_csv(out_dir / 'metrics_by_scene.csv', index=False)
    det_results_df.to_csv(out_dir / 'detection_results.csv', index=False)
    gt_results_df.to_csv(out_dir / 'gt_match_results.csv', index=False)

    print(f"Saved to {out_dir}:")
    print(f"  metrics_by_background.csv  ← primary metrics table")
    print(f"  metrics_by_scene.csv       ← per-scene breakdown")
    print(f"  detection_results.csv      ← per-detection TP/FP/TP1 with bg_type")
    print(f"  gt_match_results.csv       ← per-ADS-B-visible TP/FN\n")

    _print_summary(metrics_df, scene_metrics_df)

    return metrics_df, scene_metrics_df, det_results_df, gt_results_df
