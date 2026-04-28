"""
ground_truth.py
───────────────
Builds two reusable ground truth tables from manually annotated GeoPackages:

    1. ground_truth_visual.csv  — ALL pipeline + manual rows
                                  (source='pipeline' or 'manual', every row)
                                  is_visible label distinguishes TP/FP/FN:
                                    pipeline, is_visible=True   → TP (or TP1 if no ADS-B)
                                    pipeline, is_visible=False  → FP
                                    manual,   is_visible=True   → FN (missed detection)
                                  Used for: FP analysis, FN analysis, note_code breakdown

    2. ground_truth_adsb.csv    — all ADS-B records (visible or not)
                                  PRIMARY GROUND TRUTH for precision / recall:
                                    is_visible=True  → detectable aircraft (GT denominator)
                                    is_visible=False → not detectable (context)
                                  Used for: recall, detectable fraction

Background type encoding (background_type column, int):
    0 = outside bounds
    1 = water
    2 = vegetation
    3 = barren land
    4 = urban
    5 = snow

Note code encoding (extracted from notes column):
    FP1–FP10 : false positive reason codes
    FN1–FN6  : false negative reason codes
    TP1      : true positive with no ADS-B signal (untracked aircraft)

Entry point: run_build_ground_truth(annotations_dir)
"""

import re
import warnings
import json
import pandas as pd
import geopandas as gpd
import fiona
from pathlib import Path

from pipeline.config import (
    ANNOTATIONS_DIR,
    GROUND_TRUTH_DIR,
    TILE_CACHE_DIR,
)





# ── Constants ─────────────────────────────────────────────────────────────────

BG_TYPE_LABELS = {
    0: 'outside_bounds',
    1: 'water',
    2: 'vegetation',
    3: 'barren_land',
    4: 'urban',
    5: 'snow',
}

# All valid note codes
_NOTE_CODE_RE = re.compile(r'^(FP\d+|FN\d+|TP\d+)', re.IGNORECASE)

VISUAL_COLS = [
    'image_id', 'tile', 'date', 'background_type', 'cloud_pct',
    'source', 'lat', 'lon',
    'is_visible',
    'in_pipeline', 'in_adsb',
    'matched_callsign', 'matched_icao24', 'match_distance_m',
    'candidate_idx',
    'baroaltitude', 'speed_kmh', 'heading',
    'angle_deg', 'residual_m',
    'seamline_duplicate', 'near_bounds', 'outside_bounds',
    'note_code', 'notes',
]

ADSB_COLS = [
    'image_id', 'tile', 'date', 'background_type', 'cloud_pct',
    'icao24', 'callsign', 'matched_callsign',
    'lat', 'lon',
    'baroaltitude', 'speed_kmh', 'heading',
    'is_visible', 'is_flying', 'outside_bounds',
    'in_pipeline', 'candidate_idx',
    'match_distance_m',
    'note_code', 'notes',
]


# ── Private helpers ───────────────────────────────────────────────────────────

def _parse_note_code(note):
    """
    Extracts the FP/FN/TP code from the start of a notes string.
    Returns uppercased code string or None if absent.
    e.g. 'FP3: vehicle' → 'FP3'
         'FN2: aircraft visible' → 'FN2'
         None → None
    """
    if not note or pd.isna(note):
        return None
    m = _NOTE_CODE_RE.match(str(note).strip())
    return m.group(1).upper() if m else None


def _load_cache():
    """Loads tile params cache for cloud_pct lookup."""
    cache_path = Path(TILE_CACHE_DIR) / 'tile_params_cache.json'
    if not cache_path.exists():
        return {}
    with open(cache_path) as f:
        return json.load(f)


def _read_annotated_layer(f):
    """
    Reads the annotated layer from a GeoPackage file.
    Handles two naming conventions:
      1. {image_id}_eval     — imported from GeoJSON then saved in QGIS
      2. {image_id}_eval_ann — saved directly as GeoPackage in QGIS
    """
    layers   = fiona.listlayers(str(f))
    image_id = f.stem.replace('_eval_ann', '').replace('_eval', '')

    layers_lower       = {l.lower(): l for l in layers}
    candidate_eval     = f"{image_id}_eval_ann".lower()
    candidate_eval_alt = f"{image_id}_eval".lower()

    if candidate_eval in layers_lower:
        layer = layers_lower[candidate_eval]
    elif candidate_eval_alt in layers_lower:
        layer = layers_lower[candidate_eval_alt]
    elif layers:
        raise ValueError(
            f"No recognised layer found in {f.name}. "
            f"Expected '{image_id}_eval_ann'. Found: {layers}"
        )
    else:
        raise ValueError(f"No layers found in {f.name}")

    image_id = image_id.upper()

    print(f"  Reading layer: {layer}")
    return gpd.read_file(str(f), layer=layer), image_id


def _normalise(gdf, image_id, cache):
    """
    Normalises a single annotated GeoDataFrame:
    - fixes image_id on ADS-B rows
    - strips trailing spaces from string columns
    - coerces dtypes
    - derives lat/lon from geometry for manual points
    - coerces background_type to int
    - adds note_code column parsed from notes
    - adds missing optional columns
    - derives tile and date from image_id
    - adds cloud_pct from cache
    """
    gdf = gdf.copy()

    # Fix image_id — uppercase, fill NaN
    gdf['image_id'] = gdf['image_id'].fillna(
        image_id.upper().replace('_eval_ann', '')
    ).str.upper()

    # Strip trailing spaces from string columns
    for col in ['matched_callsign', 'matched_icao24',
                'icao24', 'callsign', 'source', 'image_id']:
        if col in gdf.columns and gdf[col].dtype == object:
            gdf[col] = gdf[col].str.strip().replace('', None)

    # Coerce dtypes
    gdf['match_distance_m'] = pd.to_numeric(
        gdf['match_distance_m'], errors='coerce'
    )
    gdf['candidate_idx'] = pd.to_numeric(
        gdf['candidate_idx'], errors='coerce'
    ).astype('Int64')

    # Coerce background_type to nullable int
    if 'background_type' in gdf.columns:
        gdf['background_type'] = pd.to_numeric(
            gdf['background_type'], errors='coerce'
        ).astype('Int64')
    else:
        gdf['background_type'] = pd.NA

    # Fix outside_bounds str → bool
    if 'outside_bounds' in gdf.columns:
        if gdf['outside_bounds'].dtype == object:
            gdf['outside_bounds'] = gdf['outside_bounds'].map(
                {'True': True, 'False': False,
                 True: True, False: False}
            )

    # Add missing optional columns
    for col in ['seamline_duplicate', 'near_bounds',
                'is_flying', 'callsign', 'angle_deg', 'residual_m']:
        if col not in gdf.columns:
            gdf[col] = None

    # Parse note_code from notes column
    gdf['note_code'] = gdf['notes'].apply(_parse_note_code)

    # Derive lat/lon from geometry for manual points
    mask = gdf['lat'].isna() | gdf['lon'].isna()
    if mask.any():
        gdf.loc[mask, 'lon'] = gdf.loc[mask, 'geometry'].x
        gdf.loc[mask, 'lat'] = gdf.loc[mask, 'geometry'].y

    # Derive tile and date from image_id
    # image_id format: YYYYMMDD_TILE e.g. 20190705_31UDQ
    parts       = gdf['image_id'].iloc[0].split('_')
    gdf['date'] = parts[0]
    gdf['tile'] = parts[1] if len(parts) > 1 else None

    # Add cloud_pct from cache
    img_id           = gdf['image_id'].iloc[0]
    cached           = cache.get(img_id, {})
    gdf['cloud_pct'] = gdf.get('cloud_pct', cached.get('cloud_pct'))

    return gdf


def _load_annotated_files(annotations_dir):
    """
    Reads all *_eval_ann.gpkg files from annotations directory.
    Returns one concatenated GeoDataFrame of all annotated scenes.
    """
    ann_dir = Path(annotations_dir)
    files   = sorted(ann_dir.glob('*_eval_ann.gpkg'))

    if not files:
        raise FileNotFoundError(
            f"No *_eval_ann.gpkg files found in {annotations_dir}"
        )

    cache = _load_cache()
    print(f"Found {len(files)} annotated files:")
    gdfs  = []

    for f in files:
        gdf, image_id = _read_annotated_layer(f)
        gdf           = _normalise(gdf, image_id, cache)

        n_pipeline  = (gdf['source'] == 'pipeline').sum()
        n_adsb      = (gdf['source'] == 'adsb').sum()
        n_manual    = (gdf['source'] == 'manual').sum()
        n_visible   = gdf['is_visible'].sum()
        n_inspected = gdf['inspected'].sum() if 'inspected' in gdf.columns else 'N/A'

        print(f"  {f.name}")
        print(f"    rows     : {len(gdf)} "
              f"(pipeline={n_pipeline}, adsb={n_adsb}, manual={n_manual})")
        print(f"    inspected: {n_inspected}/{len(gdf)}")
        print(f"    visible  : {n_visible}/{len(gdf)}")

        gdfs.append(gdf)

    combined = pd.concat(gdfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows across "
          f"{combined['image_id'].nunique()} scenes\n")
    return combined


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_background_completeness(gdf):
    """
    Safeguard 1: All rows must have a non-null background_type.
    Rows with background_type=0 (outside bounds) are valid but noted.

    Raises ValueError listing every offending row so
    the user can return to QGIS and assign background_type before
    running again.
    """
    missing = gdf['background_type'].isna()
    outside = (gdf['background_type'] == 0)

    if missing.any():
        lines = [
            f"Safeguard 1 — Missing background_type: "
            f"{missing.sum()} row(s) have no background_type assigned.\n"
            f"  → Fix in QGIS, then re-run.\n"
        ]
        for _, row in gdf[missing].iterrows():
            lines.append(
                f"  {row['image_id']}  source={row['source']}  "
                f"callsign={row.get('matched_callsign')}  "
                f"candidate_idx={row.get('candidate_idx')}"
            )
        raise ValueError("\n".join(lines))

    print(f"Safeguard 1 — Background completeness: OK "
          f"(all {len(gdf)} rows have background_type).")

    if outside.any():
        print(f"  Note: {outside.sum()} row(s) have background_type=0 "
              f"(outside bounds) — excluded from stratified metrics.")


def _validate_background_consistency(gdf):
    """
    Safeguard 2: For matched pairs (same image_id + matched_callsign),
    the background_type of the detection (pipeline or manual) must prevail
    over the ADS-B row's background_type.

    Overrides ADS-B background_type in-place where mismatches are found.
    Returns the corrected GeoDataFrame.
    """
    gdf = gdf.copy()

    # Only operate on rows that have a matched_callsign
    det_rows = gdf[
        gdf['source'].isin(['pipeline', 'manual']) &
        gdf['matched_callsign'].notna()
    ]

    overrides = 0
    for _, det_row in det_rows.iterrows():
        adsb_mask = (
            (gdf['source'] == 'adsb') &
            (gdf['image_id'] == det_row['image_id']) &
            (gdf['matched_callsign'] == det_row['matched_callsign'])
        )
        for idx in gdf[adsb_mask].index:
            adsb_bg = gdf.at[idx, 'background_type']
            det_bg  = det_row['background_type']
            if pd.isna(det_bg):
                continue
            if adsb_bg != det_bg:
                print(f"  BACKGROUND OVERRIDE: {det_row['image_id']} "
                      f"callsign={det_row['matched_callsign']} "
                      f"ADS-B bg={adsb_bg} "
                      f"→ {det_bg} "
                      f"(detection source={det_row['source']})")
                gdf.at[idx, 'background_type'] = det_bg
                overrides += 1

    if overrides:
        print(f"Background consistency: {overrides} ADS-B background_type(s) "
              f"overridden by detection value.\n")
    else:
        print(f"Background consistency: OK — all matched pairs agree.\n")

    return gdf


def _validate_matched_are_visible(gdf):
    """
    Safeguard 3: All rows with matched=True must have is_visible=True.
    A matched row that is not visible is a labelling inconsistency —
    you cannot match something that isn't there.

    Raises ValueError listing every offending row.
    """
    if 'matched' not in gdf.columns:
        print("Safeguard 3 — 'matched' column not found; skipping.")
        return

    violations = gdf[
        (gdf['matched'] == True) & (gdf['is_visible'] == False)
    ]

    if not violations.empty:
        lines = [
            f"Safeguard 3 — Matched-but-not-visible: "
            f"{len(violations)} row(s) have matched=True but is_visible=False.\n"
            f"  → Fix in QGIS (un-match or mark visible), then re-run.\n"
        ]
        for _, row in violations.iterrows():
            lines.append(
                f"  {row['image_id']}  source={row['source']}  "
                f"callsign={row.get('matched_callsign')}  "
                f"candidate_idx={row.get('candidate_idx')}  "
                f"notes={row.get('notes')}"
            )
        raise ValueError("\n".join(lines))

    print("Safeguard 3 — Matched-visible check: OK "
          "(all matched rows are is_visible=True).")


def _validate_adsb_has_counterpart(gdf):
    """
    Safeguard 4: Every ADS-B row with matched=True must have a corresponding
    pipeline or manual row with the same image_id + matched_callsign.

    Raises ValueError listing every orphaned ADS-B record.
    """
    if 'matched' not in gdf.columns:
        adsb_to_check = gdf[
            (gdf['source'] == 'adsb') & (gdf['is_visible'] == True)
        ]
    else:
        adsb_to_check = gdf[
            (gdf['source'] == 'adsb') & (gdf['matched'] == True)
        ]

    orphaned = []
    for _, row in adsb_to_check.iterrows():
        callsign = row['matched_callsign']
        image_id = row['image_id']

        if pd.isna(callsign):
            continue

        has_pipeline = (
            (gdf['source'] == 'pipeline') &
            (gdf['image_id'] == image_id) &
            (gdf['matched_callsign'] == callsign)
        ).any()

        has_manual = (
            (gdf['source'] == 'manual') &
            (gdf['image_id'] == image_id) &
            (gdf['matched_callsign'] == callsign)
        ).any()

        if not has_pipeline and not has_manual:
            orphaned.append({
                'image_id': image_id,
                'callsign': callsign,
                'icao24'  : row.get('icao24'),
            })

    if orphaned:
        lines = [
            f"Safeguard 4 — ADS-B without counterpart: "
            f"{len(orphaned)} matched ADS-B record(s) have no pipeline "
            f"or manual row.\n"
            f"  → Add a manual point in QGIS for each, then re-run.\n"
        ]
        for o in orphaned:
            lines.append(
                f"  {o['image_id']}  callsign={o['callsign']}  "
                f"icao24={o['icao24']}"
            )
        raise ValueError("\n".join(lines))

    print("Safeguard 4 — ADS-B counterpart check: OK "
          "(all matched ADS-B records have a pipeline or manual row).")


def _run_all_validations(gdf):
    """
    Runs all four safeguard validations in sequence.

    Safeguards 1, 3, 4 raise ValueError immediately on
    any violation, listing every offending row so the user can return
    to QGIS and fix before re-running.

    Safeguard 2 (background consistency) silently corrects ADS-B
    background_type to match the detection's value and logs each override.

    Returns the (possibly corrected) GeoDataFrame.
    """
    print(f"\n── Validation ─────────────────────────────────────────────\n")

    _validate_background_completeness(gdf)        # raises if missing
    gdf = _validate_background_consistency(gdf)   # corrects in-place
    _validate_matched_are_visible(gdf)            # raises if violated
    _validate_adsb_has_counterpart(gdf)           # raises if orphaned

    print(f"\n── Validation complete: all checks passed. ──\n")
    return gdf


# ── Ground truth table builders ───────────────────────────────────────────────

def _build_visual_gt(gdf):
    """
    Builds ground_truth_visual — ALL pipeline and manual rows.

    Includes:
        source='pipeline', is_visible=True   → TP (detected, visible)
        source='pipeline', is_visible=False  → FP (detected, not visible)
        source='pipeline', note_code='TP1'   → untracked (visible, no ADS-B)
        source='manual',   is_visible=True   → FN (missed, visible)

    The is_visible and note_code columns distinguish these cases.
    Excludes ADS-B rows — those go in ground_truth_adsb.
    """
    visual_gt = gdf[
        gdf['source'].isin(['pipeline', 'manual'])
    ].copy()

    cols = [c for c in VISUAL_COLS if c in visual_gt.columns]
    visual_gt = visual_gt[cols].reset_index(drop=True)

    visual_gt.insert(0, 'gt_id', [
        f"gt_vis_{i:04d}" for i in range(len(visual_gt))
    ])

    return visual_gt


def _build_adsb_gt(gdf):
    """
    Builds ground_truth_adsb — all ADS-B records.

    PRIMARY GROUND TRUTH for recall:
        is_visible=True  → detectable aircraft (denominator for recall)
        is_visible=False → not detectable

    background_type on ADS-B rows has already been corrected by
    _validate_background_consistency() to match the detection's value.
    """
    adsb_gt = gdf[gdf['source'] == 'adsb'].copy()

    # Deduplicate column list (ADSB_COLS may repeat matched_callsign)
    seen = set()
    cols = [c for c in ADSB_COLS
            if c in adsb_gt.columns and not (c in seen or seen.add(c))]
    adsb_gt = adsb_gt[cols].reset_index(drop=True)

    adsb_gt.insert(0, 'gt_id', [
        f"gt_adsb_{i:04d}" for i in range(len(adsb_gt))
    ])

    return adsb_gt


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(visual_gt, adsb_gt):
    """Prints summary statistics stratified by background_type."""
    print("── Ground Truth Summary ──────────────────────────────────────\n")

    # ── Visual GT (pipeline + manual) ────────────────────────────────────
    pipeline_rows = visual_gt[visual_gt['source'] == 'pipeline']
    manual_rows   = visual_gt[visual_gt['source'] == 'manual']
    tp_rows       = pipeline_rows[
        (pipeline_rows['is_visible'] == True) &
        (pipeline_rows['note_code'] != 'TP1')
    ]
    fp_rows       = pipeline_rows[pipeline_rows['is_visible'] == False]
    tp1_rows      = pipeline_rows[pipeline_rows['note_code'] == 'TP1']

    print("Visual ground truth (pipeline + manual rows):")
    print(f"  Total pipeline detections  : {len(pipeline_rows)}")
    print(f"    TP (ADS-B matched)       : {len(tp_rows)}")
    print(f"    TP1 (no ADS-B, visible)  : {len(tp1_rows)}")
    print(f"    FP (not visible)         : {len(fp_rows)}")
    print(f"  Manual (FN) rows           : {len(manual_rows)}")

    if 'note_code' in visual_gt.columns:
        fp_codes = fp_rows['note_code'].value_counts()
        fn_codes = manual_rows['note_code'].value_counts()
        if len(fp_codes):
            print(f"\n  FP breakdown by code:")
            for code, n in fp_codes.items():
                print(f"    {code}: {n}")
        if len(fn_codes):
            print(f"\n  FN breakdown by code:")
            for code, n in fn_codes.items():
                print(f"    {code}: {n}")

    # ── ADS-B GT ──────────────────────────────────────────────────────────
    print(f"\nADS-B ground truth (primary recall denominator):")
    print(f"  Total ADS-B records        : {len(adsb_gt)}")
    print(f"  Visually detectable        : "
          f"{(adsb_gt['is_visible'] == True).sum()}")
    print(f"  Not detectable             : "
          f"{(adsb_gt['is_visible'] == False).sum()}")

    n_vis = (adsb_gt['is_visible'] == True).sum()
    frac  = n_vis / len(adsb_gt) * 100 if len(adsb_gt) > 0 else 0
    print(f"  Detectable fraction        : {frac:.1f}%")
    print(f"  Detected by pipeline       : "
          f"{(adsb_gt['in_pipeline'] == True).sum()}")

    # ── Background breakdown ──────────────────────────────────────────────
    print(f"\nADS-B visible records by background_type:")
    visible_adsb = adsb_gt[adsb_gt['is_visible'] == True]
    for bg_int, bg_label in BG_TYPE_LABELS.items():
        if bg_int == 0:
            continue
        n_total   = (visible_adsb['background_type'] == bg_int).sum()
        n_detected = (
            (visible_adsb['background_type'] == bg_int) &
            (visible_adsb['in_pipeline'] == True)
        ).sum()
        if n_total > 0:
            print(f"  {bg_label:<20}: {n_total:>4} visible, "
                  f"{n_detected:>4} detected "
                  f"({n_detected/n_total*100:.0f}%)")


# ── Public entry point ────────────────────────────────────────────────────────

def run_build_ground_truth(annotations_dir=ANNOTATIONS_DIR):
    """
    Builds two reusable ground truth tables from annotated GeoPackages.

    Changes from previous version:
    - background_map parameter removed: background_type is now per-detection
      (int column in GeoPackage, annotated in QGIS).
    - ground_truth_visual now includes ALL pipeline + manual rows (not just
      is_visible=True), enabling FP analysis alongside TP/FN analysis.
    - ground_truth_adsb is the PRIMARY ground truth for precision/recall:
      visible ADS-B records (is_visible=True) are the recall denominator.
    - Four validation safeguards run before building tables:
        1. Background completeness (all rows have background_type)
        2. Background consistency (detection bg_type overrides ADS-B)
        3. Matched rows must be visible
        4. Matched ADS-B records must have a pipeline or manual counterpart

    Args:
        annotations_dir : directory containing *_eval_ann.gpkg files
                          (default: ANNOTATIONS_DIR from config)

    Returns:
        visual_gt : DataFrame — all pipeline + manual rows with labels
        adsb_gt   : DataFrame — all ADS-B records with is_visible flag
    """
    print(f"\n{'─'*55}")
    print(f"Building Ground Truth Tables")
    print(f"{'─'*55}\n")

    # Load all annotated files
    gdf = _load_annotated_files(annotations_dir)

    # Run all safeguard validations (background consistency corrects in-place)
    gdf = _run_all_validations(gdf)

    # Build tables
    visual_gt = _build_visual_gt(gdf)
    adsb_gt   = _build_adsb_gt(gdf)

    # Save
    out_dir = Path(GROUND_TRUTH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    visual_path = out_dir / 'ground_truth_visual.csv'
    adsb_path   = out_dir / 'ground_truth_adsb.csv'

    visual_gt.to_csv(visual_path, index=False)
    adsb_gt.to_csv(adsb_path,    index=False)

    print(f"Saved ground_truth_visual.csv → {visual_path}")
    print(f"Saved ground_truth_adsb.csv   → {adsb_path}\n")

    # Print summary
    _print_summary(visual_gt, adsb_gt)

    return visual_gt, adsb_gt