"""
evaluate.py
───────────
Orchestrates the full Step 5 evaluation pipeline for all tile/date
combinations found in the detections CSV.

Entry point: run_evaluation(detections_path)
"""

import pandas as pd
from pathlib import Path

from pipeline.s5_evaluation.tile_params  import get_cached_tile_params
from pipeline.s5_evaluation.adsb_query   import run_adsb_query
from pipeline.s5_evaluation.adsb_filter  import run_adsb_filter
from pipeline.s5_evaluation.adsb_match   import run_adsb_match
from pipeline.config import CONFIRMED_DIR


def _load_detections():
    """
    Loads the combined detections CSV and filters to confirmed detections.
    Normalises tile column by stripping leading T.
    """
    df = pd.read_csv(Path(CONFIRMED_DIR) / "all_confirmed.csv")
    
    # Normalise date — cast int to string first, then format
    df['date'] = pd.to_datetime(
        df['date'].astype(str), format='%Y%m%d'
    ).dt.strftime('%Y-%m-%d')

    # Normalise tile — strip leading T
    df['tile'] = df['tile'].str.lstrip('T')

    # Reconstruct image_id to match normalised tile (without T)
    df['image_id'] = df['date'].str.replace('-', '') + '_' + df['tile']

    # Filter to confirmed detections only
    df_confirmed = df[df['confirmed'] == True].copy()

    print(f"Total candidates  : {len(df):,}")
    print(f"Confirmed         : {len(df_confirmed):,}")
    print(f"Unique tile/dates : "
          f"{df_confirmed[['tile', 'date']].drop_duplicates().shape[0]}")

    return df_confirmed


def _get_unique_pairs(df):
    """Returns list of unique (tile, date) tuples from detections DataFrame."""
    return (
        df[['tile', 'date']]
        .drop_duplicates()
        .values.tolist()
    )


def _run_single(tile, date, df_all_detections):
    """
    Runs the full evaluation pipeline for a single tile/date combination.
    Returns a GeoDataFrame with matched results.
    """
    # tile params 
    params = get_cached_tile_params(tile, date)

    # DEBUG
    print(f"Looking for image_id: {params['image_id']}")
    print(f"Available image_ids: {df_all_detections['image_id'].unique()}")
    print(f"Columns available: {df_all_detections.columns.tolist()}")

    # filters detections to this date + tile
    df_detections = df_all_detections[
        df_all_detections['image_id'] == params['image_id']
    ].copy()

    if df_detections.empty:
        print(f"No confirmed detections for {params['image_id']} — skipping")
        return None

    # runs cross matching
    df_adsb_raw      = run_adsb_query(params)
    df_adsb_filtered = run_adsb_filter(df_adsb_raw, params)
    gdf_matched = run_adsb_match(df_detections, df_adsb_filtered, params)

    return gdf_matched


# public entry point

def run_evaluation():
    """
    Runs the full Step 5 evaluation for all tile/date combinations
    found in the detections CSV.

    Args:
        detections_path : str or Path — path to combined detections CSV
    Returns:
        dict mapping image_id → GeoDataFrame of matched results
    """
    print(f"\n{'─'*50}")
    print(f"Step 5: ADS-B Evaluation")
    print(f"{'─'*50}\n")

    # loads detections
    df_confirmed = _load_detections()
    pairs        = _get_unique_pairs(df_confirmed)

    print(f"\nProcessing {len(pairs)} tile/date combinations...\n")

    # processes each tile + date
    results = {}
    for tile, date in pairs:
        print(f"\n{'─'*40}")
        print(f"Processing {tile} - {date}")
        print(f"{'─'*40}")

        gdf = _run_single(tile, date, df_confirmed)
        if gdf is not None:
            image_id          = f"{date.replace('-', '')}_{tile}"
            results[image_id] = gdf

    # summary 
    print(f"\n{'─'*40}")
    print(f"Evaluation complete - {len(results)} scenes processed")
    print(f"{'─'*40}\n")

    total_det     = sum(
        len(gdf[gdf['source'] == 'pipeline']) for gdf in results.values()
    )
    total_matched = sum(
        len(gdf[(gdf['source'] == 'pipeline') & (gdf['matched'])]) 
        for gdf in results.values()
    )
    total_adsb    = sum(
        len(gdf[gdf['source'] == 'adsb']) for gdf in results.values()
    )

    print(f"Total detections       : {total_det}")
    print(f"Matched to ADS-B       : {total_matched} / {total_det}")
    print(f"Total ADS-B records    : {total_adsb}")

    return results