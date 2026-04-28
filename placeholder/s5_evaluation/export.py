"""
export.py
─────────
Exports matched ADS-B / detection results as GeoJSON for QGIS inspection.
One GeoJSON file per image_id, containing all points (both sources)
with match metadata and blank in_visual / adsb_visually_checked columns
for manual labelling in QGIS.

Entry point: run_export(results, output_dir)
"""

import geopandas as gpd
from pathlib import Path
import datetime
from datetime import timedelta
import ee
import json
import pandas as pd

from pipeline.config import ADSB_ANNOTATIONS_DIR, TILE_CACHE_DIR



def _prepare_gdf(gdf, image_id):
    """
    Prepares a GeoDataFrame for export by ensuring all required columns
    exist, adding blank visual inspection columns, and cleaning up
    columns not needed in QGIS.
    """
    gdf = gdf.copy()

    # Add image_id if not present
    if 'image_id' not in gdf.columns:
        gdf['image_id'] = image_id

    # Add inspection columns with correct defaults
    gdf['inspected']     = False   # flipped to True in QGIS when done
    gdf['is_visible']    = False    # filled during inspection
    gdf['is_flying']     = False    # ADS-B only — manual override
    gdf['seamline_duplicate'] = False
    gdf['near_bounds'] = False
    gdf['notes']         = None    # free text
    # Add scene-level metadata
    gdf['cloud_pct'] = _load_cloud_pct(image_id)

    # outside_bounds — auto computed for ADS-B rows, None for others
    if 'outside_bounds' not in gdf.columns:
        gdf['outside_bounds'] = False
    else:
        gdf['outside_bounds'] = gdf['outside_bounds'].fillna(False)

    # Round floats for cleaner QGIS display
    for col in ['match_distance_m', 'lat', 'lon',
                'baroaltitude', 'geoaltitude', 'speed_kmh',
                'angle_deg', 'residual_m']:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce').round(2)

    # Column order — most useful columns first for QGIS attribute table
    col_order = [
        # identifiers
        'image_id', 'source',
        # inspection columns — first so they're visible in QGIS attribute table
        'inspected', 'is_visible', 'is_flying', 'outside_bounds', 'near_bounds', 'seamline_duplicate', 'notes',
        # match info
        'in_pipeline', 'in_adsb',
        'matched', 'match_distance_m',
        'matched_icao24', 'matched_callsign',
        # position
        'lat', 'lon',
        # ADS-B fields
        'icao24', 'callsign',
        'baroaltitude', 'geoaltitude',
        'speed_kmh', 'heading',
        # pipeline fields
        'angle_deg', 'residual_m', 'candidate_idx',
        'geometry', 'cloud_pct',
    ]
    col_order = [c for c in col_order if c in gdf.columns]
    extras    = [c for c in gdf.columns if c not in col_order]
    gdf       = gdf[col_order + extras]

    return gdf


def _export_single(gdf, image_id, output_dir):
    """Exports a single GeoDataFrame to GeoJSON."""
    out_path = Path(output_dir) / f"{image_id}_eval.geojson"
    gdf.to_file(out_path, driver='GeoJSON')

    n_pipeline = len(gdf[gdf['source'] == 'pipeline'])
    n_adsb     = len(gdf[gdf['source'] == 'adsb'])
    n_matched  = len(gdf[gdf['matched'] == True])

    print(f"  {image_id}_eval.geojson")
    print(f"    pipeline : {n_pipeline}")
    print(f"    adsb     : {n_adsb}")
    print(f"    matched  : {n_matched}")

    return out_path


def _load_cloud_pct(image_id):
    """
    Reads cloud cover percentage for a given image_id from
    the tile params cache. Returns None if not found.
    """
    cache_path = Path(TILE_CACHE_DIR) / "tile_params_cache.json"
    if not cache_path.exists():
        return None
    with open(cache_path, "r") as f:
        cache = json.load(f)
    return cache.get(image_id, {}).get("cloud_pct")


# public entry point

def run_export(results, ):
    """
    Exports matched results as GeoJSON files for QGIS inspection.
    One file per image_id with blank in_visual and
    adsb_visually_checked columns ready for manual labelling.

    Args:
        results    : dict mapping image_id → GeoDataFrame
                     (output of run_evaluation())
        output_dir : directory to write GeoJSON files
    Returns:
        list of output file paths
    """
    Path(ADSB_ANNOTATIONS_DIR).mkdir(parents=True, exist_ok=True)

    print(f"\n── Exporting {len(results)} scenes to {ADSB_ANNOTATIONS_DIR} ──\n")

    out_paths = []
    for image_id, gdf in results.items():
        gdf_clean = _prepare_gdf(gdf, image_id)
        out_path  = _export_single(gdf_clean, image_id, ADSB_ANNOTATIONS_DIR)
        out_paths.append(out_path)

    print(f"\nExported {len(out_paths)} GeoJSON files to {ADSB_ANNOTATIONS_DIR}")
    return out_paths




#-------
#export S-2 tile from GEE
#-------

def export_rgb_tile_to_drive(tile, date):
    """
    Exports true colour (B4/B3/B2) composite of a tile to Google Drive.
    """
    date_next = (datetime.datetime.strptime(date, '%Y-%m-%d') 
                 + timedelta(days=1)).strftime('%Y-%m-%d')
    
    image = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        .filter(ee.Filter.eq('MGRS_TILE', tile))
        .filterDate(date, date_next)
        .first())
    
    rgb      = image.select(['B4', 'B3', 'B2'])
    filename = f"{date.replace('-','')}_{tile}_rgb"
    aoi      = image.geometry()
    native_crs = image.select('B3').projection().getInfo()['crs']

    task = ee.batch.Export.image.toDrive(
        image          = rgb,
        description    = filename,
        folder         = 's2_rgb',
        fileNamePrefix = filename,
        scale          = 10,
        region         = aoi,
        crs            = native_crs,
        maxPixels      = 1e13,
        fileFormat     = 'GeoTIFF'
    )
    task.start()
    print(f"Task ID:  {task.id}")
    print(f"Status:   {task.status()['state']}")
    print(f"Filename: {filename}.tif")
    print(f"Check progress at code.earthengine.google.com/tasks")



