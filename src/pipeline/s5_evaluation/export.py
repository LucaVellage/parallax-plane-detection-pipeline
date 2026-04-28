"""
exports matched geodataframe as GeoJsON
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

    gdf = gdf.copy()

    if 'image_id' not in gdf.columns:
        gdf['image_id'] = image_id

    # Prepares file for annotation
    gdf['inspected']     = False
    gdf['is_visible']    = False   
    gdf['is_flying']     = False   
    gdf['seamline_duplicate'] = False
    gdf['near_bounds'] = False
    gdf['notes']         = None    #

    gdf['cloud_pct'] = _load_cloud_pct(image_id)

    # autocomputed: outside bounds
    if 'outside_bounds' not in gdf.columns:
        gdf['outside_bounds'] = False
    else:
        gdf['outside_bounds'] = gdf['outside_bounds'].fillna(False)

    for col in ['match_distance_m', 'lat', 'lon',
                'baroaltitude', 'geoaltitude', 'speed_kmh',
                'angle_deg', 'residual_m']:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce').round(2)


    col_order = [
        'image_id', 'source',
        'inspected', 'is_visible', 'is_flying', 'outside_bounds', 'near_bounds', 'seamline_duplicate', 'notes',
        'in_pipeline', 'in_adsb',
        'matched', 'match_distance_m',
        'matched_icao24', 'matched_callsign',
        'lat', 'lon',
        'icao24', 'callsign',
        'baroaltitude', 'geoaltitude',
        'speed_kmh', 'heading',
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


# public function

def run_export(results, ):

    Path(ADSB_ANNOTATIONS_DIR).mkdir(parents=True, exist_ok=True)

    print(f"\n── Exporting {len(results)} scenes to {ADSB_ANNOTATIONS_DIR} ──\n")

    out_paths = []
    for image_id, gdf in results.items():
        gdf_clean = _prepare_gdf(gdf, image_id)
        out_path  = _export_single(gdf_clean, image_id, ADSB_ANNOTATIONS_DIR)
        out_paths.append(out_path)

    print(f"\nExported {len(out_paths)} GeoJSON files to {ADSB_ANNOTATIONS_DIR}")
    return out_paths


#export S-2 tile from GEE
#Downloads in GEE to connected Google drive from where tiles need to be downloaded manually
#Task can be viewed in Gee Task manager


def export_rgb_tile_to_drive(tile, date):
    """
    Exports true colour of a tile to Google Drive.
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



