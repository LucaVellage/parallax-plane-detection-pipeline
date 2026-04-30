"""
Fetches Sentinel-2 tile acquisition parameters from GEE and caches them locally so each tile/date combination is only queried once.
"""

import json
import datetime
from datetime import timezone, timedelta
from pathlib import Path
import os
import ee

from dotenv import load_dotenv

load_dotenv()
GEE_PROJECT = os.getenv('GEE_PROJECT')

from pipeline.utils import io
from pipeline.config import (
    ADSB_BUFFER_S,
    TILE_HEIGHT, 
    SCAN_DURATION_S, 
    TILE_CACHE_DIR, 
    MASK_DIR
)

cache_file = Path(f"{TILE_CACHE_DIR}/tile_params_cache.json")

def _load_cache():
    """Loads the tile params if available"""
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def _fetch_from_gee(tile, date):
    """
    Fetches Sentinel-2 acquisition parameters from GEE. 
    Returns dict with all parameters
    """
    date_next = (
        datetime.datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    image = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filter(ee.Filter.eq("MGRS_TILE", tile))
        .filterDate(date, date_next)
        .first()
    )

    if image.getInfo() is None:
        raise ValueError(f"No S2 image found for tile {tile} on {date}")

    props = image.getInfo()["properties"]

    # Acquisition times
    t_ms = props["system:time_start"]
    t_top = datetime.datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)
    t_bottom = t_top + timedelta(seconds=SCAN_DURATION_S)

    # ADS-B query window
    t_query_start = t_top    - timedelta(seconds=ADSB_BUFFER_S)
    t_query_end = t_bottom + timedelta(seconds=ADSB_BUFFER_S)

    # Bounding box
    # handles both image ID s starting with or without T 
    mask_candidates = [
        Path(MASK_DIR) / f"{date.replace('-', '')}_T{tile}_candidates.tif",
        Path(MASK_DIR) / f"{date.replace('-', '')}_{tile}_candidates.tif",
    ]
    mask_path = next((p for p in mask_candidates if p.exists()), None)

    if mask_path is not None:
        print(f"  AOI bbox from mask : {mask_path.name}")
        bbox = io.get_mask_bbox_wgs84(mask_path)
    else:
        print(f"WARNING: mask not found for {tile} {date} "
              f"falling back to full tile bbox from GEE")
        gee_bbox = image.geometry().bounds().getInfo()['coordinates'][0]
        lons     = [c[0] for c in gee_bbox]
        lats     = [c[1] for c in gee_bbox]
        bbox     = {
            'west'  : min(lons),
            'east'  : max(lons),
            'south' : min(lats),
            'north' : max(lats),
        }

    return {
        "tile"          : tile,
        "date"          : date,
        "t_top"         : t_top.isoformat(),
        "t_bottom"      : t_bottom.isoformat(),
        "t_query_start" : t_query_start.isoformat(),
        "t_query_end"   : t_query_end.isoformat(),
        "west"          : bbox['west'],
        "east"          : bbox['east'],
        "south"         : bbox['south'],
        "north"         : bbox['north'],
        "cloud_pct"     : props.get("CLOUDY_PIXEL_PERCENTAGE"),
    }


def _deserialise_params(raw):
    """
    Converts a cached params to py objects
    """
    return {
        **raw,
        "t_top"         : datetime.datetime.fromisoformat(raw["t_top"]),
        "t_bottom"      : datetime.datetime.fromisoformat(raw["t_bottom"]),
        "t_query_start" : datetime.datetime.fromisoformat(raw["t_query_start"]),
        "t_query_end"   : datetime.datetime.fromisoformat(raw["t_query_end"]),
    }


#public entry point
def get_cached_tile_params(tile, date):
    """
    function gets params for S-2 tile and date from cache if available or from GEE and saves to cache.

    Args:
        tile
        date
    Returns:
        dict with param keys
    """
    cache_file = Path(f"{TILE_CACHE_DIR}/tile_params_cache.json")
    
    # Normalise tile: strip leading T if present
    tile     = tile.lstrip("T")
    image_id = f"{date.replace('-', '')}_{tile}"

    # cache
    cache = _load_cache()
    if image_id in cache:
        print(f"tile_params cache hit:  {image_id}")
        params = _deserialise_params(cache[image_id])
        params["image_id"] = image_id
        return params

    # if not in cache, get from GEE
    print(f"Fetching from GEE:      {image_id}")
    raw              = _fetch_from_gee(tile, date)
    raw["image_id"]  = image_id
    cache[image_id]  = raw
    _save_cache(cache)
    print(f"Cached to {cache_file}")

    params = _deserialise_params(raw)
    params["image_id"] = image_id
    return params
