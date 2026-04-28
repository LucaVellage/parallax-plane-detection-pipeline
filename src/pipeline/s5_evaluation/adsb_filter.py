"""
Filters returned ADS-B records to get unique record at satellite overpass time
"""

import pandas as pd
import numpy as np
from pathlib import Path

from pipeline.config import (
    ADSB_MIN_ALTITUDE_M,
    MIN_VELO_MS,
    ADSB_FILTERED_DIR,
    TILE_HEIGHT, 
    SCAN_DURATION_S
)

def _apply_filters(df, params):
    """
    Applies filter to state vectors
    """
    n0 = len(df)

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Temporal filter
    df = df[
        (df['datetime'] >= params['t_query_start']) &
        (df['datetime'] <= params['t_query_end'])
    ]
    print(f"After temporal filter:  {len(df):,} / {n0:,} records")

    # Spatial filter: clips to AOI if smaller than full tile
    df = df[
        (df['lon'] >= params['west'])  &
        (df['lon'] <= params['east'])  &
        (df['lat'] >= params['south']) &
        (df['lat'] <= params['north'])
    ]
    print(f"After spatial filter:   {len(df):,} records")

    # Altitude filter to remove e.g. taxiing aircraft
    df = df[
        (df['onground'] == False) &
        (df['baroaltitude'].notna()) &
        (df['baroaltitude'] >= ADSB_MIN_ALTITUDE_M) &
        (df['velocity'].notna()) &
        (df['velocity'] >= MIN_VELO_MS)
    ]
    print(f"  After ground filter    : {len(df):,} records")

    # Drop NA
    df = df.dropna(subset=['lat', 'lon'])
    df = df.reset_index(drop=True)
    print(f"After dropna:           {len(df):,} records")

    return df


def _select_best_ping(group, params):
    """
    Selects pings closest to row correctedt acquisition time based on location
    """
    group     = group.sort_values('datetime').copy()
    lat_range = params['north'] - params['south']

    #find ping at tile center
    t_centre  = params['t_top'] + pd.Timedelta(seconds=SCAN_DURATION_S / 2)
    group['dt'] = (group['datetime'] - t_centre).abs()
    best      = group.loc[group['dt'].idxmin()]
    lat_0     = best['lat']

    # eatimates row from lat
    row_est   = int(((params['north'] - lat_0) / lat_range) * TILE_HEIGHT)
    row_est   = max(0, min(TILE_HEIGHT - 1, row_est))

    # computes acquistion time for that row
    t_correct = params['t_top'] + pd.Timedelta(
        seconds=(row_est / TILE_HEIGHT) * SCAN_DURATION_S
    )

    # selects ping closest to row corrected time
    group['dt_corrected'] = (group['datetime'] - t_correct).abs()
    best_corrected        = group.loc[group['dt_corrected'].idxmin()]

    return {
        **best_corrected.to_dict(),
        'row_estimate' : row_est,
        't_correct'    : t_correct.isoformat(),
        'speed_kmh'    : best_corrected['velocity'] * 3.6,
    }



def _get_unique_aircraft(df, params):
    """
    returns unique aircraft per scene
    """
    records = [
        _select_best_ping(group, params)
        for _, group in df.groupby('icao24')
    ]
    unique = pd.DataFrame(records).reset_index(drop=True)

    # Add outside_bounds flag
    unique['outside_bounds'] = ~(
        (unique['lat'] >= params['south']) &
        (unique['lat'] <= params['north']) &
        (unique['lon'] >= params['west'])  &
        (unique['lon'] <= params['east'])
    )

    print(f"  Unique aircraft        : {len(unique)}")
    print(f"    likely inside bounds : {(~unique['outside_bounds']).sum()}")
    print(f"    likely outside bounds: {unique['outside_bounds'].sum()}")

    return unique


#public fct
def run_adsb_filter(df_raw, params):

    print(f"\n── Filtering {params['image_id']} ──")
    print(f"  Tile scan window:")
    print(f"    t_top         : {params['t_top']}")
    print(f"    t_bottom      : {params['t_bottom']}")
    print(f"    t_query_start : {params['t_query_start']}")
    print(f"    t_query_end   : {params['t_query_end']}")
    print(f"  Tile bbox:")
    print(f"    N={params['north']:.4f}  S={params['south']:.4f}  "
          f"W={params['west']:.4f}  E={params['east']:.4f}")
    
    df_filtered = _apply_filters(df_raw, params)
    unique      = _get_unique_aircraft(df_filtered, params)

    return unique