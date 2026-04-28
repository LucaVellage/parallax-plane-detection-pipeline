"""
Script matches pipeline detections and unique adsb points based on 3km buffer
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

from pipeline.config import ADSB_MATCH_THRESHOLD_M, CONFIRMED_DIR


def _haversine_m(lat1, lon1, lat2, lon2):
    R    = 6_371_000 
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a    = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _find_best_match(lat, lon, df_adsb, threshold_m):
    """
    finds nearest match for single detection
    """
    if df_adsb.empty:
        return None, None, None

    dists = _haversine_m(lat, lon, df_adsb['lat'].values, df_adsb['lon'].values)
    idx   = np.argmin(dists)

    if dists[idx] <= threshold_m:
        row = df_adsb.iloc[idx]
        return float(dists[idx]), row['icao24'], row.get('callsign', None)

    return None, None, None


def _build_detection_rows(df_detections, df_adsb, threshold_m):
    """
    For annotation, builds one row per detection
    """
    rows = []
    for _, det in df_detections.iterrows():
        dist, icao24, callsign = _find_best_match(
            det['lat'], det['lon'], df_adsb, threshold_m
        )
        rows.append({
            'source'           : 'pipeline',
            'in_pipeline'      : True,
            'in_adsb'          : dist is not None,
            'matched'          : dist is not None,
            'match_distance_m' : dist,
            'matched_icao24'   : icao24,
            'matched_callsign' : callsign,
            'lat'              : det['lat'],
            'lon'              : det['lon'],
            'image_id'         : det.get('image_id'),
            'candidate_idx'    : det.get('candidate_idx'),
            'speed_kmh'        : det.get('speed_kmh'),
            'angle_deg'        : det.get('angle_deg'),
            'residual_m'       : det.get('residual_m'),
        })
    return rows


def _build_adsb_rows(df_adsb, matched_icao24s, params):
    """
    one row per adsb record
    """
    rows = []
    for _, ac in df_adsb.iterrows():
        matched      = ac['icao24'] in matched_icao24s
        outside_bounds = not (
            params['west']  <= ac['lon'] <= params['east'] and
            params['south'] <= ac['lat'] <= params['north']
        )
        rows.append({
            'source'           : 'adsb',
            'in_pipeline'      : matched,
            'in_adsb'          : True,
            'matched'          : matched,
            'match_distance_m' : None,
            'matched_icao24'   : ac['icao24'] if matched else None,
            'matched_callsign' : ac.get('callsign'),
            'lat'              : ac['lat'],
            'lon'              : ac['lon'],
            'image_id'         : ac.get('image_id'),
            'candidate_idx'    : None,
            'speed_kmh'        : ac.get('speed_kmh'),
            'angle_deg'        : None,
            'residual_m'       : None,
            # ADS-B specific
            'icao24'           : ac['icao24'],
            'baroaltitude'     : ac.get('baroaltitude'),
            'heading'          : ac.get('heading'),
            'outside_bounds'   :outside_bounds,
        })
    return rows


#public functionm

def run_adsb_match(df_detections, df_adsb, params):
    """
    Matches pipeline detections with nearby ads-b records. Returns GeoDataframe with one row per point
    I.e. in a match, both a pipeline detection and an adsb point are represented in individual rows

    """
    image_id = params['image_id']
    print(f"\nMatching {image_id}")
    print(f"Detections : {len(df_detections)}")
    print(f"ADS-B      : {len(df_adsb)}")
    print(f"Threshold  : {ADSB_MATCH_THRESHOLD_M}m")

    # matching detections with ADS-B
    det_rows = _build_detection_rows(df_detections, df_adsb, ADSB_MATCH_THRESHOLD_M)

    # Collects matched icao24
    matched_icao24s = {
        r['matched_icao24']
        for r in det_rows
        if r['matched_icao24'] is not None
    }

    # build adsb rows
    adsb_rows = _build_adsb_rows(df_adsb, matched_icao24s, params)

    all_rows = det_rows + adsb_rows
    gdf      = gpd.GeoDataFrame(
        all_rows,
        geometry=[Point(r['lon'], r['lat']) for r in all_rows],
        crs='EPSG:4326',
    )

    # summary
    n_det        = len(df_detections)
    n_adsb       = len(df_adsb)
    n_matched    = len(matched_icao24s)
    n_unmatched_det  = sum(1 for r in det_rows  if not r['matched'])
    n_unmatched_adsb = sum(1 for r in adsb_rows if not r['matched'])

    print(f"\nMatch summary:")
    print(f"  Detections matched to ADS-B : {n_matched} / {n_det}")
    print(f"  Detections unmatched        : {n_unmatched_det} / {n_det}")
    print(f"  ADS-B unmatched             : {n_unmatched_adsb} / {n_adsb}")

    matched_dists = [
        r['match_distance_m'] for r in det_rows
        if r['match_distance_m'] is not None
    ]
    if matched_dists:
        print(f"  Match distance: "
              f"mean: {np.mean(matched_dists):.0f}m  "
              f"max: {np.max(matched_dists):.0f}m  "
              f"min: {np.min(matched_dists):.0f}m")

    return gdf