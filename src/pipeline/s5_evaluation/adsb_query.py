"""
Queries OpenSky Trino for ADS-B state vectors covering a full Sentinel-2 tile.
"""

import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import trino

from pipeline.config import (
    ADSB_CACHE_DIR,
    ADSB_MIN_ALTITUDE_M,
)

load_dotenv()

from pipeline.config import TRINO_HOST, TRINO_PORT, TRINO_CATALOG, TRINO_SCHEMA, QUERY_COLS



def _get_connection():
    """Opens a Trino connection using OpenSky credentials from .env."""
    username = os.getenv("OPENSKY_USERNAME")
    if not username:
        raise EnvironmentError(
            "OPENSKY_USERNAME not found in .env — "
            "add it with: echo 'OPENSKY_USERNAME=yourusername' >> .env"
        )
    return trino.dbapi.connect(
        host=TRINO_HOST,
        port=TRINO_PORT,
        user=username.lower(),
        catalog=TRINO_CATALOG,
        schema=TRINO_SCHEMA,
        http_scheme="https",
        auth=trino.auth.OAuth2Authentication(),
    )


def _get_hour_partitions(t_start, t_end):
    """
    Returns list of Unix timestamps for each hour partition
    """
    t = t_start.replace(minute=0, second=0, microsecond=0)
    hours = []
    while t <= t_end:
        hours.append(int(t.timestamp()))
        t += pd.Timedelta(hours=1)
    return hours


def _build_query(params, hour_partitions):
    """
    Builds Trino SQL query for the given tile params and hour partitions.
    Always filters on the partition column (hour) first.
    """
    t_start_unix = int(params['t_query_start'].timestamp())
    t_end_unix = int(params['t_query_end'].timestamp())

    hour_list = ", ".join(str(h) for h in hour_partitions)
    cols = ", ".join(QUERY_COLS)

    return f"""
        SELECT {cols}
        FROM state_vectors_data4
        WHERE hour IN ({hour_list})
          AND time BETWEEN {t_start_unix} AND {t_end_unix}
          AND lat  BETWEEN {params['south']} AND {params['north']}
          AND lon  BETWEEN {params['west']}  AND {params['east']}
          AND onground = false
          AND time - lastcontact <= 15
        ORDER BY time
    """

def _cache_path(image_id):
    """Returns the parquet cache path for a given image_id to avoid sending queries repeatedly."""
    path = Path(ADSB_CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{image_id}_adsb_raw.parquet"


def _query_trino(params, hour_partitions):
    """Runs Trino query and returns a DataFrame."""
    query = _build_query(params, hour_partitions)
    #print(f"Connecting to OpenSky Trino...")
    #print(f"Query window: {params['t_query_start']} → {params['t_query_end']}")
    #print(f"Hour partitions: {hour_partitions}")
    #print(f"Bbox: W={params['west']:.4f} E={params['east']:.4f} "
    #      f"S={params['south']:.4f} N={params['north']:.4f}")

    conn = _get_connection()
    cursor = conn.cursor()

    t0 = time.time()
    cursor.execute(query)
    rows = cursor.fetchall()
    elapsed = time.time() - t0

    df = pd.DataFrame(rows, columns=QUERY_COLS)
    print(f"Query returned {len(df):,} records in {elapsed:.1f}s")
    return df


#public entry point 
def run_adsb_query(params):
    """
    Returns raw ADS-B state vectors for an-2 tile.
    Loads from parquet cache if available, otherwise sends query to trino
    """
    image_id   = params['image_id']
    cache_file = _cache_path(image_id)

    if cache_file.exists():
        print(f"Loading from cache: {cache_file.name}")
        return pd.read_parquet(cache_file)

    hour_partitions = _get_hour_partitions(
        params['t_query_start'],
        params['t_query_end'],
    )
    df = _query_trino(params, hour_partitions)

    # saves queried tile to cache
    df.to_parquet(cache_file, index=False)
    print(f"Cached to {cache_file}")

    return df