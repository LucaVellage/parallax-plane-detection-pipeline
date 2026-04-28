"""
File downloads image chips of filtered candidates 
"""

import ee
import io
import requests
import zipfile
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipeline.utils.io import list_files
from pipeline.utils import gee_auth
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta


from pipeline.utils.io import (
    list_files,
    parse_filename,
)

from pipeline.config import (
    SEAM_FILTERED_DIR, 
    CHIPS_DIR,
    CHIP_SIZE_PX, 
    CHIP_RADIUS_M,
    BANDS,
    SCALE
)


def _load_survivors():
    """
    Loads surviving centroid candidates from seamline-filtered csv files.
    """
    files = list_files(SEAM_FILTERED_DIR, pattern="*_seam_filtered.csv")
    if not files:
        raise FileNotFoundError(f"No seam-filtered CSVs found in {SEAM_FILTERED_DIR}")
    
    dfs = [pd.read_csv(f) for f in sorted(files)]
    df  = pd.concat(dfs, ignore_index=True)
    df  = df[df["seam_exclude"] == False].reset_index(drop=True)
    print(f"Surviving candidates: {len(df)}")
    return df


def _load_scene_image(date, tile):
    """
    Loads the Sentinel-2 image matching date and tile.
    Returns ee.Image with bands B2, B3, B4.
    loads full images from GEE server and extracts image chip per candidate centroid, 
    instead of downloading each image chip individually.
    """
    dt    = datetime.strptime(str(date), "%Y%m%d")
    start = dt.strftime("%Y-%m-%d")
    end   = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    
    col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
            .filter(ee.Filter.date(start, end)) \
            .filter(ee.Filter.stringContains("system:index", tile)) \
            .select(BANDS)
    
    count = col.size().getInfo()
    print(f"Images found for {date} {tile}: {count}")
    return col.first()


def _download_chip(image, lon, lat, image_id, candidate_idx):
    """
    Downloads a 301×301px chip centred on (lon, lat) from image.
    Saves as GeoTIFF to CHIPS_DIR
    """
    out_path = Path(CHIPS_DIR) / f"{image_id}_{candidate_idx:04d}_chip.tif"
    #only downloads chips that are not downloaded
    if out_path.exists():
        return out_path

    point  = ee.Geometry.Point([lon, lat])
    region = point.buffer(CHIP_RADIUS_M).bounds()
    native_crs = image.select('B2').projection().getInfo()['crs']

    url = image.getDownloadURL({
        "bands":  BANDS,
        "region": region,
        #"scale":  SCALE,
        "format": "GEO_TIFF",
        "dimensions": f"{CHIP_SIZE_PX}x{CHIP_SIZE_PX}",
        "crs":        native_crs
    })

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    out_path.write_bytes(response.content)
    return out_path


def _download_scene_chips(image, df_scene, max_workers=4):
    """
    Downloads all chips for one scene in parallel.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _download_chip,
                image,
                row["lon"], row["lat"],
                row["image_id"],
                idx,
            ): idx
            for idx, row in df_scene.iterrows()
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                path = future.result()
                results.append((idx, path))
                print(f"candidate {idx:04d}")
            except Exception as e:
                print(f"failed to download candidate {idx:04d}: {e}")
    return results


#public function
def run_chip_download():
    Path(CHIPS_DIR).mkdir(parents=True, exist_ok=True)

    df = _load_survivors()
    summary = []

    for image_id, df_scene in df.groupby("image_id"):
        date = str(df_scene["date"].iloc[0])
        tile = str(df_scene["tile"].iloc[0])
        print(f"\n{image_id}: {len(df_scene)} candidates")

        image   = _load_scene_image(date, tile)
        results = _download_scene_chips(image, df_scene)

        n_ok   = sum(1 for _, p in results if p is not None)
        n_fail = len(df_scene) - n_ok
        print(f"  → downloaded {n_ok}, failed {n_fail}")

        summary.append({"image_id": image_id, "total": len(df_scene),
                        "ok": n_ok, "failed": n_fail})

    print("\n── chip download summary ──")
    print(f"{'scene':<25} {'total':>6} {'ok':>6} {'failed':>6}")
    print("─" * 46)
    for row in summary:
        print(f"{row['image_id']:<25} {row['total']:>6} {row['ok']:>6} {row['failed']:>6}")
    total = sum(r["total"] for r in summary)
    ok    = sum(r["ok"]    for r in summary)
    print("─" * 46)
    print(f"{'all scenes':<25} {total:>6} {ok:>6} {total-ok:>6}")


#inspection 

#Normalizing reflectance

def _normalize_chip(path, percentile=2):
    """
    Displays a chip as an RGB image (B4=red, B3=green, B2=blue).
    Stretches contrast using percentile clipping.
    """
    arr = tifffile.imread(path) 
    if arr.shape[2] == 3:
        arr = arr.transpose(2, 0, 1)
    rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1).astype(float)
    for c in range(3):
        lo, hi       = np.percentile(rgb[:, :, c], [percentile, 100 - percentile])
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb


def plot_chip(date, tile, candidate_idx, percentile=2):
    """Displays single chip"""
    path = Path(CHIPS_DIR) / f"{date}_{tile}_{candidate_idx:04d}_chip.tif"
    if not path.exists():
        raise FileNotFoundError(f"No chip found at {path}")

    rgb = _normalize_chip(str(path), percentile)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(path.stem)
    ax.axis("off")
    plt.tight_layout()
    plt.show()



def plot_chips(date: str, tile: str, max_chips: int = 24, percentile: int = 2) -> None:
    """
    Displays a grid of chips for a given date and tile.
    max_chips: maximum number of chips to show.
    """
    pattern = f"{date}_{tile}_*_chip.tif"
    paths   = sorted(list_files(CHIPS_DIR, pattern=pattern))[:max_chips]

    if not paths:
        raise FileNotFoundError(f"No chips found for {date}_{tile} in {CHIPS_DIR}")

    n_cols = 4
    n_rows = int(np.ceil(len(paths) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes      = np.array(axes).flatten()

    for ax, path in zip(axes, paths):
        rgb = _normalize_chip(str(path), percentile)
        ax.imshow(rgb)
        ax.set_title(Path(path).stem, fontsize=8)
        ax.axis("off")

    for ax in axes[len(paths):]:
        ax.axis("off")

    fig.suptitle(f"{date}_{tile}: {len(paths)} chips", fontsize=12)
    plt.tight_layout()
    plt.show()