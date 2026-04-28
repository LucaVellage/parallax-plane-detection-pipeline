"""
This script implements: 
- A morphological dilation to connect pixels that are already 8- but not 4-connected
- A connected component algorithm to identify pixel blobs after dilation 
- Centroid computation of pixel blobs 
- An affine geotransformation of pixel blob centroids to geocoordinates
#TODO Note: Maybe geotransform can be moved to end of script to only convert the surviving
centroids after false positive filtering
"""
import cv2
import csv
import numpy as np
import pandas as pd
import logging
from PIL import Image
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import uniform_filter
from matplotlib.colors import to_rgb

from pipeline.config import (
    MASK_DIR, 
    CENTROIDS_DIR,
    DIL_KERNEL_SIZE, 
    MAX_OBJECT_SIZE_PX
)

from pipeline.utils.io import (
    list_files,
    parse_filename,
    extract_geotransform, 
    epsg_from_tile,
    get_transformer,
    transform_coords,
    pixel_to_utm
)

# Suppressing harmless tifffile Error
logging.getLogger("tifffile").setLevel(logging.ERROR)


def _read_mask(mask_file: str) -> np.ndarray:
    """
    Function reads a binary GeoTIFF as numpy array 
    Uses PIL to avoid GDAL dependency.
    """
    mask = np.array(Image.open(mask_file))
    return mask


def _morph_dilation(mask: np.ndarray)-> np.ndarray:
    #defining structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DIL_KERNEL_SIZE, DIL_KERNEL_SIZE))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    return mask_dilated


def _get_blob_centroids(mask_dilated: np.ndarray):
    """
    Function implements a connected component algorithm using the CV2 package
    - After dilation, Algorithm connects all candidate pixels by 4-connectivity into pixel blobs
    - Implements two-pass with union find 
    - Assigns unique ID to each blobs
    - CV2 algorithm also counts pixels, only keeps blobs <= 100px
    - Computes centroid (mean point) of surviving blobs
    """
    _, _, stats, centroids = cv2.connectedComponentsWithStats(
    mask_dilated, connectivity=4, ltype=cv2.CV_32S)

    #only pixels, without background 
    blobs = stats[1:, cv2.CC_STAT_AREA]     

    #large area filter: only keeps blobs <= 100px 
    mask_keep = blobs <= MAX_OBJECT_SIZE_PX 

    #computes centroid of surviving pixel blobs
    filtered_centroids = centroids[1:][mask_keep]
    return filtered_centroids


def _extract_cand_centroids(mask_file: str) -> list[tuple[float, float]]:
    """
    Runs the transformation pipeline on single mask file: 
    1: Reads binary pixel mask as array
    2: Extracts geotransform from GeoTIFF files
    3: Runs morphological dilation 
    4: Connects pixels into labelled blobs, applies size filter + extracts blob centroids
    4: Converts centroids into geographic coordinates

    Returns a list of dicts, one per surviving candidate:
    col: column coordinate in pixel space
    row: row coordinate in pixel space      
    lon: longitude
    lat: latitude
    date: YYYYMMDD from filename
    tile: tile ID from filename
    image_id — date_tile combined identifier
    """
    fields = parse_filename(mask_file)
    date = fields["date"]
    tile = fields["tile"]

    mask = _read_mask(mask_file)
    geotransform = extract_geotransform(mask_file)
    epsg = epsg_from_tile(mask_file)
    #dilation on pixel mask (i.e. numpy array)
    mask_dilated = _morph_dilation(mask)
    filtered_centroids = _get_blob_centroids(mask_dilated)

    #loops over all centroids
    centr_coord = []

    for col, row in filtered_centroids:
       #converts raster coord to UTM easting/northing coordinates
       easting, northing = pixel_to_utm(col, row, geotransform)
       #converts UTM coordinates to lon lat degrees in epsg 4326
       lon, lat = transform_coords(easting, northing, epsg, 4326)
       centr_coord.append({
            "col":      col,
            "row":      row,
            "lon":      lon,
            "lat":      lat,
            "epsg":     epsg,
            "date":     date,
            "tile":     tile,
            "image_id": f"{date}_{tile}",
       })

    return centr_coord


def _save_centroids(centr_coord: list[dict], out_path: str) -> None:
    """
    Saving candidate centroids to csv files per tile 
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["col", "row", "lon", "lat", "epsg", "date", "tile", "image_id"]
        )
        writer.writeheader()
        writer.writerows(centr_coord)


#run transformation pipeline 
def run_transformation():

    mask_dir = Path(MASK_DIR)
    centroids_dir = Path(CENTROIDS_DIR)

    mask_files = list_files(mask_dir, "*_candidates.tif")
    candidate_centroids = []

    for mask_path in mask_files:
        candidates = _extract_cand_centroids(mask_path)
        mask = _read_mask(mask_path)

        if not candidates:
            print(f"{date}_{tile}: 0 candidates → skipping")
            continue

        date = candidates[0]["date"]
        tile = candidates[0]["tile"]

        #Saving candidates into csv / tile
        out_path = centroids_dir / f"{date}_{tile}_centroids.csv"
        _save_centroids(candidates, out_path)

        print(f"{date}_{tile}: {mask.sum()} candidate pixels → {len(candidates)} candidate centroids")
        candidate_centroids.extend(candidates)
 
    print(f"Output saved to {CENTROIDS_DIR}")
    print(f"\nTotal candidates across all images: {len(candidate_centroids)} candidate centroids")
    return candidate_centroids
 


 #visualisations

def inspect_blobs(date, tile) -> None:
    """
    Plots show  blob size distribution pre and post size filter.
    """
    mask_path = f'{MASK_DIR}/{date}_{tile}_candidates.tif'
    # run the same steps as the pipeline

    mask = _read_mask(mask_path)
    mask_dilated = _morph_dilation(mask)

    _, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_dilated, connectivity=4, ltype=cv2.CV_32S
    )

    # excluding background
    all_blobs = stats[1:, cv2.CC_STAT_AREA]
    kept_blobs = all_blobs[all_blobs <= MAX_OBJECT_SIZE_PX]
    dropped_areas = all_blobs[all_blobs > MAX_OBJECT_SIZE_PX]

    print('=' * 40)
    print('Transformation Inspection')
    print('=' * 40)
    print(f"File                         : {Path(mask_path).name}")
    print(f"Candidate px before dilation : {mask.sum()}")
    print(f"Candidate px after dilation  : {mask_dilated.sum()}")
    print(f"Pixels added by dilation     : {mask_dilated.sum() - mask.sum()}")
    print(f"Total blobs found            : {len(all_blobs)}")
    print(f"Kept (≤{MAX_OBJECT_SIZE_PX}px)                : {len(kept_blobs)}")
    print(f"Dropped (>{MAX_OBJECT_SIZE_PX}px)             : {len(dropped_areas)}")
    if len(dropped_areas) > 0:
        print(f"Largest dropped          : {dropped_areas.max()} px")
    print(f"Largest kept                 : {kept_blobs.max() if len(kept_blobs) > 0 else 'n/a'} px")
    print(f"Smallest kept                : {kept_blobs.min() if len(kept_blobs) > 0 else 'n/a'} px")


    # plot blob size distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(all_blobs, bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(MAX_OBJECT_SIZE_PX, color="red", linewidth=1.5,
                    linestyle="--", label=f"threshold ({MAX_OBJECT_SIZE_PX}px)")
    axes[0].set_title("Blob size distribution: all blobs")
    axes[0].set_xlabel("Blob size (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(kept_blobs, bins=20, color="steelblue", edgecolor="white")
    axes[1].set_title(f"Blob size distribution: kept blobs (≤{MAX_OBJECT_SIZE_PX}px)")
    axes[1].set_xlabel("Blob size (pixels)")
    axes[1].set_ylabel("Count")

    plt.suptitle(f"Detected Blobs in {date}_{tile}", fontsize=12)
    plt.tight_layout()
    plt.show()



def inspect_transformation(date, tile, zoom_size) -> None:
    """
    Visual inspection of the transformation pipeline for one mask file. 
    Shows dilation, blob labeling, size filtering + centroid computation zoomed in
    to selected pixel blobs.

    zoom_size : half-size of the zoom window in pixels
    """
    mask_path = f'{MASK_DIR}/{date}_{tile}_candidates.tif'

    mask         = _read_mask(mask_path)
    mask_dilated = _morph_dilation(mask)

    _, label_img, stats, centroids = cv2.connectedComponentsWithStats(
        mask_dilated, connectivity=4, ltype=cv2.CV_32S
    )
    all_areas = stats[1:, cv2.CC_STAT_AREA]
    mask_keep = all_areas <= MAX_OBJECT_SIZE_PX
    kept_labels = np.where(mask_keep)[0] + 1 
    dropped_labels = np.where(~mask_keep)[0] + 1
    kept_centroids = centroids[1:][mask_keep]

    #finding densest zoomregion
    density  = uniform_filter(mask.astype(float), size=zoom_size)
    row, col = np.unravel_index(density.argmax(), density.shape)
    r0 = max(0, row - zoom_size)
    r1 = min(mask.shape[0], row + zoom_size)
    c0 = max(0, col - zoom_size)
    c1 = min(mask.shape[1], col + zoom_size)

    zoom_before  = mask[r0:r1, c0:c1]
    zoom_after   = mask_dilated[r0:r1, c0:c1]
    zoom_labels  = label_img[r0:r1, c0:c1]
    added        = (zoom_after.astype(int) - zoom_before.astype(int)).clip(0)

    #plot
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    # 1: raw mask
    cmap_binary = ListedColormap(["#ffffff", "#000000"])
    axes[0].imshow(zoom_before, cmap=cmap_binary, vmin=0, vmax=1,
                   interpolation="nearest")
    axes[0].set_title(f"1. Raw mask\n{zoom_before.sum()} candidate px (zoomed area)")

    # 2: after dilation
    axes[1].imshow(zoom_after, cmap=cmap_binary, vmin=0, vmax=1,
                   interpolation="nearest")
    axes[1].set_title(f"2. After dilation\n{zoom_after.sum()} candidate px (zoomed area)")

    # 3: dilation difference
    diff_rgb = np.ones((*zoom_before.shape, 3)) 
    diff_rgb[zoom_before == 1] = to_rgb("#000000") #original pixels  
    diff_rgb[added == 1]       = to_rgb("#2ADBDE") #dilated pixels ´
    axes[2].imshow(diff_rgb, interpolation="nearest")
    axes[2].set_title("3. Dilation difference\nblack=original  cyan=added")

    # 4: blob labelling with kept/dropped colour coding
    blob_rgb = np.ones((*zoom_labels.shape, 3))
    for lbl in np.unique(zoom_labels):
        if lbl == 0:
            continue   # background
        blob_mask = zoom_labels == lbl
        if lbl in kept_labels:
            blob_rgb[blob_mask] = to_rgb("#000000") #original pixels 
        else:
            blob_rgb[blob_mask] = to_rgb("#b61111")  # dropped (too large)

    axes[3].imshow(blob_rgb, interpolation="nearest")

    # count blobs in zoom window
    zoom_kept    = len([l for l in np.unique(zoom_labels) if l in kept_labels])
    zoom_dropped = len([l for l in np.unique(zoom_labels) if l in dropped_labels])
    axes[3].set_title(
        f"4. Blob labelling\n"
        f"black=kept ({zoom_kept})  red=dropped ({zoom_dropped})"
    )

    # 5: centroids overlaid on kept blobs
    axes[4].imshow(blob_rgb, interpolation="nearest")

    # plot centroids that fall within the zoom window
    n_centroids_shown = 0
    for i, (cx, cy) in enumerate(kept_centroids):
        # cx=col, cy=row in full image space — convert to zoom space
        cx_zoom = cx - c0
        cy_zoom = cy - r0
        if 0 <= cx_zoom < zoom_labels.shape[1] and 0 <= cy_zoom < zoom_labels.shape[0]:
            axes[4].plot(cx_zoom, cy_zoom, "y+", markersize=10, markeredgewidth=1.5)
            n_centroids_shown += 1

    axes[4].set_title(
        f"5. Centroids (green +)\n"
        f"{n_centroids_shown} shown in window  |  {len(kept_centroids)} total"
    )

    # labels + format
    for ax in axes:
        ax.set_xlabel("col (px)")
        ax.set_ylabel("row (px)")

    # legend
    legend_elements = [
        mpatches.Patch(color= to_rgb("#000000"), label=f"kept (≤{MAX_OBJECT_SIZE_PX}px)"),
        mpatches.Patch(color= to_rgb("#b61111"), label=f"dropped (>{MAX_OBJECT_SIZE_PX}px)"),
        mpatches.Patch(color= to_rgb("#2ADBDE"), label=f"dilated"),
        plt.Line2D([0], [0], marker="+", color= to_rgb("#24CE68"), markersize=10,
                   markeredgewidth=1.5, linestyle="none", label="centroid"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)

    plt.suptitle(
        (f"Pixel transformation of Example Blobs in {date}_{tile}  |  zoom [{r0}:{r1}, {c0}:{c1}]"),
        fontsize=12
    )
    plt.tight_layout(pad=3.0)

    plt.show()


#inspecting distribution of blob centroids
def blob_centr_plot(date, tile):
    centr_csv = (f'{CENTROIDS_DIR}/{date}_{tile}_centroids.csv')
    df = pd.read_csv(centr_csv)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax.scatter(df["lon"], df["lat"], 
               c="steelblue", 
               s=15, 
               label=f"Candidate centroids (n={len(df)})")
    ax.set_title(f"Candidate Centroids after Transformation: {date}_{tile}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="upper left")
    plt.tight_layout(pad=0.3)
    plt.show()

    













