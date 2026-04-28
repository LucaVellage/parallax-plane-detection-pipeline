"""
Script removes cluster anomalies: 
Case: reflectance anomalies or sun-glint over water, i.e. 
cases in which centroids are grouped very close to each other

Threshold: Removed if >= 3 centroids within buffer (r = 1500m) of a
singular centroid. 

Haversine matrix to compute pairwise circle distances 
in metres between points.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from pipeline.utils.io import (
    list_files,
    parse_filename,
    epsg_from_tile, 
    transform_coords
)

from pipeline.config import (
    MASK_DIR, 
    CENTROIDS_DIR,
    CLUSTER_FILTERED_DIR,
    CLUSTER_BUFFER_M,
    CLUSTER_MAX
)

def _load_csv(directory: str, pattern: str):

    files = list_files(directory, pattern)
    all_dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(all_dfs, ignore_index=True)

    return df


def _haversine_matrix(lons: np.ndarray, lats: np.ndarray) -> np.array:
    #Earth radius
    R = 6_371_000.0 

    #converting degrees to radians
    #necessary for trigonometric fct
    lats = np.radians(lats)
    lons = np.radians(lons)

    #computing pairwise angle differences
    #produces difference matrix 
    dlat = lats[:, None] - lats[None, :]
    dlon = lons[:, None] - lons[None, :]

    #haversine formula applied element-wise across /
    #difference matrices
    #values (0-1) are stored in matrix a
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lats[:, None]) * np.cos(lats[None, :])
         * np.sin(dlon / 2) ** 2)
    
    #converting to metres
    return 2 * R * np.arcsin(np.sqrt(a))


def _exclude_clusters(candidate_centroids: list[dict]) -> list[dict]:
    """
    Function removes candidate centrois that sit at dense clusters.

    For each candidate, counts how many other candidates fall in cluster of 
    one centroid. If count >= CLUSTER_MAX, then candidate removed. 
    
    Current setting: 
    neighbours (incl self) <= 3 --> kept
    neighbours (incl self) > 4 --> removed
    """

    if len(candidate_centroids) <= CLUSTER_MAX:
        return candidate_centroids
    
    lons = np.array([c["lon"] for c in candidate_centroids])
    lats = np.array([c["lat"] for c in candidate_centroids])

    #includes distance to itself at diagnoal
    dist_matrix = _haversine_matrix(lons, lats)
    np.fill_diagonal(dist_matrix, np.inf)  # exclude self
    
    #produces boolean matrix (True: point within 1500m)
    #result: Sum of TRUE values per row (i.e. per candidate)
    neighbour_count = (dist_matrix <= CLUSTER_BUFFER_M).sum(axis=1)
    in_dense_buffer = neighbour_count >= CLUSTER_MAX 
    exclude = np.zeros(len(candidate_centroids), dtype=bool)
    for i, is_dense in enumerate(in_dense_buffer):
        if is_dense:
            # exclude i and all its neighbours
            exclude[i] = True
            exclude[dist_matrix[i] <= CLUSTER_BUFFER_M] = True

    return [c for c, ex in zip(candidate_centroids, exclude) if not ex]


def run_cluster_exclusion() -> pd.DataFrame:

    df = _load_csv(CENTROIDS_DIR, '*_centroids.csv')
    print(f"Total candidates before cluster filter: {len(df)}")

    survivors = []
    for image_id, group in df.groupby("image_id"):
        candidates = group.to_dict(orient="records")
        filtered   = _exclude_clusters(candidates)

        # inspection output per tile
        lons = np.array([c["lon"] for c in candidates])
        lats = np.array([c["lat"] for c in candidates])
        
        dist_matrix = _haversine_matrix(lons, lats)
        dist_inspect = dist_matrix.copy()
        np.fill_diagonal(dist_inspect, np.inf)
        min_distances = dist_inspect.min(axis=1)
        
        print(f"\n{image_id}:")
        print(f"  candidates before : {len(candidates)}")
        print(f"  candidates after  : {len(filtered)}")
        print(f"  excluded          : {len(candidates) - len(filtered)}")
        print(f"  min distance      : {min_distances.min():.1f} m")
        print(f"  max distance      : {min_distances.max():.1f} m")
      
        print(f"{image_id}: {len(candidates)} → {len(filtered)}")

        survivors.extend(filtered)

    df_filtered = pd.DataFrame(survivors)
    print(f"\nTotal candidates after cluster filter: {len(df_filtered)}")

    #save for now into separate csv files per tile
    cluster_filtered_dir = Path(CLUSTER_FILTERED_DIR)
    cluster_filtered_dir.mkdir(parents=True, exist_ok=True)

    for image_id, group in df_filtered.groupby("image_id"):
        out_path = cluster_filtered_dir / f"{image_id}_cluster_filtered.csv"
        group.to_csv(out_path, index=False)
    
    print(f"Updated centroids saved to {CLUSTER_FILTERED_DIR}")
 
    return df_filtered
    


#Inspection plots 
def cluster_filtered_plot(date, tile):

    centr_dil_csv = (f'{CENTROIDS_DIR}/{date}_{tile}_centroids.csv')
    scene_before = pd.read_csv(centr_dil_csv)

    centr_cluster_filtered_csv = (f'{CLUSTER_FILTERED_DIR}/{date}_{tile}_cluster_filtered.csv')
    scene_after = pd.read_csv(centr_cluster_filtered_csv)

    #surviving tuples
    merged  = scene_before.merge(scene_after[["col", "row"]], on=["col", "row"], how="left", indicator=True)
    removed = scene_before[merged["_merge"] == "left_only"]

    print('=' * 40)
    print('Cluster Filter Inspection')
    print('=' * 40)
    print(f"Before:     {len(scene_before)} centroids")
    print(f"Survivors:  {len(scene_after)} centroids")
    print(f"Removed:    {len(removed)} centroids")

    mask_path = Path(MASK_DIR) / f"{date}_{tile}_candidates.tif"
    epsg       = epsg_from_tile(mask_path)
    lat_centre = scene_before["lat"].mean()
    lon_centre = scene_before["lon"].mean()
    e1, n1     = transform_coords(lon_centre,       lat_centre, 4326, epsg)
    e2, n2     = transform_coords(lon_centre + 1.0, lat_centre, 4326, epsg)
    m_per_deg  = np.sqrt((e2 - e1)**2 + (n2 - n1)**2)
    deg_radius = CLUSTER_BUFFER_M / m_per_deg

    #plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Cluster exclusion: buffer: {CLUSTER_BUFFER_M}m, threshold: {CLUSTER_MAX} neighbours",
                 fontsize=12)


    # left: all candidates with removed ones highlighted
    axes[0].scatter(scene_before["lon"], scene_before["lat"],
                    c="steelblue", s=30, zorder=3, label=f"surviving ({len(scene_after)})")
    axes[0].scatter(removed["lon"], removed["lat"],
                    c="red", s=60, marker="x", zorder=4, linewidths=1.5,
                    label=f"excluded (cluster) ({len(removed)})")

    #buffer circles around removed candidates
    for _, row in removed.iterrows():
        #approx 1500m in degrees at this latitude
        axes[0].add_patch(plt.Circle(
            (row["lon"], row["lat"]), deg_radius,
            color="red", fill=False, linewidth=0.8, alpha=0.4
        ))
    axes[0].set_title(f"{date}_{tile}: Before cluster filter")

    #right: survivors only
    axes[1].scatter(scene_after["lon"], scene_after["lat"],
                    c="steelblue", s=30, zorder=3, label=f"surviving ({len(scene_after)})")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()
