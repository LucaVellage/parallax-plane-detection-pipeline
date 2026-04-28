import math
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.config import COLINEAR_ANGLE_MIN, COL_DIR, REFL_DIR

def _angle_at_vertex(a, v, b):
    """
    Angle at vertex V in triangle A-V-B

    Uses dot product: cos(θ) = (VA · VB) / (|VA| |VB|)
    Returns value in [0, 180].
    Returns 0.0 if either vector has zero length (coincident points).
    """

    #vectors relative to vertex V
    va = np.array([a[0] - v[0], a[1] - v[1]], dtype=float)
    vb = np.array([b[0] - v[0], b[1] - v[1]], dtype=float)

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    cos_theta = np.clip(np.dot(va, vb) / (norm_a * norm_b), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


#single chip entry point
def compute_triplet_angles(b2_candidates, b3, b4_candidates):
    """
    Computes angles of all possible triplets
    """
    all_triplets = []
    for b2 in b2_candidates:
        for b4 in b4_candidates:
            angle = _angle_at_vertex(b2, b3, b4)
            all_triplets.append({
                'c_b2_col':  b2[0], 'c_b2_row':  b2[1], 'c_b2_area': b2[2],
                'c_b3_col':  b3[0], 'c_b3_row':  b3[1], 'c_b3_area': b3[2],
                'c_b4_col':  b4[0], 'c_b4_row':  b4[1], 'c_b4_area': b4[2],
                'angle_deg': angle,
                'colinear_found': angle >= COLINEAR_ANGLE_MIN,
            })
    return all_triplets

#batch entry point
def run_collinearity_check() -> pd.DataFrame:
    """
    Runs collinearity check on all available centroid triplets
    """
    Path(COL_DIR).mkdir(parents=True, exist_ok=True)

    in_path = Path(REFL_DIR) / "all_reflectance.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Reflectance CSV not found: {in_path}")

    df   = pd.read_csv(in_path)
    rows = []

    for (image_id, candidate_idx), chip_df in df.groupby(['image_id', 'candidate_idx']):
            meta = chip_df.iloc[0][['tile', 'date', 'lon', 'lat', 'epsg']].to_dict()

            b2_rows = chip_df[chip_df['band'] == 'B2'][['col', 'row', 'area']].itertuples(index=False, name=None)
            b3_rows = chip_df[chip_df['band'] == 'B3'][['col', 'row', 'area']].itertuples(index=False, name=None)
            b4_rows = chip_df[chip_df['band'] == 'B4'][['col', 'row', 'area']].itertuples(index=False, name=None)

            b2_list = list(b2_rows)
            b3_list = list(b3_rows)
            b4_list = list(b4_rows)

            base = {
                'image_id':      image_id,
                'candidate_idx': candidate_idx,
                **meta,
            }

            # band missing
            if not b3_list or not b2_list or not b4_list:
                rows.append({
                    **base,
                    'colinear_found': False,
                    'angle_deg':  float('nan'),
                    'c_b2_col':   float('nan'), 'c_b2_row': float('nan'),
                    'c_b2_area':  float('nan'),
                    'c_b3_col':   float('nan'), 'c_b3_row': float('nan'),
                    'c_b3_area':  float('nan'),
                    'c_b4_col':   float('nan'), 'c_b4_row': float('nan'),
                    'c_b4_area':  float('nan'),
                })
                continue

            b3 = b3_list[0] 

            triplet_angles = compute_triplet_angles(b2_list, b3, b4_list)

            for triplet_angle in triplet_angles:
                rows.append({**base, **triplet_angle})
                
    result_df = pd.DataFrame(rows)

    out_path = Path(COL_DIR) / "all_collinearity.csv"
    result_df.to_csv(out_path, index=False)

    n_chips    = df.groupby(['image_id', 'candidate_idx']).ngroups
    n_colinear = result_df[result_df['colinear_found'] == True]['candidate_idx'].nunique()
    n_combos_passing = result_df['colinear_found'].sum()
    n_combos_notpassing = (result_df['colinear_found'] == False).sum()
    n_combos_total = result_df['angle_deg'].notna().sum()
    #n_no_b2b4  = result_df[result_df['colinear_found'] == False].groupby('candidate_idx').ngroups

    print(f"\n{'=' * 40}")
    print(f"Collinearity Check Summary")
    print(f"{'=' * 40}")
    print(f"Total chips          : {n_chips}")
    print(f"With colinear combos : {n_colinear}")
    print(f"No colinear combos   : {n_chips - n_colinear}")
    print(f"Total combinations   : {n_combos_total}")
    print(f"Passing combos       : {n_combos_passing}")
    print(f"Non-passing combos   : {n_combos_notpassing}")
    

    print(f"\nPer-tile breakdown:")
    tile_summary = (result_df.groupby('tile').agg(chips    =('candidate_idx', 'nunique'), colinear =('colinear_found', 'sum')))    

    print(tile_summary.to_string())
    print(f"\nOutput written to    : {out_path}")

    #return result_df
    
    