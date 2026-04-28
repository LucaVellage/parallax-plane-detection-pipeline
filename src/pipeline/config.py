# Time range
START_DATE = '2023-08-15'
END_DATE   = '2023-08-30'

#AOI Boudnding Box 
AOI_BBOX_COLS = ["aoi_lon_min", "aoi_lon_max", "aoi_lat_min", "aoi_lat_max"]

# Step 1: candidate screening (global threshold)
REFL_THRESHOLD = 0.05

# Transform: 
DIL_KERNEL_SIZE = 3 
MAX_OBJECT_SIZE_PX = 100 #to filter large objects

# Step 2: false positive filtering
# cluster
CLUSTER_BUFFER_M  = 1500
CLUSTER_MAX  = 3

#road
ROAD_KERNEL_HALF = 25
ROAD_ANGLE_STEP = 10
ROAD_FREQUENCY_THRESHOLD = 5

#seamline
SEAM_MAX_DIST_M = 50
SEAM_MIN_INLIERS = 3 
SEAM_ITERATIONS = 7
SEAM_DIR_MIN = 68
SEAM_DIR_MAX = 82

# Step 3: image chip download
CHIP_SIZE_PX  = 301
CHIP_RADIUS_M = 1500
BANDS = ["B2", "B3", "B4"]
SCALE = 10

#step 4: confirmation 
#reflectance
FOCAL_RADIUS_PX = 10 
FOCAL_STD_MULT = 2.0 
CHIP_CENTRE_PX = 150
MAX_CAND_BAND = 3

#collinearity
COLINEAR_ANGLE_MIN = 170

#displacement
DISPLACEMENT_RATIO = 0.9151 
DISPLACEMENT_OFFSET = 0.7333
DISPLACEMENT_TOLERANCE = 25.0
METRES_PER_PIXEL= 10.0
T_BAND2_3 = 0.527
T_BAND2_4 = 1.005

#evaluation
ADSB_BUFFER_S = 30 
ADSB_MIN_ALTITUDE_M = 300

# Step 5: ADS-B
ADSB_BUFFER_S = 30     # seconds buffer either side of scan
ADSB_MIN_ALTITUDE_M = 300  
MIN_VELO_MS = 30
ADSB_MATCH_THRESHOLD_M = 3000

#Trino Connection 
TRINO_HOST = "trino.opensky-network.org"
TRINO_PORT = 443
TRINO_CATALOG = "minio"
TRINO_SCHEMA = "osky"

QUERY_COLS = [
    "time", "icao24", "callsign",
    "lat", "lon",
    "baroaltitude", "geoaltitude",
    "velocity", "heading",
    "onground", "lastcontact",
]

#Tile Param
TILE_HEIGHT     = 10980  
SCAN_DURATION_S = 15.0

#Step 6: Metrics
VISUAL_MATCH_RADIUS_M = 500 

# Paths
MASK_DIR     = '../db/step1_masks'
CENTROIDS_DIR = '../db/step2_centroids'
CLUSTER_FILTERED_DIR = "../db/step2_cluster_filtered"
ROAD_FREQ_DIR = "../db/step2_road_frequency"
ROAD_FILTERED_DIR = "../db/step2_road_filtered"
SEAM_FILTERED_DIR = "../db/step2_seam_filtered"
CHIPS_DIR = "../db/step3_chips"
REFL_DIR = "../db/step4_refl_confirmed"
COL_DIR = "../db/step4_col_confirmed"
CONFIRMED_DIR = "../db/step4_final_confirmed"
ADSB_CACHE_DIR = "../db/step5_adsb_chache"
ADSB_FILTERED_DIR = "../db/step5_adsb_filtered"
ADSB_ANNOTATIONS_DIR = "../db/step5_annotations"
TILE_CACHE_DIR = "../db/step5_tile_cache"
ANNOTATIONS_DIR = "../db/step5_annotations/QGIS_annotated"
GROUND_TRUTH_DIR = "../db/step5_annotations/ground_truth"
ANALYSIS_DIR = "../db/step5_annotations/analysis_plots"
REFL_EVAL_DIR = "../db/step4_results_eval/refl_eval"
CONF_EVAL_DIR = "../db/step4_results_eval/conf_eval"

OUTPUT_PATH  = 'outputs/results/'
FIGURES_PATH = 'outputs/figures/'