# parallax-plan-detection-pipeline


A pipeline for detecting flying aircraft in Sentinel-2 multispectral imagery using inter-band parallax effects, reimplementing the method of Liu et al. (2020) with extensions.

---

## Overview

The pipeline proceeds in five stages:

```
Step 1: GEE        : Band-3 reflectance anomaly detection → binary mask per image
Step 2: Local      : Sequential false positive filtering → candidate centroids
Step 3: GEE        : Image chip download around surviving candidates
Step 4: Local      : Aircraft confirmation via inter-band displacement
Step 5: Local      : ADS-B cross-reference and performance evaluation
```

A full end-to-end demonstration is provided in `notebooks/Pipeline_run.ipynb`.

---

## Repository Structure

```
parallax-plane-detection-pipeline/
├── src/
│   └── pipeline/
│       ├── config.py                   ← all pipeline parameters
│       ├── config_local.py             ← credentials and local paths (not tracked)
│       ├── utils/
│       │   ├── gee_auth.py
│       │   └── io.py
│       ├── s1_gee_candidates/          ← Step 1: candidate screening
│       ├── s2_filter/                  ← Step 2: false positive filtering
│       │   ├── transform.py
│       │   ├── cluster.py
│       │   ├── road.py
│       │   └── seamline.py
│       ├── s3_download/                ← Step 3: chip download
│       ├── s4_confirmation/            ← Step 4: aircraft confirmation
│       └── s5_evaluation/              ← Step 5: ADS-B evaluation
├── notebooks/
│   └── Pipeline_run.ipynb              ← end-to-end demonstration
├── annotated_data/
│   └── evaluation_set_annotated.zip    ← annotated GeoPackages for evaluation
├── db.zip                              ← pre-computed pipeline outputs
├── pyproject.toml
└── requirements.txt
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Unzip replication data

Pre-computed outputs for all pipeline stages are provided in `db.zip` and need to be unzipped

```bash
unzip db.zip
```

This created `db/` directory with the following structure:

```
db/
├── step1_masks/                ← binary GeoTIFF masks (Step 1)
├── step2_centroids/            ← centroid CSVs after transformation (Step 2a)
├── step2_cluster_filtered/     ← after cluster exclusion (Step 2b)
├── step2_road_filtered/        ← after road exclusion (Step 2c)
├── step2_seam_filtered/        ← after seam-line exclusion (Step 2d)
├── step3_chips/                ← image chips (Step 3)
├── step4_reflectance/          ← reflectance segmentation results (Step 4)
├── step4_collinearity/         ← collinearity check results (Step 4)
├── step4_confirmed/            ← confirmed detections (Step 4)
├── step5_adsb_cache/           ← cached ADS-B state vectors (Step 5)
└── ground_truth/               ← ground truth tables and metrics (Step 5)
```

### 3. Unzip annotated evaluation data

in order to inspect the annotated GeoPackage files per scene used in the study:
```bash
unzip `annotated_data/evaluation_set_annotated.zip` 
```
**Note:** This is not needed to run the demonstration pipeline

---

## Credentials and Authentication
### Google Earth Engine

Steps 1 and 3 require a Google Earth Engine account with a registered cloud project. Accounts can be requested at [earthengine.google.com](https://earthengine.google.com). Once approved, create a cloud project at [console.cloud.google.com](https://console.cloud.google.com).

Store the project ID in a `.env` file in format:

```python
GEE_PROJECT = your-gee-project-id
```

> Users without a GEE account can skip Steps 1 and 3 and run the pipeline from Step 2 onwards using the pre-computed masks in `db/step1_masks/`.

### OpenSky Network (ADS-B)

Step 5 requires an OpenSky Network research account for Trino API access. Accounts can be requested at [opensky-network.org](https://opensky-network.org). Once approved, store your username in a `.env` file in format:

```python
OPENSKY_USERNAME = your-username
```
---


## Configuration

All pipeline parameters are defined in `src/pipeline/config.py` and can be edited here.

## Data Sources

| Data | Source | Access |
|---|---|---|
| Sentinel-2 imagery | ESA Copernicus via Google Earth Engine | GEE account required |
| ADS-B state vectors | OpenSky Network Trino API | Research account required |
| Annotated ground truth | Manual annotation in QGIS | Provided in `annotated_data/evaluation_set_annotated.zip` |

---

## Running the Pipeline

Open `notebooks/Pipeline_run.ipynb` for a step-by-step walkthrough of the full pipeline. Each step is documented with inputs, outputs, and inspection plota. Pre-computed outputs are provided so no data needs to be downlaoded from the above mentioned sources.
