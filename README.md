# AirTS-Forecast

A modular, production-ready Python pipeline for climate data analysis. AirTS-Forecast ingests, processes, and visualizes 3D environmental data (time, latitude, longitude) from HDF5 files—such as ERA5 datasets—with built-in support for spatial analysis, timeseries modeling, and geographic boundary filtering.

## ✨ Key Features

- **Auto-Discovering File System**: Automatically discovers and validates HDF5 files using strict regex patterns (`era5_3d_YYYY_MM.h5`), sorts them chronologically, and routes data without manual intervention.

- **Dependency Injection Architecture**: Core data pipelines are decoupled from mathematical operations, allowing flexible composition of models (spatial averaging, timeseries flattening, etc.) at runtime.

- **Vectorized Spatial Processing**: Flattens 3D tensors into Pandas MultiIndex DataFrames, leveraging underlying C++ libraries for efficient spatial binning and granularity adjustments.

- **Automated Sinusoidal Fitting**: Uses Levenberg-Marquardt optimization (SciPy) to fit 12-month trigonometric waves to regional timeseries, with equation output and visualization of fitted models.

- **Geographic Boundary Filtering**: Leverages OpenStreetMap (OSMnx) and Shapely to extract high-precision borders (e.g., Italy) and generates interactive Folium maps with valid/invalid grid intersections.

- **Automated Reporting**: Exports spatial and temporal statistics to multi-sheet Excel workbooks and compiles hundreds of 3D Matplotlib visualizations into multi-page PDF reports.

## 📁 Directory Structure

```
AirTS-Forecast/
│
├── era5_monthly_data/                     # Input: 3D HDF5 climate data files
│   ├── era5_3d_2004_06.h5
│   └── era5_3d_2004_07.h5
│
├── Excel exported statistical summaries/  # Output: Excel reports (auto-generated)
├── Exported pdf plots/                    # Output: PDF visualizations (auto-generated)
│
├── environmental_data_retrieval.py        # Core: HDF5 I/O & file routing
├── environmental_monthly_mean_analysis.py # Module: Spatial statistics & 3D plots
├── environmental_data_2d_timeplots.py     # Module: Timeseries & sinusoidal fitting
└── italy_grid_classification.py           # Module: Geographic boundary masking
```

## 📦 Installation

It is recommended to use a virtual environment (venv or conda).

```bash
pip install numpy pandas h5py matplotlib scipy openpyxl osmnx shapely folium geopandas
```

## 🧩 Module Overview

### 1. Core Engine (`environmental_data_retrieval.py`)

The backbone of the pipeline. Handles safe HDF5 file I/O, regex-based filename parsing, and directory exploration.

- **Key Function**: `file_var_retrieval` — a higher-order function that accepts mathematical models as callable objects and applies them to HDF5 data.
- **Responsibilities**: File validation, data extraction, route management, fail-fast error handling.

### 2. Spatial Analysis (`environmental_monthly_mean_analysis.py`)

Generates spatial tables (Latitude × Longitude) and calculates regional statistics.

- **Calculations**: Grand means, zonal means (latitudinal), and meridional means (longitudinal).
- **Visualizations**: Interactive 3D surface plots with 2D contour projections onto the ocean floor.
- **Exports**: Writes spatial data to `.xlsx` files; compiles multi-page `.pdf` reports.

### 3. Timeseries Analysis (`environmental_data_2d_timeplots.py`)

Tracks temporal evolution of specific regions and identifies periodic patterns.

- **Vectorization**: Uses Pandas MultiIndex to flatten spatial axes and compute regional binning instantly.
- **Concatenation**: Seamlessly stitches months/years into continuous timeseries using `pd.concat`.
- **Sinusoidal Fitting**: Applies `scipy.optimize.curve_fit` to overlay 12-month trigonometric waves on correlated regions; outputs equations to console.

### 4. Geographic Masking (`italy_grid_classification.py`)

Ensures all analyses are geographically relevant by filtering data to target regions.

- **Boundary Extraction**: Fetches precise national borders (Italy) from OpenStreetMap.
- **Grid Validation**: Validates grid coordinates against Shapely polygons (`polygon.contains(point)`).
- **Map Generation**: Outputs an interactive Folium HTML map overlaying valid (green) and invalid (red) grid cells on CartoDB basemap.

## 🚀 Quick Start

Each module can be executed standalone for testing or imported into a larger pipeline.

**Generate a geographic grid and create a visual map:**
```bash
python italy_grid_classification.py
```

**Generate 3D spatial plots and export Excel/PDF reports:**
```bash
python environmental_monthly_mean_analysis.py
```

**Extract continuous timeseries and fit sinusoidal curves:**
```bash
python environmental_data_2d_timeplots.py
```

## 📝 Notes

- Designed for robust, fail-fast environmental data processing.
- All modules follow modular design principles for easy extension and testing.
- Suitable for ERA5 climate datasets and similar 3D environmental data formats.

---

**Part of the AirTS-Forecast Project** | Climate Data Analysis Pipeline
