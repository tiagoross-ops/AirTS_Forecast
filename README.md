AirTS-Forecast: Climate Data Exploration & Analysis Pipeline
Section 1: Data Gathering, Exploration, and Visualization

A highly optimized, modular Python pipeline designed to ingest, process, and visualize 3D environmental climate data (Time, Latitude, Longitude) from HDF5 files (e.g., ERA5 datasets). This project leverages advanced data structures, vectorized Pandas operations, and Dependency Injection to create a fast, scalable, and memory-efficient ETL (Extract, Transform, Load) architecture.

🚀 Key Features
Auto-Discovering File System: Automatically parses directories using strict Regex (era5_3d_YYYY_MM.h5), validates data, and sorts files chronologically without requiring manual date inputs.

Dependency Injection Architecture: Core reading loops are decoupled from mathematical operations. Mathematical models (like spatial averaging or timeseries flattening) are injected dynamically into the HDF5 readers as factory functions.

Vectorized Spatial Coarsening: Bypasses slow Python for loops by flattening 3D tensors into Pandas MultiIndex DataFrames, allowing underlying C++ libraries to handle spatial binning and granularity adjustments instantly.

Automated Mathematical Modeling: Automatically attempts Levenberg-Marquardt optimization (scipy.optimize) to fit 12-month sinusoidal waves to regional timeseries, plotting idealized models when correlation exceeds 95%.

Geographic Boundary Filtering: Utilizes OpenStreetMap (OSMnx) and Shapely to calculate high-precision geospatial polygons (e.g., Italy's borders) and generates interactive Folium HTML maps to visually verify granularity grids.

Automated Reporting: Exports statistics to multi-sheet Excel workbooks and compiles hundreds of 3D Matplotlib plots into centralized, multi-page PDF reports.

🗂️ Directory Structure
The pipeline expects and generates the following directory structure:

Plaintext
AirTS-Forecast/
│
├── era5_monthly_data/                     # Input directory for 3D HDF5 climate data
│   ├── era5_3d_2004_06.h5
│   └── era5_3d_2004_07.h5
│
├── Excel exported statistical summaries/  # Auto-generated Excel reports
├── Exported pdf plots/                    # Auto-generated multi-page 3D PDF plots
│
├── environmental_data_retrieval.py        # Core Engine: HDF5 I/O & File Routing
├── environmental_monthly_mean_analysis.py # Module: Spatial Statistics & 3D Plots
├── environmental_data_2d_timeplots.py     # Module: Timeseries & Sinusoidal Fitting
└── italy_grid_classification.py           # Module: OSMnx Geographic Boundary Masking
📦 Dependencies
Ensure you have the following libraries installed. It is recommended to use a virtual environment (venv or conda).

Bash
pip install numpy pandas h5py matplotlib scipy openpyxl osmnx shapely folium geopandas
🧩 Module Overview
1. Core Engine (environmental_data_retrieval.py)
The backbone of the pipeline. It handles safe HDF5 file opening, Regex-based file name comprehension, and directory exploration. It exposes file_var_retrieval, a higher-order function that accepts Callable math functions to execute against datasets.

2. Spatial Analysis (environmental_monthly_mean_analysis.py)
Handles the generation of spatial tables (Lat x Lon).

Math: Calculates Grand Means, Zonal Means (Latitudinal), and Meridional Means (Longitudinal).

Visuals: Generates interactive 3D surface plots with 2D shadow contours mapped to the ocean floor.

Exports: Routes spatial data to .xlsx files and compiles .pdf visualization reports.

3. Timeseries Analysis (environmental_data_2d_timeplots.py)
Tracks the temporal evolution of specific regions over time.

Vectorization: Flattens spatial axes using Pandas MultiIndexes to instantly calculate regional binning.

Period Concatenation: Seamlessly stitches together months/years of data into a continuous pd.DataFrame timelines using pd.concat.

Sinusoidal Fitting: Uses scipy.optimize.curve_fit to overlay perfect 12-month trigonometric waves on highly correlated regions, printing the mathematical equations directly to the console.

4. Geographic Masking (italy_grid_classification.py)
Ensures data is geographically relevant.

Boundary Extraction: Fetches highly accurate national borders (Italy) using OpenStreetMap data.

Grid Validation: Tests mathematical grid coordinates against Shapely Polygons (polygon.contains(point)).

Map Generation: Spits out an interactive Folium HTML map overlaying the valid (Green) and invalid (Red) grid intersections over a CartoDB basemap.

🛠️ Usage Examples
Each module is designed to be executable as a standalone script for testing or imported as part of a larger pipeline.

To test the geographic grid and generate a visual map:

Bash
python italy_grid_classification.py
To generate 3D spatial plots and export Excel/PDF reports for a specific file:

Bash
python environmental_monthly_mean_analysis.py
To extract continuous timeseries across an entire folder of data and fit sinusoidal curves:

Bash
python environmental_data_2d_timeplots.py
Part of the AirTS-Forecast Project. Designed for robust, fail-fast environmental data processing.
