"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: environmental_data_exp_spatial_avg.py
Author: Tiago TOLOCZKO ROSS

Description:
Core monthly analysis module. Orchestrates the extraction, mathematical
aggregation, and visualization of 3D climate data into 2D spatial maps.
Fully modularized using Dependency Injection for data retrieval.
"""

import logging
import warnings
from pathlib import Path
from typing import Callable, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Explicitly import the retrieval engine and the new orchestrator
from environmental_data_exp_retrieval import file_var_retrieval, grand_mean_retrieval
from environmental_data_exp_visualization_orchestration_core import visualization_orchestration

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 1. RETRIEVAL & FACTORY FUNCTIONS
# =============================================================================
def granular_spatial_average(
        granularity: float = None
) -> Callable[[str, h5py.File | h5py.Group], Optional[pd.DataFrame]]:
    """
    Factory function returning a customized data retrieval function embedded
    with the desired spatial granularity.
    """
    def spatial_mean_retrieval(
            var: str,
            dataset: h5py.File | h5py.Group,
    ) -> Optional[pd.DataFrame]:
        """
        Extracts coordinates, calculates temporal mean, and applies spatial coarsening.
        """
        try:
            coord_group_name = f"{var}_coordinates"
            if coord_group_name not in dataset:
                return None

            coord_group = dataset[coord_group_name]
            if 'latitude' not in coord_group or 'longitude' not in coord_group:
                return None

            lats = coord_group['latitude'][:]
            lons = coord_group['longitude'][:]
            ds = dataset[var]

            if len(ds.shape) < 3:
                return None

            # Suppress the ocean NaN warning and calculate the temporal mean
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                time_mean_2d = np.nanmean(ds[:], axis=0)

            if time_mean_2d.shape != (len(lats), len(lons)):
                return None

            # Construct the formal Pandas DataFrame
            native_grid_df = pd.DataFrame(time_mean_2d, index=lats, columns=lons)
            native_grid_df.index.name = "Latitude"
            native_grid_df.columns.name = "Longitude"

            # Spatial Coarsening
            if granularity:
                binned_lats = np.round(native_grid_df.index / granularity) * granularity
                binned_lons = np.round(native_grid_df.columns / granularity) * granularity

                coarsened_grid = native_grid_df.groupby(binned_lats).mean()
                coarsened_grid = coarsened_grid.T.groupby(binned_lons).mean().T

                coarsened_grid.index = np.round(coarsened_grid.index, 4)
                coarsened_grid.columns = np.round(coarsened_grid.columns, 4)
                coarsened_grid.index.name = "Latitude"
                coarsened_grid.columns.name = "Longitude"
                return coarsened_grid
            else:
                return native_grid_df

        except Exception as var_e:
            logger.error(f"Unexpected error processing variable '{var}': {var_e}")
            return None

    return spatial_mean_retrieval

# =============================================================================
# 2. PERIOD OVERALL FUNCTION
# =============================================================================
def overall_period_spatial_average(
        period_data_dict: dict[str, list[pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """
    Calculates the overall temporal mean for each spatial position (lat, lon)
    across a list of DataFrames. Returns a dictionary of DataFrames to be appended
    by the orchestrator.
    """
    overall_stats = {}
    for var, df_list in period_data_dict.items():
        if not df_list:
            logger.warning(f"No data available for variable '{var}' to calculate spatial average.")
            continue

        # 1. Stack the underlying 2D NumPy arrays into a 3D tensor (Time, Lat, Lon)
        stacked_arrays = np.stack([df.values for df in df_list], axis=0)

        # 2. Calculate the mean along the Time axis (axis=0), safely ignoring NaNs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_matrix = np.nanmean(stacked_arrays, axis=0)

        # 3. Reconstruct the Pandas DataFrame
        var_period_spatial_mean = pd.DataFrame(
            mean_matrix,
            index=df_list[0].index,     # Latitudes
            columns=df_list[0].columns  # Longitudes
        )

        overall_stats[var] = var_period_spatial_mean

    return overall_stats


# =============================================================================
# 3. VISUALIZATION ENGINE AND PLOTTING FUNCTIONS
# =============================================================================
def plot_3d_surface_on_axis(
        ax: plt.Axes,
        lons: np.ndarray,
        lats: np.ndarray,
        data_2d: np.ndarray,
        var_name: str
) -> None:
    """
    Renders a 3D surface plot onto a provided Matplotlib Axis,
    and projects a 2D geographical contour onto the lowest XY plane.
    """
    z_max = np.nanmax(data_2d)
    z_min = np.nanmin(data_2d)
    z_range = z_max - z_min

    # Define the floor level (10% below the absolute lowest data point)
    floor_z = z_min - (z_range * 0.1) if z_range != 0 else z_min - 1

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    surf = ax.plot_surface(lon_grid, lat_grid, data_2d, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.contourf(lon_grid, lat_grid, data_2d, zdir='z', offset=floor_z, cmap='viridis', alpha=0.6)

    ax.set_zlim(floor_z, z_max)
    ax.set_title(f"Variable: {var_name.upper()}", fontweight='bold', fontsize=12)
    ax.set_xlabel("Longitude", labelpad=10)
    ax.set_ylabel("Latitude", labelpad=10)
    ax.set_zlabel("Time Average Value", labelpad=10)

    ax.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    ax.view_init(elev=25, azim=230)


def spatial_3d_plotter_factory(*args, **kwargs) -> Callable[[dict[str, pd.DataFrame], str], plt.Figure] | None:
    """
    Conforms to the universal orchestrator signature. Accepts a single-variable
    dictionary from the orchestrator loop and generates a standalone 3D plt.Figure.
    """
    def plot_generator(
            extracted_data: dict[str, pd.DataFrame],
            plot_title: str
    ) -> plt.Figure | None:
        if not extracted_data:
            return None

        # The orchestrator strictly passes 1 variable at a time in its plotting loop
        var_name, df = next(iter(extracted_data.items()))

        fig = plt.figure(figsize=(10, 8), *args, **kwargs)
        fig.suptitle(plot_title, fontsize=16, fontweight='bold')
        ax = fig.add_subplot(111, projection='3d')

        plot_3d_surface_on_axis(
            ax=ax,
            lons=df.columns.values,
            lats=df.index.values,
            data_2d=df.values,
            var_name=var_name
        )
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        return fig

    return plot_generator

# =============================================================================
# 4. TABULAR VISUALIZATION AND EXPORTING FUNCTIONS
# =============================================================================

def calculate_monthly_spatial_statistics(
        month_file: Path,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Calculates aggregated spatial and temporal averages for a specific month.

    Extracts the spatial grid, calculates the zonal mean (latitude), the
    meridional mean (longitude), and retrieves the overall grand mean.

    Args:
        month_file (Path): Path to the target monthly .h5 file.
        target_variable (str, optional): Specific variable to extract. If None,
                                         processes all available variables.
        granularity (float, optional): Spatial binning resolution. Defaults to 1.0.

    Returns:
        dict: A nested dictionary mapped by variable name. Structure:
              {
                  "variable_name": {
                      "spatial_grid": pd.DataFrame,
                      "zonal_mean": pd.DataFrame,
                      "meridional_mean": pd.DataFrame,
                      "grand_mean": float
                  }
              }
    """
    extracted_dataframes = file_var_retrieval(
        month_file=month_file,
        retrieval_func=granular_spatial_average(granularity)
    )

    if not extracted_dataframes:
        logger.warning(f"No data available to aggregate in {month_file.name}.")
        return {}

    grand_mean_dict = file_var_retrieval(
        month_file=month_file,
        retrieval_func=grand_mean_retrieval
    )

    tabular_results_dict = {}

    for var, coarsened_grid in extracted_dataframes.items():
        if target_variable and var != target_variable:
            continue

        logger.info(f"Aggregating '{var}' statistics at {granularity}° granularity...")

        # Zonal: Average across longitudes for each latitude
        zonal_mean_df = coarsened_grid.mean(axis=1).to_frame(name="Zonal_Mean")

        # Meridional: Average across latitudes for each longitude
        meridional_mean_df = coarsened_grid.mean(axis=0).to_frame(name="Meridional_Mean")

        if var in grand_mean_dict and not grand_mean_dict[var].empty:
            grand_mean = float(grand_mean_dict[var].iat[0, 0])
        else:
            grand_mean = float('nan')

        tabular_results_dict[var] = {
            "spatial_grid": coarsened_grid,
            "zonal_mean": zonal_mean_df,
            "meridional_mean": meridional_mean_df,
            "grand_mean": grand_mean
        }

    return tabular_results_dict


def print_monthly_spatial_summaries(
        month_file: Path,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Pipeline node that calculates spatial statistics and prints a formatted
    summary to the console before returning the data dictionary.

    Args:
        month_file (Path): Path to the target monthly .h5 file.
        target_variable (str, optional): Specific variable to filter by.
        granularity (float, optional): Spatial binning resolution. Defaults to 1.0.

    Returns:
        dict: The fully populated nested statistics dictionary.
    """
    stats_dict = calculate_monthly_spatial_statistics(
        month_file=month_file,
        target_variable=target_variable,
        granularity=granularity
    )

    if not stats_dict:
        return {}

    for var, stats in stats_dict.items():
        print(f"\n{'='*60}")
        print(f"STATISTICAL SUMMARY: {var.upper()} ({month_file.stem}) | Granularity: {granularity}°")
        print(f"{'='*60}")
        print(f"Overall Grand Mean: {stats['grand_mean']:.4f}\n")

        print("--- Zonal Averages (By Latitude) ---")
        print(stats['zonal_mean'].head(5))
        print("...\n")

        print("--- Meridional Averages (By Longitude) ---")
        print(stats['meridional_mean'].head(5))
        print("...\n")

        print("--- Spatial Matrix Head (Lat x Lon) ---")
        print(stats['spatial_grid'].iloc[:5, :5])
        print("\n")

    return stats_dict


def export_stats_to_excel(
        stats_dict: dict[str, dict],
        output_filename: str | Path = "climate_statistics.xlsx",
        output_dir: str | Path = "Excel exported statistical summaries"
) -> Path | None:
    """
    Exports a nested statistics dictionary to a multi-sheet Excel workbook.
    Automatically splits distinct data views (Spatial, Zonal, Meridional)
    into separate sheets and builds an overall Grand Mean summary sheet.

    Args:
        stats_dict (dict): The nested dictionary generated by calculate_monthly_spatial_statistics.
        output_filename (str | Path, optional): Name of the final Excel file. Defaults to "climate_statistics.xlsx".
        output_dir (str | Path, optional): Target directory for the export. Defaults to "Excel exported statistical summaries".

    Returns:
        Path | None: Absolute path to the generated Excel file, or None if the export fails.
    """
    if not stats_dict:
        logger.warning("No statistics provided to the Excel exporter.")
        return None

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    file_name = Path(output_filename).name
    output_path = target_dir / file_name

    summary_data = []

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for var, stats in stats_dict.items():

                # Excel restricts sheet names to 31 characters.
                # We slice the variable name to 15 chars to ensure suffix fits safely.
                safe_var_name = var[:15]

                spatial_df = stats.get("spatial_grid")
                if spatial_df is not None:
                    spatial_df.to_excel(writer, sheet_name=f"{safe_var_name}_Spatial")

                zonal_df = stats.get("zonal_mean")
                if zonal_df is not None:
                    zonal_df.to_excel(writer, sheet_name=f"{safe_var_name}_Zonal")

                meridional_df = stats.get("meridional_mean")
                if meridional_df is not None:
                    meridional_df.to_excel(writer, sheet_name=f"{safe_var_name}_Meridional")

                grand_mean = stats.get("grand_mean")
                if grand_mean is not None:
                    summary_data.append({"Variable": var, "Grand_Mean": grand_mean})

            # Create the master summary sheet if grand means exist
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="00_Overall_Summary", index=False)

        logger.info(f"Successfully exported statistical tables to {output_path.absolute()}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to export to Excel: {e}")
        return None


# =============================================================================
# 5. EXECUTION BLOCK
# =============================================================================
if __name__ == '__main__':
    source_dir = Path("Analysis - 02 round/era5_monthly_data")
    test_file = source_dir / "era5_3d_2021_06.h5"

    if test_file.exists():
        logger.info(f"Starting analysis and visualization for: {test_file.name}\n")

        alpha = .1
        # 1. Spatial Tables
        stats_dictionary = print_monthly_spatial_summaries(
            month_file=test_file,
            granularity=alpha
        )

        # 2. Excel Export
        if stats_dictionary:
            excel_filename = f"statistics_summary_{test_file.stem}.xlsx"
            export_stats_to_excel(
                stats_dict=stats_dictionary,
                output_filename=excel_filename
            )

        # 3. PDF Export utilizing the master Orchestrator
#        pdf_filename = f"3d_visualizations_{test_file.stem}.pdf"
#
#        visualization_orchestration(
#            input_path=source_dir,
#            study="Spatial Average",
#            retrieval_func=granular_spatial_average(granularity=alpha),
#            plot_generator_func=spatial_3d_plotter_factory(),
#            overall_analysis=overall_period_spatial_average,
#            objective='export',
#            output_filename=pdf_filename,
#            verbose=True
#        )

        # 4. Interactive Display utilizing the master Orchestrator
        logger.info("\nLaunching Interactive Display...")
        visualization_orchestration(
            input_path=source_dir,
            study="Spatial Average",
            retrieval_func=granular_spatial_average(granularity=alpha),
            plot_generator_func=spatial_3d_plotter_factory(),
            overall_analysis=overall_period_spatial_average,
            objective='show',
            verbose=False
        )

        # 'show' objective returns the active figures to RAM, so we trigger matplotlib here
        plt.show()
        logger.info("Pipeline execution completed successfully.")

    else:
        logger.error(f"Test file not found at: {test_file.absolute()}")