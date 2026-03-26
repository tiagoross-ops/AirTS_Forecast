"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: environmental_monthly_mean_analysis.py
Author: Tiago TOLOCZKO ROSS

Description:
Core monthly analysis module. Orchestrates the extraction, mathematical
aggregation, and visualization of 3D climate data into 2D spatial maps.
Fully modularized using Dependency Injection for data retrieval.
"""

import logging
import math
import warnings
from pathlib import Path
from typing import Callable, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Explicitly import the retrieval engine and the grand mean function
from environmental_data_retrieval import file_var_retrieval, grand_mean_retrieval

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# --- 1. TRANSFORM (RETRIEVAL) FACTORY ---

def spatial_mean_granular(
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


# --- 2. GRANULAR STATISTICAL AGGREGATION ---

def monthly_granular_spatial_tables(
        month_file: Path,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Calculates aggregated spatial and temporal averages using the centralized
    data retrieval function. Returns the data as a nested dictionary of DataFrames.
    """
    # 1. Delegate HDF5 Extraction to the core engine.
    extracted_dataframes = file_var_retrieval(
        month_file=month_file,
        retrieval_func=spatial_mean_granular(granularity)
    )

    if not extracted_dataframes:
        logger.warning(f"No data available to aggregate in {month_file.name}.")
        return {}

    # Pre-fetch the grand means for all variables to save I/O time
    grand_mean_dict = file_var_retrieval(
        month_file=month_file,
        retrieval_func=grand_mean_retrieval
    )

    results = {}

    for var, coarsened_grid in extracted_dataframes.items():
        if target_variable and var != target_variable:
            continue

        logger.info(f"Aggregating '{var}' at {granularity}° granularity...")

        # Zonal Means (Average across longitudes for each latitude -> axis=1)
        zonal_mean_df = coarsened_grid.mean(axis=1).to_frame(name="Zonal_Mean")

        # Meridional Means (Average across latitudes for each longitude -> axis=0)
        meridional_mean_df = coarsened_grid.mean(axis=0).to_frame(name="Meridional_Mean")

        # Safely extract Grand Mean from the pre-fetched dictionary
        if var in grand_mean_dict and not grand_mean_dict[var].empty:
            grand_mean = grand_mean_dict[var].iat[0, 0]
        else:
            grand_mean = float('nan')

        # Store in results dictionary
        results[var] = {
            "spatial_grid": coarsened_grid,
            "zonal_mean": zonal_mean_df,
            "meridional_mean": meridional_mean_df,
            "grand_mean": grand_mean
        }

    return results


# --- 3. PRINTING & EXPORTING ---

def print_monthly_spatial_summaries(
        month_file: Path,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Retrieves aggregated spatial data and prints a formatted statistical
    summary to the console.
    """
    stats_dictionary = monthly_granular_spatial_tables(
        month_file=month_file,
        target_variable=target_variable,
        granularity=granularity
    )

    if not stats_dictionary:
        return {}

    for var, stats in stats_dictionary.items():
        grand_mean = stats["grand_mean"]
        zonal_mean_df = stats["zonal_mean"]
        meridional_mean_df = stats["meridional_mean"]
        coarsened_grid = stats["spatial_grid"]

        print(f"\n{'='*60}")
        print(f"STATISTICAL SUMMARY: {var.upper()} ({month_file.stem}) | Granularity: {granularity}°")
        print(f"{'='*60}")
        print(f"Overall Grand Mean: {grand_mean:.4f}\n")

        print("--- Zonal Averages (By Latitude) ---")
        print(zonal_mean_df.head(5))
        print("...\n")

        print("--- Meridional Averages (By Longitude) ---")
        print(meridional_mean_df.head(5))
        print("...\n")

        print("--- Spatial Matrix Head (Lat x Lon) ---")
        print(coarsened_grid.iloc[:5, :5])
        print("\n")

    return stats_dictionary


def export_stats_to_excel(
        stats_dict: dict[str, dict],
        output_filename: str | Path = "climate_statistics.xlsx"
) -> Path:
    """
    Exports the nested statistics dictionary to a multi-sheet Excel workbook.
    """
    target_dir = Path(r"Excel exported statistical summaries")
    target_dir.mkdir(parents=True, exist_ok=True)

    file_name = Path(output_filename).name
    output_path = target_dir / file_name

    summary_data = []

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for var, stats in stats_dict.items():
                spatial_df = stats.get("spatial_grid")
                if spatial_df is not None:
                    spatial_df.to_excel(writer, sheet_name=f"{var[:15]}_Spatial")

                zonal_df = stats.get("zonal_mean")
                if zonal_df is not None:
                    zonal_df.to_excel(writer, sheet_name=f"{var[:15]}_Zonal")

                meridional_df = stats.get("meridional_mean")
                if meridional_df is not None:
                    meridional_df.to_excel(writer, sheet_name=f"{var[:15]}_Meridional")

                grand_mean = stats.get("grand_mean")
                if grand_mean is not None:
                    summary_data.append({"Variable": var, "Grand_Mean": grand_mean})

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="00_Overall_Summary", index=False)

        logger.info(f"Successfully exported statistical tables to {output_path.absolute()}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to export to Excel: {e}")
        return None


# --- 4. VISUALIZATION ENGINE ---

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


def plot_generation(
        valid_plots_data: dict[str, pd.DataFrame],
        plot_title: str
) -> Optional[plt.Figure]:
    """
    Generates an interactive 3D plot window with floor contours for all
    variables provided in the dataset dictionary.
    """
    if not valid_plots_data:
        logger.warning("No data provided to the rendering engine.")
        return None

    num_vars = len(valid_plots_data)
    cols = math.ceil(math.sqrt(num_vars))
    rows = math.ceil(num_vars / cols)

    fig = plt.figure(figsize=(7 * cols, 6 * rows))
    fig.suptitle(plot_title, fontsize=16, fontweight='bold')

    for index, (var_name, df) in enumerate(valid_plots_data.items()):
        ax = fig.add_subplot(rows, cols, index + 1, projection='3d')
        plot_3d_surface_on_axis(
            ax=ax,
            lons=df.columns.values,
            lats=df.index.values,
            data_2d=df.values,
            var_name=var_name
        )

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


def generate_monthly_visualizations(
        month_file: Path,
        granularity: float = None,
        verbose: bool = False
) -> None:
    """
    Integration function: Reads a monthly climate file and coordinates
    the generation of interactive 3D visualizations.
    """
    logger.info(f"Preparing visualizations for {month_file.name}...")

    # Added granularity parameter so you can visualize coarsened data if you want to!
    valid_plots_data = file_var_retrieval(
        month_file,
        retrieval_func=spatial_mean_granular(granularity),
        verbose=verbose
    )

    if not valid_plots_data:
        logger.warning(f"Aborting visualization: No valid data found in {month_file.name}.")
        return

    plot_title = f"Monthly Temporal Means: {month_file.stem}"
    fig = plot_generation(valid_plots_data=valid_plots_data, plot_title=plot_title)

    if fig:
        plt.show()


def export_3d_plots_to_pdf(
        month_file: Path,
        retrieval_func: Callable[[str, h5py.File | h5py.Group], Optional[pd.DataFrame]],
        output_filename: str | Path = "climate_plots.pdf",
        verbose: bool = False
) -> Path:
    """
    Acts as a pure output router: Requests data extraction, utilizes the central
    rendering engine, and saves perfectly centralized separate PDF pages.
    """
    if not month_file.exists():
        logger.error(f"Target file does not exist: {month_file.absolute()}")
        return None

    target_dir = Path("Exported pdf plots")
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / Path(output_filename).name

    valid_plots_data = file_var_retrieval(month_file, retrieval_func, verbose=verbose)

    if not valid_plots_data:
        logger.warning(f"No variables passed validation for PDF plotting in {month_file.name}.")
        return None

    logger.info(f"Generating PDF with {len(valid_plots_data)} pages in '{target_dir.name}'...")

    try:
        with PdfPages(output_path) as pdf:
            for var_name, df in valid_plots_data.items():
                single_var_dict = {var_name: df}
                plot_title = f"Temporal Mean: {var_name.upper()} ({month_file.stem})"

                fig = plot_generation(valid_plots_data=single_var_dict, plot_title=plot_title)

                if fig:
                    pdf.savefig(fig, bbox_inches='tight', pad_inches=0.5)
                    plt.close(fig)

                    if verbose:
                        logger.info(f"Rendered centralized PDF page for variable: {var_name}")

        logger.info(f"Successfully exported 3D plots to PDF: {output_path.absolute()}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to construct PDF: {e}")
        return None


if __name__ == '__main__':
    source_dir = Path("era5_monthly_data")
    test_file = source_dir / "era5_3d_2004_06.h5"

    if test_file.exists():
        logger.info(f"Starting analysis and visualization for: {test_file.name}\n")

        # 1. Spatial Tables
        stats_dictionary = print_monthly_spatial_summaries(
            month_file=test_file,
            granularity=1.0
        )

        # 2. Excel Export
        if stats_dictionary:
            excel_filename = f"statistics_summary_{test_file.stem}.xlsx"
            export_stats_to_excel(
                stats_dict=stats_dictionary,
                output_filename=excel_filename
            )

        # 3. PDF Export
        pdf_filename = f"3d_visualizations_{test_file.stem}.pdf"
        export_3d_plots_to_pdf(
            month_file=test_file,
            retrieval_func=spatial_mean_granular(),
            output_filename=pdf_filename,
            verbose=True
        )

        # 4. Interactive Display
        generate_monthly_visualizations(test_file, verbose=True)
        logger.info("Pipeline execution completed successfully.")

    else:
        logger.error(f"Test file not found at: {test_file.absolute()}")