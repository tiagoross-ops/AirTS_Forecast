"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: Period_Statistics.py
Author: Tiago TOLOCZKO ROSS

Description:
Period statistical aggregation module. Iterates over a designated time range of
monthly 3D HDF5 climate data, leveraging the monthly aggregation function to
efficiently calculate the overall spatial, zonal, meridional, and grand means
for the entire period without overwhelming system memory.
"""

import logging
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# --- Import from your existing modules ---
# (Adjust module names if your files are named differently)
from environmental_monthly_mean_analysis import (
    export_stats_to_excel,
    var_description_by_month,
    plot_3d_surface_on_axis
)

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def monthly_directory_iteration(
        data_dir: Path,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
) -> list[Path]:
    """
    Scans the target directory and generates an ordered list of valid HDF5 file
    paths for a specific multi-month period.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir.absolute()}")
        return []

    target_files = []
    for y in range(start_year, end_year + 1):
        m_start = start_month if y == start_year else 1
        m_end = end_month if y == end_year else 12
        for m in range(m_start, m_end + 1):
            expected_file = data_dir / f"era5_3d_{y}_{m:02d}.h5"
            if expected_file.exists():
                target_files.append(expected_file)
            else:
                logger.warning(f"Missing expected file in sequence: {expected_file.name}")

    if not target_files:
        logger.error("No valid HDF5 files found for the specified period.")

    return target_files


def generate_period_spatial_average_tables(
        data_dir: Path,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, list[pd.DataFrame]]:
    """
    Iterates through monthly HDF5 files using `var_description_by_month`,
    applies spatial coarsening if required, and accumulates them into a list.
    """
    target_files = monthly_directory_iteration(
        data_dir, start_year, start_month, end_year, end_month
    )

    if not target_files:
        return {}

    accumulated_grids = {}

    logger.info(f"Initiating period extraction across {len(target_files)} months...")

    for file in target_files:
        logger.debug(f"Extracting spatial grid from {file.name}...")

        # Extract monthly dataframes using the refactored monthly function
        monthly_dataframes = var_description_by_month(month_file=file, verbose=False)

        for var, df in monthly_dataframes.items():
            # Filter by target_variable if specified
            if target_variable and var != target_variable:
                continue

            # Apply Spatial Coarsening (Granularity)
            if granularity:
                binned_lats = np.round(df.index / granularity) * granularity
                binned_lons = np.round(df.columns / granularity) * granularity

                df = df.groupby(binned_lats).mean()
                df = df.T.groupby(binned_lons).mean().T

                df.index = np.round(df.index, 4)
                df.columns = np.round(df.columns, 4)
                df.index.name = "Latitude"
                df.columns.name = "Longitude"

            # Store in accumulation list
            if var not in accumulated_grids:
                accumulated_grids[var] = []

            accumulated_grids[var].append(df)

    return accumulated_grids


def period_averages_calculation(
        data_dir: Path,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Coordinates the extraction of monthly grids and calculates the final
    mathematical overall period averages (Spatial, Zonal, Meridional, Grand).
    """
    accumulated_grids = generate_period_spatial_average_tables(
        data_dir, start_year, start_month, end_year, end_month, target_variable, granularity
    )

    period_results = {}

    if not accumulated_grids:
        logger.warning("No accumulated grids to calculate period averages.")
        return period_results

    for var, grid_list in accumulated_grids.items():
        logger.info(f"Calculating final period mathematical averages for '{var}'...")

        stacked_grids = np.stack([df.values for df in grid_list], axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            period_mean_values = np.nanmean(stacked_grids, axis=0)

        period_spatial_grid = pd.DataFrame(
            period_mean_values,
            index=grid_list[0].index,
            columns=grid_list[0].columns
        )

        zonal_mean_df = period_spatial_grid.mean(axis=1).to_frame(name="Period_Zonal_Mean")
        meridional_mean_df = period_spatial_grid.mean(axis=0).to_frame(name="Period_Meridional_Mean")
        grand_mean = float(np.nanmean(period_mean_values))

        period_results[var] = {
            "spatial_grid": period_spatial_grid,
            "zonal_mean": zonal_mean_df,
            "meridional_mean": meridional_mean_df,
            "grand_mean": grand_mean
        }

    return period_results


def print_period_spatial_summaries(
        data_dir: Path,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Retrieves aggregated spatial data for a time period and prints a formatted
    statistical summary to the console.
    """
    logging.getLogger().setLevel(logging.WARNING)

    stats_dictionary = period_averages_calculation(
        data_dir, start_year, start_month, end_year, end_month, target_variable, granularity
    )

    logging.getLogger().setLevel(logging.INFO)

    if not stats_dictionary:
        return {}

    period_str = f"{start_year}/{start_month:02d} to {end_year}/{end_month:02d}"

    for var, stats in stats_dictionary.items():
        grand_mean = stats["grand_mean"]
        zonal_mean_df = stats["zonal_mean"]
        meridional_mean_df = stats["meridional_mean"]
        coarsened_grid = stats["spatial_grid"]

        print(f"\n{'='*70}")
        print(f"PERIOD STATISTICAL SUMMARY: {var.upper()} | Range: {period_str} | Granularity: {granularity}°")
        print(f"{'='*70}")
        print(f"Overall Period Grand Mean: {grand_mean:.4f}\n")

        print("--- Period Zonal Averages (By Latitude) ---")
        print(zonal_mean_df.head(5))
        print("...\n")

        print("--- Period Meridional Averages (By Longitude) ---")
        print(meridional_mean_df.head(5))
        print("...\n")

        print("--- Period Spatial Matrix Head (Lat x Lon) ---")
        print(coarsened_grid.iloc[:5, :5])
        print("\n")

    return stats_dictionary


def plot_period_3d_surfaces(
        stats_dictionary: dict[str, dict],
        period_label: str
) -> None:
    """
    Generates interactive 3D surface subplots for all variables, using the
    centralized plotting engine.
    """
    if not stats_dictionary:
        logger.warning("No data provided for period 3D plotting.")
        return

    num_vars = len(stats_dictionary)

    cols = math.ceil(math.sqrt(num_vars))
    rows = math.ceil(num_vars / cols)

    fig = plt.figure(figsize=(7 * cols, 6 * rows))
    fig.suptitle(f"Period Temporal Means: {period_label}", fontsize=16, fontweight='bold')

    for index, (var, stats) in enumerate(stats_dictionary.items()):
        logger.info(f"Rendering 3D surface and bottom contour for period variable: {var}")

        spatial_df = stats["spatial_grid"]
        lats = spatial_df.index.values
        lons = spatial_df.columns.values
        data_2d = spatial_df.values

        ax = fig.add_subplot(rows, cols, index + 1, projection='3d')

        # Call the unified plotting engine
        plot_3d_surface_on_axis(
            ax=ax,
            lons=lons,
            lats=lats,
            data_2d=data_2d,
            var_name=f"{var} ({period_label})"
        )

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


def export_period_3d_plots_to_pdf(
        stats_dictionary: dict[str, dict],
        period_label: str,
        output_filename: str | Path = "period_climate_plots.pdf",
        verbose: bool = False
) -> Path:
    """
    Saves each variable's 3D surface plot as a centralized page in a PDF document
    using the unified plotting engine.
    """
    if not stats_dictionary:
        logger.warning("No data provided for PDF export.")
        return None

    target_dir = Path("Exported pdf plots")
    target_dir.mkdir(parents=True, exist_ok=True)

    file_name = Path(output_filename).name
    output_path = target_dir / file_name

    logger.info(f"Generating Period PDF with {len(stats_dictionary)} pages in '{target_dir.name}'...")

    try:
        with PdfPages(output_path) as pdf:
            for var, stats in stats_dictionary.items():

                fig = plt.figure(figsize=(11.69, 8.27))
                ax = fig.add_subplot(111, projection='3d')

                spatial_df = stats["spatial_grid"]
                lats = spatial_df.index.values
                lons = spatial_df.columns.values
                data_2d = spatial_df.values

                # Call the unified plotting engine
                plot_3d_surface_on_axis(
                    ax=ax,
                    lons=lons,
                    lats=lats,
                    data_2d=data_2d,
                    var_name=f"{var} ({period_label})"
                )

                # Nudge the master title so it doesn't overlap the plot title
                fig.suptitle(
                    f"Period Temporal Mean: {var.upper()}",
                    fontsize=16,
                    fontweight='bold',
                    y=0.96
                )

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.5)
                plt.close(fig)

                if verbose:
                    logger.info(f"Rendered centralized PDF page with contour for: {var}")

        logger.info(f"Successfully exported 3D period plots with contours to PDF: {output_path.absolute()}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to construct PDF: {e}")
        return None


if __name__ == '__main__':
    target_directory = Path("era5_monthly_data")

    # Example Period: March 2004 to April 2005
    s_year, s_month = 2004, 3
    e_year, e_month = 2005, 4
    period_label_str = f"{s_year}/{s_month:02d} to {e_year}/{e_month:02d}"

    # 1. Execute calculations
    period_stats = print_period_spatial_summaries(
        data_dir=target_directory,
        start_year=s_year,
        start_month=s_month,
        end_year=e_year,
        end_month=e_month,
        granularity=.5 # Set back to 1.0 to prevent massive console output
    )

    if period_stats:
        # 2. Export to Excel
        try:
            excel_filename = f"period_stats_{s_year}_{s_month:02d}_to_{e_year}_{e_month:02d}.xlsx"
            export_stats_to_excel(period_stats, output_filename=excel_filename)
        except Exception as e:
            logger.error(f"Failed to export Excel: {e}")

        # 3. Export to Centralized PDF
        pdf_filename = f"period_visualizations_{s_year}_{s_month:02d}_to_{e_year}_{e_month:02d}.pdf"
        export_period_3d_plots_to_pdf(
            stats_dictionary=period_stats,
            period_label=period_label_str,
            output_filename=pdf_filename,
            verbose=True
        )

        # 4. Show Interactive Window (Optional)
        plot_period_3d_surfaces(
            stats_dictionary=period_stats,
            period_label=period_label_str
        )