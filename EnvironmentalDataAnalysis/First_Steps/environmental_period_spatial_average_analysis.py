"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: environmental_period_spatial_average_analysis.py
Author: Tiago TOLOCZKO ROSS
(File is becoming obsolete. Avoid using)

Description:
Period statistical aggregation module. Iterates over a designated time range of
monthly 3D HDF5 climate data, leveraging the monthly aggregation function to
efficiently calculate the overall spatial, zonal, meridional, and grand means
for the entire period without overwhelming system memory.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# --- Explicit Core Imports ---
from environmental_data_retrieval import monthly_data_directory_exploration
from environmental_data_visualization_orchestration import pdf_exporting_function

# --- Explicit Analysis Imports ---
from environmental_spatial_average_analysis import (
    monthly_granular_spatial_tables,
    plot_3d_surface_on_axis,
    granular_spatial_average,
    export_stats_to_excel
)

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def period_granular_spatial_average_tables(
        data_dir: Path,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, list[pd.DataFrame]]:
    """
    Leverages the directory explorer to find valid files, and routes them
    through the monthly aggregation function to extract and accumulate the
    coarsened spatial grids chronologically.
    """
    target_files, period_bounds = monthly_data_directory_exploration(data_dir)

    if not target_files:
        logger.warning(f"No valid files found in {data_dir.name} to process for the period.")
        return {}

    accumulated_grids = {}

    if period_bounds:
        s_m, s_y, e_m, e_y = period_bounds
        logger.info(f"Initiating period extraction across {len(target_files)} months "
                    f"({s_m:02d}/{s_y} to {e_m:02d}/{e_y})...")

    for file in target_files:
        logger.debug(f"Processing monthly stats for {file.name}...")

        # This single call natively handles HDF5 extraction, target filtering, AND spatial coarsening
        monthly_stats = monthly_granular_spatial_tables(
            month_file=file,
            target_variable=target_variable,
            granularity=granularity
        )

        for var, stats in monthly_stats.items():
            if var not in accumulated_grids:
                accumulated_grids[var] = []

            # Extract ONLY the formatted 2D spatial grid
            accumulated_grids[var].append(stats["spatial_grid"])

    return accumulated_grids


def period_averages_calculation(
        data_dir: Path,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Coordinates the extraction of monthly grids and calculates the final
    mathematical overall period averages (Spatial, Zonal, Meridional, Grand).
    """
    accumulated_grids = period_granular_spatial_average_tables(
        data_dir=data_dir,
        target_variable=target_variable,
        granularity=granularity
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
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Retrieves aggregated spatial data for the auto-discovered time period
    and prints a formatted statistical summary to the console.
    """
    # Temporarily suppress lower level logs during calculation
    logging.getLogger().setLevel(logging.WARNING)
    stats_dictionary = period_averages_calculation(data_dir, target_variable, granularity)
    logging.getLogger().setLevel(logging.INFO)

    if not stats_dictionary:
        return {}

    _, bounds = monthly_data_directory_exploration(data_dir)
    period_str = f"{bounds[0]:02d}/{bounds[1]} to {bounds[2]:02d}/{bounds[3]}" if bounds else "Unknown Range"

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
        valid_plots_data: dict[str, list[pd.DataFrame] | pd.DataFrame],
        plot_title: str
) -> dict[str, list[plt.Figure]]:
    """
    Generates interactive 3D surface subplots for variables across multiple time steps.
    Outputs a dictionary keyed by variable, containing a list of figures where
    Index 0 is the mathematical spatial average of the period, and Index 1..N
    are the individual monthly spatial maps.
    """
    if not valid_plots_data:
        logger.warning("No data provided to the rendering engine for 3D plotting.")
        return {}

    generated_figures: dict[str, list[plt.Figure]] = {}

    for var, data in valid_plots_data.items():
        # Normalize to ensure we are always working with a list
        df_list = data if isinstance(data, list) else [data]
        num_steps = len(df_list)
        var_figures = []

        # ==========================================
        # 1. OVERALL PERIOD PLOT (Index 0)
        # ==========================================
        logger.info(f"Rendering OVERALL spatial average for: {var.upper()}")

        # Stack all monthly arrays and calculate the NaN-safe mathematical mean
        stacked = np.stack([df.values for df in df_list], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_vals = np.nanmean(stacked, axis=0)

        overall_df = pd.DataFrame(mean_vals, index=df_list[0].index, columns=df_list[0].columns)

        fig_overall = plt.figure(figsize=(10, 8))
        fig_overall.suptitle(f"{plot_title}: {var.upper()} (OVERALL AVERAGE)", fontsize=16, fontweight='bold')
        ax_overall = fig_overall.add_subplot(111, projection='3d')

        plot_3d_surface_on_axis(
            ax=ax_overall,
            lons=overall_df.columns.values,
            lats=overall_df.index.values,
            data_2d=overall_df.values,
            var_name=var
        )
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        var_figures.append(fig_overall)

        # ==========================================
        # 2. MONTHLY STEP PLOTS (Index 1..N)
        # ==========================================
        for step, df_step in enumerate(df_list):
            step_title = f"{plot_title}: {var.upper()} (Step {step + 1}/{num_steps})" if num_steps > 1 else f"{plot_title}: {var.upper()}"

            fig_step = plt.figure(figsize=(10, 8))
            fig_step.suptitle(step_title, fontsize=16, fontweight='bold')
            ax_step = fig_step.add_subplot(111, projection='3d')

            plot_3d_surface_on_axis(
                ax=ax_step,
                lons=df_step.columns.values,
                lats=df_step.index.values,
                data_2d=df_step.values,
                var_name=var
            )
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            var_figures.append(fig_step)

        # Assign the bundled list to the variable key
        generated_figures[var] = var_figures

    return generated_figures


def export_period_3d_plots_to_pdf(
        stats_dictionary: dict[str, dict],
        period_label: str,
        output_filename: str | Path = "period_climate_plots.pdf",
        verbose: bool = False
) -> Optional[Path]:
    """
    Saves the calculated period averages as centralized pages in a PDF document.
    Delegates rendering logic to plot_period_3d_surfaces to maintain DRY principles.
    """
    if not stats_dictionary:
        logger.warning("No data provided for PDF export.")
        return None

    target_dir = Path("../Exported pdf plots")
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / Path(output_filename).name

    logger.info(f"Generating Period PDF with {len(stats_dictionary)} pages in '{target_dir.name}'...")

    try:
        with PdfPages(output_path) as pdf:
            for var, stats in stats_dictionary.items():

                # Format data to fit the universal plotter signature
                single_var_data = {var: stats["spatial_grid"]}
                plot_title = f"Period Temporal Mean: {var.upper()} ({period_label})"

                # Delegate to the standard plotter (Returns dict[str, list[plt.Figure]])
                returned_figs = plot_period_3d_surfaces(single_var_data, plot_title)

                if returned_figs and var in returned_figs:
                    # Index 0 is the mathematical OVERALL average figure
                    fig_overall = returned_figs[var][0]
                    pdf.savefig(fig_overall, bbox_inches='tight', pad_inches=0.5)

                    if verbose:
                        logger.info(f"Rendered centralized PDF page with contour for: {var}")

                    # Safely close all figures generated for this variable to free up RAM
                    for fig in returned_figs[var]:
                        plt.close(fig)

        logger.info(f"Successfully exported 3D period plots to PDF: {output_path.absolute()}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to construct PDF: {e}")
        return None


if __name__ == '__main__':
    target_directory = Path("../Analysis - first round/era5_monthly_data")

    if not target_directory.exists():
        logger.error(f"Cannot run tests: The directory '{target_directory}' does not exist.")
    else:
        # Dynamically fetch the period string for file naming and titles
        _, bounds = monthly_data_directory_exploration(target_directory)
        period_label_str = f"{bounds[0]:02d}-{bounds[1]} to {bounds[2]:02d}-{bounds[3]}" if bounds else "Unknown_Period"
        clean_period_str = period_label_str.replace(" ", "_").replace("/", "_")

        # 1. Calculate Period Statistics
        logger.info("\n--- 1. Calculating Period Statistics ---")
        period_stats = print_period_spatial_summaries(
            data_dir=target_directory,
            granularity=.5  # Kept at 1.0 to prevent console flood
        )

        if period_stats:
            # 2. Export Aggregated Period Stats to Excel
            logger.info("\n--- 2. Exporting Period Data to Excel ---")
            excel_filename = f"period_stats_{clean_period_str}.xlsx"
            export_stats_to_excel(period_stats, output_filename=excel_filename)

            # 3. Export Aggregated Period Stats to PDF (Using specific Period PDF generator)
            logger.info("\n--- 3. Exporting Period Data to PDF ---")
            pdf_filename = f"period_visualizations_{clean_period_str}.pdf"
            export_period_3d_plots_to_pdf(
                stats_dictionary=period_stats,
                period_label=period_label_str,
                output_filename=pdf_filename,
                verbose=True
            )

            # 4. Batch Export ALL Monthly Files to PDFs using the Universal Router

            # 5. Display Interactive Period Dashboard (Blocks execution until closed)
            logger.info("\n--- 5. Launching Interactive Dashboard ---")
            plot_period_3d_surfaces(
                valid_plots_data={var: stats["spatial_grid"] for var, stats in period_stats.items()},
                plot_title=f"Overall Period Mean ({period_label_str})"
            )
            plt.show()