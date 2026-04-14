"""
AirTS-Forecast Project
Main Execution Module
Author: Tiago TOLOCZKO ROSS

Description:
The User Control Panel for the Environmental Data Analysis Pipeline.
Centralizes configuration and execution for all analysis modules:
    1. 3D Spatial Averages
    2. Regional Timeseries
    3. Global Grand Means
    4. 3D Climate Animations
    5. Statistical Excel Exports
"""

import logging
from pathlib import Path

# --- Import Master Orchestrators & Retrieval ---
from environmental_data_retrieval import period_retrieval_function, grand_mean_retrieval
from environmental_data_visualization_orchestration import visualization_orchestration

# --- Import Pipeline 1: Spatial Averages ---
from environmental_spatial_average_analysis import (
    granular_spatial_average,
    spatial_3d_plotter_factory,
    overall_period_spatial_average,
    calculate_monthly_spatial_statistics,
    export_stats_to_excel
)

# --- Import Pipeline 2: Regional Timeseries ---
from environmental_timeseries import (
    timeseries_by_region,
    timeseries_by_region_plotter_factory,
    overall_period_timeseries,
    print_sinusoidal_summaries,
    export_sinusoidal_stats_to_excel
)

# --- Import Pipeline 3: Grand Mean ---
from environmental_grand_mean_visualization import (
    grand_mean_plotter_factory,
    overall_period_grand_mean,
    print_grand_mean_sinusoidal_summaries,
    export_grand_mean_stats_to_excel
)

# --- Import Pipeline 4: Animations ---
from environmental_period_evolution import animate_spatial_evolution

# Configure global logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AirTS_Controller")

# Disable pyplot max open warning for large batch jobs
# plt.rcParams['figure.max_open_warning'] = 0


# =============================================================================
# USER CONFIGURATION PANEL
# =============================================================================

# 1. Target Data
TARGET_DIRECTORY = Path("era5_monthly_data")

# 2. Global Execution Settings
OBJECTIVE = 'show'          # Options: 'show' (Interactive Dashboard) or 'export' (PDF Generation)
SAVE_PNG_BACKUPS = True       # If objective is 'show', also saves 300dpi PNG copies
VERBOSE_LOGGING = False       # Set to True for deep page-by-page debugging

# 3. Spatial Resolution (Degrees)
# Note: Animations and 3D plotting require lots of RAM. A coarser grid (e.g., 2.0 or 5.0) is highly recommended.
SPATIAL_GRANULARITY = .1
TIMESERIES_GRANULARITY = .5

# 4. Pipeline Execution Toggles (Turn specific analyses True/False)
RUN_SPATIAL_AVERAGES = False
RUN_REGIONAL_TIMESERIES = False
RUN_GRAND_MEANS = False
RUN_ANIMATION_GIF = True     # Set True to generate the .gif evolution
IMAGE_SIZE = (10,12)

# 5. Output path
OUTPUT_ROOT_DIR = "Analysis - first round"
# =============================================================================
# PIPELINE EXECUTION ENGINE
# =============================================================================

def run_environmental_analysis():
    """Reads user configurations and triggers the requested analysis pipelines."""

    if not TARGET_DIRECTORY.exists():
        logger.error(f"FATAL: Target directory not found at {TARGET_DIRECTORY.absolute()}")
        return

    logger.info(f"=== Starting AirTS Analysis Pipeline ===")
    logger.info(f"Target: {TARGET_DIRECTORY.name} | Objective: {OBJECTIVE.upper()} | Granularity: {SPATIAL_GRANULARITY}°")

    # ---------------------------------------------------------
    # PIPELINE 1: 3D SPATIAL AVERAGES
    # ---------------------------------------------------------
    if RUN_SPATIAL_AVERAGES:
        logger.info("\n--- [1/4] Running 3D Spatial Average Pipeline ---")

        # 1. Orchestrate Visualization
        visualization_orchestration(
            input_path=TARGET_DIRECTORY,
            study="Spatial Average",
            retrieval_func=granular_spatial_average(granularity=SPATIAL_GRANULARITY),
            plot_generator_func=spatial_3d_plotter_factory(),
            overall_analysis=overall_period_spatial_average,
            objective=OBJECTIVE,
            save_png_backups=SAVE_PNG_BACKUPS,
            output_dir=f"{OUTPUT_ROOT_DIR}/Spatial Average Visualization",
            output_filename="spatial_averages.pdf",
            verbose=VERBOSE_LOGGING
        )

        # 2. Orchestrate Tabular Export (Using the final file in the dir as a sample)
        sample_file = list(TARGET_DIRECTORY.glob("*.h5"))[-1]
        stats_dict = calculate_monthly_spatial_statistics(sample_file, granularity=SPATIAL_GRANULARITY)
        export_stats_to_excel(stats_dict, output_filename=f"spatial_stats_{sample_file.stem}.xlsx",
                              output_dir=f"{OUTPUT_ROOT_DIR}/Excel exported statistical summaries")


    # ---------------------------------------------------------
    # PIPELINE 2: REGIONAL TIMESERIES & SINUSOIDAL FITS
    # ---------------------------------------------------------
    if RUN_REGIONAL_TIMESERIES:
        logger.info("\n--- [2/4] Running Regional Timeseries Pipeline ---")
        # 1. Orchestrate Visualization
        visualization_orchestration(
            input_path=TARGET_DIRECTORY,
            study="Regional Timeseries",
            retrieval_func=timeseries_by_region(),
            plot_generator_func=timeseries_by_region_plotter_factory(granularity=TIMESERIES_GRANULARITY,
                                                                     verbose=VERBOSE_LOGGING,
                                                                     figsize=IMAGE_SIZE),
            overall_analysis=overall_period_timeseries,
            objective=OBJECTIVE,
            save_png_backups=SAVE_PNG_BACKUPS,
            output_dir=f"{OUTPUT_ROOT_DIR}/Regional Timeseries Visualization",
            output_filename="regional_timeseries.pdf",
            verbose=VERBOSE_LOGGING
        )

        # 2. Orchestrate Tabular Export (Requires fetching the whole period)
        period_data = period_retrieval_function(TARGET_DIRECTORY, timeseries_by_region())
        if period_data:
            continuous_data = overall_period_timeseries(period_data)
            timeseries_stats = print_sinusoidal_summaries(continuous_data, granularity=TIMESERIES_GRANULARITY)
            export_sinusoidal_stats_to_excel(timeseries_stats,
                                             output_dir=f"{OUTPUT_ROOT_DIR}/Excel exported statistical summaries")


    # ---------------------------------------------------------
    # PIPELINE 3: GLOBAL GRAND MEANS
    # ---------------------------------------------------------
    if RUN_GRAND_MEANS:
        logger.info("\n--- [3/4] Running Global Grand Mean Pipeline ---")

        # 1. Orchestrate Visualization
        visualization_orchestration(
            input_path=TARGET_DIRECTORY,
            study="Global Grand Mean",
            retrieval_func=grand_mean_retrieval,
            plot_generator_func=grand_mean_plotter_factory(),
            overall_analysis=overall_period_grand_mean,
            objective=OBJECTIVE,
            save_png_backups=SAVE_PNG_BACKUPS,
            output_dir=f"{OUTPUT_ROOT_DIR}/Grand Mean Visualization",
            output_filename="grand_mean_analysis.pdf",
            verbose=VERBOSE_LOGGING
        )

        # 2. Orchestrate Tabular Export
        period_data = period_retrieval_function(TARGET_DIRECTORY, grand_mean_retrieval)
        if period_data:
            continuous_grand_means = overall_period_grand_mean(period_data)
            gm_stats_df = print_grand_mean_sinusoidal_summaries(continuous_grand_means)
            export_grand_mean_stats_to_excel(gm_stats_df,
                                             output_dir=f"{OUTPUT_ROOT_DIR}/Excel exported statistical summaries")


    # ---------------------------------------------------------
    # PIPELINE 4: SPATIAL ANIMATION
    # ---------------------------------------------------------
    if RUN_ANIMATION_GIF:
        logger.info("\n--- [4/4] Running Spatial Animation Pipeline ---")
        animate_spatial_evolution(
            data_dir=TARGET_DIRECTORY,
            granularity=TIMESERIES_GRANULARITY,
#            save_gif_path=f"{OUTPUT_ROOT_DIR}/Exported_Animations/spatial_climate_evolution.gif",
            fps=2
        )

    logger.info("\n=== Analysis Pipeline Completed Successfully ===")

    # Trigger Matplotlib display if 'show' was selected and visualizations were generated
    if OBJECTIVE == 'show':
        logger.info("Opening active interactive dashboards...")


if __name__ == '__main__':
    run_environmental_analysis()