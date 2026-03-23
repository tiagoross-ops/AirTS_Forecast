"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: Data_Visualization.py
Author: Tiago TOLOCZKO ROSS

Description:
Data visualization and statistical analysis module for the AirTS-Forecast pipeline.
This script reads 3-dimensional climate data from HDF5 files, performs strict
structural validation, calculates temporal means across the spatial grid, and
generates interactive 3D surface plots using Matplotlib. It is designed to verify
the integrity and spatial distribution of the ETL pipeline outputs.
"""
import warnings
import logging
import math
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def var_description_by_month(month_file: Path, verbose: bool = False) -> None:
    """
    Reads 3D climate data, validates HDF5 structure, calculates temporal means,
    and generates a single window containing 3D surface subplots.
    """
    if not month_file.exists():
        logger.error(f"Target file does not exist: {month_file.absolute()}")
        return

    # List to store successfully validated and extracted data
    valid_plots_data = []

    # --- PASS 1: Data Extraction and Validation ---
    try:
        with h5py.File(month_file, mode="r") as h5f:

            # 1. Type Checking: Strictly isolate h5py.Dataset objects
            # This automatically ignores groups like '_coordinates'
            main_vars = [key for key in h5f.keys() if isinstance(h5f[key], h5py.Dataset)]

            if not main_vars:
                logger.warning(f"No valid HDF5 Datasets found in the root of {month_file.name}.")
                return

            for var in main_vars:
                try:
                    # 2. Key Validation: Ensure coordinate group exists
                    coord_group_name = f"{var}_coordinates"
                    if coord_group_name not in h5f:
                        logger.warning(f"Skipping '{var}': Coordinate group '{coord_group_name}' not found.")
                        continue

                    coord_group = h5f[coord_group_name]

                    # 3. Key Validation: Ensure latitude and longitude exist
                    if 'latitude' not in coord_group or 'longitude' not in coord_group:
                        logger.warning(f"Skipping '{var}': Missing 'latitude' or 'longitude' in its coordinate group.")
                        continue

                    lats = coord_group['latitude'][:]
                    lons = coord_group['longitude'][:]

                    # 4. Dimensionality Validation: Ensure the data is at least 3D (Time, Lat, Lon)
                    ds = h5f[var]
                    if len(ds.shape) < 3:
                        logger.warning(f"Skipping '{var}': Expected 3D tensor, got shape {ds.shape}.")
                        continue

                    # Extract the array and calculate the temporal mean (Axis 0)
                    data_3d = ds[:]
                    time_mean_2d = np.nanmean(data_3d, axis=0)

                    # 5. Geometry Validation: Ensure the 2D mean matches the coordinate grid size
                    if time_mean_2d.shape != (len(lats), len(lons)):
                        logger.warning(
                            f"Skipping '{var}': Shape mismatch. Mean is {time_mean_2d.shape}, "
                            f"but coordinates dictate ({len(lats)}, {len(lons)})."
                        )
                        continue

                    # If all checks pass, store the data for plotting
                    valid_plots_data.append({
                        'var_name': var,
                        'mean_2d': time_mean_2d,
                        'lats': lats,
                        'lons': lons
                    })

                    if verbose:
                        logger.info(f"Successfully validated and processed '{var}'.")

                except Exception as var_e:
                    # Catch variable-specific errors so the loop can process the remaining variables
                    logger.error(f"Unexpected error processing variable '{var}': {var_e}")
                    continue

    except OSError as e:
        logger.error(f"OS Error: Failed to open or read HDF5 file {month_file.name}: {e}")
        return
    except Exception as e:
        logger.critical(f"Critical Error during extraction of {month_file.name}: {e}")
        return

    # --- PASS 2: Grid Calculation and Visualization ---
    num_vars = len(valid_plots_data)

    if num_vars == 0:
        logger.warning(f"No variables passed validation for plotting in {month_file.name}.")
        return

    # Calculate optimal subplot grid
    cols = math.ceil(math.sqrt(num_vars))
    rows = math.ceil(num_vars / cols)

    # Initialize dynamic figure
    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    fig.suptitle(f"Monthly Temporal Means: {month_file.stem}", fontsize=16, fontweight='bold')

    # Render each validated dataset
    for index, plot_data in enumerate(valid_plots_data):
        ax = fig.add_subplot(rows, cols, index + 1, projection='3d')
        plot_3d_surface_on_axis(
            ax=ax,
            lons=plot_data['lons'],
            lats=plot_data['lats'],
            data_2d=plot_data['mean_2d'],
            var_name=plot_data['var_name']
        )

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


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
    # Calculate Z-axis boundaries to place the floor map correctly
    z_max = np.nanmax(data_2d)
    z_min = np.nanmin(data_2d)
    z_range = z_max - z_min

    # Define the floor level (10% below the absolute lowest data point)
    floor_z = z_min - (z_range * 0.1)

    # Create the 2D Cartesian grid required for 3D surfaces
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # 1. Render the main 3D surface plot
    surf = ax.plot_surface(
        lon_grid, lat_grid, data_2d,
        cmap='viridis',
        edgecolor='none',
        alpha=0.9
    )

    # 2. Render the 2D geographical contour onto the floor (XY plane)
    ax.contourf(
        lon_grid, lat_grid, data_2d,
        zdir='z',
        offset=floor_z,
        cmap='viridis',
        alpha=0.6  # Slightly transparent to look like a shadow map
    )

    # Lock the Z-axis limits so the floor sits tightly at the bottom
    ax.set_zlim(floor_z, z_max)

    # Format axis labels and title
    ax.set_title(f"Variable: {var_name}", fontweight='bold', fontsize=12)
    ax.set_xlabel("Longitude", labelpad=10)
    ax.set_ylabel("Latitude", labelpad=10)
    ax.set_zlabel("Time Average Value", labelpad=10)

    # Attach a color bar mapped to the surface values, linked to the parent figure
    ax.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)

    # Set default camera viewing angle (adjusted slightly to match your period plots)
    ax.view_init(elev=25, azim=230)


def generate_monthly_spatial_average_tables(
        month_file: Path,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Calculates aggregated spatial and temporal averages from a 3D HDF5 dataset.
    Returns the data strictly as a nested dictionary of Pandas DataFrames.

    Args:
        month_file (Path): Path to the target .h5 file.
        target_variable (str, optional): Specific variable to analyze.
        granularity (float): The spatial resolution for the output tables in degrees.

    Returns:
        dict: A nested dictionary containing the aggregated DataFrames.
    """
    if not month_file.exists():
        logger.error(f"File not found: {month_file.absolute()}")
        return {}

    results = {}

    try:
        with h5py.File(month_file, mode="r") as h5f:

            if target_variable:
                if target_variable not in h5f:
                    logger.error(f"Variable '{target_variable}' not found in {month_file.name}.")
                    return {}
                main_vars = [target_variable]
            else:
                main_vars = [key for key in h5f.keys() if isinstance(h5f[key], h5py.Dataset)]

            for var in main_vars:
                logger.info(f"Aggregating '{var}' at {granularity}° granularity...")

                coord_group = h5f[f"{var}_coordinates"]
                lats = coord_group['latitude'][:]
                lons = coord_group['longitude'][:]
                data_3d = h5f[var][:]

                # 1. Temporal Mean
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    time_mean_2d = np.nanmean(data_3d, axis=0)

                native_grid_df = pd.DataFrame(time_mean_2d, index=lats, columns=lons)

                # 2. Spatial Coarsening
                binned_lats = np.round(native_grid_df.index / granularity) * granularity
                binned_lons = np.round(native_grid_df.columns / granularity) * granularity

                coarsened_grid = native_grid_df.groupby(binned_lats).mean()
                coarsened_grid = coarsened_grid.T.groupby(binned_lons).mean().T

                coarsened_grid.index = np.round(coarsened_grid.index, 4)
                coarsened_grid.columns = np.round(coarsened_grid.columns, 4)
                coarsened_grid.index.name = "Latitude"
                coarsened_grid.columns.name = "Longitude"

                # 3. Zonal Means (Average across longitudes for each latitude -> axis=1)
                zonal_mean_df = coarsened_grid.mean(axis=1).to_frame(name="Zonal_Mean")

                # 4. Meridional Means (Average across latitudes for each longitude -> axis=0)
                meridional_mean_df = coarsened_grid.mean(axis=0).to_frame(name="Meridional_Mean")

                # 5. Grand Mean
                grand_mean = float(np.nanmean(data_3d))

                # Store in results dictionary
                results[var] = {
                    "spatial_grid": coarsened_grid,
                    "zonal_mean": zonal_mean_df,
                    "meridional_mean": meridional_mean_df,
                    "grand_mean": grand_mean
                }

    except Exception as e:
        logger.error(f"Error calculating statistics for {month_file.name}: {e}")

    return results


def print_monthly_spatial_summaries(
        month_file: Path,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Retrieves aggregated spatial data and prints a formatted statistical
    summary to the console.

    Returns:
        dict: The generated statistics dictionary, so it can be passed
              to other functions (like Excel export) after printing.
    """
    # Call the calculation function
    stats_dictionary = generate_monthly_spatial_average_tables(
        month_file=month_file,
        target_variable=target_variable,
        granularity=granularity
    )

    if not stats_dictionary:
        return {}

    # Print the formatted summaries
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
    Exports the nested statistics dictionary to a multi-sheet Excel workbook
    and saves it inside a dedicated "Excel exported statistical summaries" folder.

    Args:
        stats_dict (dict): The output dictionary from generate_monthly_spatial_average_tables.
        output_filename (str | Path): The target Excel file name.

    Returns:
        Path: The absolute path to the generated Excel file.
    """
    # 1. Define and create the target directory
    target_dir = Path(r"Excel exported statistical summaries")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. Extract just the file name (in case a full path was passed) and append it to the new directory
    file_name = Path(output_filename).name
    output_path = target_dir / file_name

    summary_data = []

    try:
        # Use pd.ExcelWriter to handle multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for var, stats in stats_dict.items():

                # 1. Export Spatial Grid
                spatial_df = stats.get("spatial_grid")
                if spatial_df is not None:
                    # Truncate variable name if necessary to fit Excel's 31-char limit
                    spatial_df.to_excel(writer, sheet_name=f"{var[:15]}_Spatial")

                # 2. Export Zonal Means (Latitudes)
                zonal_df = stats.get("zonal_mean")
                if zonal_df is not None:
                    zonal_df.to_excel(writer, sheet_name=f"{var[:15]}_Zonal")

                # 3. Export Meridional Means (Longitudes)
                meridional_df = stats.get("meridional_mean")
                if meridional_df is not None:
                    meridional_df.to_excel(writer, sheet_name=f"{var[:15]}_Meridional")

                # 4. Collect Grand Mean for the summary sheet
                grand_mean = stats.get("grand_mean")
                if grand_mean is not None:
                    summary_data.append({"Variable": var, "Grand_Mean": grand_mean})

            # 5. Create the Overall Summary Sheet
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                # Save summary as the first/last sheet
                summary_df.to_excel(writer, sheet_name="00_Overall_Summary", index=False)

        logger.info(f"Successfully exported statistical tables to {output_path.absolute()}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to export to Excel: {e}")
        return None


def export_3d_plots_to_pdf(
        month_file: Path,
        output_filename: str | Path = "climate_plots.pdf",
        verbose: bool = False
) -> Path:
    """
    Reads 3D climate data, validates the structure, and saves each variable's
    3D surface plot as a separate page in a multi-page PDF document.
    Automatically routes the output to an "Exported pdf plots" folder.

    Args:
        month_file (Path): Path to the target .h5 file.
        output_filename (str | Path): The target PDF file name.
        verbose (bool): If True, enables debug logging.

    Returns:
        Path: The absolute path to the generated PDF file.
    """
    if not month_file.exists():
        logger.error(f"Target file does not exist: {month_file.absolute()}")
        return None

    # 1. Define and create the target directory for PDFs
    target_dir = Path("Exported pdf plots")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. Extract just the file name and append it to the new directory
    file_name = Path(output_filename).name
    output_path = target_dir / file_name

    valid_plots_data = []

    # --- PASS 1: Data Extraction and Validation ---
    try:
        with h5py.File(month_file, mode="r") as h5f:
            main_vars = [key for key in h5f.keys() if isinstance(h5f[key], h5py.Dataset)]
            if not main_vars:
                logger.warning(f"No valid HDF5 Datasets found in {month_file.name}.")
                return None

            for var in main_vars:
                try:
                    coord_group = h5f.get(f"{var}_coordinates")
                    if not coord_group or 'latitude' not in coord_group or 'longitude' not in coord_group:
                        continue

                    lats = coord_group['latitude'][:]
                    lons = coord_group['longitude'][:]
                    ds = h5f[var]

                    if len(ds.shape) < 3:
                        continue

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        time_mean_2d = np.nanmean(ds[:], axis=0)

                    if time_mean_2d.shape == (len(lats), len(lons)):
                        valid_plots_data.append({
                            'var_name': var,
                            'mean_2d': time_mean_2d,
                            'lats': lats,
                            'lons': lons
                        })
                except Exception as var_e:
                    logger.error(f"Error validating '{var}': {var_e}")
                    continue
    except Exception as e:
        logger.critical(f"Critical Error extracting {month_file.name}: {e}")
        return None

    if not valid_plots_data:
        logger.warning(f"No variables passed validation for PDF plotting in {month_file.name}.")
        return None

    # --- PASS 2: PDF Generation ---
    logger.info(f"Generating PDF with {len(valid_plots_data)} pages in '{target_dir.name}'...")

    try:
        # Use PdfPages context manager to handle the multi-page document
        with PdfPages(output_path) as pdf:
            for plot_data in valid_plots_data:
                # Create a fresh figure for each page
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Render the 3D surface using your existing modular function
                plot_3d_surface_on_axis(
                    ax=ax,
                    lons=plot_data['lons'],
                    lats=plot_data['lats'],
                    data_2d=plot_data['mean_2d'],
                    var_name=plot_data['var_name']
                )

                # Add a master title to the top of the PDF page
                fig.suptitle(
                    f"Temporal Mean: {plot_data['var_name'].upper()} ({month_file.stem})",
                    fontsize=16,
                    fontweight='bold'
                )

                plt.tight_layout()

                # Save the current figure to the PDF document as a new page
                pdf.savefig(fig)

                # CRITICAL: Close the figure to free memory.
                plt.close(fig)

                if verbose:
                    logger.info(f"Rendered page for variable: {plot_data['var_name']}")

        logger.info(f"Successfully exported 3D plots to PDF: {output_path.absolute()}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to construct PDF: {e}")
        return None


if __name__ == '__main__':
    # Use a relative path so the script executes correctly on any cloned repository
    # Assumes the script is run from the root of the AirTS-Forecast project
    source_dir = Path("era5_monthly_data")
    test_file = source_dir / "era5_3d_2004_07.h5"

    # Check if the data file exists to prevent unhandled OS errors
    if test_file.exists():
        logger.info(f"Starting analysis and visualization for: {test_file.name}\n")

        # 1. Generate the spatial statistics dictionary
        # Set target_variable=None to process all variables, and adjust granularity as needed
        stats_dictionary = print_monthly_spatial_summaries(
            month_file=test_file,
            granularity=1.0
        )


        # 2. Export the statistics to a multi-sheet Excel workbook
        if stats_dictionary:
            excel_filename = f"statistics_summary_{test_file.stem}.xlsx"
            export_stats_to_excel(
                stats_dict=stats_dictionary,
                output_filename=excel_filename
            )

        # 3. Export the 3D surface plots to a multi-page PDF document
        pdf_filename = f"3d_visualizations_{test_file.stem}.pdf"
        export_3d_plots_to_pdf(
            month_file=test_file,
            output_filename=pdf_filename,
            verbose=True
        )

        # 4. Show Interactive 3D Matplotlib Window (Optional)
        # Note: This will pause the script until you manually close the window.
        # It is active by default here, but you can comment it out if running a bulk automated pipeline.
        var_description_by_month(test_file, verbose=True)

        logger.info("Pipeline execution completed successfully.")

    else:
        logger.error(f"Test file not found at: {test_file.absolute()}")
        logger.info("Ensure the ETL pipeline has successfully generated the .h5 data before running the visualization.")