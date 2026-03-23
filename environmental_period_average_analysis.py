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
from pathlib import Path

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Import the monthly calculation function from your existing module
# Adjust 'Data_Statistics' to match the actual name of your file if different
from environmental_monthly_mean_analysis import generate_monthly_spatial_average_tables
from matplotlib.backends.backend_pdf import PdfPages

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_period_spatial_average_tables(
        data_dir: Path,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Calculates aggregated spatial averages across a multi-month period.

    Args:
        data_dir (Path): Directory containing the monthly .h5 files.
        start_year (int): Starting year of the period.
        start_month (int): Starting month.
        end_year (int): Ending year.
        end_month (int): Ending month.
        target_variable (str, optional): Specific variable to analyze.
        granularity (float): Spatial resolution for the tables in degrees.

    Returns:
        dict: A nested dictionary containing the aggregated period DataFrames.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir.absolute()}")
        return {}

    # 1. Generate a list of expected files within the temporal range
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
        return {}

    # 2. Accumulate the monthly 2D spatial grids
    # Structure: { 't2m': [df_month1, df_month2, ...], 'stl4': [...] }
    accumulated_grids = {}

    logger.info(f"Initiating period aggregation across {len(target_files)} months...")

    for file in target_files:
        logger.debug(f"Extracting spatial grid from {file.name}...")

        # We call your existing monthly function!
        monthly_stats = generate_monthly_spatial_average_tables(
            month_file=file,
            target_variable=target_variable,
            granularity=granularity
        )

        for var, stats in monthly_stats.items():
            if var not in accumulated_grids:
                accumulated_grids[var] = []
            # Extract just the spatial grid (2D DataFrame) and store it
            accumulated_grids[var].append(stats['spatial_grid'])

    # 3. Calculate the overall period averages
    period_results = {}

    for var, grid_list in accumulated_grids.items():
        logger.info(f"Calculating final period mathematical averages for '{var}'...")

        # Convert list of DataFrames into a 3D NumPy array (Months x Lat x Lon)
        stacked_grids = np.stack([df.values for df in grid_list], axis=0)

        # Calculate the mean across the Months axis (axis=0)
        period_mean_values = np.nanmean(stacked_grids, axis=0)

        # Reconstruct the DataFrame using the index and columns from the first month
        period_spatial_grid = pd.DataFrame(
            period_mean_values,
            index=grid_list[0].index,
            columns=grid_list[0].columns
        )

        # Calculate Zonal, Meridional, and Grand means strictly from the period grid
        zonal_mean_df = period_spatial_grid.mean(axis=1).to_frame(name="Period_Zonal_Mean")
        meridional_mean_df = period_spatial_grid.mean(axis=0).to_frame(name="Period_Meridional_Mean")
        grand_mean = float(np.nanmean(period_mean_values))

        # Store in results dictionary
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
    # Disable INFO logging temporarily to prevent the monthly function from spamming the console
    logging.getLogger().setLevel(logging.WARNING)

    # Call the calculation function
    stats_dictionary = generate_period_spatial_average_tables(
        data_dir, start_year, start_month, end_year, end_month, target_variable, granularity
    )

    # Restore logging level
    logging.getLogger().setLevel(logging.INFO)

    if not stats_dictionary:
        return {}

    # Print the formatted summaries
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
    Reads the aggregated period statistics dictionary and generates a single
    window containing interactive 3D surface subplots for all variables.
    Projects a 2D map contour of the geographical area (e.g., Italy) onto
    the lowest XY plane.

    Args:
        stats_dictionary (dict): Output from generate_period_spatial_average_tables.
        period_label (str): A descriptive string for the master title.
    """
    if not stats_dictionary:
        logger.warning("No data provided for period 3D plotting.")
        return

    num_vars = len(stats_dictionary)

    # Calculate optimal subplot grid
    cols = math.ceil(math.sqrt(num_vars))
    rows = math.ceil(num_vars / cols)

    # Initialize dynamic figure
    fig = plt.figure(figsize=(7 * cols, 6 * rows))
    fig.suptitle(f"Period Temporal Means: {period_label}", fontsize=16, fontweight='bold')

    for index, (var, stats) in enumerate(stats_dictionary.items()):
        logger.info(f"Rendering 3D surface and bottom contour for period variable: {var}")

        # Extract the 2D grid and the coordinates directly from the Pandas DataFrame
        spatial_df = stats["spatial_grid"]
        lats = spatial_df.index.values
        lons = spatial_df.columns.values
        data_2d = spatial_df.values

        # Calculate Z-axis boundaries to place the floor map correctly
        z_max = np.nanmax(data_2d)
        z_min = np.nanmin(data_2d)
        z_range = z_max - z_min

        # Define the floor level (e.g., 10% below the absolute lowest data point)
        floor_z = z_min - (z_range * 0.1)

        # Create the specific 3D subplot axis for this variable
        ax = fig.add_subplot(rows, cols, index + 1, projection='3d')

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
        # zdir='z' projects it flat, offset drops it to our calculated floor_z
        ax.contourf(
            lon_grid, lat_grid, data_2d,
            zdir='z',
            offset=floor_z,
            cmap='viridis',
            alpha=0.6  # Slightly more transparent to look like a shadow/map
        )

        # Lock the Z-axis limits so the floor doesn't float into empty space
        ax.set_zlim(floor_z, z_max)

        # Format axis labels and title
        ax.set_title(f"Variable: {var}", fontweight='bold', fontsize=12)
        ax.set_xlabel("Longitude", labelpad=10)
        ax.set_ylabel("Latitude", labelpad=10)
        ax.set_zlabel("Period Mean Value", labelpad=10)

        # Attach a color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)

        # Set default camera viewing angle
        ax.view_init(elev=25, azim=230)

    # Adjust layout and display
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Ensure logger is configured
logger = logging.getLogger(__name__)

def export_period_3d_plots_to_pdf(
        stats_dictionary: dict[str, dict],
        period_label: str,
        output_filename: str | Path = "period_climate_plots.pdf",
        verbose: bool = False
) -> Path:
    """
    Reads the aggregated period statistics dictionary and saves each variable's
    3D surface plot as a perfectly centralized page in a PDF document.
    Projects a 2D geographical contour of the data onto the lowest XY plane.

    Args:
        stats_dictionary (dict): Output from generate_period_spatial_average_tables.
        period_label (str): Descriptive string for the title (e.g., "2004 to 2005").
        output_filename (str | Path): The target PDF file name.
        verbose (bool): If True, enables debug logging.

    Returns:
        Path: The absolute path to the generated PDF file.
    """
    if not stats_dictionary:
        logger.warning("No data provided for PDF export.")
        return None

    # 1. Define and create the target directory for PDFs
    target_dir = Path("Exported pdf plots")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. Extract just the file name and append it to the directory
    file_name = Path(output_filename).name
    output_path = target_dir / file_name

    logger.info(f"Generating Period PDF with {len(stats_dictionary)} pages in '{target_dir.name}'...")

    try:
        with PdfPages(output_path) as pdf:
            for var, stats in stats_dictionary.items():

                # Standard A4 Landscape sizing for perfect PDF pages
                fig = plt.figure(figsize=(11.69, 8.27))
                ax = fig.add_subplot(111, projection='3d')

                # Extract pre-calculated data directly from the dictionary
                spatial_df = stats["spatial_grid"]
                lats = spatial_df.index.values
                lons = spatial_df.columns.values
                data_2d = spatial_df.values

                # Calculate Z-axis boundaries to place the floor map correctly
                z_max = np.nanmax(data_2d)
                z_min = np.nanmin(data_2d)
                z_range = z_max - z_min
                floor_z = z_min - (z_range * 0.1)  # Drop the floor 10% below lowest point

                lon_grid, lat_grid = np.meshgrid(lons, lats)

                # 1. Render the main 3D surface
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
                    alpha=0.6
                )

                # Lock the Z-axis limits so the floor sits securely at the bottom of the visual box
                ax.set_zlim(floor_z, z_max)

                # Add a master title to the top of the PDF page
                fig.suptitle(
                    f"Period Temporal Mean: {var.upper()}\n({period_label})",
                    fontsize=16,
                    fontweight='bold',
                    y=0.95 # Nudges the title slightly up to prevent overlap
                )

                # Format axes
                ax.set_xlabel("Longitude", labelpad=10)
                ax.set_ylabel("Latitude", labelpad=10)
                ax.set_zlabel("Period Mean Value", labelpad=10)

                # Attach colorbar, shrunk to fit the centralized aesthetic
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)

                # Set default camera viewing angle
                ax.view_init(elev=25, azim=230)

                # Enforce tight layout to calculate the absolute bounding box of the graph
                plt.tight_layout()

                # Save the page with a forced even margin to perfectly centralize it
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.5)

                # CRITICAL: Close the figure to free memory
                plt.close(fig)

                if verbose:
                    logger.info(f"Rendered centralized PDF page with contour for: {var}")

        logger.info(f"Successfully exported 3D period plots with contours to PDF: {output_path.absolute()}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to construct PDF: {e}")
        return None


if __name__ == '__main__':
    # Define target directory and period bounds
    target_directory = Path("era5_monthly_data")

    # Example Period: March 2004 to December 2005
    s_year, s_month = 2004, 3
    e_year, e_month = 2005, 12
    period_label_str = f"{s_year}/{s_month:02d} to {e_year}/{e_month:02d}"

    # 1. Execute calculations and print to console
    period_stats = print_period_spatial_summaries(
        data_dir=target_directory,
        start_year=s_year,
        start_month=s_month,
        end_year=e_year,
        end_month=e_month,
        granularity=.1
    )

    if period_stats:
        # 2. Export to Excel (safely reusing the function from environmental_monthly_mean_analysis)
        try:
            from environmental_monthly_mean_analysis import export_stats_to_excel
            excel_filename = f"period_stats_{s_year}_{s_month:02d}_to_{e_year}_{e_month:02d}.xlsx"
            export_stats_to_excel(period_stats, output_filename=excel_filename)
        except ImportError:
            logger.warning("Could not import export_stats_to_excel. Ensure environmental_monthly_mean_analysis.py is in the same folder.")

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

    if period_stats:
        # 2. Export to Excel (reusing the function from the other module)
        try:
            from environmental_monthly_mean_analysis import export_stats_to_excel
            excel_filename = f"period_stats_{s_year}_{s_month:02d}_to_{e_year}_{e_month:02d}.xlsx"
            export_stats_to_excel(period_stats, output_filename=excel_filename)
        except ImportError:
            from environmental_monthly_mean_analysis import export_stats_to_excel
            logger.warning("Could not import export_stats_to_excel. Skipping Excel export.")
