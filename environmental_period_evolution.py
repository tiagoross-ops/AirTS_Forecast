"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: Data_Animation.py
"""

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Import your existing functions (adjust module names as needed)
from environmental_period_average_analysis import period_granular_spatial_average_tables
from environmental_monthly_average_analysis import plot_3d_surface_on_axis

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def var_evolution(
        data_dir: Path,
        start_year: int,
        start_month: int = 1,
        end_year: int = 2004,
        end_month: int = 12,
        frequency: str = 'm',
        target_variable: str = None,
        granularity: float = 1.0,
        save_gif_path: str = None
) -> None:
    """
    Generates a time-animated 3D series of the available spatial data,
    visualizing the evolution of variables throughout the studied period.
    """
    # 1. Frequency check correction
    if frequency not in ['m', 'd']:
        logger.error(f"Expected frequency 'm' or 'd', got frequency '{frequency}'")
        return

    # 2. Extract Data
    try:
        data_dictionary = period_granular_spatial_average_tables(data_dir, start_year, start_month)
    except Exception as e:
        logger.error(f"Problem exploring directory {data_dir} in the given range: {e}")
        return

    if not data_dictionary:
        logger.warning("No data found for the animation.")
        return

    variables = list(data_dictionary.keys())
    num_months = len(data_dictionary[variables[0]])
    num_vars = len(variables)

    # 3. Calculate Global Z-Limits for Animation Stability
    # We must find the absolute highest and lowest points across ALL months
    # so the 3D box doesn't resize itself every frame.
    global_z_limits = {}
    for var in variables:
        all_matrices = [df.values for df in data_dictionary[var]]
        z_max = np.nanmax(all_matrices)
        z_min = np.nanmin(all_matrices)
        z_range = z_max - z_min
        floor_z = z_min - (z_range * 0.1) # 10% drop for the floor contour
        global_z_limits[var] = (floor_z, z_max)

    # 4. Setup Figure
    cols = math.ceil(math.sqrt(num_vars))
    rows = math.ceil(num_vars / cols)
    fig = plt.figure(figsize=(7 * cols, 6 * rows))

    # 5. Define the Animation Update Function
    def update(frame_index: int):
        # Clear the figure to prevent RAM bloat and overlapping colorbars
        fig.clear()

        # Calculate current year/month for the dynamic title
        current_month = start_month + frame_index
        current_year = start_year + ((current_month - 1) // 12)
        display_month = ((current_month - 1) % 12) + 1

        fig.suptitle(
            f"Climate Evolution: {current_year}-{display_month:02d} "
            f"(Month {frame_index + 1} of {num_months})",
            fontsize=16, fontweight='bold'
        )

        for index, var_name in enumerate(variables):
            df = data_dictionary[var_name][frame_index]
            ax = fig.add_subplot(rows, cols, index + 1, projection='3d')

            lats = df.index.values
            lons = df.columns.values
            data_2d = df.values

            # Render the 3D plot and floor contour
            plot_3d_surface_on_axis(
                ax=ax,
                lons=lons,
                lats=lats,
                data_2d=data_2d,
                var_name=var_name
            )

            # OVERRIDE the Z-limits to lock the 3D box to the global limits
            ax.set_zlim(global_z_limits[var_name][0], global_z_limits[var_name][1])

    # 6. Execute the Animation
    logger.info("Compiling animation frames. This may take a moment...")

    # interval=500 means 500 milliseconds (half a second) per frame
    ani = animation.FuncAnimation(
        fig, update, frames=num_months, interval=500, repeat=True
    )

    # 7. Output Handling
    if save_gif_path:
        logger.info(f"Saving animation to {save_gif_path} (Requires Pillow)...")
        # Ensure the directory exists
        Path(save_gif_path).parent.mkdir(parents=True, exist_ok=True)
        ani.save(save_gif_path, writer='pillow', fps=2)
        logger.info("Animation saved successfully!")
    else:
        plt.show()


if __name__ == '__main__':
    target_directory = Path("era5_monthly_data")

    # Example: Visualize the evolution from March 2004 to December 2005
    var_evolution(
        data_dir=target_directory,
        start_year=2004,
        start_month=3,
        end_year=2005,
        end_month=12,
        granularity=.1,
        #save_gif_path="Exported_Animations/climate_evolution.gif"
    )