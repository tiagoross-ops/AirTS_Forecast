"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: environmental_data_animation.py
Author: Tiago TOLOCZKO ROSS

Description:
Time-series animation module. Extracts spatial data across a multi-file period
using the core retrieval engines, calculates global Z-axis limits for visual stability,
and generates a multi-variable 3D surface animation (.gif) to visualize spatial
evolution over time.
"""

import logging
import math
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

# Import the core data extraction and metadata engines
from environmental_data_retrieval import (
    period_retrieval_function,
    monthly_data_directory_exploration,
    file_name_comprehension
)

# Import the specific spatial mathematical tools and plotting helpers
from environmental_spatial_average_analysis import (
    granular_spatial_average,
    plot_3d_surface_on_axis
)

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)



def calculate_global_z_limits(
        period_data_dict: dict[str, list[pd.DataFrame]]
) -> dict[str, tuple[float, float]]:
    """
    Scans the entire chronological dataset to find the absolute maximum and minimum
    values for each variable. This prevents the 3D Matplotlib box from jarringly
    resizing itself every single frame.
    """
    global_z_limits = {}
    for var, df_list in period_data_dict.items():
        # Stack all time-steps for this variable into one array to find absolute bounds
        all_matrices = np.stack([df.values for df in df_list])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            z_max = float(np.nanmax(all_matrices))
            z_min = float(np.nanmin(all_matrices))

        z_range = z_max - z_min
        floor_z = z_min - (z_range * 0.1) if z_range != 0 else z_min - 1

        global_z_limits[var] = (floor_z, z_max)

    return global_z_limits


def animate_spatial_evolution(
        data_dir: Path,
        granularity: float = 1.0,
        save_gif_path: Optional[str | Path] = None,
        fps: int = 2
) -> None:
    """
    Generates a time-animated 3D series of the available spatial data,
    visualizing the evolution of variables throughout the studied period.

    Args:
        data_dir (Path): Directory containing the chronological .h5 files.
        granularity (float, optional): Spatial binning resolution. Defaults to 1.0.
        save_gif_path (str | Path, optional): Filepath to save the .gif. If None,
                                              displays interactively. Defaults to None.
        fps (int, optional): Frames per second for the animation. Defaults to 2.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        logger.error(f"Invalid directory provided: {data_dir.absolute()}")
        return

    # 1. Utilize the Master Retrieval Engines
    logger.info(f"Extracting period data from {data_dir.name} at {granularity}° resolution...")
    data_dictionary = period_retrieval_function(
        data_dir=data_dir,
        retrieval_func=granular_spatial_average(granularity=granularity),
        verbose=False
    )

    if not data_dictionary:
        logger.warning("No valid data extracted for the animation.")
        return

    # 2. Extract Exact Chronological Metadata
    # This replaces the mathematical guessing of start_year/start_month
    files, _ = monthly_data_directory_exploration(data_dir)
    months_list = [file_name_comprehension(f)[:2] for f in files]  # List of (Year, Month)

    variables = list(data_dictionary.keys())
    num_months = len(months_list)
    num_vars = len(variables)

    # 3. Calculate Global Z-Limits for Animation Stability
    global_z_limits = calculate_global_z_limits(data_dictionary)

    # 4. Setup Dynamic Matplotlib Figure
    cols = math.ceil(math.sqrt(num_vars))
    rows = math.ceil(num_vars / cols)
    fig = plt.figure(figsize=(8 * cols, 7 * rows))

    # 5. Define the Animation Update Engine
    def update(frame_index: int):
        # Clear the figure to prevent massive RAM leaks and overlapping colorbars
        fig.clear()

        # Extract the exact year and month from our metadata list
        current_year, current_month = months_list[frame_index]

        fig.suptitle(
            f"Spatial Climate Evolution: {current_year} - {current_month:02d}\n"
            f"(Step {frame_index + 1} of {num_months})",
            fontsize=16, fontweight='bold'
        )

        for index, var_name in enumerate(variables):
            df = data_dictionary[var_name][frame_index]
            ax = fig.add_subplot(rows, cols, index + 1, projection='3d')

            # Render the 3D plot and floor contour using our injected spatial tool
            plot_3d_surface_on_axis(
                ax=ax,
                lons=df.columns.values,
                lats=df.index.values,
                data_2d=df.values,
                var_name=var_name
            )

            # OVERRIDE the dynamic Z-limits to lock the 3D box to our global limits
            ax.set_zlim(global_z_limits[var_name][0], global_z_limits[var_name][1])

    # 6. Execute the Animation Compilation
    logger.info(f"Compiling {num_months}-frame 3D animation. This may take a moment...")

    interval_ms = int(1000 / fps)
    ani = animation.FuncAnimation(
        fig, update, frames=num_months, interval=interval_ms, repeat=True
    )

    # 7. Output Routing
    if save_gif_path:
        target_path = Path(save_gif_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving animation to disk (Requires Pillow): {target_path.name} ...")

        try:
            ani.save(target_path, writer='pillow', fps=fps)
            logger.info("Animation saved successfully!")
        except Exception as e:
            logger.error(f"Failed to save GIF. Ensure 'Pillow' is installed. Error: {e}")

    else:
        logger.info("Launching interactive animation dashboard...")
        plt.show()


# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == '__main__':
    target_directory = Path("era5_monthly_data")

    if target_directory.exists():
        # Objective 1: Save as a GIF
        animate_spatial_evolution(
            data_dir=target_directory,
            granularity=.1,  # Highly recommended to use coarsened data (e.g., 2.0 or 5.0) for 3D animations to save RAM!
#            save_gif_path="Exported Animations/spatial_climate_evolution.gif",
#            fps=2
        )
