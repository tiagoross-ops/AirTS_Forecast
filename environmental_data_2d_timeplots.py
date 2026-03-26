"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: environmental_data_2d_timeplots.py
Author: Tiago TOLOCZKO ROSS

Description:
Time-series visualization module. Reads 3D hourly climate data (Time, Lat, Lon),
groups the spatial grid into regional bins based on a specified granularity,
and generates 2D line plots showing the temporal evolution of each region.
Fully integrated with the core data retrieval engine and includes automated
12-month sinusoidal curve fitting for strong temporal correlations.
"""

import logging
import math
import warnings
from pathlib import Path
from typing import Callable, Optional

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Explicitly import only the required core extraction and exploration tools
from environmental_data_retrieval import (
    file_name_comprehension,
    file_var_retrieval,
    monthly_data_directory_exploration
)

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 1. RETRIEVAL & FACTORY FUNCTIONS
# =============================================================================

def timeseries_by_region(
        granularity: float,
        year: int,
        month: int
) -> Callable[[str, h5py.File | h5py.Group], Optional[pd.DataFrame]]:
    """
    Factory function that creates a customized data retrieval function embedded
    with the desired spatial granularity.

    Args:
        granularity (float): Spatial resolution for regional grouping in degrees.
        year (int): The dataset's year.
        month (int): The dataset's month.

    Returns:
        Callable: A function matching the signature required by the core engine.
                  Returns a DataFrame where Index = Time, Columns = Regions.
    """
    def timeseries_retrieval_func(var: str, dataset: h5py.File | h5py.Group) -> Optional[pd.DataFrame]:
        try:
            coord_group_name = f"{var}_coordinates"
            if coord_group_name not in dataset:
                return None

            coord_group = dataset[coord_group_name]
            if 'latitude' not in coord_group or 'longitude' not in coord_group or 'valid_time' not in coord_group:
                return None

            raw_time = coord_group['valid_time'][:]
            lats = coord_group['latitude'][:]
            lons = coord_group['longitude'][:]
            data_3d = dataset[var][:]  # Shape: (Time, Lat, Lon)

            if len(data_3d.shape) < 3:
                return None

            # Format the Time Array
            formated_time = [f"{year:04d}{month:02d}{t+1:02d}" for t in raw_time]
            clean_time = [t.decode('utf-8') if isinstance(t, bytes) else t for t in formated_time]
            time_datetime = np.array([pd.to_datetime(str(t)) for t in clean_time])

            # Define Spatial Bins
            binned_lats = np.round(lats / granularity) * granularity
            binned_lons = np.round(lons / granularity) * granularity
            unique_lat_bins = np.unique(binned_lats)
            unique_lon_bins = np.unique(binned_lons)

            timeseries_dict = {}

            # Calculate spatial mean for each bin while preserving the Time axis
            for lat_b in unique_lat_bins:
                lat_mask = (binned_lats == lat_b)
                for lon_b in unique_lon_bins:
                    lon_mask = (binned_lons == lon_b)

                    sub_grid = data_3d[:, lat_mask, :][:, :, lon_mask]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        ts_mean = np.nanmean(sub_grid, axis=(1, 2))

                    if not np.all(np.isnan(ts_mean)):
                        label = f"Lat {lat_b:.1f}°, Lon {lon_b:.1f}°"
                        timeseries_dict[label] = ts_mean

            if not timeseries_dict:
                return None

            # Construct a DataFrame where rows are Time and columns are Spatial Regions
            df = pd.DataFrame(timeseries_dict, index=time_datetime)
            df.index.name = "Time"
            return df

        except Exception as var_e:
            logger.error(f"Unexpected error processing variable '{var}': {var_e}")
            return None

    return timeseries_retrieval_func


def period_time_series_by_region(
        data_dir: Path,
        granularity: float = 1.0,
        target_variable: str = None,
        verbose: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Iterates over an auto-discovered period of monthly files, extracts the
    regional timeseries for each month, and concatenates them into a single,
    continuous temporal DataFrame for each variable.
    """
    target_files, bounds = monthly_data_directory_exploration(data_dir)

    if not target_files:
        logger.warning(f"No valid files found in {data_dir.name} to process for the period timeseries.")
        return {}

    if bounds:
        s_m, s_y, e_m, e_y = bounds
        logger.info(f"Initiating continuous timeseries extraction across {len(target_files)} files "
                    f"({s_m:02d}/{s_y} to {e_m:02d}/{e_y}) at {granularity}° granularity...")

    accumulated_timeseries: dict[str, list[pd.DataFrame]] = {}

    for file in target_files:
        if verbose:
            logger.info(f"Extracting timeseries from {file.name}...")

        try:
            year, month, _ = file_name_comprehension(file)
            custom_retrieval_logic = timeseries_by_region(granularity, year, month)

            monthly_data = file_var_retrieval(
                month_file=file,
                retrieval_func=custom_retrieval_logic,
                verbose=False
            )

            for var, df in monthly_data.items():
                if target_variable and var != target_variable:
                    continue

                if var not in accumulated_timeseries:
                    accumulated_timeseries[var] = []

                accumulated_timeseries[var].append(df)

        except Exception as e:
            logger.error(f"Failed to process timeseries for {file.name}: {e}")
            continue

    if not accumulated_timeseries:
        logger.warning("No timeseries data was accumulated.")
        return {}

    logger.info("Concatenating monthly blocks into continuous period timeseries...")
    final_period_timeseries = {}

    for var, df_list in accumulated_timeseries.items():
        continuous_df = pd.concat(df_list, axis=0)
        continuous_df = continuous_df.sort_index()
        final_period_timeseries[var] = continuous_df

    logger.info(f"Successfully generated continuous period timeseries for {len(final_period_timeseries)} variables.")
    return final_period_timeseries


# =============================================================================
# 2. MATHEMATICAL MODELING
# =============================================================================

def fit_sinusoidal(time_idx: pd.DatetimeIndex, data: pd.Series) -> Optional[tuple[np.ndarray, list]]:
    """
    Fits a 12-month period sinusoidal function to the given time-series data.

    Equation: y(t) = A * sin(omega * t + phi) + C
    where omega is locked to a 12-month (365.2425 days) period.

    Returns:
        tuple containing the fitted numpy array and the optimized parameters [A, phi, C]
        if the absolute Pearson correlation > 0.95, otherwise None.
    """
    valid_mask = ~np.isnan(data)
    if valid_mask.sum() < 12:
        return None

    t_full = mdates.date2num(time_idx)
    t_valid = t_full[valid_mask]
    y_valid = data[valid_mask].values

    period_days = 365.2425
    omega = 2 * np.pi / period_days

    def sin_func(t, A, phi, C):
        return A * np.sin(omega * t + phi) + C

    guess_C = np.mean(y_valid)
    guess_A = (np.max(y_valid) - np.min(y_valid)) / 2.0
    guess_phi = 0.0

    try:
        popt, _ = curve_fit(sin_func, t_valid, y_valid, p0=[guess_A, guess_phi, guess_C], maxfev=5000)
        y_fit_full = sin_func(t_full, *popt)
        y_fit_valid = sin_func(t_valid, *popt)

        correlation = np.corrcoef(y_valid, y_fit_valid)[0, 1]

        if abs(correlation) > 0.95:
            return y_fit_full, popt  # <--- Modified to return parameters
        else:
            return None

    except Exception as e:
        logger.debug(f"Curve fitting failed to converge: {e}")
        return None


# =============================================================================
# 3. RENDERING ENGINE
# =============================================================================

def _populate_timeseries_axis(
        ax: plt.Axes,
        df: pd.DataFrame,
        var: str,
        granularity: float
) -> None:
    """
    Internal helper function to populate a single Matplotlib axis with regional
    timeseries data, format the grid, and evaluate/print sinusoidal fits.

    Args:
        ax (plt.Axes): The Matplotlib axis to draw on.
        df (pd.DataFrame): The timeseries data (Index = Time, Columns = Regions).
        var (str): The name of the environmental variable being plotted.
        granularity (float): The spatial binning resolution (used for the legend).
    """
    time_array = df.index

    for region_label in df.columns:
        # 1. Plot the raw data line and capture its object
        line, = ax.plot(time_array, df[region_label], label=region_label, linewidth=1.5, alpha=0.8)

        # 2. Attempt the 12-month Sinusoidal Fit
        fit_result = fit_sinusoidal(time_array, df[region_label])

        if fit_result is not None:
            fitted_y, popt = fit_result
            A, phi, C = popt
            omega = 2 * np.pi / 365.2425

            # Print the mathematical equation to the console
            logger.info(
                f"Strong correlation fit for {var.upper()} [{region_label}]: "
                f"y(t) = {A:.2f} * sin({omega:.4f}*t + {phi:.2f}) + {C:.2f}"
            )

            # Plot the fitted wave (omitting 'label' keeps it out of the legend)
            ax.plot(
                time_array, fitted_y,
                linestyle=':', color=line.get_color(), linewidth=2.0, alpha=0.9
            )

    # 3. Apply standard formatting
    ax.set_title(var.upper(), fontweight='bold')
    ax.set_xlabel("Time", fontweight='bold')
    ax.set_ylabel(f"{var.upper()} Value", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ax.legend(title=f"Regions ({granularity}° Bins)",
              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')


def _render_timeseries_plots(
        extracted_data: dict[str, pd.DataFrame],
        granularity: float,
        visual: str,
        title_str: str
) -> dict[str, plt.Figure]:
    """
    Coordinates the rendering of time-series plots, routing data to either a
    single unified dashboard or multiple individual figure windows.

    Args:
        extracted_data (dict[str, pd.DataFrame]): Dictionary mapping variables to DataFrames.
        granularity (float): The spatial binning resolution.
        visual (str): 's' for single dashboard, 'm' for multiple windows.
        title_str (str): The master title to display on the figure(s).

    Returns:
        dict[str, plt.Figure]: A dictionary containing the generated Matplotlib Figure objects.
    """
    generated_figures = {}

    if visual == 's':
        logger.info("Rendering Single Window Dashboard...")
        num_vars = len(extracted_data)
        cols = math.ceil(math.sqrt(num_vars))
        rows = math.ceil(num_vars / cols)

        fig = plt.figure(figsize=(9 * cols, 6 * rows))
        fig.suptitle(f"Temporal Evolution ({title_str})", fontsize=16, fontweight='bold')

        # Iterate and delegate subplot creation
        for idx, (var, df) in enumerate(extracted_data.items()):
            ax = fig.add_subplot(rows, cols, idx + 1)
            _populate_timeseries_axis(ax, df, var, granularity)

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        generated_figures['combined'] = fig

    elif visual == 'm':
        logger.info("Rendering Multiple Individual Windows...")

        # Iterate and delegate full-figure creation
        for var, df in extracted_data.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.suptitle(f"Temporal Evolution: {var.upper()} ({title_str})", fontsize=16, fontweight='bold')

            _populate_timeseries_axis(ax, df, var, granularity)

            plt.tight_layout()
            generated_figures[var] = fig

    return generated_figures


# =============================================================================
# 4. EXPOSED ORCHESTRATION FUNCTIONS
# =============================================================================

def environmental_timeseries(
        file_path: Path,
        granularity: float = 1.0,
        visual: str = 's',
        verbose: bool = False
) -> dict[str, plt.Figure]:
    """
    Extracts hourly data for a single month, applies spatial coarsening, and
    generates time-series line plots utilizing the core extraction engine.
    """
    if not file_path.exists():
        logger.error(f"Target file does not exist: {file_path.absolute()}")
        return {}

    if visual not in ['s', 'm']:
        logger.warning(f"Invalid visual parameter '{visual}'. Defaulting to 's' (single).")
        visual = 's'

    try:
        year, month, _ = file_name_comprehension(file_path)
    except Exception as e:
        logger.error(f"Could not parse temporal data from {file_path.name}: {e}")
        return {}

    custom_retrieval_logic = timeseries_by_region(granularity, year, month)

    extracted_data = file_var_retrieval(month_file=file_path, retrieval_func=custom_retrieval_logic, verbose=verbose)

    if not extracted_data:
        logger.warning("No valid data extracted for visualization.")
        return {}

    return _render_timeseries_plots(extracted_data, granularity, visual, file_path.stem)


def period_environmental_timeseries(
        data_dir: Path,
        granularity: float = 1.0,
        visual: str = 's',
        target_variable: str = None,
        verbose: bool = False
) -> dict[str, plt.Figure]:
    """
    Wraps the period timeseries aggregation logic into the rendering engine
    to output plots for the entire continuous timeframe.
    """
    extracted_data = period_time_series_by_region(
        data_dir=data_dir,
        granularity=granularity,
        target_variable=target_variable,
        verbose=verbose
    )

    if not extracted_data:
        return {}

    _, bounds = monthly_data_directory_exploration(data_dir)
    title_str = f"{bounds[0]:02d}/{bounds[1]} to {bounds[2]:02d}/{bounds[3]}" if bounds else "Period"

    return _render_timeseries_plots(extracted_data, granularity, visual, title_str)


# =============================================================================
# 5. EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    target_dir = Path("era5_monthly_data")
    test_file = target_dir / "era5_3d_2004_06.h5"

    if test_file.exists():
        logger.info(f"\nTesting period timeseries on directory {target_dir.name}...")

        period_figures = period_environmental_timeseries(
            data_dir=target_dir,
            granularity=2.0,
            visual='m',
            verbose=True
        )

        plt.show()
    else:
        logger.error(f"Test file not found at: {test_file.absolute()}")