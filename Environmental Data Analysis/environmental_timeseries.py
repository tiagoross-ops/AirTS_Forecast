"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: environmental_timeseries.py
Author: Tiago TOLOCZKO ROSS

Description:
Time-series visualization module. Reads 3D hourly climate data (Time, Lat, Lon),
groups the spatial grid into regional bins based on a specified granularity,
and generates 2D line plots showing the temporal evolution of each region.
Fully integrated with the core data retrieval engine and includes automated
12-month sinusoidal curve fitting for strong temporal correlations.
"""

import logging
import warnings
from pathlib import Path
from typing import Callable, Optional, Any

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Explicitly import the retrieval engine and the new orchestrator
from environmental_data_retrieval import (file_name_comprehension, period_retrieval_function)
from environmental_data_visualization_orchestration import visualization_orchestration

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 1. RETRIEVAL & FACTORY FUNCTIONS
# =============================================================================

def timeseries_by_region() -> Callable[[str, h5py.File | h5py.Group], Optional[pd.DataFrame]]:
    """
    Simplified retrieval function that strictly extracts native HDF5 data into a
    DataFrame with a proper Time index and a [Latitude, Longitude] MultiIndex.
    """

    def retrieval_func(var: str, dataset: h5py.File | h5py.Group) -> Optional[pd.DataFrame]:
        try:
            # Dynamically extract the year and month from the current file's name
            current_file_path = Path(dataset.file.filename)
            year, month, _ = file_name_comprehension(current_file_path)

            coord_group_name = f"{var}_coordinates"
            if coord_group_name not in dataset:
                return None

            coord_group = dataset[coord_group_name]
            if 'latitude' not in coord_group or 'longitude' not in coord_group or 'valid_time' not in coord_group:
                return None

            lats = coord_group['latitude'][:]
            lons = coord_group['longitude'][:]
            data_3d = dataset[var][:]  # Shape: (Time, Lat, Lon)

            time_steps = data_3d.shape[0]
            if time_steps < 3:
                return None

            # Generate proper hourly DatetimeIndex
            start_date = pd.Timestamp(year=year, month=month, day=1)
            time_datetime = pd.date_range(start=start_date, periods=time_steps, freq='D')

            # Flatten spatial dimensions: (Time, Lat, Lon) -> (Time, Lat * Lon)
            data_2d = data_3d.reshape(time_steps, -1)

            # Create a MultiIndex for the columns combining Lat and Lon
            multi_cols = pd.MultiIndex.from_product([lats, lons], names=['Latitude', 'Longitude'])

            native_df = pd.DataFrame(data_2d, index=time_datetime, columns=multi_cols)
            native_df.index.name = "Time"

            # Return the raw native DataFrame (Binning happens later in the plotter!)
            return native_df

        except Exception as var_e:
            logger.error(f"Unexpected error processing variable '{var}': {var_e}")
            return None

    return retrieval_func


# =============================================================================
# 2. PERIOD OVERALL FUNCTION
# =============================================================================
def overall_period_timeseries(
        period_data_dict: dict[str, list[pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """
    Mathematical Aggregation Engine: Iterates over an auto-discovered period
    of monthly files, extracting the regional timeseries for each month, and
    seamlessly concatenates them into a continuous temporal DataFrame.
    """
    overall_stats = {}

    for var, df_list in period_data_dict.items():
        if not df_list:
            logger.warning(f"No data available for variable '{var}' to concatenate.")
            continue

        # 1. Seamlessly stitch the list of DataFrames together along the Time axis (rows)
        continuous_timeseries_df = pd.concat(df_list, axis=0)

        # 2. Sort the index to absolutely guarantee strict chronological order
        continuous_timeseries_df = continuous_timeseries_df.sort_index()

        # 3. Store the finalized continuous DataFrame
        overall_stats[var] = continuous_timeseries_df

    return overall_stats


# =============================================================================
# 3. MATHEMATICAL MODELING
# =============================================================================

def fit_sinusoidal(time_idx: pd.DatetimeIndex, data: pd.Series) -> tuple[Any, Any, Any] | None:
    """
    Fits a 12-month period sinusoidal function to the given time-series data.
    """
    valid_mask = ~np.isnan(data)
    if valid_mask.sum() < 12:
        return None

    t_full = mdates.date2num(time_idx)
    t_valid = t_full[valid_mask]
    y_valid = data[valid_mask].values

    period_days = 365.2425
    omega = 2 * np.pi / period_days

    def sin_func(t, a, phi, c):
        return a * np.sin(omega * t + phi) + c

    guess_c = np.mean(y_valid)
    guess_a = (np.max(y_valid) - np.min(y_valid)) / 2.0
    guess_phi = 0.0

    try:
        popt, _ = curve_fit(sin_func, t_valid, y_valid, p0=[guess_a, guess_phi, guess_c], maxfev=5000)
        y_fit_full = sin_func(t_full, *popt)
        y_fit_valid = sin_func(t_valid, *popt)

        correlation = np.corrcoef(y_valid, y_fit_valid)[0, 1]

        if abs(correlation) > 0.8:
            # ---> NEW: Return the correlation alongside the fit data <---
            return y_fit_full, popt.tolist(), correlation
        else:
            return None

    except Exception as exc:
        logger.debug(f"Curve fitting failed to converge: {exc}")
        return None


# =============================================================================
# 4. VISUALIZATION ENGINE & PLOTTING FUNCTIONS
# =============================================================================
def _populate_timeseries_axis(
        ax: plt.Axes,
        df: pd.DataFrame,
        var: str,
        granularity: float,
        verbose: bool = False,

) -> None:
    """
    Internal helper function to populate a single Matplotlib axis with regional
    timeseries data, format the grid, and evaluate/print sinusoidal fits.
    Safely absorbs and applies arbitrary styling arguments to the main data lines.
    """
    time_array = df.index

    for region_label in df.columns:
        # Unpack the merged style dictionary into the main data plot
        line, = ax.plot(
            time_array,
            df[region_label],
            label=region_label,
            linewidth=1.5, alpha=0.8
        )

        fit_result = fit_sinusoidal(time_array, df[region_label])

        if fit_result is not None:
            # Inside _populate_timeseries_axis...
            fit_result = fit_sinusoidal(time_array, df[region_label])

            if fit_result is not None:
                # ---> NEW: Add the `_` to safely ignore the correlation in the plotter <---
                fitted_y, popt, corr = fit_result
                a, phi, c = popt
                omega = 2 * np.pi / 365.2425

                if verbose:
                    # We can even include it in our console printout now!
                    logger.info(
                        f"Strong correlation ({abs(corr)*100:.1f}%) fit for {var.upper()} [{region_label}]: "
                        f"y(t) = {a:.2f} * sin({omega:.4f}*t + {phi:.2f}) + {c:.2f}"
                    )

            # Keep the fitted line visually distinct (dotted) but match the region's color
            ax.plot(
                time_array, fitted_y,
                linestyle=':', color=line.get_color(), linewidth=2.0, alpha=0.9
            )

    ax.set_title(var.upper(), fontweight='bold')
    ax.set_xlabel("Time", fontweight='bold')
    ax.set_ylabel(f"{var.upper()} Value", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

#    ax.legend(title=f"Regions ({granularity}° Bins)",
#              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')


def timeseries_by_region_plotter_factory(
        granularity: float = 1.0,
        verbose: bool = False,
        *args,
        **kwargs
) -> Callable[..., plt.Figure | None]:
    """
    Conforms to the universal orchestrator signature. Accepts a single-variable
    dictionary from the orchestrator loop, applies spatial binning, and
    generates a standalone 2D plt.Figure timeseries.
    """

    def plot_generator(
            extracted_data: dict[str, pd.DataFrame],
            plot_title: str,
    ) -> plt.Figure | None:

        if not extracted_data:
            return None

        # The orchestrator strictly passes 1 variable at a time in its plotting loop
        var: str
        native_df: pd.DataFrame
        var, native_df = next(iter(extracted_data.items()))
        logger.debug(f"Applying {granularity}° spatial binning and rendering timeseries for: {var.upper()}")

        if granularity and isinstance(native_df.columns, pd.MultiIndex):
            lats_idx = native_df.columns.get_level_values('Latitude')
            lons_idx = native_df.columns.get_level_values('Longitude')

            binned_lats = np.round(lats_idx / granularity) * granularity
            binned_lons = np.round(lons_idx / granularity) * granularity

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                coarsened_df = native_df.T.groupby([binned_lats, binned_lons]).mean().T

            coarsened_df = coarsened_df.dropna(axis=1, how='all')
            coarsened_df.columns = [f"Lat {lat:.1f}°, Lon {lon:.1f}°" for lat, lon in coarsened_df.columns]

        else:
            coarsened_df = native_df.copy()
            if isinstance(coarsened_df.columns, pd.MultiIndex):
                coarsened_df.columns = [f"Lat {lat:.1f}°, Lon {lon:.1f}°" for lat, lon in coarsened_df.columns]

        # Generate the Plot
        fig, ax = plt.subplots(*args, **kwargs)
        fig.suptitle(f"Temporal Evolution: {var.upper()} ({plot_title})", fontsize=16, fontweight='bold')

        # Pass the arbitrary arguments down to the axis populator
        _populate_timeseries_axis(
            ax, coarsened_df, var, granularity, verbose=verbose
        )

        plt.tight_layout()

        return fig

    return plot_generator


# =============================================================================
# 5. TABULAR VISUALIZATION AND EXPORTING FUNCTIONS
# =============================================================================
def _coarsen_spatial_timeseries(native_df: pd.DataFrame, granularity: float) -> pd.DataFrame:
    """
    Internal helper to apply spatial binning to a native MultiIndex timeseries DataFrame.
    """
    if granularity and isinstance(native_df.columns, pd.MultiIndex):
        lats_idx = native_df.columns.get_level_values('Latitude')
        lons_idx = native_df.columns.get_level_values('Longitude')

        binned_lats = np.round(lats_idx / granularity) * granularity
        binned_lons = np.round(lons_idx / granularity) * granularity

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            coarsened_df = native_df.T.groupby([binned_lats, binned_lons]).mean().T

        coarsened_df = coarsened_df.dropna(axis=1, how='all')
        coarsened_df.columns = [f"Lat {lat:.1f}°, Lon {lon:.1f}°" for lat, lon in coarsened_df.columns]
        return coarsened_df

    elif isinstance(native_df.columns, pd.MultiIndex):
        coarsened_df = native_df.copy()
        coarsened_df.columns = [f"Lat {lat:.1f}°, Lon {lon:.1f}°" for lat, lon in coarsened_df.columns]
        return coarsened_df

    return native_df.copy()


def calculate_sinusoidal_statistics(
        continuous_data_dict: dict[str, pd.DataFrame],
        granularity: float = 1.0
) -> dict[str, pd.DataFrame]:
    """
    Analyzes the continuous period timeseries, applies spatial binning, and
    attempts a sinusoidal fit for every region. Filters out regions with weak
    correlations and returns the mathematical parameters of the successful fits.
    """
    stats_dict = {}
    omega = 2 * np.pi / 365.2425

    for var, native_df in continuous_data_dict.items():
        logger.info(f"Calculating sinusoidal fits for '{var}' at {granularity}° granularity...")

        binned_df = _coarsen_spatial_timeseries(native_df, granularity)
        var_stats = []

        for region_label in binned_df.columns:
            fit_result = fit_sinusoidal(binned_df.index, binned_df[region_label])

            if fit_result is not None:
                # ---> NEW: Unpack the correlation value <---
                _, popt, correlation = fit_result
                a, phi, c = popt

                var_stats.append({
                    "Region": region_label,
                    "Correlation (%)": f"{abs(correlation) * 100:.2f}%",  # Format as percentage
                    "Amplitude (A)": a,
                    "Phase Shift (phi)": phi,
                    "Vertical Shift/Mean (C)": c,
                    "Equation": f"y(t) = {a:.2f} * sin({omega:.4f}*t + {phi:.2f}) + {c:.2f}"
                })

        if var_stats:
            stats_df = pd.DataFrame(var_stats).set_index("Region")
            # Sort by highest correlation first
            stats_df = stats_df.sort_values(by="Correlation (%)", ascending=False)
            stats_dict[var] = stats_df
        else:
            logger.info(f"No regions in '{var}' met the >0.95 correlation threshold for a seasonal fit.")

    return stats_dict


def print_sinusoidal_summaries(
        continuous_data_dict: dict[str, pd.DataFrame],
        granularity: float = 1.0
) -> dict[str, pd.DataFrame]:
    """
    Calculates the sinusoidal statistics and prints a clean, formatted
    summary to the console for quick verification.
    """
    stats_dict = calculate_sinusoidal_statistics(continuous_data_dict, granularity)

    if not stats_dict:
        logger.warning("No significant sinusoidal correlations found to print.")
        return {}

    for var, df_stats in stats_dict.items():
        print(f"\n{'='*80}")
        print(f"SEASONAL FIT SUMMARY: {var.upper()} | Granularity: {granularity}°")
        print(f"Regions with R > 0.95: {len(df_stats)}")
        print(f"{'='*80}")

        # Print the top 5 most extreme amplitudes
        sorted_stats = df_stats.sort_values(by="Amplitude (A)", ascending=False)
        print(sorted_stats.head(5).to_string())
        print("...\n")

    return stats_dict


def export_sinusoidal_stats_to_excel(
        stats_dict: dict[str, pd.DataFrame],
        output_filename: str | Path = "sinusoidal_fit_statistics.xlsx",
        output_dir: str | Path = "Excel exported statistical summaries"
) -> Path | None:
    """
    Exports the successful sinusoidal fit parameters to a multi-sheet Excel workbook.
    Each variable gets its own sheet containing the Regions, parameters, and equations.

    Args:
        stats_dict (dict): The dictionary generated by calculate_sinusoidal_statistics.
        output_filename (str | Path, optional): Name of the Excel file.
        output_dir (str | Path, optional): Target directory.

    Returns:
        Path | None: Absolute path to the generated Excel file.
    """
    if not stats_dict:
        logger.warning("No sinusoidal statistics provided to the Excel exporter.")
        return None

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / Path(output_filename).name

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for var, df_stats in stats_dict.items():

                # Excel restricts sheet names to 31 characters.
                safe_var_name = var[:25] + "_Fits"

                # Write the DataFrame, auto-adjusting the region index
                df_stats.to_excel(writer, sheet_name=safe_var_name)

        logger.info(f"Successfully exported sinusoidal tables to {output_path.absolute()}")
        return output_path

    except Exception as exc:
        logger.error(f"Failed to export sinusoidal stats to Excel: {exc}")
        return None


# =============================================================================
# 6. EXECUTION BLOCK
# =============================================================================
if __name__ == '__main__':
    target_dir = Path("era5_monthly_data")

    if target_dir.exists():
        logger.info(f"\nInitiating Batch Period Timeseries Export on directory: {target_dir.name}...")

        try:
            # 1. Retrieve the multi-month data using your standard retrieval function
            period_data = period_retrieval_function(target_dir, timeseries_by_region())

            # 2. Stitch it into a continuous dictionary
            continuous_overall_data = overall_period_timeseries(period_data)

            # 3. Calculate and Export the Tables
            stats_dict = print_sinusoidal_summaries(continuous_overall_data, granularity=2.0)
            export_sinusoidal_stats_to_excel(stats_dict)

            # PDF APPLICATION OF THE MASTER ORCHESTRATOR
            exported_files = visualization_orchestration(
                input_path=target_dir,
                study="Regional Timeseries",
                retrieval_func=timeseries_by_region(),
                plot_generator_func=timeseries_by_region_plotter_factory(granularity=2.0),
                overall_analysis=overall_period_timeseries,
                objective='show',
                output_dir="Exported timeseries plots",
                verbose=False
            )

            if exported_files:
                logger.info(f"\nSUCCESS: Pipeline completed!!")

            # 'show' objective returns the active continuous figures to RAM, so we trigger matplotlib here
            plt.show()

        except Exception as e:
            logger.error(f"Batch Analysis failed: {e}")

    else:
        logger.error(f"Target directory not found at: {target_dir.absolute()}")
