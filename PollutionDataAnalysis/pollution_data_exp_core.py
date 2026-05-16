"""
AirTS-Forecast Project
Section 1.2: Pollution Data Gathering, Exploration, and Spectral Analysis
File: pollution_data_exp_core.py
Author: Tiago TOLOCZKO ROSS

Description:
    Core module for ingesting, parsing, cleaning, and performing diagnostic
    visualizations on pollutant concentration datasets. Implements
    vectorized temporal resampling, outlier remediation, Fourier Transform
    analysis, and ADF Stationarity tests for cyclical trend detection.

    Website: https://www.geodair.fr/donnees/api
    ZAS Code: FR76ZAG01
    Start Date: 21/03/2021 00:01
    End Date: 21/03/2026 00:01
    Data type: a1
"""
import os
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Callable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as stt
from matplotlib.animation import FuncAnimation
from statsmodels.tsa.seasonal import STL
from sympy.printing.pretty.pretty_symbology import line_width

# --- CONSTANTS & CONFIGURATION ---

# Replace data directory by the folder in which the data is stored
DATA_DIRECTORY = Path(r"C:\Users\Tiago\Documents - PC\UTTOP\Enseignements\M1.2\Projet AirTS - Forecast\GEODAIR_POLLUTION_DATA")
TARGET_FILENAME = "O3_20210321_20260321.csv"
PERIODICITY = "hourly"  # Options: 'hourly', 'daily', 'weekly', 'monthly', 'annually'
DIFFERENTIATION_ORDER = 1

ANALYSIS_KWARGS = {
    # File settings
    "sep": ";",
    "encoding": "utf-8",

    # Cleaning & Math settings
    "apply_cleaning": False,        # Data cleaning methods -> experimental
    "outlier_method": "iqr",
    "iqr_multiplier": 2.0,

    "rolling_window": 24*30*6,            # e.g. 24 periods = 1 day rolling trend for 'hourly' data
    "autolag": 'AIC',
    "export_diagnostics": False,    # Toggles the pre-cleaning outlier plot
    "stl_period": 24*30,
    "stl_seasonal": 24*30,             # Should cover an expected cycle (must be odd integer)
    "stl_robust": True,            # Enhances resistance to remaining outliers
    "nlags": 168,                    # ACF/PACF memory lookback limit
    "alpha": 0.05,                  # Significance threshold for ACF/PACF intervals
    "pacf_method": 'ywm',

    # Figure styling settings
    "figsize": (6, 6),
    "color": "teal",
    "marker": "o",
    "markerfacecolor": "red",
    "markeredgecolor": "red",
    "xlabel": "Observation Timestamp",
    "ylabel": "Concentration",
    "unit": "µg/m³",

    # Data animation settings
    "animate": False,
    "animation_filepath": "NO2_Trend_Animation.gif",
    "fps": 24
}

# ======================================================================================================================
# SECTION 1: DATA EXTRACTION AND BASIC STATISTICS
# ======================================================================================================================
def parse_pollution_metadata(filename: str) -> Optional[Dict[str, Any]]:
    """Extracts pollutant identity and temporal bounds from standardized filenames."""
    pattern = r"^(?P<pollutant>.+?)_(?P<start>\d{8})_(?P<end>\d{8})"
    match = re.match(pattern, filename)

    if not match:
        logging.warning(f"METADATA_ERROR: Filename '{filename}' violates naming convention.")
        return None

    try:
        return {
            "pollutant": match.group("pollutant"),
            "start_date": datetime.strptime(match.group("start"), "%Y%m%d"),
            "end_date": datetime.strptime(match.group("end"), "%Y%m%d")
        }
    except ValueError as e:
        logging.error(f"DATE_FORMAT_ERROR: {e}")
        return None

def ingest_pollution_data(file_path: Path, **kwargs: Any) -> Optional[pd.DataFrame]:
    """High-performance CSV ingestion explicitly filtering kwargs."""
    if not file_path.is_file():
        logging.error(f"IO_ERROR: Path does not exist: {file_path}")
        return None

    file_sep = kwargs.get("sep", ";")
    file_encoding = kwargs.get("encoding", "utf-8")

    try:
        df = pd.read_csv(file_path, sep=file_sep, encoding=file_encoding)
        logging.info(f"INGESTION_SUCCESS: {file_path.name} | Shape: {df.shape} | Sep: '{file_sep}'")
        return df
    except Exception as e:
        logging.error(f"PARSING_FATAL: Failed to process {file_path.name}. Details: {e}")
        return None


def flag_pollution_outliers(data_series: pd.Series, method: str = "iqr", **kwargs: Any) -> pd.Series:
    """Identifies statistical outliers in a time-series dataset."""
    method = method.lower()
    if data_series.empty or data_series.dropna().empty:
        return pd.Series(False, index=data_series.index)

    if method == "zscore":
        threshold = kwargs.get("z_threshold", 3.0)
        std = data_series.std()
        if std == 0: return pd.Series(False, index=data_series.index)
        z_scores = np.abs((data_series - data_series.mean()) / std)
        outliers = z_scores > threshold
    elif method == "iqr":
        multiplier = kwargs.get("iqr_multiplier", 1.5)
        q1 = data_series.quantile(0.25)
        q3 = data_series.quantile(0.75)
        iqr = q3 - q1
        outliers = (data_series < (q1 - multiplier * iqr)) | (data_series > (q3 + multiplier * iqr))
    else:
        return pd.Series(False, index=data_series.index)

    logging.info(f"DATA_CLEANING: '{method}' method flagged {outliers.sum()} outliers.")
    return outliers

def pollution_data_resampling(
        data_path: Path,
        periodicity: str = "hourly",
        diff_order: int = 0,
        **kwargs: Any
) -> Tuple[pd.DataFrame, Union[pd.Series, Tuple[pd.Series, ...]]]:
    """
    Performs spatial validation, outlier remediation, dynamic resampling,
    and optional differentiation to enforce mathematical stationarity.

    Args:
        data_path (Path): Filepath to the raw dataset.
        periodicity (str): Target resampling frequency (e.g., 'hourly', 'daily').
        diff_order (int): The n-th differentiation order expected
        **kwargs: Configuration dictionary containing analysis parameters.

    Returns:
        Tuple[pd.DataFrame, Union[pd.Series, Tuple[pd.Series, ...]]]:
            - The cleaned raw DataFrame.
            - If differentiation order == 0: The resampled time series.
            - If differentiation order > 0: An n-tuple of differentiated series,
              where the n-th order differentiation is at index 0.
    """
    # 1. KWARGS EXTRACTION & SANITIZATION
    pd_kwargs = kwargs.copy()
    viz_keys = ['figsize', 'color', 'marker', 's', 'xlabel', 'ylabel', 'title', 'export_diagnostics', 'animate']
    for key in viz_keys:
        pd_kwargs.pop(key, None)

    # 2. INGESTION
    df = ingest_pollution_data(data_path, **pd_kwargs)
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    lats, lons = df["Latitude"].unique(), df["Longitude"].unique()
    logging.info(f"GEOSPATIAL_AUDIT: Lats {lats} | Lons {lons}")

    df["Date de début"] = pd.to_datetime(df["Date de début"])
    df = df.set_index("Date de début").sort_index()

    # 3. OUTLIER REMEDIATION
    if kwargs.get("apply_cleaning", True):
        outlier_mask = flag_pollution_outliers(df["valeur"], **kwargs)

        if kwargs.get("export_diagnostics", False):
            diag_fig = visualize_outlier_diagnostics(df["valeur"], outlier_mask, "Outlier Diagnostics", **kwargs)
            plt.show()
            plt.close(diag_fig)

        df.loc[outlier_mask, "valeur"] = np.nan

    # 4. TEMPORAL RESAMPLING
    freq_map = {"hourly": "h", "daily": "D", "weekly": "W", "monthly": "ME", "annually": "YE"}
    resample_rule = freq_map.get(periodicity.lower(), periodicity)

    try:
        resampled_series = df["valeur"].resample(resample_rule).mean()
        logging.info(f"TEMPORAL_ANALYSIS: Resampled to '{periodicity}' ({len(resampled_series)} valid points).")
    except ValueError as e:
        logging.error(f"RESAMPLING_ERROR: {e}")
        return df, pd.Series(dtype=float)

    # 5. STATIONARITY TRANSFORMATION (DIFFERENTIATION)
    if diff_order > 0:
        original_length = len(resampled_series)
        diff_list = []
        current_series = resampled_series

        # Sequentially apply the difference and store each order
        for step in range(diff_order):
            current_series = current_series.diff().dropna()
            diff_list.append(current_series)

        logging.info(f"DATA_TRANSFORMATION: Applied order-{diff_order} differentiation. Final size reduced from {original_length} to {len(current_series)}.")

        # Defensive check in case the dataset was too small for the requested order
        if current_series.empty:
            logging.error("DATA_TRANSFORMATION_ERROR: Differentiation resulted in an empty dataset. Consider lowering the differentiation order.")

        # Reverse the list so the highest (n-th) order is at position 0
        diff_list.reverse()
        diff_list.append(resampled_series)
        return df, tuple(diff_list)

    # If no differentiation is requested, return the single series
    return df, resampled_series


# =====================================================================
# 1.1 TIMESERIES VISUALIZATION
# =====================================================================
def visualize_outlier_diagnostics(raw_series: pd.Series, outlier_mask: pd.Series, title: str, **kwargs: Any) -> plt.Figure:
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))
    ax.plot(raw_series.index, raw_series.values, color="gray", alpha=0.5, label="Raw Data", zorder=1)

    outlier_data = raw_series[outlier_mask]
    if not outlier_data.empty:
        ax.scatter(outlier_data.index, outlier_data.values, color="red", marker="x", label=f"Outliers ({len(outlier_data)})", zorder=2)

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    return fig

def visualize_pollution_timeseries(
        data_input: Union[pd.Series, Tuple[pd.Series, ...]],
        title: str,
        **kwargs: Any
) -> Union[plt.Figure, Tuple[plt.Figure, animation.FuncAnimation], None]:
    """
    Standardized scatter/line plot generation with optional timeseries animation.
    Dynamically supports single Series or an n-tuple of Series (creating n subplots).
    """
    # 1. STANDARDIZE INPUT
    is_single = isinstance(data_input, pd.Series)
    series_tuple = (data_input,) if is_single else data_input

    # 1.1 DEFENSE: Filter out empty series and drop NAs
    cleaned_series = [s.dropna() for s in series_tuple if not s.dropna().empty]
    n_plots = len(cleaned_series)
    ordered_series = cleaned_series[::-1]

    if n_plots == 0:
        logging.warning("VISUALIZATION_WARNING: Empty data provided. Aborting scatter render.")
        return None

    # 2. FIGURE CONFIGURATION
    base_width = kwargs.get("figsize", (10, 6))[0]
    fig, axes = plt.subplots(n_plots, 1, figsize=(base_width, 5 * n_plots), sharex=True)

    # 2.1 DEFENSE: Ensure axes is iterable even for a single plot
    if n_plots == 1:
        axes = [axes]

    # 2.2  STYLE: getting kwargs
    base_color = kwargs.get("color", "teal")
    marker_style = kwargs.get("marker", "o")
    marker_size = kwargs.get("marker_size", 2)
    markerfacecolor = kwargs.get("markerfacecolor", "black")
    markeredgecolor = kwargs.get("markeredgecolor", "black")
    line_style = kwargs.get("line_style", "dashed")
    line_width = kwargs.get("line_width", .5)
    is_animated = kwargs.get("animate", False)

    # 3. STATIC PLOTTING
    if not is_animated:
        # 3.1 Defining different series
        for i, (ax, series) in enumerate(zip(axes, ordered_series)):
            ax.plot(series.index, series.values,
                    color=base_color,
                    marker=marker_style, markersize=marker_size,
                    markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor,
                    linestyle=line_style, linewidth=line_width)

            # 3.2 Formatting
            concentration_label = kwargs.get("ylabel", "Concentration")
            concentration_unit =  kwargs.get("unit", None)
            if i != 0:
                data_label = f"{concentration_label} order {i}\n$[{concentration_unit}/h^{i}]$"
            else:
                data_label = f"{concentration_label} $[{concentration_unit}]$"

            ax.set_title(title if is_single else f"{title} - Time series order {i}")
            ax.set_ylabel(data_label)
            ax.grid(True, linestyle="--", alpha=0.6)

        axes[-1].set_xlabel(kwargs.get("xlabel", "Date"))

        #        fig.tight_layout()
        return fig

    # 4. EXTRA: ANIMATED PLOTTING
    else:
        lines = []
        points = []
        max_frames = 0

        # Setup axis limits and initial objects for each subplot
        for i, (ax, series) in enumerate(zip(axes, ordered_series)):
            y_min, y_max = series.min(), series.max()
            # Prevent zero-division/flatline padding errors
            y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 1.0

            ax.set_xlim(series.index.min(), series.index.max())
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

            line, = ax.plot([], [], color=base_color, linewidth=2, label="Trend")
            point, = ax.plot([], [], color="red", marker=marker_style, markersize=6, label="Current")
            lines.append(line)
            points.append(point)

            ax.set_title(title if is_single else f"{title} - Transformation Step {i}")
            ax.set_ylabel(kwargs.get("ylabel", "Concentration"))
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper right")

            if len(series) > max_frames:
                max_frames = len(series)

        axes[-1].set_xlabel(kwargs.get("xlabel", "Date"))
        fig.tight_layout()

        # Animation update function
        def init() -> tuple:
            for line, point in zip(lines, points):
                line.set_data([], [])
                point.set_data([], [])
            return tuple(lines + points)

        def update(frame_idx: int) -> tuple:
            if frame_idx == 0:
                return tuple(lines + points)

            for line, point, series in zip(lines, points, cleaned_series):
                # .iloc handles out-of-bounds gracefully if series have different lengths
                current_slice = series.iloc[:frame_idx]
                if not current_slice.empty:
                    line.set_data(current_slice.index, current_slice.values)
                    point.set_data([current_slice.index[-1]], [current_slice.values[-1]])

            return tuple(lines + points)

        # Generate Animation
        anim = animation.FuncAnimation(fig, update, frames=max_frames + 1, init_func=init, blit=False)

        try:
            anim.save(kwargs.get("animation_filepath", "anim.gif"), writer='pillow', fps=kwargs.get("fps", 10))
            logging.info("ANIMATION: Saved successfully.")
        except Exception as e:
            logging.error(f"ANIMATION_ERROR: {e}")

        return fig, anim


# ======================================================================================================================
# SECTION 2: FREQUENCY ANALYSIS - POWER SPECTRAL DENSITY AND PERIODOGRAM
# ======================================================================================================================
def fourier_analysis(
        data_series: pd.Series,
        periodicity: str,
        top_k: int = 5
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Applies Fast Fourier Transform to detect cyclical trends."""
    if data_series.dropna().empty:
        return np.array([]), np.array([]), {}

    signal = data_series.interpolate(method='linear').bfill().ffill()
    signal_detrended = signal - signal.mean()
    n_samples = len(signal_detrended)

    if n_samples < 2: return np.array([]), np.array([]), {}

    fft_values = np.fft.rfft(signal_detrended.values)
    amplitudes = np.abs(fft_values) / n_samples
    frequencies = np.fft.rfftfreq(n_samples, d=1.0)

    valid_idx = frequencies > 0
    valid_freqs, valid_amps = frequencies[valid_idx], amplitudes[valid_idx]
    periods = 1 / valid_freqs

    peak_indices = np.argsort(valid_amps)[::-1][:top_k]
    dominant_periods = periods[peak_indices]
    dominant_amplitudes = valid_amps[peak_indices]

    context_map = {
        "hourly": "Daily Cycle (Hours)", "daily": "Weekly Cycle (Days)",
        "weekly": "Monthly/Seasonal Cycle (Weeks)", "monthly": "Annual Cycle (Months)"
    }

    summary = {
        "periodicity_base": periodicity,
        "expected_cycle_label": context_map.get(periodicity.lower(), "Unknown Cycle"),
        "dominant_periods_raw": np.round(dominant_periods, 2).tolist(),
        "dominant_amplitudes": np.round(dominant_amplitudes, 4).tolist(),
        "signal_variance_captured": np.round(np.sum(dominant_amplitudes**2) / np.sum(valid_amps**2) * 100, 2)
    }
    return periods, valid_amps, summary


def visualize_fourier_spectrum(
        periods: np.ndarray,
        amplitudes: np.ndarray,
        summary: Dict[str, Any],
        save_file: bool = False,
        **kwargs: Any
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))
    if len(periods) == 0: return fig

    # 1. Plot the linear amplitudes
    ax.plot(periods, amplitudes, color=kwargs.get("fft_color", "purple"), linewidth=1.5)

    top_p = summary.get("dominant_periods_raw", [])
    top_a = summary.get("dominant_amplitudes", [])

    if len(top_p) > 0:
        ax.scatter(top_p, top_a, color="red", zorder=3, label="Dominant Cycles")

        # 2. Smart Annotation (Un-clustering Text)
        for i, (p, a) in enumerate(zip(top_p, top_a)):
            # Alternate vertical offsets (15px vs 35px) to prevent text collision
            y_offset = 15 if i % 2 == 0 else 35
            ax.annotate(
                f"T={p}: Amp={a:.2f}",
                (p, a),
                textcoords="offset points",
                xytext=(5, -3),      # Shift 15 points right, 20 points up
                ha='left',            # Anchor the left side of the text to the offset point
                va='bottom',          # Anchor the bottom of the text
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color='black', alpha=0.6) # Connect text to peak
            )

    # 3. Handle X-Axis Clustering
    # We restore the X-axis log scale by default to prevent horizontal clustering of short periods.
    # We also add a kwarg override just in case you ever strictly need a linear X-axis.
    if kwargs.get("force_linear_x", False):
        # If forced linear, we zoom in slightly to ignore infinite long-tails
        max_period = max(top_p) * 1.5 if top_p else periods.max()
        ax.set_xlim(0, max_period)
    else:
        ax.set_xscale("log")

    base = summary.get('periodicity_base', 'Unknown').capitalize()
    ax.set_title(f"Spectral Analysis Periodogram ({base})")
    ax.set_xlabel(f"Period Length ({base} units) - Logarithmic scale")
    ax.set_ylabel("Amplitude")

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    # 4. File Saving Execution
    if save_file:
        # Extract path and resolution settings from kwargs, providing safe defaults
        save_path = kwargs.get("save_path", f"fourier-analysis.png")
        save_folder = kwargs.get("save_folder", "./")

        dpi = kwargs.get("dpi", 300)

        try:
            # Validate and create target directory if it does not exist
            target_dir = save_folder
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)

            # Save the figure using tight bounding boxes to prevent label cutoff
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
            logging.info(f"VISUALIZATION: Fourier spectrum successfully saved to {save_path}")

        except Exception as e:
            logging.error(f"VISUALIZATION_ERROR: I/O failure while saving figure to {save_path}. {e}")

    return fig


# ======================================================================================================================
# SECTION 3: STATIONARITY AND TREND ANALYSIS
# ======================================================================================================================
def analyze_stationarity_trend(
        data_input: Union[pd.Series, tuple[pd.Series, ...]],
        window_size: int,
        **kwargs: Any
) -> tuple[Union[pd.Series, tuple[pd.Series, ...]], dict[str, Any]]:
    """
    Calculates a rolling average and performs an Augmented Dickey-Fuller (ADF) test.
    Dynamically supports both a single Series or a tuple of sequentially differentiated Series.

    Args:
        data_input (Union[pd.Series, Tuple[pd.Series, ...]]): The time-series data to analyze.
        window_size (int): The number of periods for the rolling average.
        **kwargs: Additional parameters (e.g., 'autolag', 'alpha' for significance threshold).

    Returns:
        Tuple:
            - The rolling mean Series (or a tuple of rolling mean Series matching the input).
            - A summary dictionary containing the ADF statistics and stationarity boolean.
    """
    # 1. STANDARDIZE INPUT
    # Wrap single series in a tuple so we can iterate consistently
    is_single_series = isinstance(data_input, pd.Series)
    series_tuple = (data_input,) if is_single_series else data_input

    trend_list = []
    master_summary = {}

    # Define significance level for stationarity (default to 5%)
    alpha = kwargs.get("alpha", 0.05)

    # 2. ITERATE AND ANALYZE
    for idx, series in enumerate(series_tuple):
        series_label = f"Order_{len(series_tuple) - idx - 1}"

        # 2.1 DEFENSE: Validation for the current series in the loop
        if series.dropna().empty or len(series) <= window_size:
            trend_list.append(pd.Series(dtype=float))
            master_summary[series_label] = {"error": "Insufficient data"}
            continue

        # 2.2: METHOD: Rolling Average Calculation
        moving_mean = series.rolling(window=window_size, min_periods=1).mean()
        clean_mean = moving_mean.dropna()
        rolled_series =  clean_mean[window_size:]

        trend_list.append(rolled_series)

        if len(clean_mean) < 3:
            master_summary[series_label] = {"error": "Insufficient data after rolling mean"}
            continue

        # 3. AUGMENTED DICKEY-FULLER TEST
        try:
            # 3.1 CALCULATION: ADF Test
            adf_tuple: tuple | float = stt.adfuller(rolled_series, autolag=kwargs.get("autolag", "AIC"))

            # 3.2 STORE: Map the tuple output to our dictionary keys
            keys = ['adf_statistic', 'p_value', 'used_lag', 'n_observations', 'critical_values', 'ic_best']
            series_summary = dict(zip(keys, adf_tuple))

            # 3.3 ANALYSIS: Calculate the boolean stationarity flag
            p_val = series_summary['p_value']
            series_summary['is_stationary'] = bool(p_val < alpha)

            master_summary[series_label] = series_summary
            logging.info(f"ADF_TEST_COMPLETE ({series_label}): p-value={p_val:.4f}, Stationary={series_summary['is_stationary']}")

        except Exception as e:
            logging.error(f"ADF_TEST_FAILURE ({series_label}): {e}")
            master_summary[series_label] = {"error": str(e)}

    # 4. ADAPTIVE RETURN
    # 4.1 Case 1: If the user provided a single series, return a single series and un-nest the dictionary
    if is_single_series:
        return trend_list[0], master_summary.get("Order_0", {})

    # 4.2 Case 2: If the user provided a tuple, return a tuple of trends of orders (n, n-1, ..., 1, 0)
    # and the fully nested dictionary
    return tuple(trend_list), master_summary


def visualize_trend_analysis(
        raw_input: Union[pd.Series, Tuple[pd.Series, ...]],
        trend_input: Union[pd.Series, Tuple[pd.Series, ...]],
        adf_summary: Dict[str, Any],
        title: str,
        save_file: bool = False,
        **kwargs: Any
) -> Optional[plt.Figure]:
    """
    Generates a dynamic multi-panel plot overlaying rolling means on raw or differentiated data.
    Supports both single Series and n-tuples of Series.

    Args:
        raw_input: The base time-series data (single Series or tuple of Series).
        trend_input: The calculated rolling trend (single Series or tuple of Series).
        adf_summary: Dictionary containing ADF test statistics.
        title: Global title for the figure.
        save_file (Bool)
        **kwargs: Matplotlib styling arguments.

    Returns:
        Optional[plt.Figure]: The rendered Matplotlib figure, or None if inputs are invalid.
    """
    # 1. STANDARDIZE INPUTS
    is_single = isinstance(raw_input, pd.Series)
    raw_tuple = (raw_input,) if is_single else raw_input
    trend_tuple = (trend_input,) if isinstance(trend_input, pd.Series) else trend_input

    n_plots = len(raw_tuple)
    if n_plots == 0:
        logging.warning("VISUALIZATION_WARNING: Empty data provided. Aborting trend render.")
        return None

    # 2. FIGURE CONFIGURATION
    # Dynamically scale height based on the number of subplots to maintain aspect ratio
    base_width = kwargs.get("figsize", (12, 6))[0]
    fig, axes = plt.subplots(n_plots, 1, figsize=(base_width, 5 * n_plots), sharex=True)

    # Ensure axes is iterable even if there is only 1 plot
    if n_plots == 1:
        axes = [axes]

    # Extract styling safely
    base_c = kwargs.get("base_color", "gray")
    trend_c = kwargs.get("trend_color", "darkorange")

    # 3. ITERATIVE RENDERING
    for i, (raw_s, trend_s) in enumerate(zip(raw_tuple, trend_tuple)):
        ax = axes[i]

        # Plot signals
        ax.plot(raw_s.index, raw_s.values, color=base_c, alpha=0.4, linewidth=1, label="Base Data", zorder=1)
        ax.plot(trend_s.index, trend_s.values, color=trend_c, linewidth=1.5, label="Rolling Trend", zorder=2)

        # Retrieve specific stats for this subplot
        stats = adf_summary if is_single else adf_summary.get(f"Order {i}", {})

        # Build statistical annotation box
        if stats and "error" not in stats:
            p_val = stats.get('p_value', 1.0)
            is_stat = stats.get('is_stationary', False)
            stat_text = f"Augmented Dickey-Fuller\np-value: {p_val:.4f}\nStationary: {is_stat}"

            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
            ax.text(0.02, 0.95, stat_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props, zorder=3)
        elif "error" in stats:
            ax.text(0.02, 0.95, f"ADF Test Failed: {stats['error']}", transform=ax.transAxes, fontsize=10, color='red', verticalalignment='top', bbox=dict(facecolor='white'))

        # Subplot formatting
        if is_single:
            ax.set_title(title)
        else:
            # Assumes order corresponds to the differentiation list (highest order at index 0)
            order_label = f"Order {len(raw_tuple)-i-1} Data"
            ax.set_title(f"Transformation: {order_label}")

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

    # 4. FINAL FORMATTING
    if not is_single:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    ax.set_xlabel(kwargs.get("xlabel", "Date"))
    #    fig.tight_layout()

    # 5. OUTPUT: FILE SAVING
    if save_file:
        # Extract path and resolution settings from kwargs, providing safe defaults
        save_path = kwargs.get("save_path", f"{title}.png")
        save_folder = kwargs.get("save_folder", "./")
        dpi = kwargs.get("dpi", 300)

        try:
            # Validate and create target directory if it does not exist
            target_dir = os.path.dirname(save_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)

            # Save the figure using tight bounding boxes to prevent label cutoff
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
            logging.info(f"VISUALIZATION: Fourier spectrum successfully saved to {save_path}")

        except Exception as e:
            logging.error(f"VISUALIZATION_ERROR: I/O failure while saving figure to {save_path}. {e}")

    return fig


# ======================================================================================================================
# SECTION 4: STL DECOMPOSITION
# ======================================================================================================================
def analyze_stl_decomposition(
        data_series: pd.Series,
        seasonal: int = 7,
        **kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Performs Seasonal and Trend decomposition using Loess (STL)."""
    if data_series.dropna().empty:
        logging.warning("STL_ANALYSIS: Series is empty. Aborting decomposition.")
        return pd.DataFrame(), {}

    if seasonal % 2 == 0:
        seasonal += 1
        logging.info(f"STL_ANALYSIS: Adjusted 'seasonal' parameter to odd integer ({seasonal}).")

    signal = data_series.interpolate(method='linear').bfill().ffill()

    try:
        period = kwargs.get("period", None)
        robust = kwargs.get("robust", False)

        stl_model = STL(signal, seasonal=seasonal, period=period, robust=robust)
        res = stl_model.fit()

        decomposition_df = pd.DataFrame({
            'observed': res.observed,
            'trend': res.trend,
            'seasonal': res.seasonal,
            'resid': res.resid
        }, index=signal.index)

        var_resid = res.resid.var()
        var_trend = (res.trend + res.resid).var()
        var_seasonal = (res.seasonal + res.resid).var()

        trend_strength = max(0, 1 - (var_resid / var_trend)) if var_trend > 0 else 0
        seasonal_strength = max(0, 1 - (var_resid / var_seasonal)) if var_seasonal > 0 else 0

        summary = {
            "seasonal_window": seasonal,
            "robust_fitting": robust,
            "trend_strength": np.round(trend_strength, 4),
            "seasonal_strength": np.round(seasonal_strength, 4),
            "residual_variance": np.round(var_resid, 4)
        }

        logging.info(f"STL_COMPLETE: Trend Strength={trend_strength:.2f} | Seasonal Strength={seasonal_strength:.2f}")
        return decomposition_df, summary

    except Exception as e:
        logging.error(f"STL_FAILURE: Could not compute decomposition. Details: {e}")
        return pd.DataFrame(), {}


def visualize_stl_decomposition(
        primary_decomp: pd.DataFrame,
        primary_summary: Dict[str, Any],
        title: str = "STL Decomposition",
        secondary_decomp: Optional[pd.DataFrame] = None,
        secondary_label: str = "Robust Fit",
        **kwargs: Any
) -> Optional[plt.Figure]:
    if primary_decomp.empty:
        logging.warning("VISUALIZATION_WARNING: Primary STL DataFrame is empty. Aborting render.")
        return None

    figsize = kwargs.get("figsize", (12, 10))
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    base_color = kwargs.get("color", "teal")

    axes[0].plot(primary_decomp.index, primary_decomp['observed'], color='gray', label='Observed')
    axes[1].plot(primary_decomp.index, primary_decomp['trend'], alpha=1, color=base_color, linewidth=.75, label='Primary Trend')
    axes[2].plot(primary_decomp.index, primary_decomp['seasonal'], color='darkblue', label='Primary Seasonal')
    axes[3].scatter(primary_decomp.index, primary_decomp['resid'], color='red', s=5, alpha=0.5, label='Primary Residuals')
    axes[3].axhline(0, color='black', linestyle='--', linewidth=1)

    t_str = primary_summary.get("trend_strength", 0)
    s_str = primary_summary.get("seasonal_strength", 0)

    axes[1].text(0.01, 0.9, f"Primary Strength: {t_str:.4f}", transform=axes[1].transAxes,
                 bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
    axes[2].text(0.01, 0.9, f"Primary Strength: {s_str:.4f}", transform=axes[2].transAxes,
                 bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

    if secondary_decomp is not None and not secondary_decomp.empty:
        overlay_color = kwargs.get("overlay_color", "darkorange")
        line_style = kwargs.get("overlay_linestyle", "--")
        alpha = kwargs.get("overlay_alpha", 0.85)

        axes[1].plot(secondary_decomp.index, secondary_decomp['trend'],
                     color=overlay_color, linestyle=line_style, linewidth=.75, alpha=.75, label=secondary_label)
        axes[2].plot(secondary_decomp.index, secondary_decomp['seasonal'],
                     color=overlay_color, linestyle=line_style, linewidth=.75,
                     alpha=alpha, label=secondary_label)
        axes[3].scatter(secondary_decomp.index, secondary_decomp['resid'],
                        color=overlay_color, marker="x", s=kwargs.get("overlay_marker_size", 15),
                        alpha=alpha, label=secondary_label)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc='upper right')

    axes[3].set_xlabel(kwargs.get("xlabel", "Date"))
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout()

    return fig


# ======================================================================================================================
# SECTION 5: MEMORY ANALYSIS - ACF AND PACF
# ======================================================================================================================
def analyze_memory(
        data_series: pd.Series,
        **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Analyzes the temporal memory of a dataset using ACF and PACF."""
    if data_series.dropna().empty:
        logging.warning("MEMORY_ANALYSIS: Series is empty. Aborting decomposition.")
        return np.array([]), np.array([]), {}

    nlags = kwargs.get('nlags', 40)
    alpha = kwargs.get('alpha', None)

    acf_kwargs = {
        'adjusted': kwargs.get('adjusted', False),
        'qstat': kwargs.get('qstat', False),
        'fft': kwargs.get('fft', True),
        'bartlett_confint': kwargs.get('bartlett_confint', True),
        'missing': kwargs.get('missing', 'none')
    }
    pacf_kwargs = {'method': kwargs.get('pacf_method', 'ywadjusted')}

    try:
        clean_data = data_series.dropna()

        # ACF Execution
        if alpha is not None or acf_kwargs['qstat']:
            acf_res = stt.acf(clean_data, nlags=nlags, alpha=alpha, **acf_kwargs)
            acf_values = acf_res[0]
            acf_confint = acf_res[1] if alpha is not None else None
            acf_extended = acf_res[1:]
        else:
            acf_values = stt.acf(clean_data, nlags=nlags, alpha=alpha, **acf_kwargs)
            acf_confint, acf_extended = None, ()

        # PACF Execution
        if alpha is not None:
            pacf_res = stt.pacf(clean_data, nlags=nlags, alpha=alpha, **pacf_kwargs)
            pacf_values = pacf_res[0]
            pacf_confint = pacf_res[1]
            pacf_extended = pacf_res[1:]
        else:
            pacf_values = stt.pacf(clean_data, nlags=nlags, alpha=alpha, **pacf_kwargs)
            pacf_confint, pacf_extended = None, ()

        sig_acf_lags, sig_pacf_lags = [], []

        if acf_confint is not None:
            for lag in range(1, len(acf_values)):
                if acf_confint[lag][0] > 0 or acf_confint[lag][1] < 0:
                    sig_acf_lags.append(lag)

        if pacf_confint is not None:
            for lag in range(1, len(pacf_values)):
                if pacf_confint[lag][0] > 0 or pacf_confint[lag][1] < 0:
                    sig_pacf_lags.append(lag)

        summary = {
            "max_lag_evaluated": len(acf_values) - 1,
            "alpha_threshold": alpha,
            "significant_acf_lags": sig_acf_lags,
            "significant_pacf_lags": sig_pacf_lags,
            "acf_extended_results": acf_extended,
            "pacf_extended_results": pacf_extended
        }

        logging.info(f"MEMORY_ANALYSIS: Found {len(sig_acf_lags)} significant ACF lags and {len(sig_pacf_lags)} significant PACF lags.")
        return acf_values, pacf_values, summary

    except Exception as e:
        logging.error(f"MEMORY_ANALYSIS: CRITICAL FAILURE - {e}")
        return np.array([]), np.array([]), {}


def visualize_correlogram(
        acf_values: np.ndarray,
        additional_stats: Dict[str, Any],
        title: str = "Memory Periodicity Analysis",
        pacf_values: Optional[np.ndarray] = None,
        **kwargs: Any
) -> Optional[plt.Figure]:
    """
    Generates a correlogram (stem plot) to visualize the temporal memory of the dataset.
    Dynamically switches to a side-by-side layout if PACF values are provided.

    Args:
        acf_values (np.ndarray): The calculated autocorrelation coefficients.
        additional_stats (Dict[str, Any]): Dictionary containing extended stats (confidence intervals).
        title (str): Global title for the figure.
        pacf_values (Optional[np.ndarray]): The calculated partial autocorrelation coefficients.
        **kwargs: Matplotlib styling arguments cascaded from the Orchestrator.

    Returns:
        Optional[plt.Figure]: The Matplotlib figure object, or None if plotting fails.
    """
    # 1. INITIAL VALIDATION
    if acf_values.size == 0:
        logging.warning("VISUALIZATION_WARNING: ACF array is empty. Aborting correlogram render.")
        return None

    # 2. DYNAMIC LAYOUT CONFIGURATION
    has_pacf = pacf_values is not None and pacf_values.size > 0
    ncols = 2 if has_pacf else 1

    # Adjust default figsize to be wider if we are plotting two charts side-by-side
    default_figsize = (14, 5) if has_pacf else (12, 6)
    fig, axes = plt.subplots(1, ncols, figsize=kwargs.get("figsize", default_figsize))

    # Normalize 'axes' into an iterable list to prevent indexing errors when ncols == 1
    if ncols == 1:
        axes = [axes]

    # Shared styling configurations
    marker_style = kwargs.get("marker", "o")
    base_color = kwargs.get("color", "teal")

    # --- 3. RENDER PLOT 1: AUTOCORRELATION (ACF) ---
    ax_acf = axes[0]
    lags_acf = np.arange(len(acf_values))

    mline_acf, slines_acf, bline_acf = ax_acf.stem(lags_acf, acf_values, basefmt="k-")
    plt.setp(mline_acf, color=base_color, marker=marker_style, markersize=5)
    plt.setp(slines_acf, color=base_color, linewidth=1.5, alpha=0.7)
    plt.setp(bline_acf, color="black", linewidth=1.2)
    plt.vlines(24,0,1,linestyle="dashed")
    plt.vlines(48,0,1,linestyle="dashed")

    # Apply ACF Confidence Interval Shading
    extended_results = additional_stats.get("extended_results")
    if extended_results and len(extended_results) > 0:
        conf_int_acf = extended_results[0]
        centered_conf_acf = conf_int_acf - acf_values[:, None]
        ax_acf.fill_between(
            lags_acf, centered_conf_acf[:, 0], centered_conf_acf[:, 1],
            color='gray', alpha=0.25, label="95% Confidence Interval"
        )
        ax_acf.legend(loc="upper right")

    # Format ACF Axis
    ax_acf.set_title("Autocorrelation (ACF)" if has_pacf else title)
    ax_acf.set_xlabel("Lag (Periods)")
    ax_acf.set_ylabel("Correlation")
    ax_acf.axhline(0, color='black', linewidth=1)
    ax_acf.grid(True, linestyle="--", alpha=0.5)
    plt.vlines(24,0,1,linestyle="dashed")
    plt.vlines(48,0,1,linestyle="dashed")
    ax_acf.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # --- 4. RENDER PLOT 2: PARTIAL AUTOCORRELATION (PACF) ---
    if has_pacf:
        ax_pacf = axes[1]
        lags_pacf = np.arange(len(pacf_values))

        # Optional: Allow a distinct color for PACF via kwargs, fallback to a contrasting color
        pacf_color = kwargs.get("pacf_color", "darkorange")

        mline_pacf, slines_pacf, bline_pacf = ax_pacf.stem(lags_pacf, pacf_values, basefmt="k-")
        plt.setp(mline_pacf, color=pacf_color, marker=marker_style, markersize=5)
        plt.setp(slines_pacf, color=pacf_color, linewidth=1.5, alpha=0.7)
        plt.setp(bline_pacf, color="black", linewidth=1.2)

        # Apply PACF Confidence Interval Shading
        # Note: Your math module must pass this key inside the 'additional_stats' dictionary
        pacf_extended = additional_stats.get("pacf_extended_results")
        if pacf_extended and len(pacf_extended) > 0:
            conf_int_pacf = pacf_extended[0]
            centered_conf_pacf = conf_int_pacf - pacf_values[:, None]
            ax_pacf.fill_between(
                lags_pacf, centered_conf_pacf[:, 0], centered_conf_pacf[:, 1],
                color='gray', alpha=0.25, label="95% Confidence Interval"
            )
            ax_pacf.legend(loc="upper right")

        # Format PACF Axis
        ax_pacf.set_title("Partial Autocorrelation (PACF)")
        ax_pacf.set_xlabel("Lag (Periods)")
        ax_pacf.set_ylabel("Partial Correlation")
        ax_pacf.axhline(0, color='black', linewidth=1)
        ax_pacf.grid(True, linestyle="--", alpha=0.5)
        ax_pacf.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Apply a global title to the entire figure when side-by-side
        fig.suptitle(title, fontsize=14, fontweight='bold')
#        fig.tight_layout()

    logging.info("VISUALIZATION: Correlogram(s) rendered successfully.")
    return fig


# ======================================================================================================================
# SECTION 6: MASTER ORCHESTRATOR
# ======================================================================================================================
def pollution_data_plotting_main() -> None:
    """Main execution routine orchestrating extraction, cleaning, and analysis."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # --- 0. INITIALIZATION --------------------------------------------------------------------------------------------
    meta = parse_pollution_metadata(TARGET_FILENAME)
    if not meta: return
    full_path = DATA_DIRECTORY / TARGET_FILENAME

    # --- 1. INGESTION, DIAGNOSTICS & RESAMPLING -----------------------------------------------------------------------
    df, processed_data = pollution_data_resampling(full_path, periodicity=PERIODICITY,
                                                    diff_order=DIFFERENTIATION_ORDER, **ANALYSIS_KWARGS)

    # Safely handle the Union return type (Series vs Tuple of Series)
    is_single_series = isinstance(processed_data, pd.Series)
    data_tuple = (processed_data,) if is_single_series else processed_data

    # Defensive check against empty datasets
    if len(data_tuple) == 0 or data_tuple[0].empty:
        logging.error("Process aborted: Empty dataset.")
        return

    # Extract the primary working series for single-series analyses (FFT, STL, Memory)
    # Note: If differentiated, index 0 holds the highest n-th order stationary series.
    target_series = data_tuple[0]

    # --- 1.1 CLEAN VISUALIZATION (STATIC VS ANIMATED) ---
    title_str = f"{PERIODICITY.capitalize()} Trend: {meta['pollutant']}"

    # Pass 'processed_data' to fully utilize the dynamic subplot generation
    # for n-order differentiation, rather than just 'target_series'.
    if ANALYSIS_KWARGS.get("animate", False):
        clean_fig, anim = visualize_pollution_timeseries(processed_data, title=title_str, **ANALYSIS_KWARGS)
        logging.info("ORCHESTRATOR: Rendering animation. Close window to proceed.")
        if clean_fig:
            plt.show()
            plt.close(clean_fig)
    else:
        clean_fig = visualize_pollution_timeseries(processed_data, title=title_str, **ANALYSIS_KWARGS)
        if clean_fig:
            plt.show()
            plt.close(clean_fig)

    print(f"\n{'#'*40}\nPOLLUTION & SPECTRAL ANALYSIS REPORT\n{'#'*40}")
    print(f"Target Pollutant:       {meta['pollutant']}")
    print(f"Temporal Bounds:        {meta['start_date'].date()} to {meta['end_date'].date()}")
    print(f"Raw Data Points:        {len(df)}")
    print(f"Cleaned Periods ({PERIODICITY[0].upper()}): {len(target_series)}")
    print(f"Global Arithmetic Mean: {df['valeur'].mean():.4f}")

    # --- 2. FOURIER TRANSFORM ANALYSIS --------------------------------------------------------------------------------
    periods, amps, fft_summary = fourier_analysis(
        target_series, periodicity=PERIODICITY, top_k=3
    )
    if fft_summary:
        fft_fig = visualize_fourier_spectrum(periods, amps, fft_summary, **ANALYSIS_KWARGS)
        plt.show()
        plt.close(fft_fig)

        print("\n--- SPECTRAL TREND INDICATORS ---")
        print(f"Primary Target Cycle: {fft_summary['expected_cycle_label']}")
        for rank, (p, a) in enumerate(zip(fft_summary['dominant_periods_raw'], fft_summary['dominant_amplitudes']), 1):
            print(f"Rank {rank} Peak: T={p:<6} | Amp={a}")
        print(f"Variance Captured:    {fft_summary['signal_variance_captured']}%")

    # --- 3. TREND & STATIONARITY ANALYSIS (ADF) -----------------------------------------------------------------------
    window_size = ANALYSIS_KWARGS.get("rolling_window", 24)

    # Pass the unified processed_data (handles both Series and Tuples natively)
    trend_data, adf_summary = analyze_stationarity_trend(
        processed_data, window_size=window_size, **ANALYSIS_KWARGS
    )

    # Extract tuple safely to check for emptiness
    trend_tuple = (trend_data,) if isinstance(trend_data, pd.Series) else trend_data

    if not trend_tuple[0].empty:
        trend_fig = visualize_trend_analysis(
            raw_input=processed_data,
            trend_input=trend_data,
            adf_summary=adf_summary,
            title=f"Trend & Stationarity: {meta['pollutant']} ({window_size}-Period Rolling)",
            **ANALYSIS_KWARGS
        )
        if trend_fig:
            plt.show()
            plt.close(trend_fig)

    if adf_summary:
        print("\n--- STATIONARITY REPORT ---")

        try:
            # 1. Ensure the summary is actually a dictionary before iterating
            if not isinstance(adf_summary, dict):
                raise TypeError("Provided adf_summary is not a dictionary object.")

            for series_key, stats in adf_summary.items():

                # 2. Ensure the inner items are dictionaries
                if not isinstance(stats, dict):
                    print(f"{series_key}: Failed -> Invalid data structure format.")
                    continue

                if "error" in stats:
                    print(f"{series_key}: Failed -> {stats['error']}")
                else:
                    # 3. Safe Extraction: Use .get() to prevent KeyErrors if data is missing
                    p_val = stats.get('p_value')
                    is_stat = stats.get('is_stationary', 'Unknown')

                    if p_val is not None:
                        print(f"{series_key}: p-value = {p_val:.4f} | Stationary = {is_stat}")
                    else:
                        print(f"{series_key}: Failed -> Missing p-value in statistics.")

        except (AttributeError, TypeError) as e:
            print(f"CRITICAL ERROR: Invalid 'adf_summary' format -> {e}")
        except Exception as e:
            print(f"CRITICAL ERROR: An unexpected failure occurred while parsing the report -> {e}")

    # --- 4. STL DECOMPOSITION (STANDARD VS ROBUST) --------------------------------------------------------------------
    seasonal_window = ANALYSIS_KWARGS.get("stl_seasonal", 13)
    period_window = ANALYSIS_KWARGS.get("stl_periodic", 12)
    logging.info(f"ORCHESTRATOR: Executing Dual STL Decomposition (Window={seasonal_window}).")

    primary_df, primary_summary = analyze_stl_decomposition(
        target_series, period=period_window, seasonal=seasonal_window, robust=False, **ANALYSIS_KWARGS
    )
    robust_df, robust_summary = analyze_stl_decomposition(
        target_series, seasonal=seasonal_window, robust=True, **ANALYSIS_KWARGS
    )

    if not primary_df.empty:
        valid_secondary = robust_df if not robust_df.empty else None
        secondary_lbl = "Robust Fit" if valid_secondary is not None else None

        stl_fig = visualize_stl_decomposition(
            primary_decomp=primary_df,
            primary_summary=primary_summary,
            title=f"STL Comparison: Standard vs Robust ({meta['pollutant']})",
            secondary_decomp=valid_secondary,
            secondary_label=secondary_lbl,
            **ANALYSIS_KWARGS
        )
        if stl_fig:
            plt.show()
            plt.close(stl_fig)

    if primary_summary:
        print("\n--- STL DECOMPOSITION INDICATORS ---")
        print(f"Seasonal Window:        {primary_summary.get('seasonal_window')}")
        print(f"Standard Trend Str:     {primary_summary.get('trend_strength', 0):.4f}")
        print(f"Standard Seasonal Str:  {primary_summary.get('seasonal_strength', 0):.4f}")
        if robust_summary and not robust_df.empty:
            print(f"Robust Trend Str:       {robust_summary.get('trend_strength', 0):.4f}")
            print(f"Robust Seasonal Str:    {robust_summary.get('seasonal_strength', 0):.4f}")

    # --- 5. MEMORY & AUTOCORRELATION ANALYSIS -------------------------------------------------------------------------
    logging.info("ORCHESTRATOR: Executing Memory Analysis (ACF/PACF).")

    memory_kwargs = ANALYSIS_KWARGS.copy()
    memory_kwargs.setdefault("nlags", 48)
    memory_kwargs.setdefault("alpha", 0.05)

    acf_vals, pacf_vals, memory_summary = analyze_memory(target_series, **memory_kwargs)

    if acf_vals.size > 0:
        plot_stats = {
            "extended_results": memory_summary.get("acf_extended_results"),
            "pacf_extended_results": memory_summary.get("pacf_extended_results")
        }
        acf_fig = visualize_correlogram(
            acf_values=acf_vals,
            pacf_values=pacf_vals,
            additional_stats=plot_stats,
            title=f"Autocorrelation (ACF): {meta['pollutant']} Memory Profile",
            **memory_kwargs
        )
        if acf_fig:
            plt.show()
            plt.close(acf_fig)

    if memory_summary:
        print("\n--- MEMORY & AUTOCORRELATION INDICATORS ---")
        print(f"Max Lag Evaluated:      {memory_summary.get('max_lag_evaluated')}")
        print(f"Significance Threshold: alpha = {memory_summary.get('alpha_threshold')}")

        acf_sig = memory_summary.get('significant_acf_lags', [])
        pacf_sig = memory_summary.get('significant_pacf_lags', [])

        print(f"Significant ACF Lags:   {acf_sig[:15]}{'...' if len(acf_sig) > 15 else ''}")
        print(f"Significant PACF Lags:  {pacf_sig[:15]}{'...' if len(pacf_sig) > 15 else ''}")

    print(f"{'#'*40}\n")


if __name__ == "__main__":
    pollution_data_plotting_main()
