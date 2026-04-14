"""
AirTS-Forecast Project
Section 1.2: Pollution Data Gathering, Exploration, and Spectral Analysis
File: pollution_data_plotting.py
Author: Tiago TOLOCZKO ROSS

Description:
    Core module for ingesting, parsing, cleaning, and performing diagnostic
    visualizations on pollutant concentration datasets. Implements
    vectorized temporal resampling, outlier remediation, and Fourier Transform
    analysis for cyclical trend detection.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- CONSTANTS & CONFIGURATION ---
DATA_DIRECTORY = Path(r"C:\Users\Tiago\Documents - PC\UTTOP\Enseignements\M1.2\Projet AirTS - Forecast\GEODAIR_POLLUTION_DATA")
TARGET_FILENAME = "PM10_20210321_20260321.csv"
PERIODICITY = "monthly"  # Options: 'hourly', 'daily', 'weekly', 'monthly', 'annually'

ANALYSIS_KWARGS = {
    "sep": ";",
    "encoding": "utf-8",
    "apply_cleaning": False,
    "export_diagnostics": True,     # Toggles the pre-cleaning outlier plot
    "outlier_method": "iqr",
    "iqr_multiplier": 2.0,
    "figsize": (12, 6),
    "color": "teal",
    "marker": "o",
    "xlabel": "Observation Timestamp",
    "ylabel": "Concentration (µg/m³)",

    "animate": True,                                  # Toggles animation routing
    "animation_filepath": "NO2_Weekly_Trend.gif",     # Output destination
    "fps": 12                                         # Rendering speed
}

# --- METADATA & INGESTION ---

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
    """
    High-performance CSV ingestion with defensive path validation.
    Strictly filters kwargs to extract only formatting parameters, ignoring
    extraneous arguments cascaded from the master orchestrator.

    Args:
        file_path (Path): The absolute or relative path to the target CSV.
        **kwargs: Arbitrary keyword arguments. Only 'sep' and 'encoding' are utilized.

    Returns:
        Optional[pd.DataFrame]: A validated DataFrame or None if the operation fails.
    """
    if not file_path.is_file():
        logging.error(f"IO_ERROR: Path does not exist: {file_path}")
        return None

    # Explicitly extract only the necessary ingestion parameters.
    # Defaulting to standard CSV formats if keys are missing from the waterfall.
    file_sep = kwargs.get("sep", ";")
    file_encoding = kwargs.get("encoding", "utf-8")

    try:
        df = pd.read_csv(
            file_path,
            sep=file_sep,
            encoding=file_encoding
        )
        logging.info(f"INGESTION_SUCCESS: {file_path.name} | Shape: {df.shape} | Sep: '{file_sep}' | Enc: {file_encoding}")
        return df

    except Exception as e:
        logging.error(f"PARSING_FATAL: Failed to process {file_path.name}. Details: {e}")
        return None

# --- STATISTICAL CLEANING ---

def flag_pollution_outliers(data_series: pd.Series, method: str = "iqr", **kwargs: Any) -> pd.Series:
    """Identifies statistical outliers in a time-series dataset."""
    method = method.lower()
    if data_series.empty or data_series.dropna().empty:
        return pd.Series(False, index=data_series.index)

    if method == "zscore":
        threshold = kwargs.get("z_threshold", 3.0)
        std = data_series.std()
        if std == 0:
            return pd.Series(False, index=data_series.index)
        z_scores = np.abs((data_series - data_series.mean()) / std)
        outliers = z_scores > threshold

    elif method == "iqr":
        multiplier = kwargs.get("iqr_multiplier", 1.5)
        q1 = data_series.quantile(0.25)
        q3 = data_series.quantile(0.75)
        iqr = q3 - q1
        outliers = (data_series < (q1 - multiplier * iqr)) | (data_series > (q3 + multiplier * iqr))
    else:
        logging.error(f"OUTLIER_ERROR: Unrecognized method '{method}'.")
        return pd.Series(False, index=data_series.index)

    logging.info(f"DATA_CLEANING: '{method}' method flagged {outliers.sum()} outliers.")
    return outliers

# --- PIPELINE ROUTING & ANALYSIS ---

def analyze_pollution_metrics(
        data_path: Path,
        periodicity: str = "daily",
        apply_cleaning: bool = True,
        **kwargs: Any
) -> Tuple[pd.DataFrame, pd.Series]:
    """Performs spatial validation, outlier remediation, and dynamic resampling."""
    pd_kwargs = kwargs.copy()
    viz_keys = ['figsize', 'color', 'marker', 's', 'xlabel', 'ylabel', 'title', 'export_diagnostics']
    for key in viz_keys:
        pd_kwargs.pop(key, None)

    df = ingest_pollution_data(data_path, **pd_kwargs)
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    lats, lons = df["Latitude"].unique(), df["Longitude"].unique()
    logging.info(f"GEOSPATIAL_AUDIT: Lats {lats} | Lons {lons}")

    df["Date de début"] = pd.to_datetime(df["Date de début"])
    df = df.set_index("Date de début").sort_index()

    # Outlier Remediation Strategy
    if apply_cleaning:
        outlier_method = kwargs.get("outlier_method", "iqr")
        outlier_mask = flag_pollution_outliers(df["valeur"], method=outlier_method, **kwargs)

        if kwargs.get("export_diagnostics", False):
            diag_fig = visualize_outlier_diagnostics(
                raw_series=df["valeur"],
                outlier_mask=outlier_mask,
                title=f"Diagnostic: {outlier_method.upper()} Outlier Detection",
                **kwargs
            )
            plt.show()
            plt.close(diag_fig)

        # Neutralize corrupted data
        df.loc[outlier_mask, "valeur"] = np.nan

    # Dynamic Resampling
    freq_map = {"hourly": "h", "daily": "D", "weekly": "W", "monthly": "ME", "annually": "YE"}
    resample_rule = freq_map.get(periodicity.lower(), periodicity)

    try:
        resampled_series = df["valeur"].resample(resample_rule).mean()
        logging.info(f"TEMPORAL_ANALYSIS: Resampled to '{periodicity}' ({len(resampled_series)} valid points).")
    except ValueError as e:
        logging.error(f"RESAMPLING_ERROR: {e}")
        resampled_series = pd.Series(dtype=float)

    return df, resampled_series

def extract_fourier_indicators(data_series: pd.Series, periodicity: str, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Applies Fast Fourier Transform to detect cyclical trends."""
    if data_series.dropna().empty:
        return np.array([]), np.array([]), {}

    signal = data_series.interpolate(method='linear').bfill().ffill()
    signal_detrended = signal - signal.mean()
    n_samples = len(signal_detrended)

    if n_samples < 2:
        return np.array([]), np.array([]), {}

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

# --- VISUALIZATION ENGINES ---

def visualize_pollution_scatter(
        data_series: pd.Series,
        title: str,
        **kwargs: Any
) -> Optional[plt.Figure]:
    """
    Standardized scatter/line plot generation with optional timeseries animation.

    Args:
        data_series (pd.Series): The resampled time-series data to plot.
        title (str): The plot title.
        **kwargs: Matplotlib styling and animation arguments cascaded from the Orchestrator.
                  Triggers animation if 'animate' == True.

    Returns:
        Optional[plt.Figure]: The Matplotlib figure object for orchestrator management.
    """
    valid_data = data_series.dropna()

    if valid_data.empty:
        logging.warning("VISUALIZATION_WARNING: The provided series contains no valid data points.")
        return None

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))

    ax.set_title(title)
    ax.set_xlabel(kwargs.get("xlabel", "Date"))
    ax.set_ylabel(kwargs.get("ylabel", "Concentration"))
    ax.grid(True, linestyle="--", alpha=0.6)

    # 1. Routing Objective: Static Export vs. Animation
    is_animated = kwargs.get("animate", False)

    if not is_animated:
        # --- STANDARD STATIC RENDERING ---
        ax.plot(
            valid_data.index,
            valid_data.values,
            color=kwargs.get("color", "teal"),
            marker=kwargs.get("marker", "o")
        )
        return fig

    else:
        # --- DYNAMIC TRACE ANIMATION RENDERING ---
        logging.info("VISUALIZATION: Initializing timeseries animation sequence...")

        # Pre-calculate global axis bounds to ensure strict visual stability
        y_min, y_max = valid_data.min(), valid_data.max()
        y_padding = (y_max - y_min) * 0.1

        ax.set_xlim(valid_data.index.min(), valid_data.index.max())
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Initialize empty graphical elements for the trace and the leading edge
        line, = ax.plot([], [], color=kwargs.get("color", "teal"), linewidth=2, label="Trend")
        point, = ax.plot([], [], color="red", marker="o", markersize=6, label="Current Observation")
        ax.legend(loc="upper right")

        def init() -> tuple:
            """Initializes the empty frame."""
            line.set_data([], [])
            point.set_data([], [])
            return line, point

        def update(frame_idx: int) -> tuple:
            """Updates the graphical elements for each chronological step."""
            if frame_idx == 0:
                return line, point

            # Slice the dataset chronologically up to the current frame
            current_slice = valid_data.iloc[:frame_idx]

            line.set_data(current_slice.index, current_slice.values)
            point.set_data([current_slice.index[-1]], [current_slice.values[-1]])

            return line, point

        # Generate the animation object
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(valid_data) + 1,
            init_func=init,
            blit=False
        )

        # Export the animation immediately to prevent RAM bloat
        export_path = kwargs.get("animation_filepath", "pollution_timeseries.gif")
        fps = kwargs.get("fps", 10)

        logging.info(f"ANIMATION_EXPORT: Rendering to {export_path} at {fps} FPS. This may take a moment.")

        try:
            # Pillow writer is natively supported for GIF export in Matplotlib
            anim.save(export_path, writer='pillow', fps=fps)
            logging.info("ANIMATION_EXPORT: Render complete.")
        except Exception as e:
            logging.error(f"ANIMATION_ERROR: Failed to save animation. Details: {e}")

        # Return the figure object so the Orchestrator can execute plt.close(fig)
        return fig


def visualize_outlier_diagnostics(raw_series: pd.Series, outlier_mask: pd.Series, title: str, **kwargs: Any) -> plt.Figure:
    """Generates a diagnostic plot overlaying raw data with flagged outliers."""
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))
    ax.plot(raw_series.index, raw_series.values, color="gray", alpha=0.5, label="Raw Data", zorder=1)

    outlier_data = raw_series[outlier_mask]
    if not outlier_data.empty:
        ax.scatter(outlier_data.index, outlier_data.values, color="red", marker="x",
                   s=kwargs.get("outlier_size", 50), label=f"Outliers ({len(outlier_data)})", zorder=2)

    ax.set_title(title)
    ax.set_xlabel(kwargs.get("xlabel", "Date"))
    ax.set_ylabel(kwargs.get("ylabel", "Concentration"))
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig

def visualize_fourier_spectrum(periods: np.ndarray, amplitudes: np.ndarray, summary: Dict[str, Any], **kwargs: Any) -> plt.Figure:
    """Generates a periodogram to visualize dominant cyclical frequencies."""
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))
    if len(periods) == 0: return fig

    ax.plot(periods, amplitudes, color=kwargs.get("fft_color", "purple"), linewidth=1.5)

    top_periods, top_amps = summary.get("dominant_periods_raw", []), summary.get("dominant_amplitudes", [])
    ax.scatter(top_periods, top_amps, color="red", zorder=3, label="Dominant Cycles")

    for p, a in zip(top_periods, top_amps):
        ax.annotate(f"T={p}\nAmp={a}", (p, a), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    ax.set_xscale("log")
    ax.set_title(f"Spectral Analysis Periodogram (Base: {summary.get('periodicity_base', 'Unknown').capitalize()})")
    ax.set_xlabel(f"Period Length ({summary.get('periodicity_base', 'units')})")
    ax.set_ylabel("Amplitude")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    return fig

# --- MASTER ORCHESTRATOR ---

def main() -> None:
    """Main execution routine orchestrating extraction, cleaning, and analysis."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 1. Initialization
    meta = parse_pollution_metadata(TARGET_FILENAME)
    if not meta: return
    full_path = DATA_DIRECTORY / TARGET_FILENAME

    # 2. Ingestion, Outlier Diagnostics (Inline), and Resampling
    df, clean_series = analyze_pollution_metrics(
        full_path, periodicity=PERIODICITY, **ANALYSIS_KWARGS
    )
    if clean_series.empty:
        logging.error("Process aborted: Empty dataset.")
        return

    # 3. Clean Visualization
    clean_fig = visualize_pollution_scatter(
        clean_series, title=f"Cleaned {PERIODICITY.capitalize()} Trend: {meta['pollutant']}", **ANALYSIS_KWARGS
    )
    plt.show()
    plt.close(clean_fig)

    # 4. Fourier Transform Analysis
    periods, amps, fft_summary = extract_fourier_indicators(
        clean_series, periodicity=PERIODICITY, top_k=5
    )
    if fft_summary:
        fft_fig = visualize_fourier_spectrum(periods, amps, fft_summary, **ANALYSIS_KWARGS)
        plt.show()
        plt.close(fft_fig)

    # 5. Console Reporting
    print(f"\n{'#'*40}\nPOLLUTION & SPECTRAL ANALYSIS REPORT\n{'#'*40}")
    print(f"Target Pollutant:       {meta['pollutant']}")
    print(f"Temporal Bounds:        {meta['start_date'].date()} to {meta['end_date'].date()}")
    print(f"Raw Data Points:        {len(df)}")
    print(f"Cleaned Periods ({PERIODICITY[0].upper()}): {len(clean_series)}")
    print(f"Global Arithmetic Mean: {df['valeur'].mean():.4f}")

    if fft_summary:
        print("\n--- SPECTRAL TREND INDICATORS ---")
        print(f"Primary Target Cycle: {fft_summary['expected_cycle_label']}")
        for rank, (p, a) in enumerate(zip(fft_summary['dominant_periods_raw'], fft_summary['dominant_amplitudes']), 1):
            print(f"Rank {rank} Peak: T={p:<6} | Amp={a}")
        print(f"Variance Captured:    {fft_summary['signal_variance_captured']}%")
    print(f"{'#'*40}\n")

if __name__ == "__main__":
    main()