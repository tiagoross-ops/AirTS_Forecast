"""
AirTS-Forecast Project
Section 1.2: Pollution Data Gathering, Exploration, and Spectral Analysis
File: pollution_data_plotting.py
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

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as stt
from matplotlib.animation import FuncAnimation
from statsmodels.tsa.seasonal import STL

# --- CONSTANTS & CONFIGURATION ---

DATA_DIRECTORY = Path(r"C:\Users\Tiago\Documents - PC\UTTOP\Enseignements\M1.2\Projet AirTS - Forecast\GEODAIR_POLLUTION_DATA")
TARGET_FILENAME = "PM10_20210321_20260321.csv"
PERIODICITY = "monthly"  # Options: 'hourly', 'daily', 'weekly', 'monthly', 'annually'

ANALYSIS_KWARGS = {
    # File settings
    "sep": ";",
    "encoding": "utf-8",

    # Cleaning & Math settings
    "apply_cleaning": False,        # Data cleaning methods -> experimental
    "outlier_method": "iqr",
    "iqr_multiplier": 2.0,

    "fft_log_scale_y": False,
    "rolling_window": 3,            # e.g. 24 periods = 1 day rolling trend for 'hourly' data
    "export_diagnostics": False,    # Toggles the pre-cleaning outlier plot
    "stl_seasonal": 13,             # Should cover an expected cycle (must be odd integer)
    "stl_robust": False,            # Enhances resistance to remaining outliers
    "nlags": 12,                    # ACF/PACF memory lookback limit
    "alpha": 0.05,                  # Significance threshold for ACF/PACF intervals
    "fft": True,

    # Figure styling settings
    "figsize": (6, 6),
    "color": "teal",
    "marker": "o",
    "xlabel": "Observation Timestamp",
    "ylabel": "Concentration (µg/m³)",

    # Data animation settings
    "animate": False,
    "animation_filepath": "NO2_Trend_Animation.gif",
    "fps": 24
}

# --- 1. METADATA & INGESTION ---

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

# --- 2. MATHEMATICAL & PIPELINE ROUTING ---

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

def analyze_pollution_metrics(
        data_path: Path,
        periodicity: str = "daily",
        **kwargs: Any
) -> Tuple[pd.DataFrame, pd.Series]:
    """Performs spatial validation, outlier remediation, and dynamic resampling."""
    pd_kwargs = kwargs.copy()
    viz_keys = ['figsize', 'color', 'marker', 's', 'xlabel', 'ylabel', 'title', 'export_diagnostics', 'animate']
    for key in viz_keys:
        pd_kwargs.pop(key, None)

    df = ingest_pollution_data(data_path, **pd_kwargs)
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    lats, lons = df["Latitude"].unique(), df["Longitude"].unique()
    logging.info(f"GEOSPATIAL_AUDIT: Lats {lats} | Lons {lons}")

    df["Date de début"] = pd.to_datetime(df["Date de début"])
    df = df.set_index("Date de début").sort_index()

    if kwargs.get("apply_cleaning", True):
        outlier_mask = flag_pollution_outliers(df["valeur"], **kwargs)

        if kwargs.get("export_diagnostics", False):
            diag_fig = visualize_outlier_diagnostics(df["valeur"], outlier_mask, "Outlier Diagnostics", **kwargs)
            plt.show()
            plt.close(diag_fig)

        df.loc[outlier_mask, "valeur"] = np.nan

    freq_map = {"hourly": "h", "daily": "D", "weekly": "W", "monthly": "ME", "annually": "YE"}
    resample_rule = freq_map.get(periodicity.lower(), periodicity)

    try:
        resampled_series = df["valeur"].resample(resample_rule).mean()
        logging.info(f"TEMPORAL_ANALYSIS: Resampled to '{periodicity}' ({len(resampled_series)} valid points).")
    except ValueError as e:
        logging.error(f"RESAMPLING_ERROR: {e}")
        resampled_series = pd.Series(dtype=float)

    return df, resampled_series

def fourier_analysis(data_series: pd.Series, periodicity: str, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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

def analyze_stationarity_trend(data_series: pd.Series, window_size: int, **kwargs: Any) -> Tuple[pd.Series, Dict[str, Any]]:
    """Calculates a rolling average and performs an Augmented Dickey-Fuller test."""
    if data_series.dropna().empty or len(data_series) <= window_size:
        return pd.Series(dtype=float), {}

    moving_mean = data_series.rolling(window=window_size, min_periods=1).mean()
    clean_mean = moving_mean.dropna()

    if len(clean_mean) < 3: return moving_mean, {}

    try:
        adf_tuple = stt.adfuller(clean_mean, autolag=kwargs.get("autolag", "AIC"))
        keys = ['adf_statistic', 'p_value', 'used_lag', 'n_observations', 'critical_values', 'ic_best']
        summary = dict(zip(keys, adf_tuple))
        logging.info(f"ADF_TEST_COMPLETE: p-value={summary['p_value']:.4f}")
        return moving_mean, summary
    except Exception as e:
        logging.error(f"ADF_TEST_FAILURE: {e}")
        return moving_mean, {}

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


# --- 3. VISUALIZATION ENGINES ---

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

def visualize_pollution_scatter(data_series: pd.Series, title: str, **kwargs: Any) -> Union[plt.Figure, Tuple[plt.Figure, FuncAnimation], None]:
    valid_data = data_series.dropna()
    if valid_data.empty: return None

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
    ax.set_title(title)
    ax.set_xlabel(kwargs.get("xlabel", "Date"))
    ax.set_ylabel(kwargs.get("ylabel", "Concentration"))
    ax.grid(True, linestyle="--", alpha=0.6)

    if not kwargs.get("animate", False):
        ax.plot(valid_data.index, valid_data.values, color=kwargs.get("color", "teal"), marker=kwargs.get("marker", "o"))
        return fig
    else:
        y_min, y_max = valid_data.min(), valid_data.max()
        y_padding = (y_max - y_min) * 0.1
        ax.set_xlim(valid_data.index.min(), valid_data.index.max())
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        line, = ax.plot([], [], color=kwargs.get("color", "teal"), linewidth=2, label="Trend")
        point, = ax.plot([], [], color="red", marker="o", markersize=6, label="Current")
        ax.legend(loc="upper right")

        def init() -> tuple:
            line.set_data([], [])
            point.set_data([], [])
            return line, point

        def update(frame_idx: int) -> tuple:
            if frame_idx == 0: return line, point
            current_slice = valid_data.iloc[:frame_idx]
            line.set_data(current_slice.index, current_slice.values)
            point.set_data([current_slice.index[-1]], [current_slice.values[-1]])
            return line, point

        anim = animation.FuncAnimation(fig, update, frames=len(valid_data) + 1, init_func=init, blit=False)
        try:
            anim.save(kwargs.get("animation_filepath", "anim.gif"), writer='pillow', fps=kwargs.get("fps", 10))
        except Exception as e:
            logging.error(f"ANIMATION_ERROR: {e}")

        return fig, anim

def visualize_fourier_spectrum(periods: np.ndarray, amplitudes: np.ndarray, summary: Dict[str, Any], **kwargs: Any) -> plt.Figure:
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))
    if len(periods) == 0: return fig

    ax.plot(periods, amplitudes, color=kwargs.get("fft_color", "purple"), linewidth=1.5)

    top_p, top_a = summary.get("dominant_periods_raw", []), summary.get("dominant_amplitudes", [])
    ax.scatter(top_p, top_a, color="red", zorder=3, label="Dominant Cycles")

    for p, a in zip(top_p, top_a):
        ax.annotate(f"T={p}\nAmp={a}", (p, a), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    # X-Axis is kept logarithmic to compress high frequencies
    ax.set_xscale("log")

    is_log_y = kwargs.get("fft_log_scale_y", False)
    if is_log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Amplitude (Log Scale)")
    else:
        ax.set_ylabel("Amplitude")

    ax.set_title(f"Spectral Analysis Periodogram ({summary.get('periodicity_base', 'Unknown').capitalize()})")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    return fig

def visualize_trend_analysis(raw_series: pd.Series, trend_series: pd.Series, adf_summary: Dict[str, Any], title: str, **kwargs: Any) -> Optional[plt.Figure]:
    if raw_series.dropna().empty or trend_series.dropna().empty: return None

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))

    ax.plot(raw_series.index, raw_series.values, color=kwargs.get("base_color", "gray"), alpha=0.4, linewidth=1, label="Raw Data", zorder=1)
    ax.plot(trend_series.index, trend_series.values, color=kwargs.get("trend_color", "darkorange"), linewidth=2.5, label="Rolling Trend", zorder=2)

    if adf_summary:
        p_val = adf_summary.get('p_value', 1.0)
        stat_text = f"Augmented Dickey-Fuller\np-value: {p_val:.4f}\nStationary: {p_val < 0.05}"
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
        ax.text(0.02, 0.95, stat_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props, zorder=3)

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    return fig

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
    axes[1].plot(primary_decomp.index, primary_decomp['trend'], color=base_color, linewidth=2, label='Primary Trend')
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
                     color=overlay_color, linestyle=line_style, alpha=alpha, label=secondary_label)
        axes[2].plot(secondary_decomp.index, secondary_decomp['seasonal'],
                     color=overlay_color, linestyle=line_style, alpha=alpha, label=secondary_label)
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

def visualize_correlogram(
        acf_values: np.ndarray,
        additional_stats: Dict[str, Any],
        title: str = "Autocorrelation (ACF) Correlogram",
        **kwargs: Any
) -> Optional[plt.Figure]:
    if acf_values.size == 0:
        logging.warning("VISUALIZATION_WARNING: ACF array is empty. Aborting correlogram render.")
        return None

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))
    lags = np.arange(len(acf_values))

    markerline, stemlines, baseline = ax.stem(lags, acf_values, basefmt="k-")

    plt.setp(markerline, color=kwargs.get("color", "teal"), marker=kwargs.get("marker", "o"), markersize=5)
    plt.setp(stemlines, color=kwargs.get("color", "teal"), linewidth=1.5, alpha=0.7)
    plt.setp(baseline, color="black", linewidth=1.2)

    extended_results = additional_stats.get("extended_results")
    if extended_results and len(extended_results) > 0:
        conf_int = extended_results[0]
        centered_conf_int = conf_int - acf_values[:, None]
        ax.fill_between(lags, centered_conf_int[:, 0], centered_conf_int[:, 1], color='gray', alpha=0.25, label="95% Confidence Interval")
        ax.legend(loc="upper right")

    ax.set_title(title)
    ax.set_xlabel("Lag (Periods)")
    ax.set_ylabel("Autocorrelation")
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    return fig

# --- 4. MASTER ORCHESTRATOR ---

def main() -> None:
    """Main execution routine orchestrating extraction, cleaning, and analysis."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # --- 1. INITIALIZATION ---
    meta = parse_pollution_metadata(TARGET_FILENAME)
    if not meta: return
    full_path = DATA_DIRECTORY / TARGET_FILENAME

    # --- 2. INGESTION, DIAGNOSTICS & RESAMPLING ---
    df, clean_series = analyze_pollution_metrics(
        full_path, periodicity=PERIODICITY, **ANALYSIS_KWARGS
    )
    if clean_series.empty:
        logging.error("Process aborted: Empty dataset.")
        return

    # --- 3. CLEAN VISUALIZATION (STATIC VS ANIMATED) ---
    title_str = f"Cleaned {PERIODICITY.capitalize()} Trend: {meta['pollutant']}"
    if ANALYSIS_KWARGS.get("animate", False):
        clean_fig, anim = visualize_pollution_scatter(clean_series, title=title_str, **ANALYSIS_KWARGS)
        logging.info("ORCHESTRATOR: Rendering animation. Close window to proceed.")
        plt.show()
        plt.close(clean_fig)
    else:
        clean_fig = visualize_pollution_scatter(clean_series, title=title_str, **ANALYSIS_KWARGS)
        plt.show()
        plt.close(clean_fig)


    # --- 4. FOURIER TRANSFORM ANALYSIS ---
    periods, amps, fft_summary = fourier_analysis(
        clean_series, periodicity=PERIODICITY, top_k=8
    )
    if fft_summary:
        fft_fig = visualize_fourier_spectrum(periods, amps, fft_summary, **ANALYSIS_KWARGS)
        plt.show()
        plt.close(fft_fig)


    # --- 5. TREND & STATIONARITY ANALYSIS (ADF) ---
    window_size = ANALYSIS_KWARGS.get("rolling_window", 24)
    trend_series, adf_summary = analyze_stationarity_trend(
        clean_series, window_size=window_size, **ANALYSIS_KWARGS
    )

    if not trend_series.empty:
        trend_fig = visualize_trend_analysis(
            clean_series, trend_series, adf_summary,
            title=f"Trend & Stationarity: {meta['pollutant']} ({window_size}-Period Rolling)",
            **ANALYSIS_KWARGS
        )
        plt.show()
        plt.close(trend_fig)


    # --- 6. STL DECOMPOSITION (STANDARD VS ROBUST) ---
    seasonal_window = ANALYSIS_KWARGS.get("stl_seasonal", 13)
    logging.info(f"ORCHESTRATOR: Executing Dual STL Decomposition (Window={seasonal_window}).")

    primary_df, primary_summary = analyze_stl_decomposition(
        clean_series, seasonal=seasonal_window, robust=False, **ANALYSIS_KWARGS
    )
    robust_df, robust_summary = analyze_stl_decomposition(
        clean_series, seasonal=seasonal_window, robust=True, **ANALYSIS_KWARGS
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
        plt.show()
        plt.close(stl_fig)


    # --- 7. MEMORY & AUTOCORRELATION ANALYSIS ---
    logging.info("ORCHESTRATOR: Executing Memory Analysis (ACF/PACF).")

    acf_vals, pacf_vals, memory_summary = analyze_memory(clean_series, **ANALYSIS_KWARGS)

    if acf_vals.size > 0:
        plot_stats = {"extended_results": memory_summary.get("acf_extended_results")}

        acf_fig = visualize_correlogram(
            acf_values=acf_vals,
            additional_stats=plot_stats,
            title=f"Autocorrelation (ACF): {meta['pollutant']} Memory Profile",
            **ANALYSIS_KWARGS
        )
        if acf_fig:
            plt.show()
            plt.close(acf_fig)

    # --- 8. CONSOLE REPORTING ---
    print(f"\n{'#'*40}\nPOLLUTION & SPECTRAL ANALYSIS REPORT\n{'#'*40}")
    print(f"Target Pollutant:       {meta['pollutant']}")
    print(f"Temporal Bounds:        {meta['start_date'].date()} to {meta['end_date'].date()}")
    print(f"Raw Data Points:        {len(df)}")
    print(f"Cleaned Periods ({PERIODICITY[0].upper()}): {len(clean_series)}")
    print(f"Global Arithmetic Mean: {df['valeur'].mean():.4f}")

    if adf_summary:
        print("\n--- STATIONARITY INDICATORS ---")
        try:
            print(f"ADF Statistic:          {adf_summary.get('adf_statistic'):.4f}")
            print(f"p-value:                {adf_summary.get('p_value'):.6f}")
            print(f"Stationary (a=0.05):    {adf_summary.get('p_value') < 0.05}")
        except TypeError:
            print("ADF Statistic:          Insufficient Data for calculation")

    if primary_summary:
        print("\n--- STL DECOMPOSITION INDICATORS ---")
        print(f"Seasonal Window:        {primary_summary.get('seasonal_window')}")
        print(f"Standard Trend Str:     {primary_summary.get('trend_strength', 0):.4f}")
        print(f"Standard Seasonal Str:  {primary_summary.get('seasonal_strength', 0):.4f}")
        if robust_summary and not robust_df.empty:
            print(f"Robust Trend Str:       {robust_summary.get('trend_strength', 0):.4f}")
            print(f"Robust Seasonal Str:    {robust_summary.get('seasonal_strength', 0):.4f}")

    if fft_summary:
        print("\n--- SPECTRAL TREND INDICATORS ---")
        print(f"Primary Target Cycle: {fft_summary['expected_cycle_label']}")
        for rank, (p, a) in enumerate(zip(fft_summary['dominant_periods_raw'], fft_summary['dominant_amplitudes']), 1):
            print(f"Rank {rank} Peak: T={p:<6} | Amp={a}")
        print(f"Variance Captured:    {fft_summary['signal_variance_captured']}%")

    if memory_summary:
        print("\n--- MEMORY & AUTOCORRELATION INDICATORS ---")
        print(f"Max Lag Evaluated:      {memory_summary.get('max_lag_evaluated')}")
        print(f"Significance Threshold: alpha = {memory_summary.get('alpha_threshold')}")

        acf_sig = memory_summary.get('significant_acf_lags', [])
        pacf_sig = memory_summary.get('significant_pacf_lags', [])

        print(f"Significant ACF Lags:   {acf_sig[:15]}{'...' if len(acf_sig) > 15 else ''}")
        print(f"Significant PACF Lags:  {pacf_sig[:15]}{'...' if len(pacf_sig) > 15 else ''}")

        acf_fft = np.fft.rfft(acf_sig)
        pacf_fft = np.fft.rfft(pacf_sig)

        # 2. Calculate the Magnitude (Amplitude) of the complex outputs
        acf_magnitude = np.abs(acf_fft)
        pacf_magnitude = np.abs(pacf_fft)

        # 3. Generate the frequency X-axis (assuming a sample spacing of 1 unit)
        n_samples = len(acf_sig)
        freqs_acf = np.fft.rfftfreq(len(acf_sig), d=1.0)
        freqs_pacf = np.fft.rfftfreq(len(pacf_sig), d=1.0)

        # 4. Visualization Setup (1 Row, 2 Columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Left Plot: Time Domain (Original Signals) ---
        ax1.plot(acf_sig, label='ACF', color='teal')
        ax1.plot(pacf_sig, label='PACF', color='darkorange', linestyle='--')
        ax1.set_title('Time/Lag Domain')
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Correlation')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        # --- Right Plot: Frequency Domain (FFT Magnitude) ---
        ax2.plot(freqs_acf, acf_magnitude, label='FFT of ACF (Power Spectrum)', color='teal')
        ax2.plot(freqs_pacf, pacf_magnitude, label='FFT of PACF', color='darkorange', linestyle='--')
        ax2.set_title('Frequency Domain (Spectral Density)')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Magnitude')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.show()


    print(f"{'#'*40}\n")


if __name__ == "__main__":
    main()