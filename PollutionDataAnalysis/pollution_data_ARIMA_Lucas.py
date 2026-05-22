"""
AirTS-Forecast Project
Section 2: Classical Time Series Models
File: pollution_data_ARIMA.py

Description:
    Classical forecasting workflow adapted to the project pollution CSVs.
    Sweeps a directory for pollutant files, builds a chronological split, and evaluates
    ARIMA, ARIMAX (w/ Fourier), and Holt-Winters models over a realistic forecast horizon.
"""

import logging
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pollution_data_exp_core import parse_pollution_metadata, pollution_data_resampling

# Globally ignore noisy statsmodels warnings regarding missing DateTime frequencies
warnings.filterwarnings("ignore")

# ======================================================================================================================
# GLOBAL CONFIGURATION
# ======================================================================================================================

DATA_DIRECTORY = Path(r"C:\Users\Tiago\Documents - PC\UTTOP\Enseignements\M1.2\Projet AirTS - Forecast\GEODAIR_POLLUTION_DATA")
OUTPUT_DIRECTORY = Path("outputs")
PERIODICITY = "daily"

ANALYSIS_KWARGS = {
    "sep": ";",
    "encoding": "utf-8",
    "apply_cleaning": False,
    "train_ratio": 0.8,
    "forecast_horizon_steps": None,

    # ARIMA
    "arima_order": (7, 1, 1),

    # ARIMAX (Fourier handles seasonality)
    "sarima_order": (1, 1, 1),
    "sarima_seasonal_order": (0, 0, 0, 0),

    # Fourier Cycles (Weekly & Annual)
    "fourier_periods": (7, 365.25),
    "fourier_order": 2,

    # Holt-Winters
    "holt_winters_trend": "add",
    "holt_winters_seasonal": "add",
    "holt_winters_seasonal_periods": 7,
}

MODEL_COLORS = {
    "ARIMA": "darkorange",
    "ARIMAX_Fourier": "royalblue",
    "Holt_Winters": "seagreen",
}

# ======================================================================================================================
# HELPERS & METRICS
# ======================================================================================================================

def train_test_split_ts(data: pd.Series, train_ratio: float, horizon: int | None) -> Tuple[pd.Series, pd.Series]:
    clean_data = data.dropna()
    split_idx = int(len(clean_data) * train_ratio)

    if split_idx <= 0 or split_idx >= len(clean_data):
        raise ValueError("Invalid train/test split.")

    if horizon is not None:
        forecast_horizon = horizon
    else:
        forecast_horizon = len(data)

    train = clean_data.iloc[:split_idx]
    test = clean_data.iloc[split_idx : split_idx + forecast_horizon]

    return train, test


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned_pred, aligned_true = y_pred.align(y_true, join="inner")
    mask = aligned_true != 0
    if not mask.any(): return np.nan
    return float(np.mean(np.abs((aligned_true[mask] - aligned_pred[mask]) / aligned_true[mask])) * 100)


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series, model_name: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    aligned_pred, aligned_true = y_pred.align(y_true, join="inner")

    if aligned_true.empty:
        metrics = {"model": model_name, "rmse": np.nan, "mae": np.nan, "mape": np.nan, "test_points": 0}
    else:
        errors = aligned_pred - aligned_true
        metrics = {
            "model": model_name,
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae": float(np.mean(np.abs(errors))),
            "mape": mape(aligned_true, aligned_pred),
            "test_points": int(len(aligned_true)),
        }

    if extra: metrics.update(extra)

    print(f"{model_name:16s}: RMSE={metrics['rmse']:8.3f}  MAE={metrics['mae']:8.3f}  MAPE={metrics['mape']:8.2f}%")
    return metrics


def build_fourier_features(length: int, periods: Tuple[float, ...], order: int) -> pd.DataFrame:
    t = np.arange(length)
    features: Dict[str, np.ndarray] = {}
    for period in periods:
        for k in range(1, order + 1):
            features[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
            features[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(features)


def export_pickle(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)

# ======================================================================================================================
# MODELING
# ======================================================================================================================

def fit_forecast_arima(train_series: pd.Series, test_series: pd.Series, **kwargs: Any) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    order = kwargs.get("arima_order", (7, 1, 1))

    logging.info(f"ARIMA_MODEL: Fitting ARIMA{order}...")
    model = ARIMA(train_series, order=order, enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit()

    forecast_result = result.get_forecast(steps=len(test_series))
    forecast = forecast_result.predicted_mean
    forecast.index = test_series.index

    conf_int = forecast_result.conf_int(alpha=0.05)
    conf_int.index = test_series.index

    metrics = evaluate_forecast(test_series, forecast, "ARIMA", extra={"order": str(order)})
    return forecast, conf_int, metrics


def fit_forecast_arimax_fourier(train_series: pd.Series, test_series: pd.Series, **kwargs: Any) -> Tuple[pd.Series, Dict[str, Any]]:
    order = kwargs.get("sarima_order", (1, 1, 1))
    fourier_periods = kwargs.get("fourier_periods", (7, 365.25))
    fourier_order = kwargs.get("fourier_order", 2)

    features = build_fourier_features(len(train_series) + len(test_series), fourier_periods, fourier_order)
    features.index = pd.concat([train_series, test_series]).index
    fourier_train = features.loc[train_series.index]
    fourier_test = features.loc[test_series.index]

    logging.info(f"ARIMAX_MODEL: Fitting ARIMAX{order} with Fourier seasonality...")
    model = SARIMAX(
        train_series,
        exog=fourier_train,
        order=order,
        seasonal_order=(0,0,0,0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)

    forecast = result.get_forecast(steps=len(test_series), exog=fourier_test).predicted_mean
    forecast.index = test_series.index

    metrics = evaluate_forecast(test_series, forecast, "ARIMAX_Fourier", extra={"order": str(order)})
    return forecast, metrics


def fit_forecast_holt_winters(train_series: pd.Series, test_series: pd.Series, **kwargs: Any) -> Tuple[pd.Series, Dict[str, Any]]:
    trend = kwargs.get("holt_winters_trend", "add")
    seasonal = kwargs.get("holt_winters_seasonal", "add")
    seasonal_periods = kwargs.get("holt_winters_seasonal_periods", 7)

    logging.info(f"HOLT_WINTERS_MODEL: Fitting Exponential Smoothing...")
    model = ExponentialSmoothing(
        train_series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    result = model.fit(optimized=True)

    forecast = result.forecast(steps=len(test_series))
    forecast.index = test_series.index

    metrics = evaluate_forecast(test_series, forecast, "Holt_Winters")
    return forecast, metrics

# ======================================================================================================================
# OUTPUTS & VISUALIZATION
# ======================================================================================================================

def plot_classical_forecasts(
        train_series: pd.Series, test_series: pd.Series, forecasts: Dict[str, pd.Series],
        arima_conf_int: pd.DataFrame, title: str,
) -> Optional[plt.Figure]:
    if not forecasts: return None

    fig, ax = plt.subplots(figsize=(14, 6))

    # Protect against index out of bounds if train_series is somehow extremely small
    context_days = min(60, len(train_series))
    visible_start = train_series.index[-context_days] if context_days > 0 else test_series.index[0]
    visible_end = test_series.index[-1]

    full_series = pd.concat([train_series, test_series])
    visible_full = full_series.loc[visible_start:visible_end]
    visible_test = test_series.loc[visible_start:visible_end]

    ax.plot(visible_full.index, visible_full.values, color="lightgray", linewidth=1.5, label="Observed History")
    ax.plot(visible_test.index, visible_test.values, color="black", linewidth=2.0, label="True Future Data")

    for model_name, forecast in forecasts.items():
        ax.plot(
            forecast.index, forecast.values,
            color=MODEL_COLORS.get(model_name, "tab:red"),
            linewidth=2, linestyle="--", label=f"{model_name} Forecast"
        )

    if not arima_conf_int.empty:
        ax.fill_between(
            arima_conf_int.index, arima_conf_int.iloc[:, 0], arima_conf_int.iloc[:, 1],
            color=MODEL_COLORS["ARIMA"], alpha=0.15, label="ARIMA 95% CI"
        )

    ax.axvline(test_series.index[0], color="blue", linestyle=":", linewidth=2, label="Forecast Start")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Concentration Level")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    plt.show()
    fig.tight_layout()
    return fig

# ======================================================================================================================
# EXECUTION ORCHESTRATOR
# ======================================================================================================================

def process_pollutant_file(file_path: Path) -> None:
    meta = parse_pollution_metadata(file_path.name)
    if not meta: return
    pollutant = meta["pollutant"]

    _, processed_data = pollution_data_resampling(file_path, periodicity=PERIODICITY, **ANALYSIS_KWARGS)

    # -> CRITICAL FIX: Safe target series extraction regardless of whether it's a Series or DataFrame
    if isinstance(processed_data, pd.Series):
        target_series = processed_data.dropna()
    elif isinstance(processed_data, pd.DataFrame):
        if pollutant in processed_data.columns:
            target_series = processed_data[pollutant].dropna()
        else:
            # Fallback to the first column if exact name match fails
            target_series = processed_data.iloc[:, 0].dropna()
    else:
        # Fallback for unexpected tuple structures
        target_series = processed_data[0].dropna()

    if target_series.empty:
        logging.error(f"Process aborted: Empty dataset for {pollutant}.")
        return

    horizon = ANALYSIS_KWARGS["forecast_horizon_steps"]

    train_series, test_series = train_test_split_ts(
        target_series,
        train_ratio=ANALYSIS_KWARGS["train_ratio"],
        horizon=horizon
    )

    print("\n" + "=" * 70)
    print(f"CLASSICAL MODEL FITTING: {pollutant} ({file_path.name})")
    print("=" * 70)

    forecasts, metrics = {}, []
    arima_conf_int = pd.DataFrame()

    try:
        forecasts["ARIMA"], arima_conf_int, m = fit_forecast_arima(train_series, test_series, **ANALYSIS_KWARGS)
        metrics.append(m)
    except Exception as exc: logging.error(f"ARIMA Failed: {exc}")

    try:
        forecasts["ARIMAX_Fourier"], m = fit_forecast_arimax_fourier(train_series, test_series, **ANALYSIS_KWARGS)
        metrics.append(m)
    except Exception as exc: logging.error(f"ARIMAX Failed: {exc}")

    try:
        forecasts["Holt_Winters"], m = fit_forecast_holt_winters(train_series, test_series, **ANALYSIS_KWARGS)
        metrics.append(m)
    except Exception as exc: logging.error(f"Holt-Winters Failed: {exc}")

    if not forecasts: return

    metrics_df = pd.DataFrame(metrics)
    export_pickle(metrics_df, OUTPUT_DIRECTORY / f"{pollutant}_classical_metrics.pkl")

    # -> CRITICAL FIX: Bulletproof JSON serialization & update handling
    model_name_mapping = {"ARIMA": "ARIMA", "ARIMAX_Fourier": "SARIMA", "Holt_Winters": "Holt-Winters"}
    formatted_metrics = {model_name_mapping.get(row["model"], row["model"]): {"RMSE": row["rmse"], "MAE": row["mae"], "MAPE": row["mape"]} for row in metrics}

    consolidated_json = OUTPUT_DIRECTORY / "classical_model_results.json"
    all_results = {}

    if consolidated_json.exists():
        try:
            with open(consolidated_json, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Corrupted JSON detected at {consolidated_json}. Starting fresh.")
            all_results = {}

    all_results[pollutant] = formatted_metrics

    with open(consolidated_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    # Render Plot
    fig = plot_classical_forecasts(
        train_series, test_series, forecasts, arima_conf_int,
        title=f"Classical Forecasts vs Reality: {pollutant} ({ANALYSIS_KWARGS['forecast_horizon_steps']}-Day Horizon)"
    )
    if fig:
        fig.savefig(OUTPUT_DIRECTORY / f"{pollutant}_classical_forecast_plot.png", dpi=150, bbox_inches="tight")

    print("\n" + metrics_df[["model", "rmse", "mae", "mape"]].to_string(index=False))


def process_directory(directory_path: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    csv_files = list(directory_path.glob("*.csv"))
    if not csv_files: return logging.error(f"[!] No CSV files found in {directory_path}")

    print(f"\n[✓] Found {len(csv_files)} CSV files. Beginning processing loop...")
    for file_path in csv_files:
        process_pollutant_file(file_path)

if __name__ == "__main__":
    process_directory(DATA_DIRECTORY)

    plt.show()