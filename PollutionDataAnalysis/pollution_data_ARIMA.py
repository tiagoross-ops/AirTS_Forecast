"""
AirTS-Forecast Project
Section 2: Traditional Statistical Time Series Analysis
File: pollution_data_ARIMA.py
Author: Tiago TOLOCZKO ROSS

Description:
    Standalone module for ARIMA (AutoRegressive Integrated Moving Average)
    mathematical modeling, forecasting, and visualization.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Tuple, Dict, Optional

from statsmodels.tsa.arima.model import ARIMA
from pollution_data_exp_core import parse_pollution_metadata, pollution_data_resampling
# --- 1. GLOBAL CONFIGURATION ---

DATA_DIRECTORY = Path(r"C:\Users\Tiago\Documents - PC\UTTOP\Enseignements\M1.2\Projet AirTS - Forecast\GEODAIR_POLLUTION_DATA")
TARGET_FILENAME = "NO2_20210321_20260321.csv"
PERIODICITY = "hourly"

ANALYSIS_KWARGS = {
    "sep": ";",
    "encoding": "utf-8",
    "apply_cleaning": False,
    "differentiation_order": 0,
    "arima_order": (2, 1, 1),
    "arima_plot_window": 200
}

# --- 1. MATHEMATICAL MODELING ENGINE ---

def fit_forecast_arima(
        data_series: pd.Series,
        forecast_steps: int = 24,
        **kwargs: Any
) -> Tuple[Any, pd.Series, pd.DataFrame, Dict[str, Any]]:
    """
    Fits an ARIMA model to the temporal dataset and generates out-of-sample forecasts.

    Args:
        data_series (pd.Series): The base time-series data. This should be the raw
                                 resampled data, as ARIMA natively handles differentiation
                                 via the 'd' parameter in the order tuple.
        forecast_steps (int): Number of future periods to forecast.
        **kwargs: Configuration dictionary containing 'arima_order' (p, d, q) and 'alpha'.

    Returns:
        Tuple:
            - The fitted statsmodels ARIMA result object.
            - A pd.Series containing the forecasted values.
            - A pd.DataFrame containing the lower and upper confidence intervals.
            - A summary dictionary of model performance metrics (AIC, BIC).
    """
    valid_data = data_series.dropna()
    if valid_data.empty:
        logging.error("ARIMA_MODEL: Input series is empty. Aborting.")
        return None, pd.Series(dtype=float), pd.DataFrame(), {}

    order = kwargs.get("arima_order", (1, 0, 1))
    alpha = kwargs.get("alpha", 0.05)

    try:
        logging.info(f"ARIMA_MODEL: Fitting model with order (p, d, q) = {order}...")
        model = ARIMA(valid_data, order=order, enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit()

        forecast_result = fitted_model.get_forecast(steps=forecast_steps)
        forecast_series = forecast_result.predicted_mean
        conf_int_df = forecast_result.conf_int(alpha=alpha)

        summary = {
            "order": order,
            "aic": np.round(fitted_model.aic, 2),
            "bic": np.round(fitted_model.bic, 2),
            "sigma2": np.round(fitted_model.params.get('sigma2', 0), 4),
            "forecast_steps": forecast_steps
        }

        logging.info(f"ARIMA_MODEL: Fit successful. AIC: {summary['aic']} | BIC: {summary['bic']}")
        return fitted_model, forecast_series, conf_int_df, summary

    except Exception as e:
        logging.error(f"ARIMA_MODEL: CRITICAL FAILURE - {e}")
        return None, pd.Series(dtype=float), pd.DataFrame(), {}


# --- 2. VISUALIZATION ENGINE ---

def visualize_arima_forecast(
        historical_series: pd.Series,
        forecast_series: pd.Series,
        conf_int_df: pd.DataFrame,
        summary: Dict[str, Any],
        title: str = "ARIMA Forecast",
        **kwargs: Any
) -> Optional[plt.Figure]:
    """
    Renders the historical data alongside the ARIMA out-of-sample forecast
    and its statistical confidence bounds.
    """
    if forecast_series.empty:
        logging.warning("VISUALIZATION_WARNING: Forecast series is empty. Aborting render.")
        return None

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))

    hist_window = kwargs.get("arima_plot_window", len(historical_series))
    recent_history = historical_series.iloc[-hist_window:]

    ax.plot(
        recent_history.index,
        recent_history.values,
        color=kwargs.get("base_color", "gray"),
        label="Historical Data",
        linewidth=1.5
    )

    forecast_color = kwargs.get("forecast_color", "darkorange")
    ax.plot(
        forecast_series.index,
        forecast_series.values,
        color=forecast_color,
        label="Forecast",
        linewidth=2.5,
        linestyle="--"
    )

    if not conf_int_df.empty:
        ax.fill_between(
            forecast_series.index,
            conf_int_df.iloc[:, 0],
            conf_int_df.iloc[:, 1],
            color=forecast_color,
            alpha=0.2,
            label=f"{(1 - kwargs.get('alpha', 0.05)) * 100:.0f}% Confidence Interval"
        )

    if summary:
        stat_text = (
            f"ARIMA Order: {summary.get('order')}\n"
            f"AIC: {summary.get('aic')}\n"
            f"BIC: {summary.get('bic')}"
        )
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
        ax.text(
            0.02, 0.95,
            stat_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props,
            zorder=3
        )

    ax.set_title(title)
    ax.set_xlabel(kwargs.get("xlabel", "Date"))
    ax.set_ylabel(kwargs.get("ylabel", "Concentration"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    return fig

# --- 4. EXECUTION ORCHESTRATOR ---

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Initialization
    meta = parse_pollution_metadata(TARGET_FILENAME)
    if not meta: return
    full_path = DATA_DIRECTORY / TARGET_FILENAME

    # Ingestion & Resampling
    df, processed_data = pollution_data_resampling(full_path, periodicity=PERIODICITY, **ANALYSIS_KWARGS)

    is_single_series = isinstance(processed_data, pd.Series)
    data_tuple = (processed_data,) if is_single_series else processed_data

    if len(data_tuple) == 0 or data_tuple[0].empty:
        logging.error("Process aborted: Empty dataset.")
        return

    target_series = data_tuple[0]

    # ARIMA Modeling
    arima_object, forecast_values, confidence_df, summary_dict = fit_forecast_arima(
        target_series,
        forecast_steps=ANALYSIS_KWARGS.get("forecast_steps", 24),
        **ANALYSIS_KWARGS
    )

    # Visualization
    if not forecast_values.empty:
        arima_fig = visualize_arima_forecast(
            historical_series=target_series,
            forecast_series=forecast_values,
            conf_int_df=confidence_df,
            summary=summary_dict,
            title=f"ARIMA Forecast: {meta['pollutant']}",
            **ANALYSIS_KWARGS
        )
        if arima_fig:
            plt.show()
            plt.close(arima_fig)

    # Console Reporting
    print("\n" + "="*40)
    print("--- ARIMA SUMMARY ---")
    if summary_dict:
        print(f"Model Order (p, d, q):  {summary_dict.get('order')}")
        print(f"Forecast Horizon:       {summary_dict.get('forecast_steps')} periods")
        print(f"Akaike Criterion (AIC): {summary_dict.get('aic')}")
        print(f"Bayesian Crit. (BIC):   {summary_dict.get('bic')}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()