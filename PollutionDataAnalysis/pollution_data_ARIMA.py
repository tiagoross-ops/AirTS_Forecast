"""
AirTS-Forecast Project
Section 2: Classical Time Series Models (4^n DOE Optimized & RSM)
File: pollution_data_ARIMA_DOE.py
"""

import logging
import json
import warnings
import itertools
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pollution_data_exp_core import parse_pollution_metadata, pollution_data_resampling

warnings.filterwarnings("ignore")

# ======================================================================================================================
# GLOBAL CONFIGURATION
# ======================================================================================================================

DATA_DIRECTORY = Path(r"C:\Users\Tiago\Documents - PC\UTTOP\Enseignements\M1.2\Projet AirTS - Forecast\GEODAIR_POLLUTION_DATA")
OUTPUT_DIRECTORY = Path("outputs/DOE_Campaign")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
PERIODICITY = "daily"

# 4^n DOE Grids (Bolder Search Space)
DOE_GRIDS = {
    "ARIMA": {"p": [0, 12, 24, 48], "d": [0, 1, 2, 3], "q": [0, 1, 2, 3]},
    "SARIMA": {"p": [0, 12, 24, 48], "d": [0, 1, 2, 3], "q": [0, 1, 2, 3]}
}

HW_DOE_GRID = {
    "trend_num": [0, 1],
    "seasonal_num": [0, 1],
    "period": [7, 14, 30, 365],
    "damped_num": [0, 1]
}

# The User Baseline "Champion" Models to challenge the Machine
USER_BASELINES = {
    "ARIMA": {"p": 2, "d": 1, "q": 0},
    "SARIMA": {"p": 3, "d": 1, "q": 3},
    "Holt-Winters": {"trend_num": 0, "seasonal_num": 1, "period": 7, "damped_num": 0}
}

# ======================================================================================================================
# HELPERS, METRICS & MATH
# ======================================================================================================================

def train_test_split_ts(data: pd.Series, train_ratio: float, horizon: int | None) -> Tuple[pd.Series, pd.Series]:
    clean_data = data.dropna()
    split_idx = int(len(clean_data) * train_ratio)
    forecast_horizon = horizon if horizon is not None else len(data)
    return clean_data.iloc[:split_idx], clean_data.iloc[split_idx : split_idx + forecast_horizon]

def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned_pred, aligned_true = y_pred.align(y_true, join="inner")
    mask = aligned_true != 0
    if not mask.any(): return np.nan
    return float(np.mean(np.abs((aligned_true[mask] - aligned_pred[mask]) / aligned_true[mask])) * 100)

def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned_pred, aligned_true = y_pred.align(y_true, join="inner")
    ss_res = np.sum((aligned_true - aligned_pred) ** 2)
    ss_tot = np.sum((aligned_true - np.mean(aligned_true)) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    aligned_pred, aligned_true = y_pred.align(y_true, join="inner")
    if aligned_true.empty: return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan}
    errors = aligned_pred - aligned_true
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))), "mae": float(np.mean(np.abs(errors))),
        "mape": mape(aligned_true, aligned_pred), "r2": r2_score(aligned_true, aligned_pred)
    }

def build_fourier_features(length: int, periods=(7, 365.25), order=4) -> pd.DataFrame:
    t = np.arange(length)
    features = {}
    for period in periods:
        for k in range(1, order + 1):
            features[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
            features[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(features)

def calculate_theoretical_minimum(df: pd.DataFrame, param_cols: List[str]) -> Optional[Dict[str, int]]:
    clean_df = df.dropna(subset=['mape'] + param_cols)
    if len(clean_df) < 10: return None

    x1, x2, x3 = clean_df[param_cols[0]], clean_df[param_cols[1]], clean_df[param_cols[2]]
    Y = clean_df['mape']

    exog = np.column_stack((
        np.ones(len(x1)), x1, x2, x3, x1**2, x2**2, x3**2, x1*x2, x1*x3, x2*x3
    ))

    try:
        model = sm.OLS(Y, exog).fit()
        b0, b1, b2, b3, b11, b22, b33, b12, b13, b23 = model.params

        b_vec = np.array([b1, b2, b3])
        B_mat = np.array([
            [b11, 0.5*b12, 0.5*b13],
            [0.5*b12, b22, 0.5*b23],
            [0.5*b13, 0.5*b23, b33]
        ])

        x_opt = -0.5 * np.linalg.pinv(B_mat).dot(b_vec)
        x_opt_clipped = np.clip(np.round(x_opt), 0, 3).astype(int)

        return {
            param_cols[0]: int(x_opt_clipped[0]),
            param_cols[1]: int(x_opt_clipped[1]),
            param_cols[2]: int(x_opt_clipped[2])
        }
    except Exception as e:
        logging.warning(f"Failed to compute theoretical minimum mathematically: {e}")
        return None

# ======================================================================================================================
# MODEL RUNNER
# ======================================================================================================================

def evaluate_single_config(train: pd.Series, test: pd.Series, model_name: str, params: dict, fourier_features=None) -> Tuple[Optional[Dict], Optional[pd.Series]]:
    try:
        if model_name == "ARIMA":
            model = ARIMA(train, order=(params["p"], params["d"], params["q"]), enforce_stationarity=False, enforce_invertibility=False).fit()
            forecast = model.get_forecast(steps=len(test)).predicted_mean

        elif model_name == "SARIMA":
            model = SARIMAX(train, exog=fourier_features.loc[train.index], order=(params["p"], params["d"], params["q"]), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            forecast = model.get_forecast(steps=len(test), exog=fourier_features.loc[test.index]).predicted_mean

        elif model_name == "Holt-Winters":
            t_val = "add" if params.get("trend_num") == 1 else None
            s_val = "add" if params.get("seasonal_num") == 1 else None
            model = ExponentialSmoothing(
                train, trend=t_val, seasonal=s_val, seasonal_periods=params.get("period"),
                damped_trend=True if params.get("damped_num") == 1 else False, initialization_method="estimated"
            ).fit(optimized=True)
            forecast = model.forecast(steps=len(test))

        forecast.index = test.index
        metrics = evaluate_forecast(test, forecast)
        return metrics, forecast
    except Exception:
        return None, None

# ======================================================================================================================
# VISUALIZATIONS
# ======================================================================================================================

def plot_3d_response_surface(df: pd.DataFrame, model_name: str, pollutant: str, param_cols: List[str], theo_params: Optional[Dict] = None):
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['mape'] + param_cols)
    if clean_df.empty: return

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"3D Quadratic Response Surface: MAPE - {model_name} ({pollutant})", fontsize=18, fontweight="bold", y=0.95)

    all_pairs = list(itertools.combinations(range(len(param_cols)), 2))
    combinations = all_pairs[:3]

    for idx, (i, j) in enumerate(combinations):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        p_x, p_y = param_cols[i], param_cols[j]
        X_data, Y_data, Z_data = clean_df[p_x].values, clean_df[p_y].values, clean_df['mape'].values

        ax.scatter(X_data, Y_data, Z_data, color='blue', marker='o', s=20, alpha=0.5, label='Empirical Runs')

        exog = np.column_stack((np.ones(len(X_data)), X_data, Y_data, X_data**2, Y_data**2, X_data * Y_data))
        try:
            ols_model = sm.OLS(Z_data, exog).fit()
            x_mesh = np.linspace(X_data.min(), X_data.max(), 20)
            y_mesh = np.linspace(Y_data.min(), Y_data.max(), 20)
            X_surf, Y_surf = np.meshgrid(x_mesh, y_mesh)

            exog_surf = np.column_stack((
                np.ones(X_surf.size), X_surf.ravel(), Y_surf.ravel(), X_surf.ravel()**2, Y_surf.ravel()**2, (X_surf * Y_surf).ravel()
            ))
            Z_surf = ols_model.predict(exog_surf).reshape(X_surf.shape)
            ax.plot_surface(X_surf, Y_surf, Z_surf, cmap='viridis_r', alpha=0.6, edgecolor='none')

            if theo_params and p_x in theo_params and p_y in theo_params:
                t_x, t_y = theo_params[p_x], theo_params[p_y]
                t_z = ols_model.predict([1, t_x, t_y, t_x**2, t_y**2, t_x*t_y])[0]
                ax.scatter(t_x, t_y, t_z, color='gold', marker='*', s=300, edgecolor='black', zorder=10, label='Theoretical Minimum')

        except Exception: pass

        ax.set_xlabel(p_x, fontweight='bold', labelpad=10)
        ax.set_ylabel(p_y, fontweight='bold', labelpad=10)
        ax.set_zlabel('MAPE (%)', fontweight='bold', labelpad=10)
        ax.set_title(f"Interaction: {p_x} & {p_y}", fontweight="bold")
        ax.view_init(elev=20, azim=135)
        if idx == 0: ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(OUTPUT_DIRECTORY / f"{pollutant}_{model_name}_DOE_3D_Surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ======================================================================================================================
# DOE ORCHESTRATORS
# ======================================================================================================================

def run_classical_doe_tournament(train: pd.Series, test: pd.Series, model_name: str, pollutant: str) -> Dict:
    logging.info(f"Initiating 4^n DOE & RSM Tournament for {model_name} ({pollutant})...")

    grid = DOE_GRIDS.get(model_name, HW_DOE_GRID)
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    fourier_features = None
    if model_name == "SARIMA":
        fourier_features = build_fourier_features(len(train) + len(test))
        fourier_features.index = pd.concat([train, test]).index

    results = []

    # 1. EMPIRICAL RUN (Grid Search with Math Explosion Failsafes)
    for params in combinations:
        metrics, forecast = evaluate_single_config(train, test, model_name, params, fourier_features)

        if metrics:
            mape_val = metrics.get("mape", np.nan)
            rmse_val = metrics.get("rmse", np.nan)

            # THE OUTLIER & EXPLOSION FILTER
            # Discards runs over 100% MAPE or insane RMSEs to protect the OLS solver
            if pd.isna(mape_val) or pd.isna(rmse_val) or mape_val > 100.0 or rmse_val > 10:
                logging.debug(f"Skipped mathematically unstable run: MAPE={mape_val}, RMSE={rmse_val}")
                continue

            results.append({**params, **metrics})

    df_results = pd.DataFrame(results)

    if df_results.empty:
        return {"model": model_name, "best_params": {}, "best_metrics": {}, "forecast_series": None}

    empirical_best_row = df_results.loc[df_results['mape'].idxmin()]
    empirical_params = {k: int(empirical_best_row[k]) for k in keys}

    # 2. THEORETICAL OPTIMUM
    theoretical_params = None
    if model_name in ["ARIMA", "SARIMA"]:
        theoretical_params = calculate_theoretical_minimum(df_results, list(keys))

    # 3. BASELINE CONFIG
    baseline_params = USER_BASELINES[model_name]

    print(f"\n--- {model_name} TOURNAMENT ---")
    tournament = []

    emp_m, emp_f = evaluate_single_config(train, test, model_name, empirical_params, fourier_features)
    if emp_m:
        tournament.append(("Empirical Best", empirical_params, emp_m, emp_f))
        print(f"Empirical Best : {empirical_params} -> MAPE: {emp_m['mape']:.2f}%")

    if theoretical_params:
        theo_m, theo_f = evaluate_single_config(train, test, model_name, theoretical_params, fourier_features)
        if theo_m:
            tournament.append(("Theoretical Min", theoretical_params, theo_m, theo_f))
            print(f"Theoretical Min: {theoretical_params} -> MAPE: {theo_m['mape']:.2f}%")

    base_m, base_f = evaluate_single_config(train, test, model_name, baseline_params, fourier_features)
    if base_m:
        tournament.append(("User Baseline", baseline_params, base_m, base_f))
        print(f"User Baseline  : {baseline_params} -> MAPE: {base_m['mape']:.2f}%")

    tournament.sort(key=lambda x: x[2]['mape'])
    champ_title, champ_params, champ_metrics, champ_forecast = tournament[0]

    print(f"🏆 CHAMPION: {champ_title} wins with {champ_metrics['mape']:.2f}% MAPE!\n")

    if model_name in ["ARIMA", "SARIMA"]:
        plot_3d_response_surface(df_results, model_name, pollutant, list(keys), theo_params=theoretical_params)

    return {
        "model": model_name,
        "champion_type": champ_title,
        "best_params": champ_params,
        "best_metrics": champ_metrics,
        "forecast_series": champ_forecast
    }


# ======================================================================================================================
# MASTER EXECUTION
# ======================================================================================================================

def process_pollutant_file(file_path: Path) -> Tuple[Optional[str], Optional[Dict]]:
    meta = parse_pollution_metadata(file_path.name)
    if not meta: return None, None
    pollutant = meta["pollutant"]

    _, processed_data = pollution_data_resampling(
        file_path, periodicity=PERIODICITY, sep=";", encoding="utf-8", apply_cleaning=False
    )

    if isinstance(processed_data, pd.Series):
        target_series = processed_data.dropna()
    elif isinstance(processed_data, pd.DataFrame):
        target_series = processed_data[pollutant].dropna() if pollutant in processed_data.columns else \
        processed_data.iloc[:, 0].dropna()
    else:
        target_series = processed_data[0].dropna()

    train_series, test_series = train_test_split_ts(target_series, train_ratio=0.8, horizon=None)

    print(f"\n{'=' * 70}\nDOE CAMPAIGN FITTING: {pollutant}\n{'=' * 70}")

    champion_payloads = {}
    downstream_metrics_payload = {}
    forecasts = {}

    for model_name in ["ARIMA", "SARIMA", "Holt-Winters"]:
        result = run_classical_doe_tournament(train_series, test_series, model_name, pollutant)

        if result["forecast_series"] is not None:
            # 1. Store Deep Parameters for the Local Audit JSON
            champion_payloads[model_name] = {
                "Champion_Origin": result["champion_type"],
                "Optimal_Parameters": result["best_params"],
                "Metrics": result["best_metrics"]
            }

            # 2. Extract and Map Metrics for the Global Comparison JSON
            downstream_metrics_payload[model_name] = {
                "RMSE": result["best_metrics"].get("rmse", np.nan),
                "MAE": result["best_metrics"].get("mae", np.nan),
                "MAPE": result["best_metrics"].get("mape", np.nan),
                "R^2": result["best_metrics"].get("r2", np.nan)
            }

            forecasts[model_name] = result["forecast_series"]

    # Local Audit Export
    local_json_path = OUTPUT_DIRECTORY / f"{pollutant}_DOE_optimized_params.json"
    with open(local_json_path, "w", encoding="utf-8") as f:
        json.dump(champion_payloads, f, indent=4)
    print(f"[✓] Theoretical/Empirical best parameters saved to: {local_json_path.name}")

    fig, ax = plt.subplots(figsize=(14, 6))

    historical_context = train_series.iloc[-90:]
    ax.plot(historical_context.index, historical_context.values, color="lightgray", linewidth=1.5,
            label="Observed History (Train)")

    ax.plot(test_series.index, test_series.values, color="black", linewidth=2, label="True Future Data")

    colors = ["darkorange", "royalblue", "seagreen"]
    for idx, (m_name, f_series) in enumerate(forecasts.items()):
        ax.plot(f_series.index, f_series.values, color=colors[idx], linestyle="--", linewidth=2,
                label=f"{m_name} (Absolute Champion)")

    ax.axvline(test_series.index[0], color="blue", linestyle=":", linewidth=2, label="Forecast Start")
    ax.set_title(f"Tournament Champion Forecasts vs Reality: {pollutant}", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    plt.savefig(OUTPUT_DIRECTORY / f"{pollutant}_champion_forecasts.png", dpi=150)
    plt.close()

    # Pass the formatted metrics back up to the main orchestrator loop
    return pollutant, downstream_metrics_payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    csv_files = list(DATA_DIRECTORY.glob("*.csv"))

    # The Global Dictionary intended for Part 4
    master_comparison_results = {}

    for file_path in csv_files:
        pol_name, pol_metrics = process_pollutant_file(file_path)
        if pol_name and pol_metrics:
            master_comparison_results[pol_name] = pol_metrics

    # Save the Master Payload exactly where the Part 4 script expects it
    # (OUTPUT_DIRECTORY is "outputs/DOE_Campaign", so we use .parent to put it directly in "outputs/")
    master_json_path = OUTPUT_DIRECTORY.parent / "classical_model_results.json"

    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master_comparison_results, f, indent=4)

    print("\n" + "=" * 80)
    print(f"[🚀 ML-OPS SUCCESS] Master Classical Results successfully exported to:")
    print(f" -> {master_json_path.absolute()}")
    print("=" * 80 + "\n")