"""
AirTS-Forecast Project
Part 3: Multivariate & Univariate Neural Networks for Time Series

Hyperparameter Research & Bayesian Optimization Module
Powered by Optuna (TPE + Hyperband Pruning + SQLite Dashboard)

Key Features:
- Fully Automated Master Loop (Iterates over Pollutants AND Architectures).
- Supports both Categorical (Discrete) and Interval (Continuous) hyperparameter domains.
- Safely consolidates the absolute best parameters into a master JSON file.
"""

import os
import time
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import optuna
from optuna.pruners import HyperbandPruner

import optuna.visualization.matplotlib as vis_mpl

# Import our core modules
import PollutionDataAnalysis.pollution_DL_models_multivariate as mv
import PollutionDataAnalysis.pollution_DL_models_single_variable as sv

# Globally ignore noisy statsmodels/matplotlib warnings during long loops
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION & STORAGE
# =====================================================================
Path("outputs/grid_search_plots_sv").mkdir(parents=True, exist_ok=True)
Path("outputs/grid_search_plots_mv").mkdir(parents=True, exist_ok=True)
Path("outputs/optuna_db/analysis_plots").mkdir(parents=True, exist_ok=True)

SQLITE_DB_URL = "sqlite:///outputs/optuna_db/hyperparameter_studies.db"
BEST_PARAMS_JSON = Path("outputs/best_parameters.json")

# =====================================================================
# DYNAMIC FACTORIES & HELPERS
# =====================================================================

def _instantiate_model(architecture: str, model_name: str, input_dim: int = 1):
    """Dynamically instantiates the requested PyTorch model class."""
    if architecture.upper() == "SV":
        if model_name == "RNN": return sv.RNNModel()
        elif model_name == "LSTM": return sv.LSTMModel()
        elif model_name == "Bi-LSTM": return sv.BiLSTMModel()
        elif model_name == "GRU": return sv.GRUModel()
    elif architecture.upper() == "MV":
        if model_name == "RNN": return mv.RNNModel(input_dim=input_dim)
        elif model_name == "LSTM": return mv.LSTMModel(input_dim=input_dim)
        elif model_name == "Bi-LSTM": return mv.BiLSTMModel(input_dim=input_dim)
        elif model_name == "GRU": return mv.GRUModel(input_dim=input_dim)

    raise ValueError(f"Unknown configuration: {architecture} - {model_name}")

def _inject_trial_parameters(trial: optuna.Trial, config_module, research_dict: dict, categorical_study: bool) -> None:
    """DRY Helper to dynamically inject parameters into the module config."""
    if research_dict is None:
        config_module.config.NUM_LAYERS = trial.suggest_int("NUM_LAYERS", 2, 6, step=2)
        config_module.config.HIDDEN_DIM = trial.suggest_categorical("HIDDEN_DIM", [32, 128, 512])
        config_module.config.BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [8, 32, 128])
        config_module.config.LOOK_BACK = trial.suggest_categorical("LOOK_BACK", [15, 45, 60])
        config_module.config.HORIZON = trial.suggest_categorical("HORIZON", [7, 14, 28])
        config_module.config.LEARNING_RATE = trial.suggest_categorical("LEARNING_RATE", [0.001, 0.01, 0.1])
    elif categorical_study:
        config_module.config.NUM_LAYERS = trial.suggest_categorical("NUM_LAYERS", research_dict["NUM_LAYERS"])
        config_module.config.HIDDEN_DIM = trial.suggest_categorical("HIDDEN_DIM", research_dict["HIDDEN_DIM"])
        config_module.config.BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", research_dict["BATCH_SIZE"])
        config_module.config.LOOK_BACK = trial.suggest_categorical("LOOK_BACK", research_dict["LOOK_BACK"])
        config_module.config.HORIZON = trial.suggest_categorical("HORIZON", research_dict["HORIZON"])
        config_module.config.LEARNING_RATE = trial.suggest_categorical("LEARNING_RATE", research_dict["LEARNING_RATE"])
    else:
        try:
            config_module.config.NUM_LAYERS = trial.suggest_int("NUM_LAYERS", *research_dict["NUM_LAYERS"])
            config_module.config.HIDDEN_DIM = trial.suggest_int("HIDDEN_DIM", *research_dict["HIDDEN_DIM"])
            config_module.config.BATCH_SIZE = trial.suggest_int("BATCH_SIZE", *research_dict["BATCH_SIZE"])
            config_module.config.LOOK_BACK = trial.suggest_int("LOOK_BACK", *research_dict["LOOK_BACK"])
            config_module.config.HORIZON = trial.suggest_int("HORIZON", *research_dict["HORIZON"])
            config_module.config.LEARNING_RATE = trial.suggest_float("LEARNING_RATE", research_dict["LEARNING_RATE"][0], research_dict["LEARNING_RATE"][1], log=True)
        except Exception as e:
            raise optuna.exceptions.TrialPruned()

# =====================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# =====================================================================

def objective_sv(trial: optuna.Trial, df_daily: pd.DataFrame, target_col: str, model_name: str, research_dict: dict, categorical_study: bool) -> float:
    _inject_trial_parameters(trial, sv, research_dict, categorical_study)

    try:
        train_loader, test_loader, scaler, y_test_orig = sv.prepare_data(df_daily, pollutant=target_col)
    except ValueError:
        raise optuna.exceptions.TrialPruned()

    model = _instantiate_model("SV", model_name)
    model, _, _ = sv.train_model(model, train_loader, test_loader, model_name=f"Trial_{trial.number}", trial=trial)

    metrics, _ = sv.evaluate_model(model, test_loader, scaler, y_test_orig)

    plot_name = (f"SV_{model_name}_Trial{trial.number}_L{sv.config.NUM_LAYERS}_HD{sv.config.HIDDEN_DIM}_B{sv.config.BATCH_SIZE}_"
                 f"LB{sv.config.LOOK_BACK}_H{sv.config.HORIZON}_LR{sv.config.LEARNING_RATE}")
    fig = sv.predict_and_plot_series(model, df_daily, target_col, test_loader, scaler, title=plot_name, save_directory="outputs/grid_search_plots_sv", model_name=model_name)
    plt.close(fig)

    return metrics["MAPE"]


def objective_multivariate(trial: optuna.Trial, df_daily: pd.DataFrame, target_col: str, feature_cols: list, model_name: str, research_dict: dict, categorical_study: bool) -> float:
    _inject_trial_parameters(trial, mv, research_dict, categorical_study)

    try:
        train_loader, test_loader, scaler, y_test_orig, num_features = mv.prepare_multivariate_data(df_daily, target_col=target_col, feature_cols=feature_cols)
    except ValueError:
        raise optuna.exceptions.TrialPruned()

    model = _instantiate_model("MV", model_name, input_dim=num_features)
    model = mv.train_model(model, train_loader, test_loader, model_name=f"Trial_{trial.number}", patience=15, trial=trial)

    plot_name = f"MV_{model_name}_Trial{trial.number}_L{mv.config.NUM_LAYERS}_H{mv.config.HIDDEN_DIM}_B{mv.config.BATCH_SIZE}"
    metrics = mv.evaluate_and_plot(model, df_daily, target_col, test_loader, scaler, y_test_orig, title=plot_name, save_directory="outputs/grid_search_plots_mv", model_name=model_name)

    return metrics["MAPE"]

# =====================================================================
# STUDY EXECUTORS & ANALYSIS
# =====================================================================

def run_optuna_search(architecture: str, model_name: str, df_daily: pd.DataFrame, target_col: str, feature_cols: list = None, n_trials: int = 50, research_dict: dict = None, categorical_study: bool = True):
    study_name = f"{architecture.upper()}_{model_name}_Optuna_{target_col}"
    print(f"\n{'='*70}\nINITIATING {study_name}\n{'='*70}")

    study = optuna.create_study(direction="minimize", pruner=HyperbandPruner(), study_name=study_name, storage=SQLITE_DB_URL, load_if_exists=True)
    study.set_user_attr("contributors", ["TOLOCZKO ROSS Tiago", "REINOSO URABAYEN Lucas"])

    if architecture.upper() == 'SV':
        study.optimize(lambda trial: objective_sv(trial, df_daily, target_col, model_name, research_dict, categorical_study), n_trials=n_trials)
    elif architecture.upper() == 'MV':
        study.optimize(lambda trial: objective_multivariate(trial, df_daily, target_col, feature_cols, model_name, research_dict, categorical_study), n_trials=n_trials)

    print(f"\n[✓] {study_name} COMPLETE | Best MAPE: {study.best_value:.2f}%")

    # Export Parquet
    df_results = study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'duration'))
    df_results.columns = [col.replace('params_', '') for col in df_results.columns]
    df_results['Time_Seconds'] = df_results['duration'].dt.total_seconds().round(2)
    df_results.rename(columns={'value': 'MAPE', 'number': 'Experiment_ID'}, inplace=True)
    df_results.to_parquet(f"outputs/optuna_db/{study_name}_results.parquet", engine="pyarrow")

    return study_name


def analyze_optuna_study(study_name: str, architecture: str, model_name: str, target_col: str):
    """Extracts best parameters and safely updates the master JSON file."""
    try:
        study = optuna.load_study(study_name=study_name, storage=SQLITE_DB_URL)
    except Exception as e:
        return logging.error(f"[!] Could not load study '{study_name}': {e}")

    # Safely load and update the JSON
    all_results = {}
    if BEST_PARAMS_JSON.exists():
        try:
            with open(BEST_PARAMS_JSON, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            pass

    if target_col not in all_results:
        all_results[target_col] = {}

    all_results[target_col][f"{architecture.upper()}-{model_name}"] = {
        "Best_MAPE": round(study.best_value, 2),
        "Hyperparameters": study.best_trial.params
    }

    with open(BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"[✓] Appended best {architecture}-{model_name} params for {target_col} to {BEST_PARAMS_JSON}")

    # Generate visual plots
    fig_hist = vis_mpl.plot_optimization_history(study)
    fig_hist.set_title(f"Optimization History ({study_name})")
    fig_hist.figure.savefig(f"outputs/optuna_db/analysis_plots/{study_name}_history.png", bbox_inches='tight', dpi=150)
    plt.close(fig_hist.figure)

# =====================================================================
# THE MASTER LOOP
# =====================================================================

def run_optuna_search_loop(
        architecture: str,
        df_daily: pd.DataFrame,
        target_list: list,
        model_list: list,
        feature_cols_map: dict = None, # Dict mapping pollutant to its specific features (for MV)
        n_trials: int = 50,
        research_dict: dict = None,
        categorical_study: bool = True,
) -> None:
    """Master loop that iterates through every target pollutant AND every model architecture."""

    for target in target_list:
        for model_name in model_list:

            # Extract features specifically for this target if doing MV
            features = feature_cols_map.get(target, []) if feature_cols_map else None

            study_name = run_optuna_search(
                architecture=architecture,
                model_name=model_name,
                df_daily=df_daily,
                target_col=target,
                feature_cols=features,
                n_trials=n_trials,
                research_dict=research_dict,
                categorical_study=categorical_study
            )

            analyze_optuna_study(study_name, architecture, model_name, target)

# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":

    POLLUTANTS_TO_TEST = ["PM25", "NO2", "O3"]
    MODELS_TO_TEST = ["LSTM", "GRU", "Bi-LSTM"]

    CATEGORICAL_GRID = {
        "NUM_LAYERS": [2, 4],
        "HIDDEN_DIM": [64, 128],
        "BATCH_SIZE": [32, 64],
        "LOOK_BACK": [15, 30],
        "HORIZON": [7],
        "LEARNING_RATE": [0.001, 0.005]
    }

    # Load MV Dataset
    df_mv = pd.read_parquet("outputs/rnn_multivariate_dataset.parquet")
    if "Timestamp" in df_mv.columns: df_mv = df_mv.set_index("Timestamp")
    df_mv.index = pd.to_datetime(df_mv.index)
    df_mv = df_mv.sort_index().interpolate(method='linear').dropna()

    # Create a map so each pollutant knows what features to include
    base_weather_features = ["sp", "u10", "v10"]
    mv_feature_map = {pol: base_weather_features + [pol] for pol in POLLUTANTS_TO_TEST}

    print("\n🚀 STARTING MASSIVE AUTOMATED ML-OPS PIPELINE")
    print(f"Targets: {POLLUTANTS_TO_TEST}")
    print(f"Models: {MODELS_TO_TEST}")

    # 1. Run Univariate Studies
    run_optuna_search_loop(
        architecture="SV",
        df_daily=df_mv,
        target_list=POLLUTANTS_TO_TEST,
        model_list=MODELS_TO_TEST,
        n_trials=10, # Keep low for testing
        research_dict=CATEGORICAL_GRID,
        categorical_study=True
    )

    # 2. Run Multivariate Studies
    run_optuna_search_loop(
        architecture="MV",
        df_daily=df_mv,
        target_list=POLLUTANTS_TO_TEST,
        model_list=MODELS_TO_TEST,
        feature_cols_map=mv_feature_map,
        n_trials=10, # Keep low for testing
        research_dict=CATEGORICAL_GRID,
        categorical_study=True
    )

    print("\n🏆 ALL OPTIMIZATIONS COMPLETE. Check outputs/best_parameters.json !")