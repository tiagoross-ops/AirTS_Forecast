"""
AirTS-Forecast Project
Part 3: Multivariate & Univariate Neural Networks for Time Series

Hyperparameter Research & Bayesian Optimization Module
Powered by Optuna (NSGA-II 4D Multi-Objective Optimization + SQLite Dashboard)

Key Features:
- 4D Optimization: Simultaneously minimizes MAPE, RMSE, MAE, and Execution Time.
- Sliced Pareto Front Analysis: Visualizes the tradeoffs between Error and Computational Cost.
- Fully Automated Master Loop: Iterates over Pollutants AND Architectures.
- Persistence: Safely consolidates the "Champion" parameters into a master JSON file.
"""

# import os
import time
import json
import logging
# import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as vis_mpl

# Import our core model training modules
import PollutionDataAnalysis.pollution_DL_models_multivariate as mv
import PollutionDataAnalysis.pollution_DL_models_single_variable as sv

# Globally ignore noisy statsmodels/matplotlib warnings during automated sweeps
# warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION & STORAGE
# =====================================================================
# Define robust, OS-agnostic paths
OUTPUT_DIR = Path("outputs")
PLOT_DIR_SV = OUTPUT_DIR / "optuna_search_sv_plots"
PLOT_DIR_MV = OUTPUT_DIR / "optuna_search_mv_plots"
OPTUNA_DB_DIR = OUTPUT_DIR / "optuna_search_db"
ANALYSIS_PLOT_DIR = OPTUNA_DB_DIR / "optuna_search_analysis_plots"

# Create directories if they do not exist
for path in [PLOT_DIR_SV, PLOT_DIR_MV, ANALYSIS_PLOT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

SQLITE_DB_URL = f"sqlite:///{OPTUNA_DB_DIR}/hyperparameter_studies.db"
BEST_PARAMS_JSON = OUTPUT_DIR / "best_parameters.json"


# =====================================================================
# DYNAMIC FACTORIES & HELPERS
# =====================================================================

def _instantiate_model(architecture: str, model_name: str, input_dim: int = 1) -> Any:
    """
    Dynamically instantiates the requested PyTorch neural network class.

    Args:
        architecture (str): 'SV' for Univariate, 'MV' for Multivariate.
        model_name (str): Type of architecture ('RNN', 'LSTM', 'Bi-LSTM', 'GRU').
        input_dim (int): Number of features. Defaults to 1 for SV.

    Returns:
        torch.nn.Module: The initialized PyTorch model.

    Raises:
        ValueError: If an unsupported architecture or model_name is passed.
    """
    arch = architecture.upper()
    if arch == "SV":
        if model_name == "RNN": return sv.RNNModel()
        elif model_name == "LSTM": return sv.LSTMModel()
        elif model_name == "Bi-LSTM": return sv.BiLSTMModel()
        elif model_name == "GRU": return sv.GRUModel()
    elif arch == "MV":
        if model_name == "RNN": return mv.RNNModel(input_dim=input_dim)
        elif model_name == "LSTM": return mv.LSTMModel(input_dim=input_dim)
        elif model_name == "Bi-LSTM": return mv.BiLSTMModel(input_dim=input_dim)
        elif model_name == "GRU": return mv.GRUModel(input_dim=input_dim)

    raise ValueError(f"Unknown configuration: {architecture} - {model_name}")


def _inject_trial_parameters(
        trial: optuna.Trial,
        config_module: Any,
        research_dict: Optional[Dict],
        categorical_study: bool = True
) -> None:
    """
    Injects Optuna's suggested parameters into the target module's Config class.

    Args:
        trial (optuna.Trial): The current optimization trial.
        config_module (module): The target module (`sv` or `mv`).
        research_dict (dict): Boundaries/options for the search space.
        categorical_study (bool): True if using discrete lists, False if using continuous intervals.

    Raises:
        optuna.exceptions.TrialPruned: If the search space intervals are mathematically invalid.
    """
    if research_dict is None:
        config_module.config.NUM_LAYERS = trial.suggest_int("NUM_LAYERS", 2, 6, step=2)
        config_module.config.HIDDEN_DIM = trial.suggest_categorical("HIDDEN_DIM", [32, 128, 512])
        config_module.config.BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [8, 32, 128])
        config_module.config.LOOK_BACK = trial.suggest_categorical("LOOK_BACK", [15, 45, 60])
        config_module.config.HORIZON = trial.suggest_categorical("HORIZON", [7, 14, 28])
        config_module.config.LEARNING_RATE = trial.suggest_categorical("LEARNING_RATE", [0.001, 0.01, 0.1])
        if not categorical_study: print("## Sorry buddy, can only do non-categorical studies with specified parameters ##")
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
        except Exception:
            raise optuna.exceptions.TrialPruned()

# =====================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# =====================================================================

def objective_sv(
        trial: optuna.Trial,
        df_daily: pd.DataFrame,
        target_col: str,
        model_name: str,
        research_dict: dict,
        categorical_study: bool
) -> Tuple[float, float, float, float]:
    """
    Objective function for Univariate (SV) 4D Multi-Objective Optimization.
    """
    start_time = time.time()
    _inject_trial_parameters(trial, sv, research_dict, categorical_study)

    try:
        train_loader, test_loader, scaler, y_test_orig = sv.prepare_data(df_daily, pollutant=target_col)
    except ValueError:
        raise optuna.exceptions.TrialPruned()

    sv.config.PATIENCE = 30
    sv.config.EPOCHS = 50

    model = _instantiate_model("SV", model_name)
    model, _, _ = sv.train_model(model, train_loader, test_loader, model_name=f"Trial_{trial.number}")

    metrics, _ = sv.evaluate_model(model, test_loader, scaler, y_test_orig)

    plot_name = (f"SV_{model_name}_Trial{trial.number}_L{sv.config.NUM_LAYERS}_HD{sv.config.HIDDEN_DIM}_B{sv.config.BATCH_SIZE}_"
                 f"LB{sv.config.LOOK_BACK}_H{sv.config.HORIZON}_LR{sv.config.LEARNING_RATE}")

    fig = sv.predict_and_plot_series(model, df_daily,
                                     target_col, test_loader, scaler,
                                     title=plot_name.replace("_", " "),
                                     save_directory=str(PLOT_DIR_SV),
                                     model_name=model_name)
    plt.close(fig)

    execution_time = time.time() - start_time
    return metrics["MAPE"], metrics["RMSE"], metrics["MAE"], execution_time


def objective_multivariate(
        trial: optuna.Trial,
        df_daily: pd.DataFrame,
        target_col: str,
        feature_cols: list,
        model_name: str,
        research_dict: dict,
        categorical_study: bool
) -> Tuple[float, float, float, float]:
    """
    Objective function for Multivariate (MV) 4D Multi-Objective Optimization.
    """
    start_time = time.time()
    _inject_trial_parameters(trial, mv, research_dict, categorical_study)

    try:
        train_loader, test_loader, scaler, y_test_orig, num_features = mv.prepare_multivariate_data(df_daily, target_col=target_col, feature_cols=feature_cols)
    except ValueError:
        raise optuna.exceptions.TrialPruned()

    model = _instantiate_model("MV", model_name, input_dim=num_features)

    mv.config.PATIENCE = 30
    mv.config.EPOCHS = 50
    model, _, _ = mv.train_model(model, train_loader, test_loader, model_name=f"Trial_{trial.number}")

    plot_name = (f"MV_{model_name}_Trial{trial.number}_L{mv.config.NUM_LAYERS}_HD{mv.config.HIDDEN_DIM}"
                 f"_B{mv.config.BATCH_SIZE}_LB{mv.config.LOOK_BACK}_H{mv.config.HORIZON}_LR{mv.config.LEARNING_RATE}")

    metrics, _ = mv.evaluate_model(model[0], test_loader, scaler, y_test_orig)

    optimized_plot = mv.predict_and_plot_series(
        model[0], df_daily,
        target_col, test_loader, scaler,
        model_name, title=plot_name,
        save_directory=str(PLOT_DIR_MV/Path(model_name))
    )

    execution_time = time.time() - start_time
    return metrics["MAPE"], metrics["RMSE"], metrics["MAE"], execution_time

# =====================================================================
# STUDY EXECUTORS & ANALYSIS
# =====================================================================

def run_optuna_search(
        architecture: str,
        model_name: str,
        df_daily: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        n_trials: int = 50,
        research_dict: Optional[Dict] = None,
        categorical_study: bool = True) -> str:
    """
    Orchestrates the Bayesian search, enforcing NSGA-II multi-objective directions.
    """
    study_name = f"{architecture.upper()}_{model_name}_Optuna_{target_col}"
    print(f"\n{'='*70}\nINITIATING {study_name} (4D MULTI-OBJECTIVE)\n{'='*70}")
    print(f"Get study progression in:\noptuna-dashboard {SQLITE_DB_URL}")

    # 4 Directions: Minimize MAPE, RMSE, MAE, and TIME
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize", "minimize"],
        study_name=study_name,
        storage=SQLITE_DB_URL,
        load_if_exists=True
    )
    study.set_user_attr("contributors", ["TOLOCZKO ROSS Tiago", "REINOSO URABAYEN Lucas"])

    if architecture.upper() == 'SV':
        study.optimize(lambda trial: objective_sv(trial, df_daily, target_col, model_name, research_dict, categorical_study), n_trials=n_trials)
    elif architecture.upper() == 'MV':
        study.optimize(lambda trial: objective_multivariate(trial, df_daily, target_col, feature_cols, model_name, research_dict, categorical_study), n_trials=n_trials)

    # Extract the Champion from the Pareto Front (Optimized primarily for MAPE)
    best_pareto_trials = study.best_trials
    champion_trial = min(best_pareto_trials, key=lambda t: t.values[0])

    print(f"\n[✓] {study_name} COMPLETE | Champion MAPE: {champion_trial.values[0]:.2f}% | Time: {champion_trial.values[3]:.2f}s")

    # Format and export history to Parquet
    df_results = study.trials_dataframe(attrs=('number', 'values', 'params', 'state', 'duration'))
    df_results.columns = [col.replace('params_', '') for col in df_results.columns]

    df_results.rename(columns={
        'values_0': 'MAPE',
        'values_1': 'RMSE',
        'values_2': 'MAE',
        'values_3': 'Processing_Time_Sec',
        'number': 'Experiment_ID'
    }, inplace=True, errors='ignore')

    df_results.to_parquet(OPTUNA_DB_DIR / f"{study_name}_results.parquet", engine="pyarrow")

    return study_name


def analyze_optuna_study(
        study_name: str,
        architecture: str,
        model_name: str,
        target_col: str
) -> None:
    """
    Extracts the champion parameters from the Pareto front and safely updates the master JSON file.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=SQLITE_DB_URL)
    except Exception as e:
        logging.error(f"[!] Could not load study '{study_name}': {e}")
        return

    best_trials = study.best_trials
    if not best_trials:
        return

    champion_trial = min(best_trials, key=lambda t: t.values[0])

    # Safely load JSON dictionary
    all_results = {}
    if BEST_PARAMS_JSON.exists():
        try:
            with open(BEST_PARAMS_JSON, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            pass

    if target_col not in all_results:
        all_results[target_col] = {}

    # Map the 4 multi-objective values into the JSON payload
    all_results[target_col][f"{architecture.upper()}-{model_name}"] = {
        "Best_MAPE": round(champion_trial.values[0], 2),
        "Best_RMSE": round(champion_trial.values[1], 2),
        "Best_MAE": round(champion_trial.values[2], 2),
        "Best_Time_Sec": round(champion_trial.values[3], 2),
        "Hyperparameters": champion_trial.params
    }

    with open(BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"[✓] Appended Champion {architecture}-{model_name} params for {target_col} to {BEST_PARAMS_JSON.name}")

    # Generate Analysis Plots
    try:
        # 1. 3D Pareto Front Slicer (Visualizing MAPE vs RMSE vs Time)
        fig_pareto = vis_mpl.plot_pareto_front(
            study,
            targets=lambda t: (t.values[0], t.values[1], t.values[3]),
            target_names=["MAPE", "RMSE", "Time (s)"]
        )
        fig_pareto.set_title(f"Pareto Front Tradeoffs ({study_name})")
        fig_pareto.figure.savefig(ANALYSIS_PLOT_DIR / f"{study_name}_pareto.png", bbox_inches='tight', dpi=150)
        plt.close(fig_pareto.figure)

        # 2. Optimization History (Tracking MAPE)
        fig_hist = vis_mpl.plot_optimization_history(study, target=lambda t: t.values[0], target_name="MAPE")
        fig_hist.set_title(f"Optimization History ({study_name})")
        fig_hist.figure.savefig(ANALYSIS_PLOT_DIR / f"{study_name}_history.png", bbox_inches='tight', dpi=150)
        plt.close(fig_hist.figure)

        # 3. Param Importance (Tracking MAPE)
        fig_param = vis_mpl.plot_param_importances(study, target=lambda t: t.values[0], target_name="MAPE")
        fig_param.set_title(f"Hyperparameter Importance ({study_name})")
        fig_param.figure.savefig(ANALYSIS_PLOT_DIR / f"{study_name}_importance.png", bbox_inches='tight', dpi=150)
        plt.close(fig_param.figure)
    except Exception as e:
        logging.warning(f"[-] Could not generate all visual plots (Needs more complete trials): {e}")


# =====================================================================
# THE MASTER LOOP
# =====================================================================

def run_optuna_search_loop(
        architecture: str,
        df_daily: pd.DataFrame,
        target_list: List[str],
        model_list: List[str],
        feature_cols_map: Optional[Dict[str, List[str]]] = None,
        n_trials: int = 50,
        research_dict: Optional[Dict] = None,
        categorical_study: bool = True,
) -> None:
    """
    Automated pipeline sweeper that iterates through specified targets and architectures.
    """
    for target in target_list:
        for model_name in model_list:
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
# EXECUTION ORCHESTRATOR
# =====================================================================
if __name__ == "__main__":

    POLLUTANTS_TO_TEST = ["PM25", "NO2", "O3"]
    MODELS_TO_TEST = ["LSTM", "GRU", "Bi-LSTM"]

    # Pre-defined Categorical Grid Example
    CATEGORICAL_GRID = {
        "NUM_LAYERS": [2, 4],
        "HIDDEN_DIM": [64, 128],
        "BATCH_SIZE": [32, 64],
        "LOOK_BACK": [15, 30],
        "HORIZON": [7],
        "LEARNING_RATE": [0.001, 0.005]
    }

    # Load MV Dataset securely
    dataset_path = OUTPUT_DIR / "rnn_multivariate_dataset.parquet"
    if not dataset_path.exists():
        logging.error(f"[!] Target dataset missing: {dataset_path}")
        exit()

    df_mv = pd.read_parquet(dataset_path)
    if "Timestamp" in df_mv.columns:
        df_mv = df_mv.set_index("Timestamp")
    df_mv.index = pd.to_datetime(df_mv.index)
    df_mv = df_mv.sort_index().interpolate(method='linear').dropna()

    base_weather_features = ["sp", "u10", "v10"]
    mv_feature_map = {pol: base_weather_features + [pol] for pol in POLLUTANTS_TO_TEST}

    print("\n🚀 STARTING 4D MULTI-OBJECTIVE ML-OPS PIPELINE")

    # Run Univariate Studies
    run_optuna_search_loop(
        architecture="SV",
        df_daily=df_mv,
        target_list=POLLUTANTS_TO_TEST,
        model_list=MODELS_TO_TEST,
        n_trials=10,
        research_dict=CATEGORICAL_GRID,
        categorical_study=True
    )

    # Run Multivariate Studies
    run_optuna_search_loop(
        architecture="MV",
        df_daily=df_mv,
        target_list=POLLUTANTS_TO_TEST,
        model_list=MODELS_TO_TEST,
        feature_cols_map=mv_feature_map,
        n_trials=10,
        research_dict=CATEGORICAL_GRID,
        categorical_study=True
    )

    print("\n🏆 ALL OPTIMIZATIONS COMPLETE. Ready for deployment analysis!")