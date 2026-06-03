"""
AirTS-Forecast Project
Part 3: Multivariate & Univariate Neural Networks for Time Series

Hyperparameter Research & Bayesian Optimization Module
Powered by Optuna (NSGA-II Multi-Objective Optimization + SQLite Dashboard)

Key Features:
- Dynamic Optimization: Supports Single-Objective (e.g., just MAPE) or Multi-Objective (MAPE, RMSE, TIME).
- Full Suite Persistence: Retrains, plots, and exports the metrics of EVERY model on the Pareto Front.
- Pruning Sandbox: Pruning logic is included but safely commented out for Multi-Objective compatibility.
- Fully Automated Master Loop: Iterates over Pollutants AND Architectures natively.
"""

import time
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.pruners import PatientPruner, MedianPruner, HyperbandPruner
import optuna.visualization.matplotlib as vis_mpl

# Import our core model training modules
import PollutionDataAnalysis.pollution_DL_models_multivariate as mv
import PollutionDataAnalysis.pollution_DL_models_single_variable as sv

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# CONFIGURATION & STORAGE
# =====================================================================
OUTPUT_DIR = Path("outputs")
PLOT_DIR_SV = OUTPUT_DIR / "optuna_search_sv_plots"
PLOT_DIR_MV = OUTPUT_DIR / "optuna_search_mv_plots"
OPTUNA_DB_DIR = OUTPUT_DIR / "optuna_search_db"
ANALYSIS_PLOT_DIR = OPTUNA_DB_DIR / "optuna_search_analysis_plots"

for path in [PLOT_DIR_SV, PLOT_DIR_MV, ANALYSIS_PLOT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

SQLITE_DB_URL = f"sqlite:///{OPTUNA_DB_DIR}/hyperparameter_studies.db"
BEST_PARAMS_JSON = OUTPUT_DIR / "best_parameters.json"
BEST_PARAMS_JSON_MV = OUTPUT_DIR / "best_parameters_mv.json"

PARETO_METRICS_CSV = OUTPUT_DIR / "pareto_front_retrained_metrics.csv"

# =====================================================================
# DYNAMIC FACTORIES & HELPERS
# =====================================================================

def _instantiate_model(architecture: str, model_name: str, input_dim: int = 1) -> Any:
    """Dynamically instantiates the requested PyTorch neural network class."""
    arch = architecture.upper()
    if arch == "SV":
        if model_name == "RNN": return sv.RNNModel()
        elif model_name == "LSTM": return sv.LSTMModel()
        elif model_name == "Bi-LSTM": return sv.BiLSTMModel()
        elif model_name == "GRU": return sv.GRUModel()
        elif model_name == "Hy-RNN-LSTM": return sv.HybridRNNLSTMModel()
        elif model_name == "CNN": return sv.CNNModel()
        elif model_name == "Hy-CNN-LSTM": return sv.HybridCNNLSTMModel()

    elif arch == "MV":
        if model_name == "RNN": return mv.RNNModel(input_dim=input_dim)
        elif model_name == "LSTM": return mv.LSTMModel(input_dim=input_dim)
        elif model_name == "Bi-LSTM": return mv.BiLSTMModel(input_dim=input_dim)
        elif model_name == "GRU": return mv.GRUModel(input_dim=input_dim)
        elif model_name == "Hy-RNN-LSTM": return mv.HybridRNNLSTMModel()
        elif model_name == "CNN": return mv.CNNModel()
        elif model_name == "Hy-CNN-LSTM": return mv.HybridCNNLSTMModel()


    raise ValueError(f"Unknown configuration: {architecture} - {model_name}")


def _inject_trial_parameters(trial: optuna.Trial, config_module: Any, research_dict: Optional[Dict], categorical_study: bool = True) -> None:
    """Injects Optuna's suggested parameters into the target module's Config class."""
    if research_dict is None:
        config_module.config.NUM_LAYERS = trial.suggest_int("NUM_LAYERS", 2, 6, step=2)
        config_module.config.HIDDEN_DIM = trial.suggest_categorical("HIDDEN_DIM", [32, 128, 512])
        config_module.config.BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [8, 32, 128])
        config_module.config.LOOK_BACK = trial.suggest_categorical("LOOK_BACK", [15, 45, 60])
        config_module.config.HORIZON = trial.suggest_categorical("HORIZON", [7, 14, 28])
        config_module.config.LEARNING_RATE = trial.suggest_categorical("LEARNING_RATE", [0.001, 0.01, 0.1])
        if not categorical_study:
            logging.warning("Categorical defaults used because research_dict is missing.")
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

def _get_dynamic_objectives(metrics: dict, exec_time: float, objectives: Union[List[str], str]) -> Union[float, Tuple[float, ...]]:
    """Safely maps the user-requested objectives to the calculated metrics."""
    if isinstance(objectives, list):
        obj_metrics = []
        for obj in objectives:
            if obj.upper() == "TIME":
                obj_metrics.append(exec_time)
            elif obj.upper() in metrics:
                obj_metrics.append(metrics[obj.upper()])
            else:
                raise ValueError(f"Objective '{obj}' not recognized.")
        return tuple(obj_metrics)
    else:
        if objectives.upper() == "TIME":
            return exec_time
        return metrics[objectives.upper()]

# =====================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# =====================================================================

def objective_sv(
        trial: optuna.Trial, df_daily: pd.DataFrame, target_col: str,
        model_name: str, research_dict: dict, objectives: Union[List[str], str], categorical_study: bool
) -> Union[float, Tuple[float, ...]]:
    """Objective function for Univariate (SV) Optimization."""
    _inject_trial_parameters(trial, sv, research_dict, categorical_study)
    start_time = time.time()
    study_name = f"SV_{model_name}_Optuna_{target_col}_{objectives}"

    try:
        train_loader, test_loader, scaler, y_test_orig = sv.prepare_data(df_daily, pollutant=target_col)
    except ValueError:
        raise optuna.exceptions.TrialPruned()

    model = _instantiate_model("SV", model_name)

    model, _, _ = sv.train_model(model, train_loader, test_loader, model_name=f"Trial_{trial.number}", trial=trial, verbose=False)

    if isinstance(objectives, str):
        save_dir = PLOT_DIR_SV/study_name
        save_dir.mkdir(parents=True, exist_ok=True)

        title = (f"SV_{model_name}_Trial{trial.number}_L{sv.config.NUM_LAYERS}_HD{sv.config.HIDDEN_DIM}_"
                 f"B{sv.config.BATCH_SIZE}_LB{sv.config.LOOK_BACK}_H{sv.config.HORIZON}")
        study_plot = sv.predict_and_plot_series(
            model, df_daily, target_col,
            test_loader, scaler, model_name,
            title, save_directory=str(save_dir)
        )
        plt.close(study_plot)

    metrics, _ = sv.evaluate_model(model, test_loader, scaler, y_test_orig)
    execution_time = time.time() - start_time

    return _get_dynamic_objectives(metrics, execution_time, objectives)


def objective_multivariate(
        trial: optuna.Trial, df_daily: pd.DataFrame, target_col: str, feature_cols: list,
        model_name: str, research_dict: dict, objectives: Union[List[str], str], categorical_study: bool
) -> Union[float, Tuple[float, ...]]:
    """Objective function for Multivariate (MV) Optimization."""
    start_time = time.time()
    _inject_trial_parameters(trial, mv, research_dict, categorical_study)
    study_name = f"SV_{model_name}_Optuna_{target_col}_{objectives}"

    try:
        train_loader, test_loader, scaler, y_test_orig, num_features = mv.prepare_multivariate_data(df_daily, target_col=target_col, feature_cols=feature_cols)
    except ValueError:
        raise optuna.exceptions.TrialPruned()

    model = _instantiate_model("MV", model_name, input_dim=num_features)
    trained_out, _, _ = mv.train_model(model, train_loader, test_loader, model_name=f"Trial_{trial.number}", trial=trial, verbose=False)

    if isinstance(objectives, str):
        title = (f"MV_{model_name}_Trial{trial.number}_L{sv.config.NUM_LAYERS}_HD{sv.config.HIDDEN_DIM}_"
                 f"B{sv.config.BATCH_SIZE}_LB{sv.config.LOOK_BACK}_H{sv.config.HORIZON}")
        study_plot = sv.predict_and_plot_series(
            model, df_daily, target_col,
            test_loader, scaler, model_name,
            title.replace('_', ' '),
            save_directory=str(PLOT_DIR_MV/study_name)
        )
        plt.close(study_plot)


    metrics, _ = sv.evaluate_model(model, test_loader, scaler, y_test_orig)
    execution_time = time.time() - start_time

    metrics, _ = mv.evaluate_model(trained_out, test_loader, scaler, y_test_orig)
    execution_time = time.time() - start_time

    return _get_dynamic_objectives(metrics, execution_time, objectives)


# =====================================================================
# STUDY EXECUTORS & ANALYSIS
# =====================================================================

def _retrain_and_plot_pareto_front(
        architecture: str, model_name: str, target_col: str, best_trials: List[optuna.trial.FrozenTrial],
        df_daily: pd.DataFrame, feature_cols: list
) -> List[Dict]:
    """
    Iterates through EVERY trial on the Pareto front, dynamically injects its parameters,
    does a verbose retraining, saves a plot, and returns the strictly evaluated metrics.
    """
    print(f"\n[🚀] Retraining {len(best_trials)} PARETO CHAMPIONS for {architecture}-{model_name} ({target_col}) to export final metrics...")

    mod = sv if architecture.upper() == "SV" else mv
    plot_dir = PLOT_DIR_SV if architecture.upper() == "SV" else PLOT_DIR_MV
    retrained_metrics_list = []

    for trial in best_trials:
        print(f"  -> Retraining Pareto Trial #{trial.number}...")

        # Inject Champion Params
        mod.config.NUM_LAYERS = trial.params["NUM_LAYERS"]
        mod.config.HIDDEN_DIM = trial.params["HIDDEN_DIM"]
        mod.config.BATCH_SIZE = trial.params["BATCH_SIZE"]
        mod.config.LOOK_BACK = trial.params["LOOK_BACK"]
        mod.config.HORIZON = trial.params["HORIZON"]
        mod.config.LEARNING_RATE = trial.params["LEARNING_RATE"]

        start_time = time.time()

        if architecture.upper() == "SV":
            train_loader, test_loader, scaler, y_test_orig = mod.prepare_data(df_daily, pollutant=target_col)
            model = _instantiate_model("SV", model_name)
        else:
            train_loader, test_loader, scaler, y_test_orig, num_features = mod.prepare_multivariate_data(df_daily, target_col=target_col, feature_cols=feature_cols)
            model = _instantiate_model("MV", model_name, input_dim=num_features)

        # Retrain with verbose=True
        trained_out = mod.train_model(model, train_loader, test_loader, model_name=f"Pareto #{trial.number}", trial=None, verbose=True)
        final_model = trained_out[0] if isinstance(trained_out, tuple) else trained_out

        # Securely evaluate the retrained model
        metrics, _ = mod.evaluate_model(final_model, test_loader, scaler, y_test_orig)
        exec_time = time.time() - start_time

        # Append to our export list
        metrics["Architecture"] = f"{architecture.upper()}-{model_name}"
        metrics["Target"] = target_col
        metrics["Trial_ID"] = trial.number
        metrics["Time_Sec"] = round(exec_time, 2)
        retrained_metrics_list.append(metrics)

        # Save individual plot securely
        plot_name = f"PARETO_TRIAL_{trial.number}_{architecture}_{model_name}_{target_col}"
        fig = mod.predict_and_plot_series(
            final_model, df_daily, target_col, test_loader, scaler,
            title=plot_name.replace("_", " "), save_directory=str(plot_dir), model_name=model_name
        )
        plt.close(fig)

    print(f"[✓] {len(best_trials)} Champion plots and metrics successfully exported.\n")
    return retrained_metrics_list


def evaluate_pareto_front(best_pareto_trials: List[optuna.trial.FrozenTrial], objectives: List[str]):
    """
    Calculates the distinct champion archetypes dynamically based on the requested objectives.
    Upgraded to use R^2 as a mathematical tie-breaker if available in the objectives list.
    """
    # Standardize objective names for safe lookups
    obj_upper = [obj.upper() for obj in objectives]

    try:
        mape_idx = obj_upper.index("MAPE")
    except ValueError:
        return best_pareto_trials[0], best_pareto_trials[0], best_pareto_trials[0] # Fallback if MAPE is missing

    # 1. Dynamically check if R^2 is being tracked in this Optuna Study
    r2_idx = None
    if "R^2" in obj_upper:
        r2_idx = obj_upper.index("R^2")
    elif "R2" in obj_upper:  # Failsafe for syntax variations
        r2_idx = obj_upper.index("R2")

    # =========================================================
    # 2. ACCURACY CHAMPION (Minimize MAPE, Maximize R^2)
    # =========================================================
    if r2_idx is not None:
        # Tuple Trick: (MAPE, -R^2). Finds the lowest MAPE. If MAPEs are equal,
        # the more negative R^2 wins, promoting the model that captures higher variance.
        accuracy_champ = min(best_pareto_trials, key=lambda t: (t.values[mape_idx], -t.values[r2_idx]))
    else:
        accuracy_champ = min(best_pareto_trials, key=lambda t: t.values[mape_idx])

    # =========================================================
    # 3. BALANCED & SPEED CHAMPIONS
    # =========================================================
    if "TIME" in obj_upper:
        time_idx = obj_upper.index("TIME")
        speed_champ = min(best_pareto_trials, key=lambda t: t.values[time_idx])

        # Extract min/max boundaries for normalization
        mapes = [t.values[mape_idx] for t in best_pareto_trials]
        times = [t.values[time_idx] for t in best_pareto_trials]
        min_mape, max_mape = min(mapes), max(mapes)
        min_time, max_time = min(times), max(times)

        def calculate_balanced_score(t):
            """Normalizes MAPE and TIME to a 0-1 scale to find the most efficient trade-off."""
            norm_mape = (t.values[mape_idx] - min_mape) / (max_mape - min_mape) if max_mape > min_mape else 0
            norm_time = (t.values[time_idx] - min_time) / (max_time - min_time) if max_time > min_time else 0
            return norm_mape + norm_time

        # Tie-breaker logic applied to the balanced score as well!
        if r2_idx is not None:
            balanced_champ = min(best_pareto_trials, key=lambda t: (calculate_balanced_score(t), -t.values[r2_idx]))
        else:
            balanced_champ = min(best_pareto_trials, key=calculate_balanced_score)

        return balanced_champ, accuracy_champ, speed_champ

    # Fallback if TIME is not tracked
    return accuracy_champ, accuracy_champ, accuracy_champ

def run_optuna_search(
        architecture: str, model_name: str, df_daily: pd.DataFrame, target_col: str, objectives: Union[List[str], str],
        feature_cols: Optional[List[str]] = None, n_trials: int = 50, research_dict: Optional[Dict] = None, categorical_study: bool = True
) -> Tuple[str, List[optuna.trial.FrozenTrial], optuna.trial.FrozenTrial]:
    """
    Orchestrates the Bayesian search, enforcing NSGA-II multi-objective directions dynamically.
    """
    study_name = f"{architecture.upper()}_{model_name}_Optuna_{target_col}_{objectives}"
    print(f"\n{'='*70}\nINITIATING {study_name}\n{'='*70}")
    print(f"Get study progression in:\noptuna-dashboard {SQLITE_DB_URL}")

    # Set up dynamic directions based on user input
    if isinstance(objectives, list):
        directions = ["minimize" for _ in objectives]
    else:
        directions = ["minimize"]

    # --- PRUNING SANDBOX ---
    # To enable pruning for Single-Objective studies, uncomment these lines and add `pruner=pruner` to create_study:
    # from optuna.pruners import PatientPruner, MedianPruner
    if isinstance(objectives, str):
        # pruner = PatientPruner(MedianPruner(n_startup_trials=5, n_warmup_steps=10), patience=10)
        pruner = HyperbandPruner()
    else:
        pruner = None

    study = optuna.create_study(
        directions=directions,
        study_name=study_name,
        storage=SQLITE_DB_URL,
        load_if_exists=True,
        pruner=pruner
    )
    study.set_user_attr("contributors", ["TOLOCZKO ROSS Tiago", "REINOSO URABAYEN Lucas"])

    if architecture.upper() == 'SV':
        study.optimize(lambda trial: objective_sv(trial, df_daily, target_col, model_name, research_dict, objectives, categorical_study), n_trials=n_trials)
    elif architecture.upper() == 'MV':
        study.optimize(lambda trial: objective_multivariate(trial, df_daily, target_col, feature_cols, model_name, research_dict, objectives, categorical_study), n_trials=n_trials)

    # DataFrame formatting
    df_results = study.trials_dataframe(attrs=('number', 'values', 'params', 'state', 'duration'))
    df_results.columns = [col.replace('params_', '') for col in df_results.columns]

    if isinstance(objectives, list):
        best_pareto_trials = study.best_trials
        balanced_champ, accuracy_champ, speed_champ = evaluate_pareto_front(best_pareto_trials, objectives)

        print(f"\n[✓] {study_name} COMPLETE | {len(best_pareto_trials)} Models on the Pareto Front")

        # Dynamically map the columns in the Parquet file
        rename_dict = {f'values_{i}': obj.upper() for i, obj in enumerate(objectives)}
        rename_dict['number'] = 'Experiment_ID'
        df_results.rename(columns=rename_dict, inplace=True, errors='ignore')
        df_results.to_parquet(OPTUNA_DB_DIR / f"{study_name}_results.parquet", engine="pyarrow")

        # Retrain ALL models on the Pareto front and log exact metrics
        retrained_metrics = _retrain_and_plot_pareto_front(architecture, model_name, target_col, best_pareto_trials, df_daily, feature_cols)

        # Append to master CSV so the user has easy access to ALL actual retrained metrics
        df_metrics = pd.DataFrame(retrained_metrics)
        if PARETO_METRICS_CSV.exists():
            df_metrics.to_csv(PARETO_METRICS_CSV, mode='a', header=False, index=False)
        else:
            df_metrics.to_csv(PARETO_METRICS_CSV, mode='w', header=True, index=False)

        return study_name, best_pareto_trials, balanced_champ

    else:
        # Single Objective Logic
        df_results.rename(columns={'value': objectives.upper(), 'number': 'Experiment_ID'}, inplace=True, errors='ignore')
        df_results.to_parquet(OPTUNA_DB_DIR / f"{study_name}_results.parquet", engine="pyarrow")
        return study_name, [study.best_trial], study.best_trial


def analyze_optuna_study(
        study_name: str,
        architecture: str,
        model_name: str,
        target_col: str,
        best_pareto_trials: List[optuna.trial.FrozenTrial],
        balanced_champ: optuna.trial.FrozenTrial,
        objectives: Union[List[str], str],
        output_best_parameters_json: str | Path = BEST_PARAMS_JSON
) -> None:
    """Updates the JSON file with the Champion and generates analytical plots."""
    try:
        study = optuna.load_study(study_name=study_name, storage=SQLITE_DB_URL)
    except Exception as e:
        return logging.error(f"[!] Could not load study '{study_name}': {e}")

    if architecture.upper() == 'MV':
        output_best_parameters_json = BEST_PARAMS_JSON_MV

    # Safely load JSON dictionary
    all_results = {}
    if output_best_parameters_json.exists():
        try:
            with open(output_best_parameters_json, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            pass

    if target_col not in all_results:
        all_results[target_col] = {}

    # Compile the full Pareto Front to save in the JSON
    pareto_front_data = []
    for t in best_pareto_trials:
        trial_data: dict[str, int | dict[str, Any] | float] = {"Trial_ID": t.number, "Hyperparameters": t.params}
        if isinstance(objectives, list):
            for i, obj in enumerate(objectives):
                trial_data[obj.upper()] = round(t.values[i], 2)
        else:
            trial_data[objectives.upper()] = round(t.value, 2)
        pareto_front_data.append(trial_data)

    # Save the balanced champion as the primary
    champ_data: dict[str, Any] | list[Any] = {"Balanced_Hyperparameters": balanced_champ.params,
                                              "Pareto_Front_Trials": pareto_front_data}
    if isinstance(objectives, list):
        for i, obj in enumerate(objectives):
            champ_data[f"Balanced_{obj.upper()}"] = round(balanced_champ.values[i], 2)
    else:
        champ_data[f"Balanced_{objectives.upper()}"] = round(balanced_champ.value, 2)

    all_results[target_col][f"{architecture.upper()}-{model_name}"] = champ_data

    with open(output_best_parameters_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    # Generate Analysis Plots
    try:
        if isinstance(objectives, list) and len(objectives) >= 2:
            fig_pareto = vis_mpl.plot_pareto_front(
                study,
                targets=lambda t: tuple(t.values[:3]) if len(objectives) >= 3 else tuple(t.values),
                target_names=[obj.upper() for obj in objectives][:3]
            )
            fig_pareto.set_title(f"Pareto Front Tradeoffs ({study_name})")
            fig_pareto.figure.savefig(ANALYSIS_PLOT_DIR / f"{study_name}_pareto.png", bbox_inches='tight', dpi=150)
            plt.close(fig_pareto.figure)

        # Plot history targeting the first objective (Usually MAPE)
        fig_hist = vis_mpl.plot_optimization_history(study, target=lambda t: t.values[0] if isinstance(objectives, list) else t.value, target_name=objectives[0].upper() if isinstance(objectives, list) else objectives.upper())
        fig_hist.set_title(f"Optimization History ({study_name})")
        fig_hist.figure.savefig(ANALYSIS_PLOT_DIR / f"{study_name}_history.png", bbox_inches='tight', dpi=150)
        plt.close(fig_hist.figure)
    except Exception as e:
        logging.warning(f"[-] Could not generate visual plots: {e}")

# =====================================================================
# THE MASTER LOOP
# =====================================================================

def run_optuna_search_loop(
        architecture: str, df_daily: pd.DataFrame, target_list: List[str], model_list: List[str], objectives: Union[List[str], str],
        feature_cols_map: Optional[Dict[str, List[str]]] = None, n_trials: int = 50, research_dict: Optional[Dict] = None, categorical_study: bool = True
) -> None:
    for target in target_list:
        for model_name in model_list:
            features = feature_cols_map.get(target, []) if feature_cols_map else None

            study_name, best_pareto_trials, balanced_champ = run_optuna_search(
                architecture=architecture, model_name=model_name, df_daily=df_daily, target_col=target,
                objectives=objectives, feature_cols=features, n_trials=n_trials,
                research_dict=research_dict, categorical_study=categorical_study
            )

            analyze_optuna_study(
                study_name,
                architecture,
                model_name, target,
                best_pareto_trials,
                balanced_champ, objectives)

# =====================================================================
# EXECUTION ORCHESTRATOR
# =====================================================================
if __name__ == "__main__":

    POLLUTANTS_TO_TEST = ["PM25", "NO2", "O3"]
    MODELS_TO_TEST = ["RNN", "LSTM", "Bi-LSTM", "GRU", "Hy-RNN-LSTM", "CNN", "Hy-CNN-LSTM"]

    # Defines the metrics to minimize. Max 3 metrics for 3D Pareto Front plotting.
    OBJECTIVES_TO_TRACK = "MAPE"

    CATEGORICAL_GRID = {
        "NUM_LAYERS": [2, 4, 6, 8],
        "HIDDEN_DIM": [64, 128],
        "BATCH_SIZE": [32, 64],
        "LOOK_BACK": [15, 30],
        "HORIZON": [7, 21],
        "LEARNING_RATE": [0.001, 0.005, .01]
    }

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

    print("\n🚀 STARTING DYNAMIC ML-OPS PIPELINE")

    run_optuna_search_loop(
        architecture="SV", df_daily=df_mv, target_list=POLLUTANTS_TO_TEST, model_list=MODELS_TO_TEST[5:],
        objectives=OBJECTIVES_TO_TRACK, n_trials=20, research_dict=CATEGORICAL_GRID, categorical_study=True
    )

    # run_optuna_search_loop(
    #     architecture="MV", df_daily=df_mv, target_list=POLLUTANTS_TO_TEST, model_list=MODELS_TO_TEST,
    #     objectives=OBJECTIVES_TO_TRACK, feature_cols_map=mv_feature_map, n_trials=50, research_dict=CATEGORICAL_GRID, categorical_study=True
    # )

    print("\n🏆 ALL OPTIMIZATIONS COMPLETE.")
    print(f"-> Full Model Configurations saved to: {BEST_PARAMS_JSON.name}")
    print(f"-> Retrained Pareto Metrics exported to: {PARETO_METRICS_CSV.name}")