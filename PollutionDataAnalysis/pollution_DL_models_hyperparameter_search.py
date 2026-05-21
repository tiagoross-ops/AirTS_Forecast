"""
AirTS-Forecast Project
Part 3: Multivariate & Univariate Neural Networks for Time Series

Hyperparameter Research & Bayesian Optimization Module
Powered by Optuna (TPE + Hyperband Pruning + SQLite Dashboard)

Key Features:
- Supports both Categorical (Discrete) and Interval (Continuous) hyperparameter domains.
- Automatically handles Pruning for mathematically invalid configurations.
- Natively connects to Optuna's SQLite Dashboard for real-time visualization.
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import optuna
from optuna.pruners import HyperbandPruner

# Import Optuna's native visualization tools for Jupyter rendering
import optuna.visualization as vis
# Import Matplotlib backend for saving static images to disk
import optuna.visualization.matplotlib as vis_mpl

# Import our core modules
import PollutionDataAnalysis.multivariate_LSTM_environmental_pollution as mv
import PollutionDataAnalysis.pollution_DL_models_single_variable as sv

# =====================================================================
# CONFIGURATION & STORAGE
# =====================================================================
# Create dedicated directories
Path("outputs/grid_search_plots_sv").mkdir(parents=True, exist_ok=True)
Path("outputs/grid_search_plots_mv").mkdir(parents=True, exist_ok=True)
Path("outputs/optuna_db").mkdir(parents=True, exist_ok=True)

# The SQLite database is what powers the Optuna Dashboard and enables pausing/resuming
SQLITE_DB_URL = "sqlite:///outputs/optuna_db/hyperparameter_studies.db"

# =====================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# =====================================================================
def objective_sv(
        trial: optuna.Trial,
        df_daily: pd.DataFrame,
        target_col: str,
        research_dict: dict = None,
        categorical_study: bool = True,
) -> float:
    """
    Evaluates a single hyperparameter configuration for the Univariate (SV) architecture.

    Args:
        trial (optuna.Trial): The Optuna trial object generating parameters.
        df_daily (pd.DataFrame): The time series dataset containing the target.
        target_col (str): The column name of the pollutant to forecast.
        research_dict (dict, optional): Custom hyperparameter search space.
        categorical_study (bool): If True, treats research_dict values as lists for categorical choice.
                                  If False, treats them as (min, max, step) tuples for interval search.

    Returns:
        float: The Mean Absolute Percentage Error (MAPE) to be minimized.
    """
    # 1. Fallback: Default Categorical Grid
    if research_dict is None:
        sv.config.NUM_LAYERS = trial.suggest_int("NUM_LAYERS", 2, 6, step=2)
        sv.config.HIDDEN_DIM = trial.suggest_categorical("HIDDEN_DIM", [32, 128, 512])
        sv.config.BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [8, 32, 128])
        sv.config.LOOK_BACK = trial.suggest_categorical("LOOK_BACK", [15, 45, 60])
        sv.config.HORIZON = trial.suggest_categorical("HORIZON", [7, 14, 28])
        sv.config.LEARNING_RATE = trial.suggest_categorical("LEARNING_RATE", [0.001, 0.01, 0.1])

    # 2. Custom Categorical Search
    elif categorical_study:
        sv.config.NUM_LAYERS = trial.suggest_categorical("NUM_LAYERS", research_dict["NUM_LAYERS"])
        sv.config.HIDDEN_DIM = trial.suggest_categorical("HIDDEN_DIM", research_dict["HIDDEN_DIM"])
        sv.config.BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", research_dict["BATCH_SIZE"])
        sv.config.LOOK_BACK = trial.suggest_categorical("LOOK_BACK", research_dict["LOOK_BACK"])
        sv.config.HORIZON = trial.suggest_categorical("HORIZON", research_dict["HORIZON"])
        sv.config.LEARNING_RATE = trial.suggest_categorical("LEARNING_RATE", research_dict["LEARNING_RATE"])

    # 3. Custom Interval Search (Expects Tuple: (low, high, step) for ints, (low, high) for floats)
    else:
        try:
            sv.config.NUM_LAYERS = trial.suggest_int("NUM_LAYERS", *research_dict["NUM_LAYERS"])
            sv.config.HIDDEN_DIM = trial.suggest_int("HIDDEN_DIM", *research_dict["HIDDEN_DIM"])
            sv.config.BATCH_SIZE = trial.suggest_int("BATCH_SIZE", *research_dict["BATCH_SIZE"])
            sv.config.LOOK_BACK = trial.suggest_int("LOOK_BACK", *research_dict["LOOK_BACK"])
            sv.config.HORIZON = trial.suggest_int("HORIZON", *research_dict["HORIZON"])

            # Learning Rate must be a float. We use log=True for proper magnitude scaling.
            sv.config.LEARNING_RATE = trial.suggest_float("LEARNING_RATE",
                                                          research_dict["LEARNING_RATE"][0],
                                                          research_dict["LEARNING_RATE"][1],
                                                          log=True)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[!] Invalid interval research_dict structure. Error: {e}")
            raise optuna.exceptions.TrialPruned()

    # Model Pipeline
    try:
        train_loader, test_loader, scaler, y_test_orig = sv.prepare_data(df_daily, pollutant=target_col)
    except ValueError:
        # Prunes trials where LOOK_BACK + HORIZON > Dataset Size
        raise optuna.exceptions.TrialPruned()

    model = sv.LSTMModel()
    model, _, _ = sv.train_model(
        model, train_loader, test_loader,
        model_name=f"Trial_{trial.number}", patience=30, trial=trial
    )

    metrics, _ = sv.evaluate_model(model, test_loader, scaler, y_test_orig)

    # Visualization
    plot_name = f"Trial{trial.number}_L{sv.config.NUM_LAYERS}_H{sv.config.HIDDEN_DIM}_B{sv.config.BATCH_SIZE}_LR{sv.config.LEARNING_RATE:.4f}"
    fig = sv.predict_and_plot_series(
        model, df_daily, target_col, test_loader, scaler,
        title=plot_name, save_directory="outputs/grid_search_plots_sv", model_name="LSTM"
    )
    plt.close(fig)

    return metrics["MAPE"]


def objective_multivariate(
        trial: optuna.Trial,
        df_daily: pd.DataFrame,
        target_col: str,
        feature_cols: list,
        research_dict: dict = None,
        categorical_study: bool = True
) -> float:
    """
    Evaluates a single hyperparameter configuration for the Multivariate (MV) architecture.
    """
    # 1. Fallback: Default Categorical Grid
    if research_dict is None:
        mv.config.NUM_LAYERS = trial.suggest_int("NUM_LAYERS", 2, 6, step=2)
        mv.config.HIDDEN_DIM = trial.suggest_categorical("HIDDEN_DIM", [32, 128, 512])
        mv.config.BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [8, 32, 128])
        mv.config.LOOK_BACK = trial.suggest_categorical("LOOK_BACK", [15, 45, 60])
        mv.config.HORIZON = trial.suggest_categorical("HORIZON", [7, 14, 28])
        mv.config.LEARNING_RATE = trial.suggest_categorical("LEARNING_RATE", [0.001, 0.01, 0.1])

    # 2. Custom Categorical Search
    elif categorical_study:
        mv.config.NUM_LAYERS = trial.suggest_categorical("NUM_LAYERS", research_dict["NUM_LAYERS"])
        mv.config.HIDDEN_DIM = trial.suggest_categorical("HIDDEN_DIM", research_dict["HIDDEN_DIM"])
        mv.config.BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", research_dict["BATCH_SIZE"])
        mv.config.LOOK_BACK = trial.suggest_categorical("LOOK_BACK", research_dict["LOOK_BACK"])
        mv.config.HORIZON = trial.suggest_categorical("HORIZON", research_dict["HORIZON"])
        mv.config.LEARNING_RATE = trial.suggest_categorical("LEARNING_RATE", research_dict["LEARNING_RATE"])

    # 3. Custom Interval Search
    else:
        try:
            mv.config.NUM_LAYERS = trial.suggest_int("NUM_LAYERS", *research_dict["NUM_LAYERS"])
            mv.config.HIDDEN_DIM = trial.suggest_int("HIDDEN_DIM", *research_dict["HIDDEN_DIM"])
            mv.config.BATCH_SIZE = trial.suggest_int("BATCH_SIZE", *research_dict["BATCH_SIZE"])
            mv.config.LOOK_BACK = trial.suggest_int("LOOK_BACK", *research_dict["LOOK_BACK"])
            mv.config.HORIZON = trial.suggest_int("HORIZON", *research_dict["HORIZON"])
            mv.config.LEARNING_RATE = trial.suggest_float("LEARNING_RATE",
                                                          research_dict["LEARNING_RATE"][0],
                                                          research_dict["LEARNING_RATE"][1],
                                                          log=True)
        except (ValueError, KeyError, TypeError):
            raise optuna.exceptions.TrialPruned()

    # Model Pipeline
    try:
        train_loader, test_loader, scaler, y_test_orig, num_features = mv.prepare_multivariate_data(
            df_daily, target_col=target_col, feature_cols=feature_cols
        )
    except ValueError:
        raise optuna.exceptions.TrialPruned()

    model = mv.LSTMModel(input_dim=num_features)
    model = mv.train_model(
        model, train_loader, test_loader,
        model_name=f"Trial_{trial.number}", patience=30, trial=trial
    )

    # Visualization
    plot_name = f"Trial{trial.number}_L{mv.config.NUM_LAYERS}_H{mv.config.HIDDEN_DIM}_B{mv.config.BATCH_SIZE}_LR{mv.config.LEARNING_RATE:.4f}"

    metrics = mv.evaluate_and_plot(
        model, df_daily, target_col, test_loader, scaler, y_test_orig,
        title=plot_name, save_directory="outputs/grid_search_plots_mv", model_name="LSTM"
    )

    return metrics["MAPE"]

# =====================================================================
# STUDY EXECUTORS
# =====================================================================

def run_optuna_search(
        architecture: str,
        df_daily: pd.DataFrame,
        target_col: str,
        feature_cols: list = None,
        n_trials: int = 50,
        research_dict: dict = None,
        categorical_study: bool = True
):
    """
    Executes the Bayesian Optimization search, saves to SQLite for the Dashboard,
    and exports a Parquet file for offline analysis.

    Args:
        architecture (str): 'SV' for Univariate, 'MV' for Multivariate.
        df_daily (pd.DataFrame): Time series dataframe.
        target_col (str): Target pollutant.
        feature_cols (list, optional): List of features required for 'MV' architecture.
        n_trials (int): Number of models to train.
        research_dict (dict, optional): Custom hyperparameter search space.
        categorical_study (bool): Flag determining how research_dict is interpreted.
    """
    study_name = f"{architecture.upper()}_Optuna_{target_col}"
    print(f"\n{'='*70}\nINITIATING {study_name}\n{'='*70}")
    # 1. Create or Load the SQLite Study
    study = optuna.create_study(
        direction="minimize",
        pruner=HyperbandPruner(),
        study_name=study_name,
        storage=SQLITE_DB_URL,
        load_if_exists=True
    )

    study.set_user_attr("contributors", ["TOLOCZKO ROSS Tiago", "REINOSO URABAYEN Lucas"])
    study.set_user_attr("dataset", "GEOD'AIR pollution data and ERA5-Land data")
    # 2. Run the Optimization
    if architecture.lower() == 'sv':
        study.optimize(lambda trial: objective_sv(
            trial, df_daily, target_col,
            research_dict=research_dict,
            categorical_study=categorical_study),
                       n_trials=n_trials
                       )
    elif architecture.lower() == 'mv':
        study.optimize(lambda trial: objective_multivariate(
            trial, df_daily, target_col, feature_cols,
            research_dict=research_dict,
            categorical_study=categorical_study),
                       n_trials=n_trials
                       )
    else:
        raise ValueError("Architecture must be 'SV' or 'MV'.")

    print("\n" + "="*70)
    print(f"[✓] OPTUNA SEARCH COMPLETE FOR {study_name}!")
    print(f"Best Trial: #{study.best_trial.number} | MAPE: {study.best_value:.2f}%")
    print("="*70 + "\n")

    # 3. Export to Parquet for offline Pandas workflows
    df_results = study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'duration'))
    df_results.columns = [col.replace('params_', '') for col in df_results.columns]
    df_results['Time_Seconds'] = df_results['duration'].dt.total_seconds().round(2)
    df_results.rename(columns={'value': 'MAPE', 'number': 'Experiment_ID'}, inplace=True)

    parquet_path = f"outputs/optuna_db/{study_name}_results.parquet"
    df_results.to_parquet(parquet_path, engine="pyarrow")
    print(f"[✓] Offline data exported to: {parquet_path}")

    return study

# =====================================================================
# NATIVE OPTUNA ANALYSIS
# =====================================================================

def analyze_optuna_study(study_name: str, target_horizon: int = None):
    """
    Loads an existing Optuna study from the SQLite database, prints the best parameters,
    and generates static Matplotlib evaluation charts.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=SQLITE_DB_URL)
    except Exception as e:
        print(f"[!] Could not load study '{study_name}'. Ensure the database exists. ({e})")
        return

    print(f"\n{'='*70}\nNATIVE BAYESIAN ANALYSIS: {study_name}\n{'='*70}")

    if target_horizon:
        trials = [t for t in study.trials if t.params.get("HORIZON") == target_horizon and t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            print(f"[!] No completed trials found for Horizon {target_horizon}.")
            return
        best_trial = min(trials, key=lambda t: t.value)
        print(f"Best Empirical Trial for Horizon {target_horizon}: #{best_trial.number} (MAPE: {best_trial.value:.2f}%)")
        print("Parameters:", best_trial.params)
    else:
        print(f"Best Overall Empirical Trial: #{study.best_trial.number} (MAPE: {study.best_value:.2f}%)")
        print("Parameters:", study.best_trial.params)

    plot_dir = Path("outputs/optuna_db/analysis_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig_hist = vis_mpl.plot_optimization_history(study)
    fig_hist.set_title(f"Optimization History ({study_name})")
    fig_hist.figure.savefig(plot_dir / f"{study_name}_history.png", bbox_inches='tight', dpi=150)
    plt.close(fig_hist.figure)

    fig_param = vis_mpl.plot_param_importances(study)
    fig_param.set_title(f"Hyperparameter Importance ({study_name})")
    fig_param.figure.savefig(plot_dir / f"{study_name}_importance.png", bbox_inches='tight', dpi=150)
    plt.close(fig_param.figure)

    print(f"[✓] Static analysis plots saved to {plot_dir}")
    print("\n" + "*"*70)
    print("🚀 TO OPEN THE INTERACTIVE DASHBOARD, RUN THIS IN YOUR TERMINAL:")
    print(f"optuna-dashboard sqlite:///outputs/optuna_db/hyperparameter_studies.db")
    print("*"*70 + "\n")


# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    target_pollutant = "PM25"

    # Example of custom categorical research dictionary
    CATEGORICAL_GRID = {
        "NUM_LAYERS": [2, 4],
        "HIDDEN_DIM": [64, 128],
        "BATCH_SIZE": [32, 64],
        "LOOK_BACK": [15, 30],
        "HORIZON": [7, 14],
        "LEARNING_RATE": [0.001, 0.005]
    }

    # Example of an interval research dictionary (low, high, step)
    # NOTE: LEARNING_RATE must be (low, high) without a step since it's a float!
    INTERVAL_GRID = {
        "NUM_LAYERS": (2, 6, 2),
        "HIDDEN_DIM": (32, 256, 32),
        "BATCH_SIZE": (16, 128, 16),
        "LOOK_BACK": (10, 60, 5),
        "HORIZON": (7, 28, 7),
        "LEARNING_RATE": (0.0001, 0.1)
    }

    df_mv = pd.read_parquet("outputs/rnn_multivariate_dataset.parquet")
    if "Timestamp" in df_mv.columns: df_mv = df_mv.set_index("Timestamp")
    df_mv.index = pd.to_datetime(df_mv.index)
    df_mv = df_mv.sort_index().interpolate(method='linear').dropna()

    wanted_features = ["sp", "u10", "v10"]
    feature_cols = [col for col in (wanted_features + [target_pollutant]) if col in df_mv.columns]

    # --- RUN OPTIMIZATIONS ---

    # Run Univariate with Categorical grid
    run_optuna_search("SV", df_mv[[target_pollutant]], target_pollutant,
                      n_trials=5, research_dict=CATEGORICAL_GRID, categorical_study=True)

    # Run Multivariate with Interval grid
    # run_optuna_search("MV", df_mv[feature_cols], target_pollutant, feature_cols,
    #                   n_trials=50, research_dict=INTERVAL_GRID, categorical_study=False)

    # --- ANALYZE EXISTING STUDIES ---
    analyze_optuna_study(study_name=f"SV_Optuna_{target_pollutant}")