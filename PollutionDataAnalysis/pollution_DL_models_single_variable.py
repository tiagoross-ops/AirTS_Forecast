"""
URBAN POLLUTION FORECASTING — TUTORIAL FOR BEGINNERS
Part 3 : Neural Networks for Time Series (Univariate)

Description:
    Core engine for training, evaluating, and visualizing Univariate (Single Variable)
    Deep Learning models. Contains standard PyTorch architectures (RNN, LSTM, Bi-LSTM, GRU)
    and a master orchestrator to load optimized parameters and train the final models.
"""

import os
import json
import pickle
import copy
import logging
import warnings
from pathlib import Path
from typing import Any, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Required strictly for early-stopping exceptions during hyperparameter searches
import optuna

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Create outputs folder
OUTPUT_DIR = Path("outputs")
SV_DIR = OUTPUT_DIR / "SV DL pred"

os.makedirs("outputs/plots", exist_ok=True)

#--------------------------------------------------------------------
# DEVICE CONFIGURATION
#--------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("[✓] Using GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[✓] Using Apple Silicon (MPS)")
else:
    device = torch.device("cpu")
    print("[!] Using CPU")


#--------------------------------------------------------------------
# CONFIGURATION CLASS
#--------------------------------------------------------------------
class Config:
    """
    Global configuration state for the Univariate models.
    These values act as defaults and are dynamically overwritten by the
    JSON file during the final optimized inference loop.
    """
    LOOK_BACK: int = 14
    HORIZON: int = 14
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    PATIENCE: int = 50
    HIDDEN_DIM: int = 256
    NUM_LAYERS: int = 2
    TEST_FRACTION: float = 0.2


config = Config()


#--------------------------------------------------------------------
# HELPER FUNCTIONS & DATA PREP
#--------------------------------------------------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Percentage Error.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The MAPE score as a percentage.
    """
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def make_sequences(data: np.ndarray, look_back: int = None, horizon: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slices a continuous 1D time series into overlapping X (input) and y (target) sequences.

    Args:
        data (np.ndarray): The continuous series to be sliced.
        look_back (int, optional): Number of historical steps to look at. Defaults to config.
        horizon (int, optional): Number of future steps to predict. Defaults to config.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X features matrix and y target matrix.
    """
    look_back = look_back if look_back is not None else config.LOOK_BACK
    horizon = horizon if horizon is not None else config.HORIZON

    data = np.asarray(data)
    X, y = [], []
    for i in range(len(data) - look_back - horizon + 1):
        X.append(data[i: i + look_back])
        y.append(data[i + look_back: i + look_back + horizon])
    return np.array(X), np.array(y)


class PollutionDataset(Dataset):
    """PyTorch Dataset wrapper optimized for converting numpy arrays to float tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def prepare_data(df: pd.DataFrame, pollutant: str, scaler: MinMaxScaler = None, test_fraction: float = None) -> Tuple[
    DataLoader, DataLoader, MinMaxScaler, np.ndarray]:
    """
    End-to-end data preparation pipeline for Univariate forecasting.
    Scales the data, splits chronologically, and converts into PyTorch DataLoaders.

    Args:
        df (pd.DataFrame): The raw chronological dataframe.
        pollutant (str): The specific column name to forecast.
        scaler (MinMaxScaler, optional): An existing scaler to apply. If None, fits a new one.
        test_fraction (float, optional): Ratio of data to reserve for testing. Defaults to config.

    Returns:
        Tuple: (train_loader, test_loader, fitted_scaler, raw_y_test_array)
    """
    test_fraction = test_fraction if test_fraction is not None else config.TEST_FRACTION
    data = df[pollutant].values.astype(np.float32)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    else:
        data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()

    X, y = make_sequences(data_scaled)
    if len(X) == 0:
        raise ValueError("Dataset is too small for this LOOK_BACK + HORIZON configuration.")

    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    _, y_test_orig = make_sequences(data)
    y_test_orig = y_test_orig[split_idx:]

    train_loader = DataLoader(PollutionDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(PollutionDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, scaler, y_test_orig


# =====================================================================
# MODELS (RNN, LSTM, Bi-LSTM, GRU)
# =====================================================================
class RNNModel(nn.Module):
    """Standard Recurrent Neural Network Architecture."""

    def __init__(self, input_dim: int = 1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                          batch_first=True)
        self.fc = nn.Linear(config.HIDDEN_DIM, config.HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.rnn(x.unsqueeze(-1))
        return self.fc(hidden[-1])


class LSTMModel(nn.Module):
    """Long Short-Term Memory Architecture."""

    def __init__(self, input_dim: int = 1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                            batch_first=True)
        self.fc = nn.Linear(config.HIDDEN_DIM, config.HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, cell) = self.lstm(x.unsqueeze(-1))
        return self.fc(hidden[-1])


class BiLSTMModel(nn.Module):
    """Bidirectional Long Short-Term Memory Architecture."""

    def __init__(self, input_dim: int = 1):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(config.HIDDEN_DIM * 2, config.HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, cell) = self.lstm(x.unsqueeze(-1))
        # Concatenate the final forward state and final backward state
        return self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))


class GRUModel(nn.Module):
    """Gated Recurrent Unit Architecture."""

    def __init__(self, input_dim: int = 1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                          batch_first=True)
        self.fc = nn.Linear(config.HIDDEN_DIM, config.HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.gru(x.unsqueeze(-1))
        return self.fc(hidden[-1])


# =====================================================================
# CORE TRAINING LOOP
# =====================================================================
def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, model_name: str = "Model",
                trial: optuna.Trial = None) -> Tuple[nn.Module, list, list]:
    """
    Standard Backpropagation/Gradient Descent training loop with early stopping.

    Args:
        model (nn.Module): The instantiated PyTorch model to train.
        train_loader (DataLoader): DataLoader containing training batches.
        test_loader (DataLoader): DataLoader containing validation batches.
        model_name (str): Identifier used for print logs.
        trial (optuna.Trial, optional): Passed exclusively during hyperparameter searches
                                        to allow Optuna to prune unpromising models early.

    Returns:
        Tuple: The trained model with best weights restored, list of train losses, list of val losses.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_losses, val_losses = [], []
    best_val_loss, patience_counter = np.inf, 0
    best_model_state = copy.deepcopy(model.state_dict())

    print(f"\n[>] Training {model_name}...")
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                val_loss += criterion(model(X.to(device)), y.to(device)).item()
        val_loss /= len(test_loader)

        # Optuna Pruning Hook
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f" [!] Trial pruned by Optuna at epoch {epoch + 1}.")
                raise optuna.exceptions.TrialPruned()

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f" [!] Early stopping triggered at epoch {epoch + 1}.")
            break

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses


# =====================================================================
# EVALUATION & VISUALIZATION
# =====================================================================
def evaluate_model(model: nn.Module, test_loader: DataLoader, scaler: MinMaxScaler, y_test_orig: np.ndarray) -> Tuple[
    Dict[str, float], np.ndarray]:
    """
    Evaluates the model on the test set, inverse-transforms the predictions,
    and calculates standard forecasting metrics.
    """
    model.eval()
    y_pred_all = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_pred_all.append(model(X_batch.to(device)).cpu().numpy())

    y_pred_orig = scaler.inverse_transform(np.vstack(y_pred_all).flatten().reshape(-1, 1)).flatten()
    y_true_orig = y_test_orig.flatten()

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))),
        "MAE": float(mean_absolute_error(y_true_orig, y_pred_orig)),
        "MAPE": mape(y_true_orig, y_pred_orig)
    }
    return metrics, y_pred_orig


def predict_and_plot_series(
        model: nn.Module,
        df_daily: pd.DataFrame,
        pollutant: str,
        test_loader: DataLoader,
        scaler: MinMaxScaler,
        model_name: str,
        title: str,
        save_directory: str
) -> plt.Figure:
    """
    Generates and exports a line chart comparing the model's test predictions against reality.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds.append(model(X_batch.to(device)).cpu().numpy()[:, 0])

    predictions_real = scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()

    # Calculate index alignment for the plot
    total_sequences = len(df_daily) - config.LOOK_BACK - config.HORIZON + 1
    split_idx = int(total_sequences * (1 - config.TEST_FRACTION))
    start_date_idx = split_idx + config.LOOK_BACK

    pred_dates = df_daily.index[start_date_idx: start_date_idx + len(predictions_real)]

    fig = plt.figure(figsize=(14, 6))
    plt.plot(df_daily.index, df_daily[pollutant], color="lightgray", label="Observed Data", alpha=0.8)
    plt.plot(pred_dates, df_daily[pollutant].loc[pred_dates], color="black", label="True Future", linewidth=1.5)
    plt.plot(pred_dates, predictions_real, color="red", label=f"{model_name} Forecast", linewidth=2.0)
    plt.axvline(pred_dates[0], color="blue", linestyle="--", alpha=0.6, label="Test Split")

    plt.title(title.replace('_', ' '), fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_directory, f"{title}.png"), dpi=150)
    return fig


# =====================================================================
# THE MASTER ORCHESTRATOR
# =====================================================================
def model_looping(
        df_daily: pd.DataFrame,
        pollutants: tuple = ("NO2", "NOx", "O3", "PM10", "PM25"),
        optimized: bool = False,
        hyperparameter_file: Path = Path("outputs/best_parameters.json")
) -> Dict[str, Any]:
    """
    Unified execution loop representing the final inference step of the ML pipeline.
    It iterates through all specified pollutants, safely loads their optimized hyperparameters
    (if available), trains all four baseline deep learning models, evaluates them,
    and dumps a consolidated Pickle file for the comparison module.

    Args:
        df_daily (pd.DataFrame): Daily resampled dataset.
        pollutants (tuple): List of targets to loop over.
        optimized (bool): If True, forces the script to override defaults using the JSON file.
        hyperparameter_file (Path): Path to the JSON containing Optuna's best results.

    Returns:
        Dict: A master dictionary containing performance metrics for every model across all pollutants.
    """
    master_results = {"rnn_results": {}, "lstm_results": {}, "bilstm_results": {}, "gru_results": {}}

    # Define suite mapping to streamline the training loop
    models_to_test = {
        "RNN": (RNNModel, master_results["rnn_results"]),
        "LSTM": (LSTMModel, master_results["lstm_results"]),
        "Bi-LSTM": (BiLSTMModel, master_results["bilstm_results"]),
        "GRU": (GRUModel, master_results["gru_results"])
    }

    # Load Hyperparameters safely into memory
    optuna_params = {}
    if optimized and hyperparameter_file.exists():
        try:
            with open(hyperparameter_file, "r", encoding="utf-8") as f:
                optuna_params = json.load(f)
            logging.info(f"[✓] Loaded Optuna JSON: {hyperparameter_file.name}")
        except json.JSONDecodeError:
            logging.warning("[!] Corrupted JSON detected. Defaulting to standard config.")

    # Execute training sweep
    for pollutant in pollutants:
        if pollutant not in df_daily.columns: continue
        print(f"\n{'=' * 70}\nTARGET: {pollutant}\n{'=' * 70}")

        for model_name, (ModelClass, results_dict) in models_to_test.items():

            # 1. Inject Hyperparameters
            if optimized:
                json_key = f"SV-{model_name}"
                if pollutant in optuna_params and json_key in optuna_params[pollutant]:
                    p = optuna_params[pollutant][json_key]["Hyperparameters"]
                    config.LOOK_BACK = p.get("LOOK_BACK", config.LOOK_BACK)
                    config.HORIZON = p.get("HORIZON", config.HORIZON)
                    config.BATCH_SIZE = p.get("BATCH_SIZE", config.BATCH_SIZE)
                    config.LEARNING_RATE = p.get("LEARNING_RATE", config.LEARNING_RATE)
                    config.HIDDEN_DIM = p.get("HIDDEN_DIM", config.HIDDEN_DIM)
                    config.NUM_LAYERS = p.get("NUM_LAYERS", config.NUM_LAYERS)
                    print(f" [⚙] Injected Optimized Parameters for {model_name}")
                else:
                    print(f" [-] No optimized params found for {json_key}. Using defaults.")

            # 2. Reslice data based on (potentially) modified LOOK_BACK and BATCH_SIZE
            try:
                train_loader, test_loader, scaler, y_test_orig = prepare_data(df_daily, pollutant)
            except ValueError as e:
                print(f" [!] Skipping {model_name}: {e}")
                continue

            # 3. Model Training
            model = ModelClass()
            model, _, _ = train_model(model, train_loader, test_loader, model_name=f"{model_name} ({pollutant})")

            # 4. Evaluation & Visualization
            metrics, _ = evaluate_model(model, test_loader, scaler, y_test_orig)
            results_dict[pollutant] = metrics
            print(
                f" [✓] {model_name} Metrics: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MAPE={metrics['MAPE']:.2f}%\n")

            fig = predict_and_plot_series(
                model, df_daily, pollutant, test_loader, scaler,
                title=f"{model_name} Prediction - {pollutant}",
                save_directory=str(SV_DIR/pollutant), model_name=model_name
            )

            # Ensures that figures render safely inside Jupyter Notebooks if returned, but close to save RAM in standard console runs
            if __name__ == "__main__":
                plt.close(fig)

    # 5. Export finalized metrics to Pickle for Part 4 Comparison
    with open("outputs/sv_nn_results.pkl", "wb") as f:
        pickle.dump(master_results, f)
    print("\n[✓] All SV Neural Network results compiled and exported!")

    return master_results


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("Loading data from Part 1...")
    try:
        df = pd.read_parquet(Path("outputs/consolidated_pollutants.parquet"))
        if "Timestamp" in df.columns:
            df = df.set_index("Timestamp")
        elif "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df_daily = df.resample("D").mean().interpolate(method='linear')
    except FileNotFoundError:
        print("[!] Dataset not found.")
        exit()

    # Pass optimized=True to automatically seek out and inject Optuna JSON parameters!
    model_looping(
        df_daily=df_daily,
        pollutants=("NO2", "NOx", "PM10", "PM25", "O3"),
        optimized=True
    )
