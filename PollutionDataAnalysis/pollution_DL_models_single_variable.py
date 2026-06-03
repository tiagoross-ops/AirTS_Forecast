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

os.makedirs(SV_DIR, exist_ok=True)

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
    PATIENCE: int = 30
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


def r_2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the R-squared (Coefficient of Determination) metric.

    Theory of Learning:
    R^2 measures how much better the model is compared to a naive baseline
    that simply predicts the mean of the target. An R^2 of 1.0 is perfect,
    0.0 is equal to the baseline, and < 0 means the model is actively failing.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The R-squared ratio.
    """
    # SS_res: The squared error of our Neural Network
    ss_res = float(np.sum((y_true - y_pred) ** 2))

    # SS_tot: The squared error of a naive "mean" baseline
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    # Failsafe: Prevent ZeroDivisionError if the true data has absolutely zero variance
    # (e.g., a perfectly flat line of true values).
    if ss_tot == 0:
        return 0.0

    return float(1 - (ss_res / ss_tot))


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


class PinballLoss(nn.Module):
    """
    Custom Asymmetric Cost Function: Pinball (Quantile) Loss.

    Theory of Learning:
    By shifting the quantile (q) away from 0.5, we force the network to over-predict
    or under-predict. A quantile of 0.90 heavily penalizes under-predictions,
    making it ideal for catching dangerous, anomalous pollution spikes.
    """
    def __init__(self, quantile: float = 0.90):
        super(PinballLoss, self).__init__()
        # Ensure the quantile is mathematically valid (between 0 and 1)
        if not (0 < quantile < 1):
            raise ValueError("Quantile must be strictly between 0 and 1.")
        self.q = quantile

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Step 1: Calculate the raw distance between reality and the prediction
        errors = targets - predictions

        # Step 2: Apply the asymmetric penalty
        # If Target > Prediction (Under-prediction): error is positive -> multiplied by q (e.g., 0.9)
        # If Target < Prediction (Over-prediction): error is negative -> multiplied by (q-1) (e.g., -0.1)
        # torch.max ensures we take the mathematically positive penalty value
        loss = torch.max((self.q - 1) * errors, self.q * errors)

        # Step 3: Return the mean loss across the entire batch for the optimizer
        return torch.mean(loss)


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


class HybridRNNLSTMModel(nn.Module):
    """Hybrid Architecture combining an RNN feature extractor with LSTM memory."""

    def __init__(self, input_dim: int = 1):
        super(HybridRNNLSTMModel, self).__init__()

        # 1. The Input Layer: Standard RNN
        # We restrict num_layers=1 here so it acts as a single transitional layer.
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True
        )

        # 2. The Deep Layer: LSTM
        # Notice the input_size is now config.HIDDEN_DIM (the output size of the RNN layer)
        self.lstm = nn.LSTM(
            input_size=config.HIDDEN_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True
        )

        # 3. The Output Layer
        self.fc = nn.Linear(config.HIDDEN_DIM, config.HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 0: Prepare input shape (Batch, Sequence Length, Features)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        # Step 1: Pass data through the RNN
        # rnn_out contains the hidden states for EVERY step in the sequence.
        # rnn_hidden is just the very last state. We want the full sequence.
        rnn_out, rnn_hidden = self.rnn(x)

        # Step 2: Feed the RNN's full sequential output into the LSTM
        lstm_out, (lstm_hidden, lstm_cell) = self.lstm(rnn_out)

        # Step 3: Extract the final hidden state of the LSTM for the fully connected layer
        return self.fc(lstm_hidden[-1])


class CNNModel(nn.Module):
    """
    1D Convolutional Neural Network Architecture for Time Series.
    Uses an Adaptive Pool to gracefully handle dynamic Optuna Look-Back sequences.
    """
    def __init__(self, input_dim: int = 1):
        super(CNNModel, self).__init__()

        # Dynamically build layers based on Optuna's NUM_LAYERS config
        layers = []
        in_channels = input_dim

        # Deep Convolutional Stacking
        # We multiply NUM_LAYERS * 4 to rapidly expand the Receptive Field.
        # By stacking standard 3-kernel convolutions without internal pooling,
        # the network mathematically "sees" further back in the sequence timeline
        # with every subsequent layer, capturing deep macro-trends.
        for i in range(config.NUM_LAYERS * 4):

            # The final conv layer outputs exactly the HIDDEN_DIM requested by Optuna.
            # Intermediate layers use a fixed 64 channels to prevent parameter explosion.
            out_channels = config.HIDDEN_DIM if i == (config.NUM_LAYERS * 4 - 1) else 64

            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1 # Padding=1 ensures the sequence length doesn't shrink prematurely
            ))
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv_block = nn.Sequential(*layers)

        # Adaptive pooling squashes whatever sequence length is left into a single value per channel
        # This makes the model crash-proof against Optuna changing the LOOK_BACK!
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Final projection to our forecasting horizon
        self.fc = nn.Linear(config.HIDDEN_DIM, config.HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If Univariate, x comes in as [Batch, Seq_Len]. We need it to be 3D.
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        # CRITICAL FIX: Transpose from (Batch, Seq_Len, Features) to (Batch, Features, Seq_Len)
        x = x.transpose(1, 2)

        # Pass through the Convolutional filters
        x = self.conv_block(x)

        # Pool across the time dimension: (Batch, Hidden_Dim, Seq_Len) -> (Batch, Hidden_Dim, 1)
        x = self.global_pool(x)

        # Squeeze the final dimension to match the Linear layer: -> (Batch, Hidden_Dim)
        x = x.squeeze(-1)

        return self.fc(x)


class HybridCNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM Architecture for Time-Series Forecasting.
    Upgraded to combat "Peak Attenuation" by using Dilated Convolutions
    instead of Max Pooling, preserving the sharp pollution spikes for the LSTM.
    """
    def __init__(self, input_dim: int = 1):
        super(HybridCNNLSTMModel, self).__init__()

        # ==========================================
        # 1. The CNN Block (Feature Extraction)
        # ==========================================
        # We use a fixed 64 channels. The CNN acts as a localized pattern scanner
        # (e.g., finding immediate correlations between wind drops and PM2.5 spikes).
        cnn_out_channels = 64

        # DILATED CONVOLUTION UPGRADE
        # dilation=2 forces the kernel to skip adjacent days, widening its "view"
        # of the macro-trend without deleting data points.
        # Math for keeping sequence length intact: padding = dilation * (kernel_size - 1) / 2
        # padding = 2 * (3 - 1) / 2 = 2
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_out_channels,
            kernel_size=3,
            padding=2,
            dilation=2
        )
        self.relu = nn.ReLU()

        # [CRITICAL ML-OPS FIX]
        # self.pool = nn.MaxPool1d(kernel_size=2) has been DELETED.
        # We do not want to compress the timeline. We want the LSTM to see the
        # exact, uncompressed sequence of features to catch anomalous spikes.

        # ==========================================
        # 2. The LSTM Block (Temporal Processing)
        # ==========================================
        # The LSTM takes the CNN's 64 complex feature maps as its input.
        # Because we removed pooling, the LSTM gets to process the full Look-Back timeline.
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True
        )

        # ==========================================
        # 3. The Output Block
        # ==========================================
        # Projects the final memory state into our required forecasting horizon
        self.fc = nn.Linear(config.HIDDEN_DIM, config.HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass handling the strict dimensional requirements of PyTorch CNNs and LSTMs.
        """
        # [FAILSAFE]: Safely convert 2D Univariate inputs to 3D.
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        # STEP 1: Dimensionality Shift for CNN -> [Batch, Channels(Features), Sequence_Length]
        x = x.transpose(1, 2)

        # STEP 2: CNN Feature Extraction
        # Data passes through the dilated filter. The sequence length is NOT halved.
        x = self.conv1d(x)
        x = self.relu(x)

        # STEP 3: Dimensionality Shift for LSTM -> [Batch, Sequence_Length, Channels(Features)]
        x = x.transpose(1, 2)

        # STEP 4: LSTM Temporal Processing
        # The LSTM tracks how the CNN's extracted patterns evolve over the full sequence.
        lstm_out, (hidden, cell) = self.lstm(x)

        # STEP 5: Final Prediction
        # hidden[-1] extracts the final memory state from the deepest layer of the LSTM
        out = self.fc(hidden[-1])

        return out
# =====================================================================
# CORE TRAINING LOOP
# =====================================================================
def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_name: str = "Model",
        trial: optuna.Trial = None,
        verbose: bool = False,
        loss_type: str = "MSE",  # <--- NEW PARAMETER
        quantile: float = 0.90        # <--- DYNAMIC PINBALL BIAS
) -> Tuple[nn.Module, list, list]:

    model = model.to(device)

    # ==========================================
    # DYNAMIC COST FUNCTION SELECTION
    # ==========================================
    if loss_type.upper() == "MSE":
        criterion = nn.MSELoss()
    elif loss_type.upper() == "MAE":
        criterion = nn.L1Loss()
    elif loss_type.upper() == "HUBER":
        # delta determines where it switches from MSE to MAE.
        # 1.0 is standard for scaled (0-1) data.
        criterion = nn.HuberLoss(delta=1.0)
    elif loss_type.upper() == "PINBALL":
        criterion = PinballLoss(quantile=quantile)
    else:
        raise ValueError(f"Unsupported Loss Function: {loss_type}")



    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_losses, val_losses = [], []
    best_val_loss, patience_counter = np.inf, 0
    best_model_state = copy.deepcopy(model.state_dict())

    if (__name__ == '__main__') & (trial is None): print(f"\n[>] Training {model_name}...")

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

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                if verbose:
                    print(f" [!] Trial pruned by Optuna at epoch {epoch + 1}.")
                raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_model_state = copy.deepcopy(model.state_dict())
            if verbose and ((epoch + 1) % 10 == 0 or epoch < 5):
                print(f" Epoch {epoch + 1:3d}/{config.EPOCHS}: Train={train_loss:.5f}, Val={val_loss:.5f} [NEW BEST]")
        else:
            patience_counter += 1
            if verbose and ((epoch + 1) % 10 == 0):
                print(f" Epoch {epoch + 1:3d}/{config.EPOCHS}: Train={train_loss:.5f}, Val={val_loss:.5f} (Patience {patience_counter}/{config.PATIENCE})")

        if (patience_counter >= config.PATIENCE) & verbose:
            print(f" [!] Early stopping triggered at epoch {epoch + 1}.")
            break

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses


# =====================================================================
# EVALUATION & VISUALIZATION
# =====================================================================
def evaluate_model(
        model: nn.Module, test_loader: DataLoader, scaler: MinMaxScaler, y_test_orig: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
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
        "MAPE": mape(y_true_orig, y_pred_orig),
        "R^2": r_2(y_true_orig, y_pred_orig)
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
    os.makedirs(save_directory, exist_ok=True)
    predictions_real = scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()

    # Calculate index alignment for the plot
    total_sequences = len(df_daily) - config.LOOK_BACK - config.HORIZON + 1
    split_idx = int(total_sequences * (1 - config.TEST_FRACTION))
    start_date_idx = split_idx + config.LOOK_BACK

    pred_dates = df_daily.index[start_date_idx: start_date_idx + len(predictions_real)]

    fig = plt.figure(figsize=(14, 6))
    # plt.plot(df_daily.index, df_daily[pollutant], color="lightgray", label="Observed Data", alpha=0.8)
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
        hyperparameter_file: Path = Path("outputs/best_parameters.json"),
        verbose: bool = False,
        loss_type: str = "MSE",  # <--- NEW PARAMETER
        quantile: float = 0.90        # <--- DYNAMIC PINBALL BIAS
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
        hyperparameter_file (Path): Path to the JSON containing Optuna's best results. Default is
            `Path("outputs/best_parameters.json")` native from `run_optuna_search_loop` function

    Returns:
        Dict: A master dictionary containing performance metrics for every model across all pollutants.
    """
    master_results = {"rnn_results": {}, "lstm_results": {}, "bilstm_results": {}, "gru_results": {},
                      "hy_rnn_lstm_results": {}, "cnn_results": {}}

    # Define suite mapping to streamline the training loop
    models_to_test = {
        "RNN": (RNNModel, master_results["rnn_results"]),
        "LSTM": (LSTMModel, master_results["lstm_results"]),
        "Bi-LSTM": (BiLSTMModel, master_results["bilstm_results"]),
        "GRU": (GRUModel, master_results["gru_results"]),
        "Hybrid-RNN-LSTM": (HybridRNNLSTMModel, master_results["hy_rnn_lstm_results"]),
        "CNN": (CNNModel, master_results["cnn_results"]),
        "Hybrid-CNN-LSTM":(HybridCNNLSTMModel, master_results["hy_rnn_lstm_results"])
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
                    p = optuna_params[pollutant][json_key]["Balanced_Hyperparameters"]
                    config.LOOK_BACK = p.get("LOOK_BACK", config.LOOK_BACK)
                    config.HORIZON = p.get("HORIZON", config.HORIZON)
                    config.BATCH_SIZE = p.get("BATCH_SIZE", config.BATCH_SIZE)
                    config.LEARNING_RATE = p.get("LEARNING_RATE", config.LEARNING_RATE)
                    config.HIDDEN_DIM = p.get("HIDDEN_DIM", config.HIDDEN_DIM)
                    config.NUM_LAYERS = p.get("NUM_LAYERS", config.NUM_LAYERS)
                    print(f" [⚙] Injected Optimized Parameters for {model_name}")
                    for hp in p:
                        print(hp, ": ", p.get(hp))
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
            model, _, _ = train_model(model, train_loader, test_loader, model_name=f"{model_name} ({pollutant})",
                                      verbose=verbose, loss_type=loss_type, quantile=quantile)

            # 4. Evaluation & Visualization
            metrics, _ = evaluate_model(model, test_loader, scaler, y_test_orig)
            results_dict[pollutant] = metrics

            # Print format updated: R^2 is kept as a raw ratio/decimal, not a percentage
            print(
                f" [✓] {model_name} Metrics: RMSE={metrics['RMSE']:.2f}, "
                f"MAE={metrics['MAE']:.2f}, MAPE={metrics['MAPE']:.2f}%, "
                f"R^2={metrics['R^2']:.3f}\n"
            )

            fig = predict_and_plot_series(
                model, df_daily, pollutant, test_loader, scaler,
                title=f"{model_name} Prediction - {pollutant}",
                save_directory=str(SV_DIR/pollutant), model_name=model_name
            )

            # Ensures that figures render safely inside Jupyter Notebooks if returned, but close to save RAM in standard console runs
            if __name__ == "__main__":
                plt.close(fig)

    # 5. Export finalized metrics to Pickle for Part 4 Comparison
    with open("outputs/sv_DL_results.pkl", "wb") as f:
        pickle.dump(master_results, f)
    print("\n[✓] All SV Neural Network results compiled and exported!")

    return master_results

def find_best_models(
        master_results: Dict[str, Dict[str, Dict[str, float]]],
        target_metric: str = "RMSE"
) -> Dict[str, Dict[str, Any]]:
    """
    Parses the master results dictionary to programmatically identify the best
    performing architecture for each individual pollutant.

    Theory of Learning Application:
    This acts as the automated "Model Selection" phase. It dynamically adjusts
    its selection logic: it seeks to MINIMIZE absolute/percentage errors (RMSE, MAPE)
    but knows to MAXIMIZE correlation/variance metrics (R^2).

    Args:
        master_results (Dict): The nested dictionary returned by model_looping.
        target_metric (str): The metric used to rank the models (e.g., 'RMSE', 'MAPE', 'R^2').

    Returns:
        Dict: A dictionary mapping each pollutant to its champion model and score.
    """
    # Clean mapping from your internal dictionary keys to readable print names
    model_name_mapping = {
        "rnn_results": "RNN",
        "lstm_results": "LSTM",
        "bilstm_results": "Bi-LSTM",
        "gru_results": "GRU",
        "hy_rnn_lstm_results": "Hybrid-RNN-LSTM",
        "cnn_results": "CNN",
        "hy_cnn_lstm_results": "Hybrid-CNN-LSTM"
    }

    print(f"\n{'='*80}\n🏆 OVERALL CHAMPIONS PER POLLUTANT (Ranked by MAPE, Tie-breaker: R^2)\n{'='*80}")

    # 1. Dynamically extract all unique pollutants that were successfully tested
    all_pollutants = set()
    for model_data in master_results.values():
        all_pollutants.update(model_data.keys())

    champions = {}

    # 2. Loop through each pollutant to find its specific champion
    for pol in sorted(list(all_pollutants)):
        best_model_key = "None"

        # [ML-Ops Failsafe]: We initialize the best score as a tuple.
        # Infinity for MAPE (worst possible error) and Infinity for inverted R^2
        best_score = (float('inf'), float('inf'))
        best_metrics = {}

        # Compare every model's performance on this specific pollutant
        for model_key, pol_data in master_results.items():
            if pol in pol_data:
                mape_score = pol_data[pol].get("MAPE")
                r2_score = pol_data[pol].get("R^2")

                if mape_score is not None and r2_score is not None:
                    # [Theory of Learning]: We create an Evaluation Tuple.
                    # Python compares tuples element by element.
                    # We use negative R^2 (-r2_score) so that a higher R^2 becomes a smaller
                    # mathematical number. This allows us to use a simple '<' for both metrics!
                    current_score = (mape_score, -r2_score)

                    # Check if this model's tuple beats the reigning champion
                    if current_score < best_score:
                        best_score = current_score
                        best_model_key = model_key
                        best_metrics = pol_data[pol]

        # 3. Save and format the champion output
        readable_name = model_name_mapping.get(best_model_key, best_model_key)

        champions[pol] = {
            "Model": readable_name,
            "MAPE": best_metrics.get("MAPE", 0),
            "R^2": best_metrics.get("R^2", 0)
        }

        # 4. Print the final cross-examined metrics
        print(
            f" -> {pol:<5} : {readable_name:<16} | "
            f"MAPE: {best_metrics.get('MAPE'):>6.2f}% | "
            f"R^2: {best_metrics.get('R^2'):>5.3f}"
        )

    print("="*80 + "\n")
    return champions

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
    output = Path("outputs/best_parameters_pinball.json")
    # Pass optimized=True to automatically seek out and inject Optuna JSON parameters!
    final_metrics_dict = model_looping(
        df_daily=df_daily,
        pollutants=("O3", "NO2", "PM25"),
        optimized=True,
        verbose=True,
        loss_type="MSE"
    )
    best_models = find_best_models(final_metrics_dict, target_metric="RMSE")
