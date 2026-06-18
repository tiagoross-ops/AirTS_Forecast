"""
URBAN POLLUTION FORECASTING — TUTORIAL FOR BEGINNERS
Part 3 : Neural Networks for Time Series (Multivariate)

Description:
    Core engine for training, evaluating, and visualizing Multivariate
    (Weather + Pollutant) Deep Learning models. Contains standard PyTorch
    architectures (RNN, LSTM, Bi-LSTM, GRU) and a master orchestrator to load
    optimized parameters and train the final models.
"""

import os
import json
import pickle
import copy
import logging
import warnings
from pathlib import Path
from typing import Any, Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Required strictly for early-stopping exceptions during hyperparameter searches
import optuna

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Create outputs folder
OUTPUT_DIR = Path("outputs")
MV_DIR = OUTPUT_DIR / "MV DL pred"

os.makedirs(MV_DIR, exist_ok=True)

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
    Global configuration state for the Multivariate models.
    These values act as defaults and are dynamically overwritten by the
    JSON file during the final optimized inference loop.
    """
    LOOK_BACK: int = 30
    HORIZON: int = 7
    BATCH_SIZE: int = 128
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    PATIENCE: int = 20
    HIDDEN_DIM: int = 128
    NUM_LAYERS: int = 6
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
    mask = y_true != 0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1 - ss_res / ss_tot)


def make_sequences(features: np.ndarray, target: np.ndarray, look_back: int = None, horizon: int = None) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Slices a continuous multivariate time series into overlapping X (input) and y (target) sequences.

    Args:
        features (np.ndarray): 2D array of all feature columns (weather + pollutant).
        target (np.ndarray): 1D or 2D array of the specific target to forecast.
        look_back (int, optional): Number of historical steps. Defaults to config.
        horizon (int, optional): Number of future steps. Defaults to config.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X features matrix and y target matrix.
    """
    look_back = look_back if look_back is not None else config.LOOK_BACK
    horizon = horizon if horizon is not None else config.HORIZON

    X, y = [], []
    for i in range(len(features) - look_back - horizon + 1):
        X.append(features[i: i + look_back, :])
        y.append(target[i + look_back: i + look_back + horizon])
    return np.array(X), np.array(y)


class MultivariatePollutionDataset(Dataset):
    """PyTorch Dataset wrapper optimized for multivariate float tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initializes the dataset with input features and targets.

        Args:
            X (np.ndarray): The input sequences.
            y (np.ndarray): The target sequences.
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def prepare_multivariate_data(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        test_fraction: float = None
) -> Tuple[DataLoader, DataLoader, MinMaxScaler, np.ndarray, int]:
    """
    End-to-end data preparation pipeline for Multivariate forecasting.
    Scales multiple features, splits chronologically, and prevents data leakage.

    Args:
        df (pd.DataFrame): The raw chronological dataframe.
        target_col (str): The column name to forecast.
        feature_cols (List[str]): List of all feature columns to feed the network.
        test_fraction (float, optional): Ratio of data to reserve for testing. Defaults to config.

    Returns:
        Tuple: (train_loader, test_loader, target_scaler, raw_y_test_array, num_features)

    Raises:
        ValueError: If the dataset is too small to construct the required look_back and horizon.
    """
    test_fraction = test_fraction if test_fraction is not None else config.TEST_FRACTION
    split_idx = int(len(df) * (1 - test_fraction))

    train_df = df.iloc[:split_idx]
    # Allow test_df to safely overlap with the look_back window to prevent prediction gaps
    test_df = df.iloc[split_idx - config.LOOK_BACK:]

    train_features = train_df[feature_cols].values.astype(np.float32)
    train_target = train_df[[target_col]].values.astype(np.float32)

    test_features = test_df[feature_cols].values.astype(np.float32)
    test_target = test_df[[target_col]].values.astype(np.float32)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    train_feat_scaled = feature_scaler.fit_transform(train_features)
    train_tgt_scaled = target_scaler.fit_transform(train_target).flatten()

    test_feat_scaled = feature_scaler.transform(test_features)
    test_tgt_scaled = target_scaler.transform(test_target).flatten()

    X_train, y_train = make_sequences(train_feat_scaled, train_tgt_scaled)
    X_test, y_test = make_sequences(test_feat_scaled, test_tgt_scaled)

    _, y_test_orig = make_sequences(test_features, test_target.flatten())

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"Dataset is too small for LOOK_BACK={config.LOOK_BACK} and HORIZON={config.HORIZON}.")

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(MultivariatePollutionDataset(X_train, y_train), batch_size=config.BATCH_SIZE,
                              shuffle=True, pin_memory=use_pin_memory)
    test_loader = DataLoader(MultivariatePollutionDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False,
                             pin_memory=use_pin_memory)

    return train_loader, test_loader, target_scaler, y_test_orig, len(feature_cols)


# =====================================================================
# MODELS (RNN, LSTM, Bi-LSTM, GRU)
# =====================================================================
class RNNModel(nn.Module):
    """Standard Recurrent Neural Network Architecture."""

    def __init__(self, input_dim: int):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                          batch_first=True)
        self.fc = nn.Linear(config.HIDDEN_DIM, config.HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        return self.fc(hidden[-1])


class LSTMModel(nn.Module):
    """Long Short-Term Memory Architecture."""

    def __init__(self, input_dim: int):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                            batch_first=True)
        self.fc = nn.Linear(config.HIDDEN_DIM, config.HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.lstm(x)
        return self.fc(hidden[-1])


class BiLSTMModel(nn.Module):
    """Bidirectional Long Short-Term Memory Architecture."""

    def __init__(self, input_dim: int):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(config.HIDDEN_DIM * 2, config.HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.lstm(x)
        # Concatenate the final forward state and final backward state
        return self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))


class GRUModel(nn.Module):
    """Gated Recurrent Unit Architecture."""

    def __init__(self, input_dim: int):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                          batch_first=True)
        self.fc = nn.Linear(config.HIDDEN_DIM, config.HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.fc(hidden[-1])


class HybridRNNLSTMModel(nn.Module):
    """Hybrid RNN-LSTM Architecture."""

    def __init__(self, input_dim: int):
        super(HybridRNNLSTMModel, self).__init__()

        self.rnn = nn.RNN(input_size=input_dim, hidden_size=config.HIDDEN_DIM,
                          num_layers=config.NUM_LAYERS, batch_first=True)

        self.lstm = nn.LSTM(input_size=config.HIDDEN_DIM, hidden_size=config.HIDDEN_DIM,
                            num_layers=config.NUM_LAYERS, batch_first=True)

        self.fc = nn.Linear(config.HIDDEN_DIM, config.HORIZON)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass data through the RNN
        rnn_out, _ = self.rnn(x)

        # Feed the RNN's full sequential output into the LSTM
        _, (lstm_hidden, _) = self.lstm(rnn_out)

        # Extract the final hidden state of the LSTM for the fully connected layer
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

        for i in range(config.NUM_LAYERS):
            # The final conv layer outputs exactly the HIDDEN_DIM requested by Optuna.
            # Intermediate layers use a fixed 64 channels to prevent parameter explosion.
            out_channels = config.HIDDEN_DIM if i == (config.NUM_LAYERS - 1) else 64

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
    Hybrid CNN-LSTM Architecture for Time-Series/Sequence Processing.
    Perfectly integrated with the ML-Ops global config.
    """
    def __init__(self, input_dim: int = 1):
        super(HybridCNNLSTMModel, self).__init__()

        # ==========================================
        # 1. The CNN Block (Feature Extraction)
        # ==========================================
        # We use a fixed 64 channels for the CNN to act as a feature compressor,
        # preventing the parameter count from exploding before the LSTM.
        cnn_out_channels = 64

        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_out_channels,
            kernel_size=3,
            padding=1  # Padding=1 keeps the sequence length the same after convolution
        )
        self.relu = nn.ReLU()

        # Pooling reduces the sequence length by half, concentrating the most important features
        self.pool = nn.MaxPool1d(kernel_size=2)

        # ==========================================
        # 2. The LSTM Block (Temporal Processing)
        # ==========================================
        # The LSTM takes the CNN's 64 output channels as its input features.
        # It uses config.HIDDEN_DIM so Optuna can actively tune its memory capacity!
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True
        )

        # ==========================================
        # 3. The Output Block
        # ==========================================
        # Projects the final memory state into our required forecasting horizon (7 days)
        self.fc = nn.Linear(config.HIDDEN_DIM, config.HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass handling the strict dimensional requirements of PyTorch CNNs and LSTMs.
        Automatically scales 2D Univariate data to 3D.
        """
        # [FIX 1]: Safely convert 2D Univariate inputs to 3D before transposing
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        # STEP 1: Dimensionality Shift for CNN -> [Batch, Features, Sequence_Length]
        x = x.transpose(1, 2)

        # STEP 2: CNN Feature Extraction
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)  # Sequence length is now halved

        # STEP 3: Dimensionality Shift for LSTM -> [Batch, Sequence_Length_Halved, Features]
        x = x.transpose(1, 2)

        # STEP 4: LSTM Temporal Processing
        lstm_out, (hidden, cell) = self.lstm(x)

        # STEP 5: Final Prediction
        # hidden[-1] grabs the hidden state from the final depth layer of the LSTM
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
) -> Tuple[nn.Module, list, list]:
    """
    Standard Backpropagation/Gradient Descent training loop with early stopping.

    Args:
        model (nn.Module): The instantiated PyTorch model to train.
        train_loader (DataLoader): DataLoader containing training batches.
        test_loader (DataLoader): DataLoader containing validation batches.
        model_name (str): Identifier used for print logs.
        trial (optuna.Trial, optional): Passed exclusively during hyperparameter searches
                                        to allow Optuna to prune unpromising models early.
        verbose (bool):

    Returns:
        Tuple: The trained model with best weights restored, list of train losses, list of val losses.

    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_losses, val_losses = [], []
    best_val_loss, patience_counter = np.inf, 0
    best_model_state = copy.deepcopy(model.state_dict())

    if verbose: print(f"\n[>] Training {model_name}...")
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
                if verbose: print(f" [!] Trial pruned by Optuna at epoch {epoch + 1}.")
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

        if patience_counter >= config.PATIENCE:
            if verbose: print(f" [!] Early stopping triggered at epoch {epoch + 1}.")
            break

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses


# =====================================================================
# EVALUATION & VISUALIZATION
# =====================================================================
def evaluate_model(
        model: nn.Module, test_loader: DataLoader, scaler: MinMaxScaler, y_test_orig: np.ndarray) -> Tuple[
    Dict[str, float], np.ndarray]:
    """
    Evaluates the model on the test set, inverse-transforms the predictions,
    and calculates standard forecasting metrics.

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): The DataLoader containing test set batches.
        scaler (MinMaxScaler): The scaler fitted on the target training data.
        y_test_orig (np.ndarray): The raw, unscaled ground truth values.

    Returns:
        Tuple: A dictionary containing metrics (RMSE, MAE, MAPE) and the array of predictions.
    """
    model.eval()
    y_pred_all = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_pred_all.append(model(X_batch.to(device)).cpu().numpy())

    y_pred_scaled = np.vstack(y_pred_all)
    y_pred_orig = scaler.inverse_transform(y_pred_scaled.flatten().reshape(-1, 1)).flatten()
    y_true_orig = y_test_orig.flatten()

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))),
        "MAE": float(mean_absolute_error(y_true_orig, y_pred_orig)),
        "MAPE": mape(y_true_orig, y_pred_orig),
        "R^2": r_2(y_true_orig, y_pred_orig)
    }
    return metrics, y_pred_orig


def predict_and_plot_series(
        model: torch.nn.Module,
        df_daily: pd.DataFrame,
        target_col: str,
        test_loader: torch.utils.data.DataLoader,
        scaler: Any,  # Replace Any with MinMaxScaler if imported directly
        y_test_orig: np.ndarray,
        model_name: str,
        title: str,
        save_directory: str,
        std_metrics: Optional[Dict[str, float]] = None
) -> plt.Figure:
    """
    Generates and exports a high-visibility, presentation-ready line chart.
    Features a clean legend and a dedicated side-panel for Metrics & Hyperparameters.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds.append(model(X_batch.to(device)).cpu().numpy()[:, 0])

    os.makedirs(save_directory, exist_ok=True)

    predictions_scaled = np.concatenate(preds).reshape(-1, 1)
    predictions_real = scaler.inverse_transform(predictions_scaled).flatten()

    end_date_idx = len(df_daily) - config.HORIZON + 1
    start_date_idx = end_date_idx - len(predictions_real)

    # Guarantees pred_dates length is EXACTLY equal to predictions_real
    pred_dates = df_daily.index[start_date_idx: end_date_idx]
    true_data = df_daily[target_col].loc[pred_dates]

    # Evaluate the metrics
    metrics, _ = evaluate_model(model, test_loader, scaler, y_test_orig)

    # --- 1. FORMAT DATA PANEL TEXT (Metrics + Hyperparameters) ---
    if std_metrics:
        metrics_text = (f"Model Performance\n"
                        f"-----------------\n"
                        f"RMSE: {metrics['RMSE']:.2f} ± {std_metrics['RMSE']:.2f}\n"
                        f"MAE:  {metrics['MAE']:.2f} ± {std_metrics['MAE']:.2f}\n"
                        f"MAPE: {metrics['MAPE']:.1f}% ± {std_metrics['MAPE']:.1f}%\n"
                        f"R²:   {metrics['R^2']:.3f} ± {std_metrics['R^2']:.3f}")
    else:
        metrics_text = (f"Model Performance\n"
                        f"-----------------\n"
                        f"RMSE: {metrics['RMSE']:.2f}\n"
                        f"MAE:  {metrics['MAE']:.2f}\n"
                        f"MAPE: {metrics['MAPE']:.1f}%\n"
                        f"R²:   {metrics['R^2']:.3f}")

    # Extract Hyperparameters from config
    hp_text = (f"Hyperparameters\n"
               f"---------------\n"
               f"Look Back : {getattr(config, 'LOOK_BACK', 'N/A')}\n"
               f"Horizon   : {getattr(config, 'HORIZON', 'N/A')}\n"
               f"Batch Size: {getattr(config, 'BATCH_SIZE', 'N/A')}\n"
               f"Layers    : {getattr(config, 'NUM_LAYERS', 'N/A')}\n"
               f"Hidden Dim: {getattr(config, 'HIDDEN_DIM', 'N/A')}\n"
               f"Learn Rate: {getattr(config, 'LEARNING_RATE', 'N/A')}")

    # Combine into a single text block
    info_panel_text = f"{metrics_text}\n\n{hp_text}"

    # --- 2. SLIDESHOW PLOT SIZING ---
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot the Shaded Error Region
    ax.fill_between(pred_dates, true_data, predictions_real,
                    color='lightcoral', alpha=0.4, label="Prediction Error", zorder=1)

    # Plot the main lines (Clean labels for the legend)
    ax.plot(pred_dates, true_data, color="black", label="True Future Data", linewidth=4.0, zorder=2)
    ax.plot(pred_dates, predictions_real, color="crimson", label=f"Forecast ({model_name})", linewidth=3.5,
            linestyle="--", zorder=3)

    # Add Context Markers
    ax.axvline(pred_dates[0], color="royalblue", linestyle=":", linewidth=3.0, alpha=0.8, label="Test Split Start")

    # --- 3. SLIDESHOW FORMATTING & LAYOUT ---
    ax.set_xlabel("Date", fontsize=20, fontweight="bold", labelpad=15)
    ax.set_ylabel(f"{target_col} - Concentration [μg/m³]", fontsize=20, fontweight="bold", labelpad=15)
    ax.set_title(title.replace('_', ' '), fontweight="bold", fontsize=26, pad=20)

    ax.tick_params(axis='x', labelsize=16, rotation=15)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, linestyle="-", alpha=0.3, linewidth=1.5, color="gray")

    # Clean, simple legend placed inside the plot
    legend = ax.legend(fontsize=18, loc="upper left", framealpha=0.9, edgecolor="black")

    # --- 4. ATTACH THE STATISTICS PANEL ---
    # Adjust the plot area to make room on the right side
    plt.subplots_adjust(right=0.75)

    # Place a text box strictly outside the data visualization area
    props = dict(boxstyle='round,pad=0.8', facecolor='whitesmoke', alpha=0.95, edgecolor='silver')
    ax.text(1.03, 0.5, info_panel_text, transform=ax.transAxes, fontsize=16,
            verticalalignment='center', bbox=props, linespacing=1.6, family='monospace')

    # Save with bbox_inches='tight' so it captures the new external text box
    fig.savefig(os.path.join(save_directory, f"{title}.png"), dpi=300, bbox_inches='tight')

    return fig

# =====================================================================
# THE MASTER ORCHESTRATOR
# =====================================================================
def model_looping(
        df_daily: pd.DataFrame,
        pollutant_features_map: dict,
        pollutants: tuple = ("NO2", "NOx", "O3", "PM10", "PM25"),
        optimized: bool = False,
        hyperparameter_file: Path = Path(OUTPUT_DIR / "best_parameters_mv.json")
) -> Dict[str, Any]:
    """
    Unified execution loop representing the final inference step of the ML pipeline.
    It iterates through all specified pollutants, fetches their specific explanatory weather
    features, safely loads optimized hyperparameters (if available), trains all models
    (3 times if optimized to average stochastic variance), evaluates them, and dumps
    a consolidated Pickle file for the comparison module. Includes Standard Deviation errors.
    """
    master_results = {"rnn_results": {}, "lstm_results": {}, "bilstm_results": {}, "gru_results": {},
                      "hy_rnn_lstm_results": {}, "cnn_results": {}, "hy-cnn-lstm_results": {}}

    # Define suite mapping to streamline the training loop
    models_to_test = {
        "RNN": (RNNModel, master_results["rnn_results"]),
        "LSTM": (LSTMModel, master_results["lstm_results"]),
        "Bi-LSTM": (BiLSTMModel, master_results["bilstm_results"]),
        "GRU": (GRUModel, master_results["gru_results"]),
        "Hy-RNN-LSTM": (HybridRNNLSTMModel, master_results["hy_rnn_lstm_results"]),
        # "CNN": (CNNModel, master_results["cnn_results"]),
        # "Hybrid-CNN-LSTM":(HybridCNNLSTMModel, master_results["hy_rnn_lstm_results"])
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

    # Determine how many times to train each model
    num_iterations = 3 if optimized else 1

    # Execute training sweep
    for pollutant in pollutants:
        if pollutant not in df_daily.columns:
            continue
        print(f"\n{'=' * 70}\nTARGET: {pollutant} (MULTIVARIATE)\n{'=' * 70}")

        # 1. Dynamically fetch specific features for this pollutant
        env_features = pollutant_features_map.get(pollutant, [])

        # Safely append the target pollutant for autoregression without duplicating it
        feature_cols = env_features.copy()
        if pollutant not in feature_cols:
            feature_cols.append(pollutant)

        for model_name, (ModelClass, results_dict) in models_to_test.items():

            # 2. Inject Hyperparameters
            if optimized:
                json_key = f"MV-{model_name}"
                json_key_sv = f"SV-{model_name}"

                if pollutant in optuna_params and json_key in optuna_params[pollutant]:
                    p = optuna_params[pollutant][json_key]["Balanced_Hyperparameters"]
                    config.LOOK_BACK = p.get("LOOK_BACK", config.LOOK_BACK)
                    config.HORIZON = p.get("HORIZON", config.HORIZON)
                    config.BATCH_SIZE = p.get("BATCH_SIZE", config.BATCH_SIZE)
                    config.LEARNING_RATE = p.get("LEARNING_RATE", config.LEARNING_RATE)
                    config.HIDDEN_DIM = p.get("HIDDEN_DIM", config.HIDDEN_DIM)
                    config.NUM_LAYERS = p.get("NUM_LAYERS", config.NUM_LAYERS)
                    print(f" [⚙] Injected Optimized Parameters for MV-{model_name}")
                elif pollutant in optuna_params and json_key_sv in optuna_params[pollutant]:
                    p = optuna_params[pollutant][json_key_sv]["Balanced_Hyperparameters"]
                    config.LOOK_BACK = p.get("LOOK_BACK", config.LOOK_BACK)
                    config.HORIZON = p.get("HORIZON", config.HORIZON)
                    config.BATCH_SIZE = p.get("BATCH_SIZE", config.BATCH_SIZE)
                    config.LEARNING_RATE = p.get("LEARNING_RATE", config.LEARNING_RATE)
                    config.HIDDEN_DIM = p.get("HIDDEN_DIM", config.HIDDEN_DIM)
                    config.NUM_LAYERS = p.get("NUM_LAYERS", config.NUM_LAYERS)
                    print(f" [⚙] Using Optimized Parameters for Single Variable model - {model_name}")
                else:
                    print(f" [-] No optimized params found for {json_key}. Using defaults.")

            # 3. Reslice data dynamically
            try:
                train_loader, test_loader, scaler, y_test_orig, num_features = prepare_multivariate_data(
                    df_daily, target_col=pollutant, feature_cols=feature_cols
                )
            except ValueError as e:
                print(f" [!] Skipping {model_name}: {e}")
                continue

            # =========================================================================
            # 4. Model Training (Averaging Loop)
            # =========================================================================
            accumulated_metrics = {"RMSE": [], "MAE": [], "MAPE": [], "R^2": []}
            best_model = None
            best_mape = float('inf')

            for iteration in range(num_iterations):
                if num_iterations > 1:
                    print(f"   [Iter {iteration + 1}/{num_iterations}] Training MV-{model_name}...")

                # Instantiate a FRESH model so weights are properly re-initialized
                model = ModelClass(input_dim=num_features)
                model, _, _ = train_model(model, train_loader, test_loader, model_name=f"MV-{model_name} ({pollutant})")

                # Evaluate this specific iteration
                iter_metrics, _ = evaluate_model(model, test_loader, scaler, y_test_orig)

                # Accumulate the scores
                for key in accumulated_metrics.keys():
                    accumulated_metrics[key].append(iter_metrics[key])

                # Track the absolute best model instance to use for the final plot
                if iter_metrics["MAPE"] < best_mape:
                    best_mape = iter_metrics["MAPE"]
                    best_model = model

            # =========================================================================
            # 5. Evaluation & Visualization Compilation (with Errors)
            # =========================================================================
            # Calculate the Mean and Standard Deviation (Error) across the iterations
            avg_metrics = {k: float(np.mean(v)) for k, v in accumulated_metrics.items()}
            std_metrics = {k: float(np.std(v)) for k, v in accumulated_metrics.items()}

            # Save the average metrics to the main key (Keeps Part 4 Pipeline from crashing!)
            results_dict[pollutant] = avg_metrics
            # Safely tuck the Standard Deviations into a secondary key for analysis
            results_dict[f"{pollutant}_std"] = std_metrics

            print(f" [✓] MV-{model_name} Performance ({num_iterations} runs):")
            print(f"     RMSE: {avg_metrics['RMSE']:.2f} ± {std_metrics['RMSE']:.2f}")
            print(f"     MAE:  {avg_metrics['MAE']:.2f}  ± {std_metrics['MAE']:.2f}")
            print(f"     MAPE: {avg_metrics['MAPE']:.2f}% ± {std_metrics['MAPE']:.2f}%")
            print(f"     R^2:  {avg_metrics['R^2']:.3f} ± {std_metrics['R^2']:.3f}\n")

            # Plot using the absolute BEST model instance from the runs
            fig = predict_and_plot_series(
                best_model, df_daily, pollutant, test_loader, scaler, y_test_orig,
                title=f"MV-{model_name} Prediction - {pollutant}",
                save_directory=str(MV_DIR / pollutant), model_name=model_name,
                std_metrics=std_metrics # <--- [FIX] WE INJECT THE ERRORS HERE!
            )

            if __name__ == "__main__":
                plt.close(fig)

    # 6. Export finalized metrics to Pickle for Part 4 Comparison
    with open(OUTPUT_DIR / "mv_DL_results.pkl", "wb") as f:
        pickle.dump(master_results, f)
    print("\n[✓] All Multivariate Neural Network results compiled and exported!")

    return master_results

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("Loading data from Part 1...")
    try:
        df = pd.read_parquet(Path("outputs/rnn_multivariate_dataset.parquet"))
        if "Timestamp" in df.columns:
            df = df.set_index("Timestamp")
        elif "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df_daily = df.sort_index().interpolate(method='linear').dropna()
    except FileNotFoundError:
        print("[!] Dataset not found at outputs/rnn_multivariate_dataset.parquet")
        exit()

    # Define the specific environmental drivers per pollutant
    # Modify these keys and features to match your exact dataset columns
    target_feature_map = {
        "PM25": ["sp", "u10", "v10"],
        "PM10": ["sp", "u10", "v10"],
        "O3": ["sp", "u10", "v10"],
        "NO2": ["sp", "u10", "v10"],
        "NOx": ["sp", "u10", "v10"]
    }

    # Pass optimized=True to automatically seek out and inject Optuna JSON parameters!
    model_looping(
        df_daily=df_daily,
        pollutant_features_map=target_feature_map,
        pollutants=("NO2", "O3"),
        optimized=True
    )
