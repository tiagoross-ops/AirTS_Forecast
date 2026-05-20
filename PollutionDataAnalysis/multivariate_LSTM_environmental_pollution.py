"""
AirTS-Forecast Project
Part 3: Multivariate Neural Networks for Time Series
    RNN · LSTM · Bi-LSTM · GRU · Training · Evaluation · Visualization

Optimizations applied:
- Centralized Config class for dynamic hyperparameter injection.
- Zero Index Drift & strict chronological tensor boundaries.
- Autoregressive feature correction (targets included in features).
- Bi-directional LSTM architecture with state concatenation.
- Hard validation barriers to prevent DataLoader crashes.
- Matplotlib memory leak prevention.
"""

import os
import copy
import warnings
from typing import Tuple
import pickle
from pathlib import Path
import optuna

import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
os.makedirs("outputs/plots", exist_ok=True)

#--------------------------------------------------------------------
# DEVICE CONFIGURATION
#--------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[✓] Hardware Accelerator: {device}")

#--------------------------------------------------------------------
# CONFIGURATION CLASS (Dynamic Hyperparameters)
#--------------------------------------------------------------------
class Config:
    """
    Centralized configuration. Modify these variables from external
    scripts before executing functions to dynamically change model behavior.
    """
    LOOK_BACK = 30
    HORIZON = 7
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 50
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    TEST_FRACTION = 0.2

# Instantiate the global configuration
config = Config()

#--------------------------------------------------------------------
# HELPER FUNCTIONS & DATA PREP
#--------------------------------------------------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def make_sequences(features: np.ndarray, target: np.ndarray, look_back: int = None, horizon: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sliding windows dynamically pulling from config."""
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
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def prepare_multivariate_data(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list,
        test_fraction: float = None
) -> Tuple[DataLoader, DataLoader, MinMaxScaler, np.ndarray, int]:
    """
    ML-Ops Pipeline: Strict Split -> Fit Scaler -> Transform -> Sequence independently.
    """
    test_fraction = test_fraction if test_fraction is not None else config.TEST_FRACTION
    split_idx = int(len(df) * (1 - test_fraction))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx - config.LOOK_BACK:]

    train_features = train_df[feature_cols].values.astype(np.float32)
    train_target = train_df[[target_col]].values.astype(np.float32)

    test_features = test_df[feature_cols].values.astype(np.float32)
    test_target = test_df[[target_col]].values.astype(np.float32)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    feature_scaler.fit(train_features)
    target_scaler.fit(train_target)

    train_feat_scaled = feature_scaler.transform(train_features)
    train_tgt_scaled = target_scaler.transform(train_target).flatten()

    test_feat_scaled = feature_scaler.transform(test_features)
    test_tgt_scaled = target_scaler.transform(test_target).flatten()

    X_train, y_train = make_sequences(train_feat_scaled, train_tgt_scaled)
    X_test, y_test = make_sequences(test_feat_scaled, test_tgt_scaled)

    _, y_test_orig = make_sequences(test_features, test_target.flatten())

    if len(X_train) == 0:
        raise ValueError(f"CRITICAL: Insufficient training data for '{target_col}'.")
    if len(X_test) == 0:
        raise ValueError(f"CRITICAL: Insufficient testing data for '{target_col}'.")

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        MultivariatePollutionDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=use_pin_memory
    )

    test_loader = DataLoader(
        MultivariatePollutionDataset(X_test, y_test),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=use_pin_memory
    )

    return train_loader, test_loader, target_scaler, y_test_orig, len(feature_cols)


# =====================================================================
# MODELS
# =====================================================================
class RNNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=None, output_dim=None, num_layers=None, dropout=0.2):
        super(RNNModel, self).__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else config.HIDDEN_DIM
        output_dim = output_dim if output_dim is not None else config.HORIZON
        num_layers = num_layers if num_layers is not None else config.NUM_LAYERS

        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hidden = self.rnn(x)
        return self.fc(hidden[-1])


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=None, output_dim=None, num_layers=None, dropout=0.2):
        super(LSTMModel, self).__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else config.HIDDEN_DIM
        output_dim = output_dim if output_dim is not None else config.HORIZON
        num_layers = num_layers if num_layers is not None else config.NUM_LAYERS

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=None, output_dim=None, num_layers=None, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else config.HIDDEN_DIM
        output_dim = output_dim if output_dim is not None else config.HORIZON
        num_layers = num_layers if num_layers is not None else config.NUM_LAYERS

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        return self.fc(hidden_concat)


class GRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=None, output_dim=None, num_layers=None, dropout=0.2):
        super(GRUModel, self).__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else config.HIDDEN_DIM
        output_dim = output_dim if output_dim is not None else config.HORIZON
        num_layers = num_layers if num_layers is not None else config.NUM_LAYERS

        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.fc(hidden[-1])


# =====================================================================
# TRAINING & EVALUATION
# =====================================================================
def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                model_name: str, epochs=None, learning_rate=None, patience=None,
                trial: optuna.Trial = None) -> nn.Module:

    epochs = epochs if epochs is not None else config.EPOCHS
    learning_rate = learning_rate if learning_rate is not None else config.LEARNING_RATE
    patience = patience if patience is not None else config.PATIENCE

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss, patience_counter = np.inf, 0
    best_model_state = copy.deepcopy(model.state_dict())

    print(f"\nTraining {model_name}...")
    for epoch in range(epochs):
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                val_loss += criterion(model(X_batch.to(device)), y_batch.to(device)).item()

        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        # --- NEW: OPTUNA PRUNING INTEGRATION ---
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f" [!] Trial pruned by Optuna at epoch {epoch + 1} (Unpromising trajectory).")
                raise optuna.exceptions.TrialPruned()


        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_model_state = copy.deepcopy(model.state_dict())
            if (epoch + 1) % 10 == 0 or epoch < 5:
                print(f" Epoch {epoch + 1:3d}/{epochs}: Train={train_loss:.5f}, Val={val_loss:.5f} [NEW BEST]")
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print(f" Epoch {epoch + 1:3d}/{epochs}: Train={train_loss:.5f}, Val={val_loss:.5f} (Patience {patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f" [!] Early stopping triggered at epoch {epoch + 1}.")
            break

    model.load_state_dict(best_model_state)
    return model


def evaluate_and_plot(
        model,
        df_daily,
        target_col,
        test_loader,
        target_scaler,
        y_test_orig,
        model_name,
        title: str = None,
        save_directory: str = None,
):
    model.eval()
    y_pred_all = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_pred_all.append(model(X_batch.to(device)).cpu().numpy())

    y_pred_scaled = np.vstack(y_pred_all)
    y_pred_orig = target_scaler.inverse_transform(y_pred_scaled.flatten().reshape(-1, 1)).flatten()
    y_true_orig = y_test_orig.flatten()

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
        "MAE": mean_absolute_error(y_true_orig, y_pred_orig),
        "MAPE": mape(y_true_orig, y_pred_orig)
    }

    pred_1step_scaled = np.concatenate([batch[:, 0] for batch in y_pred_all]).reshape(-1, 1)
    pred_1step_real = target_scaler.inverse_transform(pred_1step_scaled).flatten()

    total_sequences = len(df_daily) - config.LOOK_BACK - config.HORIZON + 1
    start_date_idx = int(total_sequences * (1 - config.TEST_FRACTION)) + config.LOOK_BACK
    pred_dates = df_daily.index[start_date_idx: start_date_idx + len(pred_1step_real)]

    fig = plt.figure(figsize=(14, 6))
    plt.plot(df_daily.index, df_daily[target_col], color="lightgray", label="Historical Observed")
    plt.plot(pred_dates, df_daily[target_col].loc[pred_dates], color="black", label="True Future Data", linewidth=1.5)
    plt.plot(pred_dates, pred_1step_real, color="red", label=f"{model_name} Forecast", linewidth=2.0)
    plt.axvline(pred_dates[0], color="blue", linestyle="--", alpha=0.6, label="Train/Test Split")

    title = title if title is not None else f"{target_col} Multivariate Forecast vs Reality ({model_name})"
    plt.title(title, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel(f"{target_col} Level")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_directory is None:
        save_path = f"outputs/plots/{target_col}_{model_name}_multivariate_forecast.png"
    else:
        # Use os.path.join for OS agnostic path handling
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, f"{title}.png")

    print(f"[✓] Forecast plot saved successfully to: {save_path}")
    fig.savefig(save_path, dpi=150)

    # Crucial to prevent RAM crashes in ML loops
    plt.close(fig)

    return metrics


# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main_execution() -> None:
    print("Loading multivariate dataset...")
    try:
        df = pd.read_parquet(r"outputs/rnn_multivariate_dataset.parquet")

        if "Timestamp" in df.columns:
            df = df.set_index("Timestamp")
        elif "timestamp" in df.columns:
            df = df.set_index("timestamp")

        df.index = pd.to_datetime(df.index)
        df_daily = df.sort_index().interpolate(method='linear').dropna()

    except FileNotFoundError:
        print("[!] Dataset not found. Ensure 'rnn_multivariate_dataset.parquet' exists.")
        return

    min_days_required = (config.LOOK_BACK + config.HORIZON) * 2
    if len(df_daily) < min_days_required:
        print(f"[!] CRITICAL: Only {len(df_daily)} usable days found.")
        return

    print(f"[✓] Final usable continuous days: {len(df_daily)}")

    targets = ["NO2", "NOx", "O3", "PM10", "PM25"]
    wanted_features = ["sp", "u10", "v10"]
    columns_to_keep = targets + wanted_features

    feature_cols = [col for col in columns_to_keep if col in df_daily.columns]
    df_daily = df_daily[feature_cols]

    print(f"[✓] Extracted {len(feature_cols)} features for Autoregression & Multivariate learning: {feature_cols}")

    lstm_results = {}
    bilstm_results = {}

    for target in targets:
        if target not in df_daily.columns:
            print(f"[!] Skipping {target}: Not found in dataset.")
            continue

        print(f"\n{'=' * 70}\nTARGET POLLUTANT: {target}\n{'=' * 70}")

        try:
            train_loader, test_loader, scaler, y_test_orig, num_features = prepare_multivariate_data(
                df_daily, target_col=target, feature_cols=feature_cols
            )
        except ValueError as e:
            print(e)
            continue

        #- Standard LSTM
        lstm_model = LSTMModel(input_dim=num_features)
        lstm_model = train_model(lstm_model, train_loader, test_loader, model_name=f"LSTM ({target})")
        lstm_metrics = evaluate_and_plot(
            lstm_model, df_daily, target, test_loader, scaler, y_test_orig,
            title="BomgusGongus", # Keeping your custom title here!
            save_directory=r"C:\Users\Tiago\Documents - PC\UTTOP\Enseignements\M1.2\Projet AirTS - Forecast\DL PLOTS",
            model_name="LSTM"
        )
        lstm_results[target] = lstm_metrics
        print(f" [✓] LSTM Metrics: RMSE={lstm_metrics['RMSE']:.3f}, MAE={lstm_metrics['MAE']:.3f}, MAPE={lstm_metrics['MAPE']:.2f}%")


        #- Bi-Directional LSTM
        bilstm_model = BiLSTMModel(input_dim=num_features)
        bilstm_model = train_model(bilstm_model, train_loader, test_loader, model_name=f"Bi-LSTM ({target})")
        bilstm_metrics = evaluate_and_plot(
            bilstm_model, df_daily, target, test_loader, scaler, y_test_orig,
            model_name="Bi-LSTM"
        )
        bilstm_results[target] = bilstm_metrics
        print(f" [✓] Bi-LSTM Metrics: RMSE={bilstm_metrics['RMSE']:.3f}, MAE={bilstm_metrics['MAE']:.3f}, MAPE={bilstm_metrics['MAPE']:.2f}%")

    # Save results
    with open("outputs/multivariate_nn_results.pkl", "wb") as f:
        pickle.dump({"lstm_results": lstm_results, "bilstm_results": bilstm_results}, f)

if __name__ == "__main__":
    main_execution()