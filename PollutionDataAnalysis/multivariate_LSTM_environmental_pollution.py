"""
AirTS-Forecast Project
Part 3: Multivariate Neural Networks for Time Series
    RNN · LSTM · Bi-LSTM · Training · Evaluation · Visualization

Optimizations applied:
- Zero Index Drift & strict chronological tensor boundaries.
- Autoregressive feature correction (targets included in features).
- Bi-directional LSTM architecture with state concatenation.
- Hard validation barriers to prevent DataLoader crashes.
"""

import os
import copy
import warnings
from typing import Tuple
import pickle
from pathlib import Path


import numpy as np
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
# DEVICE & HYPERPARAMETERS
#--------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[✓] Hardware Accelerator: {device}")

LOOK_BACK = 30
HORIZON = 7
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 50
HIDDEN_DIM = 256
NUM_LAYERS = 2
TEST_FRACTION = 0.2

#--------------------------------------------------------------------
# HELPER FUNCTIONS & DATA PREP
#--------------------------------------------------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def make_sequences(features: np.ndarray, target: np.ndarray, look_back: int = LOOK_BACK, horizon: int = HORIZON) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sliding windows. Features are 2D (time, vars), Target is 1D (time)."""
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
        test_fraction: float = TEST_FRACTION
) -> Tuple[DataLoader, DataLoader, MinMaxScaler, np.ndarray, int]:
    """
    ML-Ops Pipeline: Strict Split -> Fit Scaler -> Transform -> Sequence independently.
    Completely eliminates sequence boundary leakage and index drift.
    """
    split_idx = int(len(df) * (1 - test_fraction))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx - LOOK_BACK:]

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
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=use_pin_memory
    )

    test_loader = DataLoader(
        MultivariatePollutionDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=use_pin_memory
    )

    return train_loader, test_loader, target_scaler, y_test_orig, len(feature_cols)


# =====================================================================
# MODELS
# =====================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=HIDDEN_DIM, output_dim=HORIZON, num_layers=NUM_LAYERS, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


class BiLSTMModel(nn.Module):
    """
    Bi-directional LSTM Architecture.
    Processes the look-back window chronologically AND reverse-chronologically
    to extract deeper contextual representations before forecasting.
    """
    def __init__(self, input_dim: int, hidden_dim=HIDDEN_DIM, output_dim=HORIZON, num_layers=NUM_LAYERS, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True,
                            bidirectional=True) # Enables Bi-LSTM

        # The hidden state doubles in size (forward + backward states)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)

        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        # We extract the final forward state [-2] and final backward state [-1]
        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]

        # Concatenate them along the feature dimension
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)

        return self.fc(hidden_concat)


# =====================================================================
# TRAINING & EVALUATION
# =====================================================================
def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, model_name: str) -> nn.Module:
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss, patience_counter = np.inf, 0
    best_model_state = copy.deepcopy(model.state_dict())

    print(f"\nTraining {model_name}...")
    for epoch in range(EPOCHS):
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

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_model_state = copy.deepcopy(model.state_dict())
            if (epoch + 1) % 10 == 0 or epoch < 5:
                print(f" Epoch {epoch + 1:3d}/{EPOCHS}: Train={train_loss:.5f}, Val={val_loss:.5f} [NEW BEST]")
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print(f" Epoch {epoch + 1:3d}/{EPOCHS}: Train={train_loss:.5f}, Val={val_loss:.5f} (Patience {patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f" [!] Early stopping triggered at epoch {epoch + 1}.")
            break

    model.load_state_dict(best_model_state)
    return model

# =====================================================================
# SECTION D : EVALUATION & VISUALIZATION
# =====================================================================
def evaluate_model(model, test_loader, scaler, y_test_orig):
    model.eval()
    y_pred_all = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred_all.append(model(X_batch).cpu().numpy())

    y_pred_scaled = np.vstack(y_pred_all)
    pred_flat = y_pred_scaled.flatten().reshape(-1, 1)
    y_pred_orig = scaler.inverse_transform(pred_flat).flatten()

    y_true_orig = y_test_orig.flatten()
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    mape_val = mape(y_true_orig, y_pred_orig)

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape_val}, y_pred_orig


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

    total_sequences = len(df_daily) - LOOK_BACK - HORIZON + 1
    start_date_idx = int(total_sequences * (1 - TEST_FRACTION)) + LOOK_BACK
    pred_dates = df_daily.index[start_date_idx: start_date_idx + len(pred_1step_real)]

    plt.figure(figsize=(14, 6))
    plt.plot(df_daily.index, df_daily[target_col], color="lightgray", label="Historical Observed")
    plt.plot(pred_dates, df_daily[target_col].loc[pred_dates], color="black", label="True Future Data", linewidth=1.5)
    plt.plot(pred_dates, pred_1step_real, color="red", label=f"{model_name} Forecast", linewidth=2.0)
    plt.axvline(pred_dates[0], color="blue", linestyle="--", alpha=0.6, label="Train/Test Split")

    if title is not None:
        title = title
    else:
        title=f"{target_col} Multivariate Forecast vs Reality ({model_name})"

    plt.title(title, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel(f"{target_col} Level")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_directory is None:
        save_directory = f"outputs/plots/{target_col}_{model_name}_multivariate_forecast.png"
    else:
        save_directory = save_directory+"/"+title+".png"

    print(save_directory)
    plt.savefig(Path(save_directory), dpi=150)
    #plt.show()

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

    min_days_required = (LOOK_BACK + HORIZON) * 2
    if len(df_daily) < min_days_required:
        print(f"[!] CRITICAL: Only {len(df_daily)} usable days found.")
        return

    print(f"[✓] Final usable continuous days: {len(df_daily)}")

    targets = ["NO2", "NOx", "O3", "PM10", "PM25"]

    # 1. Define exactly what we want to keep
    wanted_features = ["sp", "u10", "v10"]

    # 2. Combine wanted weather features with our target pollutants
    columns_to_keep = targets + wanted_features

    # 3. Filter safely (only keep columns that actually exist in the dataframe)
    feature_cols = [col for col in columns_to_keep if col in df_daily.columns]

    # 4. Overwrite the dataframe with only the selected columns
    df_daily = df_daily[feature_cols]

    print(f"[✓] Extracted {len(feature_cols)} features for Autoregression & Multivariate learning. - {feature_cols}")

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
        lstm_model = LSTMModel(input_dim=num_features, hidden_dim=HIDDEN_DIM, output_dim=HORIZON, num_layers=NUM_LAYERS)
        lstm_model = train_model(lstm_model, train_loader, test_loader, model_name=f"LSTM ({target})")
        lstm_metrics = evaluate_and_plot(lstm_model, df_daily, target, test_loader, scaler, y_test_orig, title="BomgusGongus",
                                         save_directory=r"C:\Users\Tiago\Documents - PC\UTTOP\Enseignements\M1.2\Projet AirTS - Forecast\DL PLOTS",
                                         model_name="LSTM")
        lstm_results[target] = lstm_metrics
        print(
            f" LSTM Metrics: RMSE={lstm_metrics['RMSE']:.3f}, MAE={lstm_metrics['MAE']:.3f}, MAPE={lstm_metrics['MAPE']:.2f}%")


        #- Bi-Directional LSTM
        bilstm_model = BiLSTMModel(input_dim=num_features, hidden_dim=HIDDEN_DIM, output_dim=HORIZON, num_layers=NUM_LAYERS)
        bilstm_model = train_model(bilstm_model, train_loader, test_loader, model_name=f"Bi-LSTM ({target})")
        bilstm_metrics = evaluate_and_plot(bilstm_model, df_daily, target, test_loader, scaler, y_test_orig, model_name="Bi-LSTM")
        bilstm_results[target] = bilstm_metrics
        lstm_results[target] = lstm_metrics
        print(
            f" LSTM Metrics: RMSE={bilstm_metrics['RMSE']:.3f}, MAE={bilstm_metrics['MAE']:.3f}, MAPE={bilstm_metrics['MAPE']:.2f}%")


    # Save results
    with open("outputs/multivariate_nn_results.pkl", "wb") as f:
        pickle.dump({"lstm_results": lstm_results, "bilstm_results": bilstm_results}, f)

if __name__ == "__main__":
    main_execution()