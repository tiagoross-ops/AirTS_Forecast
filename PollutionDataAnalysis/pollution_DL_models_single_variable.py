"""URBAN POLLUTION FORECASTING — TUTORIAL FOR BEGINNERS
Part 3 : Neural Networks for Time Series
    RNN · LSTM · Bi-LSTM · GRU · Training · Evaluation · Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import optuna

warnings.filterwarnings("ignore")
import os
import pickle
import copy  # For safe model state saving
from typing import Any
import random
# PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Create outputs folder
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)  # New folder for our forecast plots

#--------------------------------------------------------------------
# DEVICE CONFIGURATION (CPU / GPU / MPS)
#--------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("[✓] Using GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[✓] Using Apple Silicon (MPS)")
else:
    device = torch.device("cpu")
    print("[!] Using CPU (slow, but works everywhere)")

#--------------------------------------------------------------------
# CONFIGURATION CLASS (Dynamic Hyperparameters)
#--------------------------------------------------------------------
class Config:
    """
    Centralized configuration class.
    Allows external scripts to import this module and modify hyperparameters
    dynamically before running functions.
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

# Instantiate the global config object
config = Config()

#--------------------------------------------------------------------
# HELPER FUNCTIONS
#--------------------------------------------------------------------
def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def make_sequences(data, look_back=None, horizon=None):
    """Create sliding window sequences from time series data."""
    # Dynamically pull from config if not explicitly provided
    look_back = look_back if look_back is not None else config.LOOK_BACK
    horizon = horizon if horizon is not None else config.HORIZON

    data = np.asarray(data)
    X, y = [], []
    for i in range(len(data) - look_back - horizon + 1):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back:i + look_back + horizon])
    return np.array(X), np.array(y)

class PollutionDataset(Dataset):
    """PyTorch Dataset wrapper for pollution time series."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(df, pollutant, scaler=None, test_fraction=None):
    """Full data preparation pipeline."""
    test_fraction = test_fraction if test_fraction is not None else config.TEST_FRACTION

    data = df[pollutant].values.astype(np.float32)

    #- Step 1: Scale to [0, 1]
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    else:
        data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()

    #- Step 2: Create sequences
    X, y = make_sequences(data_scaled)

    #- Step 3: Split train/test chronologically
    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    _, y_test_orig = make_sequences(data)
    y_test_orig = y_test_orig[split_idx:]

    #- Step 4: Create PyTorch dataloaders
    train_dataset = PollutionDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_dataset = PollutionDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, scaler, y_test_orig

# =====================================================================
# SECTION A & B : MODELS (RNN, LSTM, bi-LSTM, GRU)
# =====================================================================
class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=None, output_dim=None, num_layers=None, dropout=0.2):
        super(RNNModel, self).__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else config.HIDDEN_DIM
        output_dim = output_dim if output_dim is not None else config.HORIZON
        num_layers = num_layers if num_layers is not None else config.NUM_LAYERS

        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        output, hidden = self.rnn(x)
        out = self.fc(hidden[-1])
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=None, output_dim=None, num_layers=None, dropout=0.2):
        super(LSTMModel, self).__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else config.HIDDEN_DIM
        output_dim = output_dim if output_dim is not None else config.HORIZON
        num_layers = num_layers if num_layers is not None else config.NUM_LAYERS

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        output, (hidden, cell) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=None, output_dim=None, num_layers=None, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else config.HIDDEN_DIM
        output_dim = output_dim if output_dim is not None else config.HORIZON
        num_layers = num_layers if num_layers is not None else config.NUM_LAYERS

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        output, (hidden, cell) = self.lstm(x)
        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.fc(hidden_concat)
        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=None, output_dim=None, num_layers=None, dropout=0.2):
        super(GRUModel, self).__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else config.HIDDEN_DIM
        output_dim = output_dim if output_dim is not None else config.HORIZON
        num_layers = num_layers if num_layers is not None else config.NUM_LAYERS

        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        # GRU only returns (output, hidden) - no cell state!
        output, hidden = self.gru(x)
        out = self.fc(hidden[-1])
        return out


    def forward(self, x):
        x = x.unsqueeze(-1)
        # GAD specifics
        output, hidden = self.gru(x)
        out = self.fc(hidden[-1])
        return out


# =====================================================================
# SECTION C : TRAINING LOOP
# =====================================================================
def train_model(
        model,
        train_loader, test_loader,
        epochs=None, learning_rate=None, patience=None,
        model_name="Model",
        trial: optuna.Trial = None,
):
    epochs = epochs if epochs is not None else config.EPOCHS
    learning_rate = learning_rate if learning_rate is not None else config.LEARNING_RATE
    patience = patience if patience is not None else config.PATIENCE

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    best_val_loss = np.inf
    patience_counter = 0
    best_model_state = copy.deepcopy(model.state_dict())

    print(f"\nTraining {model_name}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f" [!] Trial pruned by Optuna at epoch {epoch + 1} (Unpromising trajectory).")
                raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_model_state = copy.deepcopy(model.state_dict())
            if (epoch + 1) % 10 == 0 or epoch < 5:
                print(f" Epoch {epoch + 1:3d}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f} [NEW BEST]")
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print(f" Epoch {epoch + 1:3d}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f} (patience {patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\n Early stopping triggered at epoch {epoch + 1}.")
            break



    model.load_state_dict(best_model_state)
    print(f"[✓] Training complete. Best val_loss: {best_val_loss:.6f}")
    return model, train_losses, val_losses


# =====================================================================
# SECTION D : GENETIC ALGORITHM
# =====================================================================
def evaluate_fitness(model: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> float:
    """Helper function to evaluate a network's fitness (Lower Loss = Higher Fitness)"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item()

    avg_loss = total_loss / len(data_loader)
    # Fitness is the inverse of loss. We add a tiny epsilon to prevent division by zero.
    return 1.0 / (avg_loss + 1e-8), avg_loss

def train_model_ga(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_name: str = "GA_LSTM",
        pop_size: int = 20,
        generations: int = 50,
        mutation_rate: float = 0.05,
        mutation_scale: float = 0.1
):
    """
    Trains a PyTorch model using a Genetic Algorithm (Neuroevolution) instead of Backpropagation.
    """
    print(f"\n🧬 Evolving {model_name} using Genetic Algorithm...")
    model = model.to(device)
    criterion = nn.MSELoss()

    # 1. Flatten the base model's weights into a single 1D vector
    base_vector = nn_utils.parameters_to_vector(model.parameters()).detach()
    num_params = len(base_vector)
    print(f"Total DNA sequence length (parameters): {num_params:,}")

    # 2. Initialize Population (Randomly mutate the base vector)
    population = []
    for _ in range(pop_size):
        # Create a mutated clone
        individual = base_vector + (torch.randn(num_params, device=device) * mutation_scale)
        population.append(individual)

    best_global_vector = None
    best_global_loss = np.inf
    train_losses = []

    for gen in range(generations):
        fitness_scores = []
        losses = []

        # --- EVALUATION PHASE ---
        for individual in population:
            # Load this individual's "DNA" into the PyTorch model
            nn_utils.vector_to_parameters(individual, model.parameters())
            fitness, loss = evaluate_fitness(model, train_loader, criterion)
            fitness_scores.append(fitness)
            losses.append(loss)

        # Track the best performer
        best_idx = np.argmax(fitness_scores)
        if losses[best_idx] < best_global_loss:
            best_global_loss = losses[best_idx]
            best_global_vector = population[best_idx].clone()

        train_losses.append(best_global_loss)

        if (gen + 1) % 5 == 0 or gen == 0:
            print(f" Generation {gen + 1:3d}/{generations} | Best Loss: {best_global_loss:.6f}")

        # --- SELECTION PHASE (Tournament Selection) ---
        new_population = []
        # Elitism: Always keep the absolute best individual untouched
        new_population.append(best_global_vector.clone())

        while len(new_population) < pop_size:
            # Pick 3 random individuals, let them fight, the fittest becomes a parent
            tournament = random.sample(range(pop_size), 3)
            parent1_idx = max(tournament, key=lambda idx: fitness_scores[idx])

            tournament = random.sample(range(pop_size), 3)
            parent2_idx = max(tournament, key=lambda idx: fitness_scores[idx])

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # --- CROSSOVER PHASE (Uniform Crossover) ---
            # 50/50 chance to take a gene (weight) from parent 1 or parent 2
            mask = torch.rand(num_params, device=device) > 0.5
            child = torch.where(mask, parent1, parent2)

            # --- MUTATION PHASE ---
            # Add random noise to a small percentage of weights
            mutation_mask = torch.rand(num_params, device=device) < mutation_rate
            noise = torch.randn(num_params, device=device) * mutation_scale
            child = child + (mutation_mask * noise)

            new_population.append(child)

        population = new_population

    # Load the absolute best DNA back into the model before returning
    nn_utils.vector_to_parameters(best_global_vector, model.parameters())
    print(f"[✓] Evolution complete. Best training loss: {best_global_loss:.6f}")

    return model, train_losses, []

# =====================================================================
# SECTION E : EVALUATION & VISUALIZATION
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


def predict_and_plot_series(
        model,
        df_daily,
        pollutant,
        test_loader,
        scaler,
        model_name="LSTM",
        title: str = None,
        save_directory: str = None
):
    model.eval()
    predictions_scaled = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            predictions_scaled.append(y_pred[:, 0])

    predictions_scaled = np.concatenate(predictions_scaled).reshape(-1, 1)
    predictions_real = scaler.inverse_transform(predictions_scaled).flatten()

    total_sequences = len(df_daily) - config.LOOK_BACK - config.HORIZON + 1
    split_idx = int(total_sequences * (1 - config.TEST_FRACTION))
    start_date_idx = split_idx + config.LOOK_BACK

    pred_dates = df_daily.index[start_date_idx: start_date_idx + len(predictions_real)]

    fig = plt.figure(figsize=(14, 6))

    plt.plot(df_daily.index, df_daily[pollutant], color="lightgray", label="Observed Data", alpha=0.8)
    plt.plot(pred_dates, df_daily[pollutant].loc[pred_dates], color="black", label="True Future Data", linewidth=1.5)
    plt.plot(pred_dates, predictions_real, color="red", label=f"{model_name} Forecast", linewidth=2.0)
    plt.axvline(pred_dates[0], color="blue", linestyle="--", alpha=0.6, label="Train/Test Split")

    title = title if title else f"{pollutant} Concentration Forecast vs Reality ({model_name})"
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"{pollutant} Level", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_directory is not None:
        save_path = os.path.join(save_directory, f"{title}.png")
    else:
        save_path = f"outputs/plots/{pollutant}_{model_name}_forecast.png"

    fig.savefig(save_path, dpi=150)
    print(f"[✓] Forecast plot saved successfully to: {save_path}")

    return fig


# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main_execution() -> None:
    print("Loading data from Part 1...")
    try:
        parquet_path = r"C:\Users\Tiago\IdeaProjects\AirTS_Forecast\PollutionDataAnalysis\outputs\consolidated_pollutants.parquet"
        df = pd.read_parquet(parquet_path)

        if "Timestamp" in df.columns:
            df = df.set_index("Timestamp")
        elif "timestamp" in df.columns:
            df = df.set_index("timestamp")

        df_daily = df.resample("D").mean().interpolate(method='linear')
    except FileNotFoundError:
        print("[!] Dataset not found. Ensure the parquet file exists at the specified path.")
        return

    print(f"[✓] Loaded: {len(df_daily)} days")
    pollutants = ["NO2", "NOx", "O3", "PM10", "PM25"]
    rnn_results, lstm_results, bi_lstm_results, gru_results = {}, {}, {}, {}

    for pollutant in pollutants:
        if pollutant not in df_daily.columns:
            print(f"[!] Skipping {pollutant}: Not found in dataset.")
            continue

        print(f"\n{'=' * 70}\nPOLLUTANT: {pollutant}\n{'=' * 70}")

        train_loader, test_loader, scaler, y_test_orig = prepare_data(df_daily, pollutant)
        print(f" Train samples ({(1-config.TEST_FRACTION)*100:.0f}%): {len(train_loader.dataset)}")
        print(f" Test samples ({config.TEST_FRACTION*100:.0f}%): {len(test_loader.dataset)}")

        #- RNN
        rnn_model = RNNModel()
        rnn_model, _, _ = train_model(rnn_model, train_loader, test_loader, model_name=f"RNN ({pollutant})")
        rnn_metrics, _ = evaluate_model(rnn_model, test_loader, scaler, y_test_orig)
        rnn_results[pollutant] = rnn_metrics
        print(f" RNN Metrics: RMSE={rnn_metrics['RMSE']:.3f}, MAE={rnn_metrics['MAE']:.3f}, MAPE={rnn_metrics['MAPE']:.2f}%")
        pred_rnn = predict_and_plot_series(rnn_model, df_daily, pollutant, test_loader, scaler, model_name="RNN")
        plt.close(pred_rnn)

        #- LSTM
        lstm_model = LSTMModel()
        lstm_model, _, _ = train_model(lstm_model, train_loader, test_loader, model_name=f"LSTM ({pollutant})")
        lstm_metrics, _ = evaluate_model(lstm_model, test_loader, scaler, y_test_orig)
        lstm_results[pollutant] = lstm_metrics
        print(f" LSTM Metrics: RMSE={lstm_metrics['RMSE']:.3f}, MAE={lstm_metrics['MAE']:.3f}, MAPE={lstm_metrics['MAPE']:.2f}%")
        pred_lstm = predict_and_plot_series(lstm_model, df_daily, pollutant, test_loader, scaler, model_name="LSTM")
        plt.close(pred_lstm)

        #- Bi-LSTM
        bi_lstm_model = BiLSTMModel()
        bi_lstm_model, _, _ = train_model(bi_lstm_model, train_loader, test_loader, model_name=f"Bi-LSTM ({pollutant})")
        bi_lstm_metrics, _ = evaluate_model(bi_lstm_model, test_loader, scaler, y_test_orig)
        bi_lstm_results[pollutant] = bi_lstm_metrics
        print(f"Bi-LSTM Metrics: RMSE={bi_lstm_metrics['RMSE']:.3f}, MAE={bi_lstm_metrics['MAE']:.3f}, MAPE={bi_lstm_metrics['MAPE']:.2f}%")
        pred_bi_lstm = predict_and_plot_series(bi_lstm_model, df_daily, pollutant, test_loader, scaler, model_name="Bi-LSTM")
        plt.close(pred_bi_lstm)

        #- GRU
        gru_model = GRUModel()
        gru_model, _, _ = train_model(gru_model, train_loader, test_loader, model_name=f"GRU ({pollutant})")
        gru_metrics, _ = evaluate_model(gru_model, test_loader, scaler, y_test_orig)
        gru_results[pollutant] = gru_metrics
        print(f" GRU Metrics: RMSE={gru_metrics['RMSE']:.3f}, MAE={gru_metrics['MAE']:.3f}, MAPE={gru_metrics['MAPE']:.2f}%")
        pred_gru = predict_and_plot_series(gru_model, df_daily, pollutant, test_loader, scaler, model_name="GRU")
        plt.close(pred_gru)

        # GA on LSTM
        #- Genetic Algorithm LSTM
        ga_model = LSTMModel()

        # Notice we use train_model_ga instead of train_model!
        ga_model, _, _ = train_model_ga(
            ga_model, train_loader, test_loader,
            model_name=f"GA-LSTM ({pollutant})",
            pop_size=30,
            generations=50
        )

        ga_metrics, _ = evaluate_model(ga_model, test_loader, scaler, y_test_orig)
        print(f" GA Metrics: RMSE={ga_metrics['RMSE']:.3f}, MAE={ga_metrics['MAE']:.3f}, MAPE={ga_metrics['MAPE']:.2f}%")
        pred_ga = predict_and_plot_series(ga_model, df_daily, pollutant, test_loader, scaler, model_name="GA-LSTM")
        plt.close(pred_ga)

    print("\n" + "=" * 70 + "\nSaving results...")

    # Updated dump to include all 4 models
    with open("outputs/nn_results.pkl", "wb") as f:
        pickle.dump({
            "rnn_results": rnn_results,
            "lstm_results": lstm_results,
            "bilstm_results": bi_lstm_results,
            "gru_results": gru_results
        }, f)

    print("[✓] Neural network results saved to nn_results.pkl")
    print("\n" + "=" * 70 + "\nPart 3 Complete!\n" + "=" * 70)


if __name__ == "__main__":
    main_execution()