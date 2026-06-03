"""
URBAN POLLUTION FORECASTING — TUTORIAL FOR BEGINNERS
Part 4 : Final Comparison · Summary · Ablation Study

This module teaches you:
1. How to load results from multiple models and architectures securely.
2. Advanced Visualization: Cross-model Heatmaps and Ablation Studies.
3. Automated ML-Ops Reporting: Exporting dynamic benchmarks to CSV.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
os.makedirs("outputs", exist_ok=True)

POLLUTANTS = ["NO2", "NOx", "O3", "PM10", "PM25"]
UNITS = {"NO2": "µg/m³", "PM10": "µg/m³", "PM25": "µg/m³", "NOx": "µg/m³", "O3": "µg/m³"}

# 1. Programmatically define architectures to guarantee strict naming conventions
CLASSICAL_MODELS = ["ARIMA", "SARIMA", "Holt-Winters"]
DL_ARCHITECTURES = ["RNN", "GRU", "LSTM", "Bi-LSTM", "Hybrid-RNN-LSTM"]

# Construct the master list dynamically
MODELS = CLASSICAL_MODELS + [f"SV-{arch}" for arch in DL_ARCHITECTURES] + [f"MV-{arch}" for arch in DL_ARCHITECTURES]

# 2. Dynamic Fallback Generator
def generate_fallback_benchmarks():
    """Generates synthetic fallback data if real models haven't finished training."""
    base_scores = {
        "RNN": 18.5, "GRU": 15.2, "LSTM": 14.1, "Bi-LSTM": 13.5,
        "Hybrid-RNN-LSTM": 13.0 # "Hybrid-CNN-LSTM": 12.2, "CNN": 14.8
    }

    fallback = {}
    for pol in POLLUTANTS:
        fallback[pol] = {}
        # Classical Fallbacks
        for m_name in CLASSICAL_MODELS:
            mape = 25.0 + np.random.uniform(-3, 3)
            # [FIX]: Added synthetic R^2 values
            fallback[pol][m_name] = {"RMSE": mape*0.4, "MAE": mape*0.3, "MAPE": mape, "R^2": 0.40 + np.random.uniform(0, 0.1)}

        # Deep Learning Fallbacks
        for arch in DL_ARCHITECTURES:
            base_mape = base_scores[arch] + np.random.uniform(-2, 2)
            base_r2 = 0.65 + np.random.uniform(0, 0.2)

            # Synthesize SV performing worse than MV
            fallback[pol][f"SV-{arch}"] = {
                "RMSE": (base_mape+3)*0.4, "MAE": (base_mape+3)*0.3, "MAPE": base_mape + 3.0, "R^2": base_r2 - 0.1
            }
            # fallback[pol][f"MV-{arch}"] = {"RMSE": base_mape*0.4, "MAE": base_mape*0.3, "MAPE": base_mape, "R^2": base_r2}

    return fallback

BENCHMARK_FALLBACK = generate_fallback_benchmarks()

# =====================================================================
# HELPER FUNCTIONS (No Changes Required Here)
# =====================================================================

def _load_nn_pkl(filepath: Path, prefix: str) -> dict:
    parsed_results = {}
    try:
        with open(filepath, "rb") as f:
            pkl_data = pickle.load(f)

        mapping = {
            "rnn_results": f"{prefix}-RNN",
            "gru_results": f"{prefix}-GRU",
            "lstm_results": f"{prefix}-LSTM",
            "bilstm_results": f"{prefix}-Bi-LSTM",
            # "cnn_results": f"{prefix}-CNN",
            "hy_rnn_lstm_results": f"{prefix}-Hybrid-RNN-LSTM",
            # "hy_cnn_lstm_results": f"{prefix}-Hybrid-CNN-LSTM"
        }

        for raw_key, new_name in mapping.items():
            model_data = pkl_data.get(raw_key, {})
            for pollutant, metrics in model_data.items():
                if pollutant not in parsed_results:
                    parsed_results[pollutant] = {}
                parsed_results[pollutant][new_name] = metrics
        return parsed_results
    except FileNotFoundError:
        return {}


def load_results(classical_filepath: Path, sv_nn_filepath: Path, mv_nn_filepath: Path) -> dict:
    classical_results = {}
    if classical_filepath.exists():
        with open(classical_filepath, "r") as f:
            classical_results = json.load(f)

    sv_results = _load_nn_pkl(sv_nn_filepath, "SV")
    mv_results = _load_nn_pkl(mv_nn_filepath, "MV")

    benchmark = {}
    print("\n" + "=" * 85 + "\nDATA STATUS — Which results are real vs fallback?\n" + "=" * 85)

    for pollutant in POLLUTANTS:
        benchmark[pollutant] = {}
        status = []
        for model in MODELS:
            # 1. Check Classical
            if model in classical_results.get(pollutant, {}):
                benchmark[pollutant][model] = classical_results[pollutant][model]
                status.append(f"[✓] {model}")
            # 2. Check Univariate (SV)
            elif model in sv_results.get(pollutant, {}):
                benchmark[pollutant][model] = sv_results[pollutant][model]
                status.append(f"[✓] {model}")
            # 3. Check Multivariate (MV)  <-- THIS IS THE FIX
            elif model in mv_results.get(pollutant, {}):
                benchmark[pollutant][model] = mv_results[pollutant][model]
                status.append(f"[✓] {model}")
            # 4. Apply Fallback if missing
            else:
                benchmark[pollutant][model] = BENCHMARK_FALLBACK[pollutant][model]
                status.append(f"[-] {model} (fallback)")

        loaded_count = sum(1 for s in status if "[✓]" in s)
        print(f"{pollutant:<5}: Loaded {loaded_count}/{len(MODELS)} real models.")

    print("=" * 85 + "\n")
    return benchmark

# =====================================================================
# VISUALIZATIONS
# =====================================================================

def plot_mape_heatmap(benchmark: dict):
    """
    Generates a high-density Heatmap with Pollutants as Rows and Models as Columns.
    Includes dynamic aspect ratio, rotated labels, and NaN failsafes.
    """
    # 1. FIXED: Added the required tuple parentheses for np.zeros
    sv_models = MODELS[0:8]
    data_matrix = np.zeros((len(POLLUTANTS), len(sv_models)))

    # 2. Dynamic & Safe Extraction
    for j, pol in enumerate(POLLUTANTS):
        for i, model in enumerate(sv_models):
            # Use .get() to safely pull the value, defaulting to np.nan if missing
            data_matrix[j, i] = benchmark[pol].get(model, {}).get("MAPE", np.nan)

    # 3. FIXED: Changed figsize to a wide landscape format (16x5)
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot the heatmap
    cax = ax.imshow(data_matrix, cmap="RdYlGn_r", aspect="auto")

    # Set the ticks
    ax.set_yticks(np.arange(len(POLLUTANTS)))
    ax.set_xticks(np.arange(len(sv_models)))

    # 4. FIXED: Rotate the X-axis labels so the long model names don't overlap
    ax.set_yticklabels(POLLUTANTS, fontweight="bold", fontsize=12)
    ax.set_xticklabels(sv_models, fontweight="bold", fontsize=10, rotation=35, ha="right")

    # Safely calculate the max value for color contrast (ignoring NaNs)
    max_val = np.nanmax(data_matrix)

    # 5. Annotate cells safely
    for j in range(len(POLLUTANTS)):
        for i in range(len(sv_models)):
            val = data_matrix[j, i]

            if not np.isnan(val):
                # Dynamic contrast: Darker/redder backgrounds get white text
                text_color = "white" if val > max_val * 0.7 else "black"
                # ax.text(x, y, string) -> i is x (column), j is y (row)
                ax.text(i, j, f"{val:.1f}%", ha="center", va="center", color=text_color, fontweight="bold")
            else:
                # Fill empty cells if a model failed
                ax.text(i, j, "N/A", ha="center", va="center", color="gray", fontweight="bold")

    ax.set_title("Comprehensive Model Matrix — MAPE (%)\n(Lower is Better)", fontsize=16, pad=20, fontweight="bold")

    # Format the colorbar
    fig.colorbar(cax, ax=ax, fraction=0.02, pad=0.02, label="MAPE (%)")

    plt.tight_layout()
    plt.savefig("outputs/12_mape_heatmap_comparison.png", dpi=150, bbox_inches="tight")
    print("[✓] Inverted Heatmap saved: outputs/12_mape_heatmap_comparison.png")


def plot_sv_vs_mv_ablation(benchmark: dict):
    """
    Generates an Ablation Study chart explicitly comparing SV vs MV performance.
    Dynamically extracts real metrics from the merged benchmark dictionary.
    Includes failsafes (np.nan) to prevent crashes if a model failed to train.
    """
    fig, axes = plt.subplots(len(POLLUTANTS), 1, figsize=(14, 4 * len(POLLUTANTS)), sharex=True)
    fig.suptitle("Ablation Study: Univariate (SV) vs Multivariate (MV) Feature Importance", fontsize=18, fontweight="bold", y=0.99)

    x = np.arange(len(DL_ARCHITECTURES))
    width = 0.35

    for ax, col in zip(axes, POLLUTANTS):
        sv_mapes = []
        mv_mapes = []

        # 1. Dynamic & Safe Extraction
        for arch in DL_ARCHITECTURES:
            # We use .get() chained with np.nan.
            # If a model failed to train, it returns NaN instead of crashing the script.
            sv_val = benchmark[col].get(f"SV-{arch}", {}).get("MAPE", np.nan)
            mv_val = benchmark[col].get(f"MV-{arch}", {}).get("MAPE", np.nan)

            sv_mapes.append(sv_val)
            mv_mapes.append(mv_val)

        # 2. Matplotlib natively handles np.nan by simply leaving the bar empty
        ax.bar(x - width/2, sv_mapes, width, label='SV (Pollution Only)', color='#CCCCCC', edgecolor="black")
        ax.bar(x + width/2, mv_mapes, width, label='MV (Pollution + Weather)', color='#1D3557', edgecolor="black")

        # 3. Dynamic Annotation Logic
        for i, (sv_val, mv_val) in enumerate(zip(sv_mapes, mv_mapes)):
            # Only calculate improvement if BOTH models successfully trained
            if not np.isnan(sv_val) and not np.isnan(mv_val):
                improvement = sv_val - mv_val
                color = "green" if improvement > 0 else "red"
                sign = "-" if improvement > 0 else "+"

                ax.text(x[i], max(sv_val, mv_val) + 0.5, f"{sign}{abs(improvement):.1f}%",
                        ha='center', va='bottom', fontweight='bold', color=color, fontsize=11)

            # Provide helpful visual feedback if one is missing
            elif not np.isnan(sv_val):
                ax.text(x[i] - width/2, sv_val + 0.5, "SV Only", ha='center', va='bottom', fontsize=9, color='gray')
            elif not np.isnan(mv_val):
                ax.text(x[i] + width/2, mv_val + 0.5, "MV Only", ha='center', va='bottom', fontsize=9, color='gray')

        # 4. Standard Formatting
        ax.set_title(f"Target: {col}", fontweight="bold", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(DL_ARCHITECTURES, fontsize=11, fontweight="bold", rotation=15)
        ax.set_ylabel("MAPE (%)", fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        if col == POLLUTANTS[0]:
            ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig("outputs/13_ablation_study_sv_vs_mv.png", dpi=150, bbox_inches="tight")
    print("[✓] Figure saved: outputs/13_ablation_study_sv_vs_mv.png")


# =====================================================================
# THE UPGRADED SUMMARY EXPORT (MAPE & R^2)
# =====================================================================
def export_and_print_summary(benchmark: dict):
    """Calculates the dynamic champion using MAPE and R^2, prints the console summary, and exports to CSV."""

    # 1. Flatten the dictionary into a Pandas DataFrame
    records = []
    for pol in POLLUTANTS:
        for model in MODELS:
            records.append({
                "Target": pol,
                "Model": model,
                "MAPE": benchmark[pol][model].get("MAPE", 0),
                "RMSE": benchmark[pol][model].get("RMSE", 0),
                "MAE": benchmark[pol][model].get("MAE", 0),
                "R^2": benchmark[pol][model].get("R^2", 0) # [FIX]: Safely extracting R^2
            })

    df_metrics = pd.DataFrame(records)

    # 2. Pivot the table to create calculation matrices
    df_mape = df_metrics.pivot(index="Model", columns="Target", values="MAPE").reindex(MODELS)
    df_r2 = df_metrics.pivot(index="Model", columns="Target", values="R^2").reindex(MODELS)

    df_mape["AVG_MAPE"] = df_mape.mean(axis=1)
    df_r2["AVG_R2"] = df_r2.mean(axis=1)

    # 3. Dynamically find the Overall Best Model using the Multi-Objective Tuple Trick
    best_score = (float('inf'), float('inf'))
    champion_model = "None"

    for model in MODELS:
        avg_mape = df_mape.loc[model, "AVG_MAPE"]
        avg_r2 = df_r2.loc[model, "AVG_R2"]

        # Skip if model failed or data is missing
        if pd.isna(avg_mape) or pd.isna(avg_r2): continue

        current_score = (avg_mape, -avg_r2)

        # If the tuple beats the current best, promote the model!
        if current_score < best_score:
            best_score = current_score
            champion_model = model

    print("\n" + "="*105 + "\nCOMPREHENSIVE RESULTS — ALL MODELS × ALL POLLUTANTS\n" + "="*105)

    # Print the table nicely
    header = f"{'Model':<20}" + "".join(f" {col:>10}" for col in POLLUTANTS) + f" | {'AVG MAPE':>9} | {'AVG R^2':>7}"
    print(header)
    print("-" * 105)

    for m in MODELS:
        row = f"{m:<20}"
        for col in POLLUTANTS:
            row += f" {df_mape.loc[m, col]:>9.1f}%"

        # Output both the Average MAPE and the Average R^2
        row += f" | {df_mape.loc[m, 'AVG_MAPE']:>8.1f}% | {df_r2.loc[m, 'AVG_R2']:>7.3f}"

        if m == champion_model:
            row += "  <-- OVERALL CHAMPION"
        print(row)

    print("-" * 105)

    # 4. Dynamic Improvement Calculation (Champion vs Baseline ARIMA)
    print(f"\n{f'{champion_model} vs ARIMA (Improvement)':<38}")
    row_improvement = f"{'':<20}"
    for col in POLLUTANTS:
        arima_mape = df_mape.loc["ARIMA", col]
        champ_mape = df_mape.loc[champion_model, col]

        # Protect against division by zero
        if arima_mape != 0:
            percent_improve = ((arima_mape - champ_mape) / arima_mape) * 100
            row_improvement += f" {percent_improve:>9.0f}% ↓"
        else:
            row_improvement += f" {'N/A':>11}"

    print(row_improvement)
    print("="*105)

    # 5. Export to CSV for BI Dashboards
    csv_path = Path("outputs/14_final_benchmark_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n[✓] Raw metrics (including R^2) successfully exported to: {csv_path.name}")


import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt


def plot_dl_all_metrics_comparison(benchmark: dict):
    """
    Generates a unified subplot grid comparing SV vs MV models.
    Inverted Layout: Metrics (MAPE, R²) are Rows, Pollutants are Columns.
    UPGRADE: Enforces Global Y-Axis Scaling and Global Color Mapping per Metric.
    """
    metrics = ["MAPE", "R^2"]
    num_pol = len(POLLUTANTS)
    x = np.arange(len(DL_ARCHITECTURES))
    width = 0.35

    # =========================================================
    # 1. PRE-COMPUTATION: Find Global Min/Max per Metric
    # =========================================================
    global_scales = {}
    for metric in metrics:
        all_scores = []
        for col in POLLUTANTS:
            for arch in DL_ARCHITECTURES:
                # Safely extract all values to find the global boundaries
                all_scores.append(benchmark[col].get(f"SV-{arch}", {}).get(metric, np.nan))
                all_scores.append(benchmark[col].get(f"MV-{arch}", {}).get(metric, np.nan))

        valid_scores = np.array(all_scores)[~np.isnan(all_scores)]

        if len(valid_scores) > 0:
            g_min, g_max = valid_scores.min(), valid_scores.max()
        else:
            g_min, g_max = 0, 1  # Failsafe

        # Calculate Y-Axis Boundaries
        # Midpoint of the lowest value -> g_min * 0.5
        # If R^2 dips below zero, we multiply by 1.1 to give it breathing room at the bottom
        y_bottom = g_min * 0.5 if g_min >= 0 else g_min * 1.1
        y_top = g_max * 1.05  # Add 5% headroom so the tallest bars don't hit the ceiling

        global_scales[metric] = {
            "vmin": g_min,
            "vmax": g_max,
            "y_bottom": y_bottom,
            "y_top": y_top
        }

    # =========================================================
    # 2. MATRIX PLOTTING
    # =========================================================
    fig, axes = plt.subplots(len(metrics), num_pol, figsize=(4.5 * num_pol, 10), squeeze=False)
    fig.suptitle("Deep Learning Architecture Analysis — SV vs MV",
                 fontsize=22, fontweight="bold", y=0.98)

    for metric_idx, metric in enumerate(metrics):
        g_scale = global_scales[metric]  # Fetch our globally synchronized scales

        for pol_idx, col in enumerate(POLLUTANTS):
            ax = axes[metric_idx, pol_idx]
            sv_scores, mv_scores = [], []

            for arch in DL_ARCHITECTURES:
                sv_scores.append(benchmark[col].get(f"SV-{arch}", {}).get(metric, np.nan))
                mv_scores.append(benchmark[col].get(f"MV-{arch}", {}).get(metric, np.nan))

            # Use the GLOBAL min/max for the colormap, not the local subplot min/max!
            cmap = plt.get_cmap("RdYlGn_r") if metric == "MAPE" else plt.get_cmap("RdYlGn")
            norm = mcolors.Normalize(vmin=g_scale["vmin"], vmax=g_scale["vmax"])

            sv_colors = [cmap(norm(v)) if not np.isnan(v) else 'lightgray' for v in sv_scores]
            mv_colors = [cmap(norm(v)) if not np.isnan(v) else 'lightgray' for v in mv_scores]

            ax.bar(x - width / 2, sv_scores, width, color=sv_colors, edgecolor="black")
            ax.bar(x + width / 2, mv_scores, width, color=mv_colors, edgecolor="black", hatch='///')

            # ENFORCE GLOBAL Y-AXIS SCALE
            ax.set_ylim(g_scale["y_bottom"], g_scale["y_top"])

            # ==========================================
            # 3. MATRIX FORMATTING
            # ==========================================
            if metric_idx == 0:
                ax.set_title(f"Target: {col}", fontweight="bold", fontsize=16)

            ax.set_xticks(x)

            if metric_idx == len(metrics) - 1:
                ax.set_xticklabels(DL_ARCHITECTURES, fontsize=11, fontweight="bold", rotation=25)
            else:
                ax.set_xticklabels([])

            # Metric labels ONLY on the Far-Left Column
            if pol_idx == 0:
                if metric == "MAPE":
                    ax.set_ylabel("MAPE Error (%)", fontsize=13, fontweight="bold")
                elif metric == "R^2":
                    ax.set_ylabel("R² Ratio", fontsize=13, fontweight="bold")
            else:
                # Hide Y-Axis tick labels for inner columns to keep the matrix ultra-clean
                ax.set_yticklabels([])

            if metric == "R^2":
                ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

            if metric_idx == 0 and pol_idx == 0:
                legend_elements = [
                    Patch(facecolor='white', edgecolor='black', label='Univariate (SV)'),
                    Patch(facecolor='white', edgecolor='black', hatch='///', label='Multivariate (MV)'),
                    Patch(facecolor='none', edgecolor='none', label='* Colors map to Global Performance')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.15, wspace=0.05)  # wspace=0.05 pulls the columns tightly together

    save_path = "outputs/15_dl_metrics_comparison_inverted.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    print(f"[✓] Globally Scaled metrics comparison saved: {save_path}")

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    classical_fp = Path("outputs/classical_model_results.json")
    sv_nn_fp = Path(r"C:\Users\Tiago\IdeaProjects\AirTS_Forecast\outputs\sv_DL_results.pkl")   # Updated to match the export filename from your Unvariate script
    mv_nn_fp = Path(r"C:\Users\Tiago\IdeaProjects\AirTS_Forecast\outputs\mv_DL_results.pkl")

    benchmark = load_results(classical_filepath=classical_fp, sv_nn_filepath=sv_nn_fp, mv_nn_filepath=mv_nn_fp)

    plot_mape_heatmap(benchmark)
    plot_sv_vs_mv_ablation(benchmark)
    export_and_print_summary(benchmark)
    plot_dl_all_metrics_comparison(benchmark)

if __name__ == "__main__":
    main()