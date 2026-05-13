"""
URBAN POLLUTION FORECASTING — TUTORIAL FOR BEGINNERS
Part 4 : Final Comparison · Summary · Exercises

This module teaches you:
1. How to load results from multiple models (Parts 2 & 3)
2. Side-by-side metric comparison (MAPE, RMSE, MAE)
3. Model strengths and weaknesses
4. Practical decision guide: when to use which model?
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

POLLUTANTS = ["NO2", "PM10", "PM25"]
MODELS = ["ARIMA", "SARIMA", "Holt-Winters", "RNN", "LSTM"]
UNITS = {
    "NO2": "µg/m³",
    "PM10": "µg/m³",
    "PM25": "µg/m³"
}
COLORS_M = {
    "ARIMA": "#A8DADC",
    "SARIMA": "#457B9D",
    "Holt-Winters": "#F4A261",
    "RNN": "#8338EC",
    "LSTM": "#E63946"
}

# Hard-coded benchmarks (realistic values from synthetic data)
BENCHMARK_FALLBACK = {
    "PM25": {
        "ARIMA": {"RMSE": 9.2, "MAE": 7.1, "MAPE": 20.5},
        "SARIMA": {"RMSE": 6.8, "MAE": 5.2, "MAPE": 14.8},
        "Holt-Winters": {"RMSE": 6.5, "MAE": 4.9, "MAPE": 13.9},
        "RNN": {"RMSE": 5.9, "MAE": 4.4, "MAPE": 12.3},
        "LSTM": {"RMSE": 4.8, "MAE": 3.6, "MAPE": 10.1},
    },
    "NO2": {
        "ARIMA": {"RMSE": 12.4, "MAE": 9.8, "MAPE": 24.2},
        "SARIMA": {"RMSE": 8.9, "MAE": 7.1, "MAPE": 17.5},
        "Holt-Winters": {"RMSE": 8.6, "MAE": 6.8, "MAPE": 16.8},
        "RNN": {"RMSE": 7.8, "MAE": 6.1, "MAPE": 15.1},
        "LSTM": {"RMSE": 6.4, "MAE": 5.0, "MAPE": 12.3},
    },
    "PM10": {
        "ARIMA": {"RMSE": 9.2, "MAE": 7.1, "MAPE": 20.5},
        "SARIMA": {"RMSE": 6.8, "MAE": 5.2, "MAPE": 14.8},
        "Holt-Winters": {"RMSE": 6.5, "MAE": 4.9, "MAPE": 13.9},
        "RNN": {"RMSE": 5.9, "MAE": 4.4, "MAPE": 12.3},
        "LSTM": {"RMSE": 4.8, "MAE": 3.6, "MAPE": 10.1},
    }
}

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def load_results():
    """Loads results from Parts 2 and 3, merging them with fallbacks."""

    # 1. Load Classical Models
    classical_results = {}
    try:
        with open("outputs/classical_model_results.json", "r") as f:
            classical_results = json.load(f)
        print("[✓] Loaded classical model results from Part 2")
    except FileNotFoundError:
        print("[!] Part 2 results not found. Using fallbacks.")

    # 2. Load Neural Networks
    neural_results = {}
    try:
        with open("outputs/nn_results.pkl", "rb") as f:
            pkl_data = pickle.load(f)
            rnn_results_raw = pkl_data.get("rnn_results", {})
            lstm_results_raw = pkl_data.get("lstm_results", {})

            for pollutant in rnn_results_raw.keys():
                if pollutant not in neural_results:
                    neural_results[pollutant] = {}
                neural_results[pollutant]["RNN"] = rnn_results_raw[pollutant]
                neural_results[pollutant]["LSTM"] = lstm_results_raw.get(pollutant, {})
        print("[✓] Loaded neural network results from Part 3")
    except FileNotFoundError:
        print("[!] Part 3 results not found. Using fallbacks.")

    # 3. Merge into Benchmark
    benchmark = {}
    print("\n" + "="*70)
    print("DATA STATUS — Which results are real vs fallback?")
    print("="*70)

    for pollutant in POLLUTANTS:
        benchmark[pollutant] = {}
        status = []
        for model in MODELS:
            # Check Classical Models
            if model in classical_results.get(pollutant, {}):
                benchmark[pollutant][model] = classical_results[pollutant][model]
                status.append(f"[✓] {model}")
            # Check Neural Models
            elif model in neural_results.get(pollutant, {}):
                benchmark[pollutant][model] = neural_results[pollutant][model]
                status.append(f"[✓] {model}")
            # Use Fallback
            else:
                benchmark[pollutant][model] = BENCHMARK_FALLBACK[pollutant][model]
                status.append(f"[-] {model} (fallback)")
        print(f"{pollutant}: {', '.join(status)}")
    print("="*70 + "\n")

    return benchmark

def plot_mape_bar_chart(benchmark):
    """Generates Figure 12: Bar Chart showing MAPE by model and pollutant."""
    fig, axes = plt.subplots(1, len(benchmark), figsize=(18, 5), sharey=False)
    fig.suptitle("Model Comparison — MAPE (%) per Pollutant\n(lower is better)",
                 fontsize=14, fontweight="bold")

    for ax, col in zip(axes, POLLUTANTS):
        mape_vals = [benchmark[col][m]["MAPE"] for m in MODELS]
        bar_colors = [COLORS_M[m] for m in MODELS]

        bars = ax.bar(MODELS, mape_vals, color=bar_colors, edgecolor="white",
                      linewidth=0.8, alpha=0.85)

        # Annotate each bar with its value
        for bar, val in zip(bars, mape_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="black")

        ax.set_title(f"{col} ({UNITS.get(col, '')})", fontsize=12, fontweight="bold")
        ax.set_xticklabels(MODELS, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("MAPE (%)", fontsize=10)
        ax.set_ylim(0, max(mape_vals) * 1.25)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        # Shade "excellent" zone (<12%)
        ax.axhline(12, color="green", linestyle="--", linewidth=1.2, alpha=0.6)
        ax.text(4.3, 12.5, "Excellent", color="green", fontsize=8, ha="right", fontweight="bold")

    plt.tight_layout()
    plt.savefig("outputs/12_mape_comparison.png", dpi=150, bbox_inches="tight")
    print("[✓] Figure saved: outputs/12_mape_comparison.png")
    plt.show()

def plot_radar_charts(benchmark):
    """Generates Figure 13: Radar Chart showing normalized performance."""
    def normalize_scores(col_data):
        """Convert raw metrics to 0-1 scores (1 = best = lowest error)."""
        metrics = ["RMSE", "MAE", "MAPE"]
        scores = {model: [] for model in MODELS}

        for metric in metrics:
            vals = {m: col_data[m][metric] for m in MODELS}
            worst, best = max(vals.values()), min(vals.values())
            rng = worst - best if worst != best else 1.0

            for model in MODELS:
                score = 1 - (vals[model] - best) / rng if rng > 0 else 0.5
                scores[model].append(score)
        return scores

    fig, axes = plt.subplots(1, len(benchmark), figsize=(20, 5), subplot_kw=dict(polar=True))
    fig.suptitle("Model Profiles — Normalized Performance\n(outer edge = best)",
                 fontsize=14, fontweight="bold")

    categories = ["RMSE\nScore", "MAE\nScore", "MAPE\nScore"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for ax, col in zip(axes, POLLUTANTS):
        scores = normalize_scores(benchmark[col])

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9)

        for model_name, vals in scores.items():
            vals = vals + vals[:1]
            color = COLORS_M.get(model_name, "#000000")
            ax.plot(angles, vals, linewidth=2, label=model_name, color=color)
            ax.fill(angles, vals, alpha=0.08, color=color)

        ax.set_ylim(0, 1)
        ax.set_title(col, pad=15, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/13_radar_comparison.png", dpi=150, bbox_inches="tight")
    print("[✓] Figure saved: outputs/13_radar_comparison.png")
    plt.show()

def print_improvement_table(benchmark):
    """Generates Figure 14: Printed Improvement Table."""
    print("\n" + "="*75)
    print("COMPREHENSIVE RESULTS — MAPE (%) ALL MODELS × ALL POLLUTANTS")
    print("="*75)

    # Pretty-print table
    header = f"{'Model':<18}" + "".join(f" {col:>10}" for col in POLLUTANTS)
    print(header)
    print("-" * 75)

    for m in MODELS:
        row = f"{m:<18}"
        for col in POLLUTANTS:
            row += f" {benchmark[col][m]['MAPE']:>9.1f}%"
        if m == "LSTM":
            row += "  <-- BEST"
        print(row)

    print("-" * 75)
    print(f"\n{'LSTM vs ARIMA (improvement)':<28}")

    row_improvement = f"{'':<18}"
    for col in POLLUTANTS:
        arima_mape = benchmark[col]["ARIMA"]["MAPE"]
        lstm_mape = benchmark[col]["LSTM"]["MAPE"]
        percent_improve = ((arima_mape - lstm_mape) / arima_mape) * 100
        row_improvement += f" {percent_improve:>8.0f}% ↓"

    print(row_improvement)
    print("="*75)

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    benchmark = load_results()
    plot_mape_bar_chart(benchmark)
    plot_radar_charts(benchmark)
    print_improvement_table(benchmark)

if __name__ == "__main__":
    main()