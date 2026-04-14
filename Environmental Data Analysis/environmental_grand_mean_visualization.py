"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: environmental_grand_mean_visualization.py
Author: Tiago TOLOCZKO ROSS

Description:
Application module utilizing the core orchestrator. Analyzes the global/grand
mean of environmental variables across a continuous time period. Extracts 1x1
DataFrames containing the spatial and temporal grand mean per month, concatenates
them into a unified chronological timeseries, and delegates PDF generation.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Import specific retrievers and the master orchestrator
from environmental_data_retrieval import (grand_mean_retrieval, period_retrieval_function)
from environmental_data_visualization_orchestration import visualization_orchestration

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 1. MATHEMATICAL AGGREGATION ENGINE
# =============================================================================

def overall_period_grand_mean(
        period_data_dict: dict[str, list[pd.DataFrame]]
) -> dict[str, pd.DataFrame]:
    """
    Iterates over the period of monthly files, extracting the single 1x1
    grand mean DataFrame for each month, and seamlessly concatenates them
    into a continuous 1D temporal DataFrame.
    """
    overall_stats = {}

    for var, df_list in period_data_dict.items():
        if not df_list:
            logger.warning(f"No data available for variable '{var}' to concatenate.")
            continue

        continuous_grand_mean_df = pd.concat(df_list, axis=0, ignore_index=True)
        continuous_grand_mean_df.index.name = "Time_Step"
        overall_stats[var] = continuous_grand_mean_df

    return overall_stats


# =============================================================================
# 2. RENDERING ENGINE & FACTORY
# =============================================================================
def fit_sinusoidal_monthly(time_idx: pd.Index, data: pd.Series) -> tuple[np.ndarray, list, float] | None:
    """
    Fits a 12-month period sinusoidal function to the given 1D time-series data.
    Returns the fitted Y array, the parameters (A, phi, C), and the Pearson correlation.
    """
    valid_mask = ~np.isnan(data)

    if valid_mask.sum() < 12:
        return None

    t_full = time_idx.values
    t_valid = t_full[valid_mask]
    y_valid = data[valid_mask].values

    omega = 2 * np.pi / 12.0

    def sin_func(t, a, phi, c):
        return a * np.sin(omega * t + phi) + c

    guess_c = float(np.mean(y_valid))
    guess_a = float((np.max(y_valid) - np.min(y_valid)) / 2.0)
    guess_phi = 0.0

    try:
        popt, _ = curve_fit(sin_func, t_valid, y_valid, p0=[guess_a, guess_phi, guess_c], maxfev=5000)
        y_fit_full = sin_func(t_full, *popt)
        y_fit_valid = sin_func(t_valid, *popt)

        correlation = float(np.corrcoef(y_valid, y_fit_valid)[0, 1])

        # We use 0.85 for Grand Means because global averages have flatter amplitudes than local regions
        if abs(correlation) > 0.85:
            return y_fit_full, popt.tolist(), correlation
        else:
            return None

    except Exception as exc:
        logger.debug(f"Curve fitting failed to converge: {exc}")
        return None

def grand_mean_plotter_factory() -> Callable[[dict[str, pd.DataFrame], str], Optional[plt.Figure]]:
    """
    Conforms to the universal orchestrator signature. Accepts a single-variable
    dictionary from the orchestrator. Intelligently renders a Bar chart for
    single-month steps, or a Line chart (with optional sinusoidal fit) for the
    Overall Period evolution.
    """
    def plot_generator(
            extracted_data: dict[str, pd.DataFrame],
            plot_title: str
    ) -> Optional[plt.Figure]:

        if not extracted_data:
            return None

        var, df = next(iter(extracted_data.items()))

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"Grand Mean Evolution: {var.upper()}", fontsize=16, fontweight='bold')
        ax.set_title(plot_title, fontsize=12, color='gray')

        if len(df) == 1:
            # ---------------------------------------------------------
            # SINGLE MONTH SCENARIO (1x1 DataFrame)
            # ---------------------------------------------------------
            val = df["Grand_Mean"].iloc[0]

            ax.bar(["Monthly Mean"], [val], color='teal', alpha=0.8, edgecolor='black')
            ax.text(0, val, f"{val:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

            ax.set_ylabel(f"{var.upper()} Grand Mean", fontweight='bold')
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        else:
            # ---------------------------------------------------------
            # OVERALL PERIOD SCENARIO (Nx1 DataFrame)
            # ---------------------------------------------------------
            ax.plot(
                df.index, df["Grand_Mean"],
                marker='o', linestyle='-', color='teal',
                linewidth=2.5, markersize=8, label="Grand Mean Trend"
            )

            # Check and apply 12-Month Sinusoidal Fit
            fit_result = fit_sinusoidal_monthly(df.index, df["Grand_Mean"])

            if fit_result is not None:
                # ---> NEW: Unpack the correlation value <---
                fitted_y, popt, corr = fit_result
                a, phi, c = popt
                omega = 2 * np.pi / 12.0

                logger.info(
                    f"Strong seasonal correlation ({abs(corr)*100:.1f}%) for {var.upper()}: "
                    f"y(t) = {a:.2f} * sin({omega:.4f}*t + {phi:.2f}) + {c:.2f}"
                )

                ax.plot(
                    df.index, fitted_y,
                    linestyle=':', color='darkorange', linewidth=2.5,
                    label="12-Month Sinusoidal Fit"
                )

            ax.set_xlabel("Months", fontweight='bold')
            ax.set_ylabel(f"{var.upper()} Grand Mean", fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)

            # ---> FORCE EVERY MONTH TO DISPLAY <---
            ax.set_xticks(df.index)

            # Rotate labels to prevent overlap if the period is very long
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            ax.legend(loc='best')

        plt.tight_layout()
        return fig

    return plot_generator

# =============================================================================
# 4. TABULAR VISUALIZATION AND EXPORTING FUNCTIONS
# =============================================================================

def calculate_grand_mean_sinusoidal_statistics(
        continuous_data_dict: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Analyzes the continuous grand mean timeseries, attempts a 12-month
    sinusoidal fit for each variable, and compiles the successful fits into a
    single summary DataFrame.

    Args:
        continuous_data_dict (dict): The concatenated Overall timeseries dictionary
                                     generated by `overall_period_grand_mean`.

    Returns:
        pd.DataFrame: A unified DataFrame containing the fit parameters (A, phi, C)
                      for every globally significant variable.
    """
    stats_list = []
    omega = 2 * np.pi / 12.0

    for var, df in continuous_data_dict.items():
        if len(df) < 12:
            continue

        fit_result = fit_sinusoidal_monthly(df.index, df["Grand_Mean"])

        if fit_result is not None:
            _, popt, correlation = fit_result
            a, phi, c = popt

            stats_list.append({
                "Variable": var.upper(),
                "Correlation (%)": f"{abs(correlation) * 100:.2f}%",
                "Amplitude (A)": a,
                "Phase Shift (phi)": phi,
                "Vertical Shift/Mean (C)": c,
                "Equation": f"y(t) = {a:.2f} * sin({omega:.4f}*t + {phi:.2f}) + {c:.2f}"
            })

    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        # Sort by highest correlation first
        stats_df = stats_df.sort_values(by="Correlation (%)", ascending=False)
        return stats_df

    return pd.DataFrame()


def print_grand_mean_sinusoidal_summaries(
        continuous_data_dict: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculates the sinusoidal statistics and prints a cleanly formatted
    summary table to the console for quick verification.
    """
    stats_df = calculate_grand_mean_sinusoidal_statistics(continuous_data_dict)

    if stats_df.empty:
        logger.warning("No globally significant sinusoidal correlations found to print.")
        return stats_df

    print(f"\n{'='*90}")
    print("GLOBAL GRAND MEAN SEASONAL FIT SUMMARY")
    print(f"{'='*90}")
    # Print the entire table cleanly without the arbitrary numeric index
    print(stats_df.to_string(index=False))
    print("\n")

    return stats_df


def export_grand_mean_stats_to_excel(
        stats_df: pd.DataFrame,
        output_filename: str | Path = "grand_mean_fit_statistics.xlsx",
        output_dir: str | Path = "Excel exported statistical summaries"
) -> Path | None:
    """
    Exports the successful Grand Mean sinusoidal fit parameters to an Excel workbook.
    Because there is only one "Region" (Global), all variables are placed on a
    single, easily readable summary sheet.

    Args:
        stats_df (pd.DataFrame): The DataFrame generated by calculate_grand_mean_sinusoidal_statistics.
        output_filename (str | Path, optional): Name of the Excel file.
        output_dir (str | Path, optional): Target directory.

    Returns:
        Path | None: Absolute path to the generated Excel file.
    """
    if stats_df.empty:
        logger.warning("No grand mean statistics provided to the Excel exporter.")
        return None

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / Path(output_filename).name

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write everything to one master sheet, hiding the arbitrary row index
            stats_df.to_excel(writer, sheet_name="Global_Seasonal_Fits", index=False)

        logger.info(f"Successfully exported grand mean tables to {output_path.absolute()}")
        return output_path

    except Exception as exc:
        logger.error(f"Failed to export grand mean stats to Excel: {exc}")
        return None


# =============================================================================
# 4. EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    target_dir = Path("era5_monthly_data")

    if target_dir.exists():
        logger.info(f"\nInitiating Grand Mean Period Analysis on directory: {target_dir.name}...")

        try:
            # 1. Gather Period Data directly via the core extraction engine
            period_data = period_retrieval_function(target_dir, grand_mean_retrieval)

            if period_data:
                # 2. Stitch into continuous data
                continuous_grand_means = overall_period_grand_mean(period_data)

                # 3. Print the table to the console and export to Excel
                stats_df = print_grand_mean_sinusoidal_summaries(continuous_grand_means)
                export_grand_mean_stats_to_excel(stats_df)

            # 1. PDF Export Engine
            exported_files = visualization_orchestration(
                input_path=target_dir,
                study="Global Grand Mean",
                retrieval_func=grand_mean_retrieval,
                plot_generator_func=grand_mean_plotter_factory(),
                overall_analysis=overall_period_grand_mean,
                objective='export',
                output_dir="Exported grand mean plots",
                output_filename="grand_mean_analysis.pdf",
                verbose=False
            )

            if exported_files:
                logger.info(f"\nSUCCESS: Grand Mean Pipeline completed. Generated {len(exported_files)} PDF reports.")

            # 2. Interactive Display Engine
            logger.info("\nLaunching Interactive Display (Showing continuous Grand Mean trend)...")
            visualization_orchestration(
                input_path=target_dir,
                study="Global Grand Mean",
                retrieval_func=grand_mean_retrieval,
                plot_generator_func=grand_mean_plotter_factory(),
                overall_analysis=overall_period_grand_mean,
                objective='show',
                verbose=False
            )

            plt.show()

        except Exception as e:
            logger.error(f"Grand Mean Analysis failed: {e}")

    else:
        logger.error(f"Target directory not found at: {target_dir.absolute()}")