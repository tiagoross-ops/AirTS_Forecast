from environmental_data_retrieval import (grand_mean_retrieval)
from environmental_data_statistics_exporting import visualization_orchestration

import logging
import pandas as pd

logger = logging.getLogger(__name__)

def overall_period_grand_mean(
        period_data_dict: dict[str, list[pd.DataFrame]]
) -> dict[str, pd.DataFrame]:
    """
    Mathematical Aggregation Engine: Iterates over the period of monthly files,
    extracting the single 1x1 grand mean DataFrame for each month, and seamlessly
    concatenates them into a continuous 1D temporal DataFrame.
    """
    overall_stats = {}

    for var, df_list in period_data_dict.items():
        if not df_list:
            logger.warning(f"No data available for variable '{var}' to concatenate.")
            continue

        # 1. Concatenate the 1x1 DataFrames along the row axis.
        # ignore_index=True converts the static [var] index into a 0, 1, 2... sequence.
        continuous_grand_mean_df = pd.concat(df_list, axis=0, ignore_index=True)

        # 2. Rename the index so the plotter knows it represents chronological steps
        continuous_grand_mean_df.index.name = "Time_Step"

        # 3. Store the finalized continuous DataFrame
        overall_stats[var] = continuous_grand_mean_df

    return overall_stats

import matplotlib.pyplot as plt
from typing import Callable

def grand_mean_plotter_factory() -> Callable[[dict[str, pd.DataFrame], str], plt.Figure]:
    """
    Conforms to the universal orchestrator signature. Accepts a single-variable
    dictionary from the orchestrator. Intelligently renders a Bar chart for
    single-month steps, or a Line chart for the Overall Period evolution.
    """
    def plot_generator(
            extracted_data: dict[str, pd.DataFrame],
            plot_title: str
    ) -> plt.Figure | None:

        if not extracted_data:
            return None

        # The orchestrator strictly passes 1 variable at a time
        var, df = next(iter(extracted_data.items()))

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"Grand Mean Evolution: {var.upper()}", fontsize=16, fontweight='bold')
        ax.set_title(plot_title, fontsize=12, color='gray')

        # Branching visualization based on dataset length
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

            ax.set_xlabel("Chronological Time Step (Months)", fontweight='bold')
            ax.set_ylabel(f"{var.upper()} Grand Mean", fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)

            # Force the X-axis to only show whole integer steps
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.legend(loc='best')

        plt.tight_layout()
        return fig

    return plot_generator


from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    target_dir = Path("era5_monthly_data")

    if target_dir.exists():
        logger.info(f"\nInitiating Grand Mean Period Analysis on directory: {target_dir.name}...")

        try:
            # 1. PDF Export Engine
            # Note: grand_mean_retrieval and overall_period_grand_mean are passed WITHOUT parentheses
            # because we are injecting the functions themselves. grand_mean_plotter_factory() IS called
            # because it is a factory that returns the actual plotting function.
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

            # 'show' objective safely isolates the Overall period figures and clears the rest from RAM
            plt.show()

        except Exception as e:
            logger.error(f"Grand Mean Analysis failed: {e}")

    else:
        logger.error(f"Target directory not found at: {target_dir.absolute()}")