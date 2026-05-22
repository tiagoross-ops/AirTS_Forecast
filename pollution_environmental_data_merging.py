"""
AirTS-Forecast Project
Module: Data Merging (Environmental + Pollution)
Description:
Loads time-series data from two distinct Parquet sources (Pollutants and
Environmental Variables), standardizes their temporal indices, applies specified
periodicity (hourly/daily), and performs an inner merge to construct a finalized
multivariate dataset for RNN training.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Union
import seaborn as sns
import matplotlib.pyplot as plt

# Configure our logger to track the execution process
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_and_standardize_parquet(file_path: Path, source_name: str, periodicity: str = "daily") -> pd.DataFrame:
    """
    Loads a Parquet file, guarantees a standardized DatetimeIndex, and resamples
    the data to the specified periodicity.
    """
    try:
        df = pd.read_parquet(file_path)

        # Standardize the index to a proper Datetime format (checking common naming conventions)
        if "timestamp" in df.columns:
            df.set_index("timestamp", inplace=True)
        elif "Timestamp" in df.columns:
            df.set_index("Timestamp", inplace=True)

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Apply Periodicity Resampling
        if periodicity.lower() == "daily":
            df = df.resample('D').mean()
        elif periodicity.lower() == "hourly":
            df = df.resample('H').mean()
        else:
            logger.warning(f"Unrecognized periodicity '{periodicity}'. Leaving data at original frequency.")

        return df

    except Exception as e:
        logger.error(f"Failed to load {source_name} Parquet from {file_path.name}: {e}")
        return pd.DataFrame()

def load_environmental_data(env_path: Path, periodicity: str = "daily") -> pd.DataFrame:
    """
    Intelligently loads environmental data. Handles both a single consolidated
    Parquet file or a directory containing multiple variable-specific Parquet files.
    """
    if env_path.is_file() and env_path.suffix == ".parquet":
        return load_and_standardize_parquet(env_path, "Environmental", periodicity)

    elif env_path.is_dir():
        logger.info(f"Directory detected. Consolidating Parquet files from {env_path.name}...")
        env_dfs = []

        for p_file in env_path.glob("*.parquet"):
            df = load_and_standardize_parquet(p_file, p_file.name, periodicity)
            if not df.empty:
                env_dfs.append(df)

        if not env_dfs:
            logger.error("No valid Parquet files found in the environmental directory.")
            return pd.DataFrame()

        # Concatenate horizontally (axis=1) aligning strictly on the DatetimeIndex
        consolidated_env_df = pd.concat(env_dfs, axis=1)

        # Drop duplicated columns if any exist across the merged files
        consolidated_env_df = consolidated_env_df.loc[:, ~consolidated_env_df.columns.duplicated()]
        return consolidated_env_df

    else:
        logger.error(f"Provided environmental path is invalid: {env_path}")
        return pd.DataFrame()


def analyze_feature_correlations(
        df: pd.DataFrame,
        targets: list,
        features: list,
        top_k: int = 3,
        save_directory: Path = None,
        method: str = 'spearman',
        verboose:bool=True
) -> dict:
    """
    Calculates the correlation matrix between pollutants and weather variables.
    Plots a heatmap and returns a dictionary mapping each target to its top_k most
    correlated environmental features.
    """
    if verboose: print(f"\n{'=' * 70}\nFEATURE SELECTION: {method.capitalize()} Correlation\n{'=' * 70}")

    # Calculate the correlation matrix for all targets and features
    corr_matrix = df[targets + features].corr(method=method)

    # Generate a visual heatmap focused only on Target vs Feature relationships
    plt.figure(figsize=(10, 6))
    target_feature_corr = corr_matrix.loc[targets, features]
    sns.heatmap(target_feature_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(f"Target vs Environmental Features ({method.capitalize()} Correlation)", fontweight="bold")
    plt.tight_layout()

    # Ensure output directory exists for the plot
    Path("PollutionDataAnalysis/outputs/plots").mkdir(parents=True, exist_ok=True)
    if save_directory is not None:
        save_directory = save_directory
    else:
        save_directory = Path("PollutionDataAnalysis/outputs/plots/correlation_heatmap.png")
    plt.savefig(save_directory, dpi=150)
    if verboose: print(f"[✓] Correlation heatmap saved to 'outputs/plots/correlation_heatmap.png'")
    plt.show()

    best_features_dict = {}

    for target in targets:
        if target not in corr_matrix.columns:
            continue

        # Isolate the correlations for this specific pollutant
        target_corrs = corr_matrix[target].drop(labels=targets, errors='ignore') # Drop other pollutants

        # We care about magnitude (absolute value). A strong negative correlation
        # (e.g., wind clearing out pollution) is just as useful as a positive one!
        abs_corrs = target_corrs.abs().sort_values(ascending=False)

        # Select the top 'k' feature names
        top_features = abs_corrs.head(top_k).index.tolist()
        best_features_dict[target] = top_features

        if verboose: print(f"\n Top {top_k} Predictors for {target}:")
        for feat in top_features:
            # Print the real correlation value (with its original +/- sign)
            if verboose: print(f"   -> {feat}: {target_corrs[feat]:.3f}")

    return best_features_dict


def merge_environmental_and_pollution(
        pollution_parquet_path: Union[str, Path],
        environmental_parquet_path: Union[str, Path],
        output_parquet_path: Union[str, Path],
        periodicity: str = "daily"
) -> pd.DataFrame:
    """
    Main orchestrator: Loads pollution data as the base dataset, iterates through
    the environmental Parquet directory, iteratively merges each variable as a
    new column, and saves the final multivariate dataset based on the chosen periodicity.
    """
    logger.info(f"Step 1: Loading Base Pollution Data (Periodicity: {periodicity})...")
    merged_df = load_and_standardize_parquet(Path(pollution_parquet_path), "Pollution", periodicity)

    if merged_df.empty:
        logger.error("Merge aborted: Pollution dataset is empty or failed to load.")
        return pd.DataFrame()

    logger.info("Step 2: Iterating and Merging Environmental Data...")
    env_dir = Path(environmental_parquet_path)

    if not env_dir.is_dir():
        logger.error(f"Merge aborted: Environmental path is not a valid directory -> {env_dir}")
        return pd.DataFrame()

    env_files = list(env_dir.glob("*.parquet"))
    if not env_files:
        logger.error("Merge aborted: No Parquet files found in the environmental directory.")
        return pd.DataFrame()

    # Iteratively join each environmental variable file as a new column
    for env_file in env_files:
        if __name__ == '__main__': logger.info(f"Appending environmental feature: {env_file.name}")
        env_df = load_and_standardize_parquet(env_file, env_file.stem, periodicity)

        if env_df.empty:
            logger.warning(f"Skipping {env_file.name}: File is empty or failed to load.")
            continue

        # 'inner' join guarantees we only keep timestamps present across ALL features
        merged_df = merged_df.merge(env_df, left_index=True, right_index=True, how="inner")

        # Early stopping if a merge eliminates all overlapping data
        if merged_df.empty:
            logger.error(f"Merge failed: Zero overlapping timestamps after joining {env_file.name}.")
            return pd.DataFrame()

    logger.info(f"Step 3: Saving final multivariate dataset to {output_parquet_path}...")

    # Ensure the parent directory exists before saving
    out_path = Path(output_parquet_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to Parquet using PyArrow with Snappy compression for optimal ML read speeds
    merged_df.to_parquet(out_path, engine="pyarrow", compression="snappy")

    logger.info(f"Success! Final multivariate dataset shape: {merged_df.shape}")
    return merged_df

# ==========================================
# IMPLEMENTATION EXAMPLE
# ==========================================
if __name__ == "__main__":
    # 1. Define your file paths
    POLLUTION_FILE = "PollutionDataAnalysis/outputs/consolidated_pollutants.parquet"
    WEATHER_PARQUET_SOURCE = "EnvironmentalDataAnalysis/Exported_Parquet_Data"
    FINAL_OUTPUT_FILE = "PollutionDataAnalysis/outputs/rnn_multivariate_dataset.parquet"

    # Define desired periodicity ("daily" or "hourly")
    DATA_PERIODICITY = "daily"

    # 2. Run the pipeline
    final_dataset = merge_environmental_and_pollution(
        pollution_parquet_path=POLLUTION_FILE,
        environmental_parquet_path=WEATHER_PARQUET_SOURCE,
        output_parquet_path=FINAL_OUTPUT_FILE,
        periodicity=DATA_PERIODICITY
    )

    if not final_dataset.empty:
        print(f"\nPreview of the new {DATA_PERIODICITY} multivariate dataset ready for your RNN:")
        print(final_dataset)

        targets = ["PM10", "PM25", "NO2", "NOx", "O3"]

        # Safely identify targets that exist in the dataframe
        valid_targets = [col for col in targets if col in final_dataset.columns]

        # Identify all columns that are NOT targets (these are our weather features)
        all_weather_features = [col for col in final_dataset.columns if col not in valid_targets]

        # --- Run Feature Selection ---
        if valid_targets and all_weather_features:
            optimal_features_dict = analyze_feature_correlations(
                df=final_dataset,
                targets=valid_targets,
                features=all_weather_features,
                top_k=3,
                method='spearman'
            )