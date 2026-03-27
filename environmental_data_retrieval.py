"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: environmental_data_retrieval.py
Author: Tiago TOLOCZKO ROSS

Description:
Core data retrieval and file management module. Handles the validation, parsing,
and chronological sorting of climate data directories. Provides higher-order
functions to inject mathematical retrieval logic into file-reading loops.
"""

import logging
import re
from pathlib import Path
from typing import Callable, Optional
import warnings

import h5py
import pandas as pd
import numpy as np

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def file_name_comprehension(file_path: Path) -> tuple[int, int, str]:
    """
    Parses and validates the filename of a climate dataset to extract
    its year, month, and data type structure.

    Expected format: 'era5_{type}_{YYYY}_{MM}.h5'
    Example: 'era5_3d_2004_06.h5'

    Args:
        file_path (Path): The complete file path of the dataset.

    Returns:
        tuple[int, int, str]: A tuple containing (year, month, data_type).

    Raises:
        ValueError: If the filename does not match the expected study format,
                    is not an .h5 file, or contains an invalid month.
    """
    filename = file_path.name

    # 1. Validate file extension cleanly using Pathlib
    if file_path.suffix != '.h5':
        logger.error(f"Invalid extension: '{filename}' is not a valid .h5 format.")
        raise ValueError(f"File '{filename}' must be an .h5 file.")

    # 2. Use Regular Expressions for strict validation and extraction in one step
    pattern = re.compile(r"^era5_(3d|pd)_(\d{4})_(\d{2})\.h5$")
    match = pattern.match(filename)

    if not match:
        logger.error(f"Format mismatch: '{filename}' does not match the study format.")
        raise ValueError(f"File '{filename}' is not recognized as a valid study dataset.")

    # 3. Extract the clean components directly from the regex groups
    type_df = match.group(1)
    year = int(match.group(2))
    month = int(match.group(3))

    # 4. Final logical validation for the month
    if not (1 <= month <= 12):
        logger.error(f"Invalid temporal data: '{filename}' presents an invalid month ({month}).")
        raise ValueError(f"Month must be between 1 and 12. Found: {month}")

    return year, month, type_df


def file_var_retrieval(
        month_file: Path,
        retrieval_func: Callable[[str, h5py.File | h5py.Group], Optional[pd.DataFrame]],
        verbose: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Reads climate data from an HDF5 file, validates its structure, and applies
    a dynamically injected retrieval function to extract specific variable data
    into formatted spatial DataFrames.

    Args:
        month_file (Path): The absolute or relative path to the target .h5 file.
        retrieval_func (Callable): The specific mathematical extraction function
                                   to apply to each variable found in the dataset.
        verbose (bool): If True, enables detailed logging for each successfully
                        extracted variable.

    Returns:
        dict[str, pd.DataFrame]: A dictionary mapping string variable names to their
                                 corresponding extracted Pandas DataFrames. Returns
                                 an empty dictionary if the file is invalid or unreadable.
    """
    if not month_file.exists():
        logger.error(f"Target file does not exist: {month_file.absolute()}")
        return {}

    bidimensional_data: dict[str, pd.DataFrame] = {}

    try:
        # Open the HDF5 file safely using a context manager
        with h5py.File(month_file, mode="r") as h5f:

            # Isolate primary datasets, safely ignoring coordinate groups and metadata
            main_vars = [key for key in h5f.keys() if isinstance(h5f[key], h5py.Dataset)]

            if not main_vars:
                logger.warning(f"No valid HDF5 Datasets found in the root of {month_file.name}.")
                return {}

            for var in main_vars:
                # Delegate the mathematical extraction entirely to the injected function
                df = retrieval_func(var, h5f)

                if df is not None:
                    bidimensional_data[var] = df
                    if verbose:
                        logger.info(f"Successfully validated and extracted '{var}'.")

    except OSError as e:
        logger.error(f"OS Error: Failed to open or read HDF5 file {month_file.name}: {e}")
        return {}
    except Exception as e:
        logger.critical(f"Critical Error during extraction of {month_file.name}: {e}")
        return {}

    return bidimensional_data


def monthly_data_directory_exploration(data_dir: Path) -> tuple[list[Path], list[int]]:
    """
    Scans the target directory, delegates filename validation to the comprehension helper,
    and generates a chronologically ordered list of valid 3D HDF5 file paths
    along with their temporal boundaries.

    Args:
        data_dir (Path): Directory containing the monthly .h5 files.

    Returns:
        tuple:
            - list[Path]: An ordered list of verified pathlib.Path objects.
            - list[int]: Temporal bounds formatted as [start_month, start_year, end_month, end_year].

    Raises:
        ValueError: If any file in the directory deviates from the expected format
                    or contains invalid temporal data.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir.absolute()}")
        return [], []

    unsorted_files = []

    for file_path in data_dir.iterdir():
        # Safely ignore subdirectories; only process files
        if file_path.is_file():
            try:
                # Delegate validation and extraction to our helper function
                year, month, type_df = file_name_comprehension(file_path)

                # Isolate 3D files (ignores 'pd' dataframes if they share the folder)
                if type_df == '3d':
                    unsorted_files.append((year, month, file_path))
            except ValueError as ve:
                logger.debug(f"Skipping {file_path.name}: {ve}")
                continue

    if not unsorted_files:
        logger.error("No valid 3D HDF5 files found in the specified directory.")
        return [], []

    # Sort the list chronologically (Year first, then Month natively)
    unsorted_files.sort(key=lambda x: (x[0], x[1]))

    # Extract the cleanly sorted Path objects back into a standard list
    target_files = [item[2] for item in unsorted_files]

    # Extract boundaries from the very first and very last items in the sorted list
    start_year, start_month = unsorted_files[0][0], unsorted_files[0][1]
    end_year, end_month = unsorted_files[-1][0], unsorted_files[-1][1]

    bounds_list = [start_month, start_year, end_month, end_year]

    logger.info(
        f"Auto-discovered {len(target_files)} chronological files. "
        f"Range: {start_month:02d}/{start_year} to {end_month:02d}/{end_year}."
    )

    return target_files, bounds_list


def period_retrieval_function(
        data_dir: Path,
        retrieval_func: Callable[[str, h5py.File | h5py.Group], Optional[pd.DataFrame]],
        verbose: bool = False
) -> dict[str, list[pd.DataFrame]]:
    """
    Iterates over a directory of chronological HDF5 files, applying a specific
    data retrieval function to each file, and accumulating the results.

    Args:
        data_dir (Path): The directory containing the monthly .h5 files.
        retrieval_func (Callable): The core mathematical extraction function
                                   to apply to each variable.
        verbose (bool): If True, enables detailed debug logging.

    Returns:
        dict[str, list[pd.DataFrame]]: A dictionary where keys are variable names
                                       and values are chronological lists of DataFrames.
    """
    # 1. Fetch valid files using your auto-discovering directory iteration
    file_list, period_range = monthly_data_directory_exploration(data_dir)

    if not file_list:
        logger.warning(f"No valid files found in {data_dir.name} to process for the period.")
        return {}

    if verbose:
        s_month, s_year, e_month, e_year = period_range
        logger.info(f"Starting period retrieval across {len(file_list)} files "
                    f"({s_month:02d}/{s_year} to {e_month:02d}/{e_year})...")

    period_data_dict: dict[str, list[pd.DataFrame]] = {}

    # 2. Iterate through the chronological files
    for file in file_list:
        if verbose:
            logger.info(f"Extracting data from {file.name}...")

        # Delegate single-file extraction to your modular helper function
        file_data_dict = file_var_retrieval(
            month_file=file,
            retrieval_func=retrieval_func,
            verbose=False
        )

        # 3. Efficiently unpack and accumulate the DataFrames
        for var, df in file_data_dict.items():
            if var not in period_data_dict:
                period_data_dict[var] = []

            period_data_dict[var].append(df)

    if verbose:
        logger.info(f"Successfully compiled period data for {len(period_data_dict)} variables.")

    return period_data_dict


def grand_mean_retrieval(
        var: str,
        dataset: h5py.File | h5py.Group
) -> Optional[pd.DataFrame]:
    """
    Extracts the overall grand temporal and spatial mean of a 3D environmental dataset,
    safely ignoring missing/masked data (NaNs).

    Args:
        var (str): The name of the variable to extract.
        dataset (h5py.File | h5py.Group): The open HDF5 file or group.

    Returns:
        Optional[pd.DataFrame]: A 1x1 DataFrame containing the grand mean,
                                structured as Index=[Variable], Column=["Grand_Mean"].
    """
    try:
        # 1. Structural Validation
        coord_group_name = f"{var}_coordinates"
        if coord_group_name not in dataset:
            return None

        coord_group = dataset[coord_group_name]
        if 'latitude' not in coord_group or 'longitude' not in coord_group:
            return None

        ds = dataset[var]
        if len(ds.shape) < 3:
            return None

        # 2. Mathematical Extraction
        # Use ds[:] to load the tensor into RAM, and nanmean to ignore masked ocean bins
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            g_mean = float(np.nanmean(ds[:]))

        # 3. Construct a formal Pandas DataFrame for a scalar value
        # We wrap the mean in a list [g_mean] and explicitly define the index
        df = pd.DataFrame({"Grand_Mean": [g_mean]}, index=[var])
        df.index.name = "Variable"

        return df

    except Exception as var_e:
        logger.error(f"Unexpected error processing grand mean for '{var}': {var_e}")
        return None


if __name__ == '__main__':
    print("\n" + "="*50)
    print("TESTING MODULE: environmental_data_retrieval.py")
    print("="*50 + "\n")

    # 1. Define target testing directory
    test_dir = Path("era5_monthly_data")

    if not test_dir.exists():
        logger.error(f"Cannot run tests: The directory '{test_dir}' does not exist.")
    else:
        # --- TEST 1: Regex Comprehension ---
        logger.info("--- TEST 1: Regex & Filename Comprehension ---")
        try:
            sample_file = list(test_dir.glob("*.h5"))[0]
            year, month, type_df = file_name_comprehension(sample_file)
            logger.info(f"Successfully parsed {sample_file.name} -> Year: {year}, Month: {month}, Type: {type_df}\n")
        except IndexError:
            logger.warning(f"No .h5 files found in {test_dir.name} to test regex.\n")
        except Exception as e:
            logger.error(f"Regex parsing failed: {e}\n")

        # --- TEST 2: Directory Auto-Discovery ---
        logger.info("--- TEST 2: Directory Auto-Discovery ---")
        files, bounds = monthly_data_directory_exploration(test_dir)
        if bounds:
            logger.info(f"Found Bounds: Start {bounds[0]:02d}/{bounds[1]} | End {bounds[2]:02d}/{bounds[3]}\n")

        # --- TEST 3: Mock Retrieval Injection ---
        logger.info("--- TEST 3: Dependency Injection Extraction ---")

        # We create a tiny mock retrieval function strictly to prove the pipeline routes data correctly
        def mock_spatial_retrieval(var: str, dataset: h5py.File) -> Optional[pd.DataFrame]:
            """Mock function to test routing without importing actual math modules."""
            logger.debug(f"Mock function triggered for variable: {var}")
            # Return an empty dummy DataFrame to prove it successfully bypassed validation
            return pd.DataFrame()

        if files:
            test_target = files[0]
            logger.info(f"Testing extraction on {test_target.name}...")

            extracted_vars = file_var_retrieval(
                month_file=test_target,
                retrieval_func=mock_spatial_retrieval,
                verbose=True
            )
            grand_mean = file_var_retrieval(
                month_file=Path(r'C:\Users\Tiago\IdeaProjects\AirTS_Forecast\era5_monthly_data\era5_3d_2004_03.h5'),
                retrieval_func=grand_mean_retrieval
            )['d2m'].iat[0,0]
            print(grand_mean)
            
            if extracted_vars:
                logger.info(f"Successfully extracted {len(extracted_vars)} variables using mock injection.\n")
        else:
            logger.warning("No files available to test extraction.\n")


        print("="*50)
        print("TESTING COMPLETE")
        print("="*50 + "\n")