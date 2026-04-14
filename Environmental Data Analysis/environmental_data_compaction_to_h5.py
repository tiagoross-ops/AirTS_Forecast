"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: Data_Storage.py
Author: Tiago TOLOCZKO ROSS

Description:
Pipeline orchestration module for downloading ERA5-Land data, processing it into
pandas DataFrames, and streaming the results into dedicated monthly HDF5 files
for efficient time-series querying.
"""

import calendar
import logging
from pathlib import Path
import h5py
import pandas as pd

# Import the previously defined functions from your modules
from environmental_data_loading_AI_enhanced import era5_land_importing_by_date
from environmental_data_conversion_AI_enhanced import environmental_data_conversion_era5_to_3d

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def store_era5_to_hdf5(
        target_year: int,
        target_month: int,
        data_dict: dict[str, pd.DataFrame],
        output_directory: str | Path = "era5_monthly_data",
        verbose: bool = False
) -> Path:
    """
    Stores a pre-processed dictionary of ERA5 DataFrames into an HDF5 file.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    month_file_path = output_path / f"era5_{target_year}_{target_month:02d}.h5"

    try:
        with pd.HDFStore(str(month_file_path), mode="a", complevel=6, complib="blosc") as store:
            for var, df in data_dict.items():
                key = f"data/{var}"
                logger.debug(f"Writing variable '{var}' to {month_file_path.name} under key '{key}'...")
                store.put(key, df, format="table")
            logger.info(f"Successfully stored data for {target_year}-{target_month:02d} in {month_file_path.name}.")

    except Exception as e:
        logger.error(f"Failed to write HDF5 data for {target_year}-{target_month:02d}: {e}")

    return month_file_path


def store_era5_3d_to_hdf5(
        target_year: int,
        target_month: int,
        data_dict: dict,
        output_directory: str | Path = "era5_monthly_data",
        verbose: bool = False
) -> Path:
    """
    Stores a dictionary of 3-dimensional ERA5 NumPy arrays into an HDF5 file.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    month_file_path = output_path / f"era5_3d_{target_year}_{target_month:02d}.h5"

    try:
        # Open HDF5 file using the native h5py library
        with h5py.File(month_file_path, mode="w") as h5_file:

            for var, content in data_dict.items():
                data_3d = content["data"]
                coords = content["coordinates"]

                logger.debug(f"Writing 3D variable '{var}' to HDF5...")

                # Create or overwrite the main 3D dataset, applying gzip compression
                if var in h5_file:
                    del h5_file[var]

                # Store the main tensor (Time x Lat x Lon)
                h5_file.create_dataset(
                    name=var,
                    data=data_3d,
                    compression="gzip",
                    compression_opts=4
                )

                # Create a sub-group to store the 1-dimensional coordinate axes
                coord_group_name = f"{var}_coordinates"
                if coord_group_name in h5_file:
                    del h5_file[coord_group_name]

                coord_group = h5_file.create_group(coord_group_name)

                # Store Time, Latitude, and Longitude vectors
                for dim_name, coord_array in coords.items():
                    coord_group.create_dataset(name=dim_name, data=coord_array)

            logger.info(f"Successfully stored 3D data for {target_year}-{target_month:02d} in {month_file_path.name}.")

    except Exception as e:
        logger.error(f"Failed to write 3D HDF5 data for {target_year}-{target_month:02d}: {e}")

    return month_file_path


def run_era5_etl_pipeline_3d(
        target_year: int,
        start_month: int,
        end_month: int,
        output_directory: str | Path = "era5_monthly_data",
        verbose: bool = False
) -> Path:
    """
    Executes the full Extract, Transform, and Load (ETL) pipeline for a given time range.

    Iterates through the specified months, downloads the raw data, converts it to
    tabular format, and stores it in HDF5 files.

    Args:
        target_year (int): The year to process.
        start_month (int): The starting month (1-12).
        end_month (int): The ending month (1-12), inclusive.
        output_directory (str | Path): The directory to store the resulting .h5 files.
        verbose (bool): If True, enables debug-level logging.

    Returns:
        Path: The absolute path to the output directory containing the .h5 files.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    output_dir_path = Path(output_directory)
    logger.info(f"Initiating ERA5 ETL Pipeline. Target directory: {output_dir_path.absolute()}")

    # Iterate through the defined temporal range
    for current_month in range(start_month, end_month + 1):
        logger.info(f"--- Processing Pipeline for {target_year}-{current_month:02d} ---")

        # Determine the exact number of days for the current month/year
        day_start = 1
        day_end = calendar.monthrange(target_year, current_month)[1]

        try:
            # Step 1: EXTRACT
            logger.info("Executing Extraction (API Retrieval)...")
            target_zip, extracted_data_dir = era5_land_importing_by_date(
                target_year, current_month, day_start, day_end, verbose=verbose
            )

            # Step 2: TRANSFORM
            logger.info(f"Executing Transformation (Processing {extracted_data_dir.name})...")
            processed_data_dict = environmental_data_conversion_era5_to_3d(
                extracted_data_dir, verbose=verbose
            )

            if not processed_data_dict:
                logger.warning(f"Transformation yielded no data for {target_year}-{current_month:02d}. Skipping load phase.")
                continue

            # Step 3: LOAD
            logger.info("Executing Load (HDF5 Storage)...")
            store_era5_3d_to_hdf5(
                target_year=target_year,
                target_month=current_month,
                data_dict=processed_data_dict,
                output_directory=output_dir_path,
                verbose=verbose
            )

        except Exception as pipeline_error:
            # Catch exceptions to prevent a single month's failure from crashing the entire pipeline
            logger.critical(f"Pipeline failed for {target_year}-{current_month:02d}: {pipeline_error}")
            continue

    logger.info("Data pipeline orchestration completed successfully.")
    return output_dir_path


if __name__ == '__main__':
    # Define execution parameters

    #TODO: What if we change the year?
    year = 2021
    start_m = 3
    end_m = 12
    TARGET_OUTPUT_ROOT_DIR = "Analysis - 02 round"
    target_output_dir = f"{TARGET_OUTPUT_ROOT_DIR}/era5_monthly_data"

    # Execute the master ETL function
    final_directory = run_era5_etl_pipeline_3d(target_year=year, start_month=start_m, end_month=end_m,
                                               output_directory=target_output_dir, verbose=True)

    logger.info(f"All data successfully written to: {final_directory.absolute()}")