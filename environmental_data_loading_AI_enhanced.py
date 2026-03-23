"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: Data_Retrieval.py
Author: Tiago TOLOCZKO ROSS

Description:
Data retrieval module for fetching ERA5-Land daily statistics from the
Copernicus Climate Data Store (CDS) API and extracting the resulting archives.

About `AirTS-Forecast` :

A tutored project for ENIT/UTTOP Masters' Program, Semester M1.2

Professor (tutor): M. Raymond HOUÉ NGOUNA

Students: Lucas REINOSO URABAYEN;   Tiago TOLOCZKO ROSS

Goals: explore and analyze the influence of climate variables in air pollution using well-known databases, for educational and training purposes.
"""

import logging
import zipfile as zf
from pathlib import Path

import cdsapi

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def era5_land_importing_by_date(
        target_year: int,
        target_month: int,
        target_day_start: int,
        target_day_end: int,
        verbose: bool = False
) -> tuple[Path, Path]:
    """
    Downloads ERA5-Land daily statistics from the CDS API and extracts the zip file.

    Args:
        target_year (int): Year of the target data.
        target_month (int): Month of the target data (1-12).
        target_day_start (int): Starting day of the request.
        target_day_end (int): Ending day of the request.
        verbose (bool): If True, sets logging to DEBUG level. Defaults to False.

    Returns:
        tuple[Path, Path]: A tuple containing the Path to the downloaded zip file
                           and the Path to the extracted data directory.
    """
    if verbose:
        # Increase logging verbosity for debugging
        logger.setLevel(logging.DEBUG)

    # Apply zero-padding to month and day parameters for strict CDS API compatibility
    year_str = str(target_year)
    month_str = f"{target_month:02d}"
    day_span = [f"{i:02d}" for i in range(target_day_start, target_day_end + 1)]

    dataset = "derived-era5-land-daily-statistics"

    # Construct the API request dictionary
    request = {
        "variable": [
            "2m_dewpoint_temperature",
            "2m_temperature",
            "skin_temperature",
            "soil_temperature_level_1",
            "soil_temperature_level_2",
            "soil_temperature_level_3",
            "soil_temperature_level_4",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "forecast_albedo"
        ],
        "year": year_str,
        "month": month_str,
        "day": day_span,
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": [44.5, 10, 42, 14.5],
        "data_format": "csv",
    }

    # Define file paths using pathlib
    target_zip = Path(f"{target_year}_{month_str}_{target_day_start:02d}-{target_day_end:02d}.zip")
    target_directory = Path("stored_monthly_data") / target_zip.stem

    # Ensure the target directory exists before extraction
    target_directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initiating CDS API request for dataset: {dataset}")
    logger.debug(f"Target archive: {target_zip.name}")

    try:
        # Initialize CDS API client. Credentials must be stored locally in ~/.cdsapirc
        client = cdsapi.Client(url='https://cds.climate.copernicus.eu/api',
                               key='3897f715-96aa-4900-8608-156a4476eae5')
        # Execute the retrieval request
        client.retrieve(dataset, request, str(target_zip))
        logger.info("CDS API data retrieval completed successfully.")

    except Exception as e:
        logger.error(f"CDS API Retrieval Error: {e}")
        # Return paths to allow calling functions to handle the failure gracefully
        return target_zip, target_directory

    logger.info(f"Extracting contents to directory: {target_directory}")

    try:
        # Open the downloaded zip archive and extract to target directory
        with zf.ZipFile(target_zip, 'r') as myzip:
            myzip.extractall(path=target_directory)
        logger.info("Archive extraction completed successfully.")

    except zf.BadZipFile:
        logger.error(f"Extraction Error: {target_zip} is corrupted or invalid.")
    except Exception as e:
        logger.error(f"Unexpected IO error during extraction: {e}")

    return target_zip, target_directory


if __name__ == '__main__':
    # Define target period parameters
    year = 2004
    month = 4
    day_start = 1
    day_end = 30

    # Execute data import with debug logging enabled
    target_archive, extracted_dir = era5_land_importing_by_date(
        year, month, day_start, day_end, verbose=True
    )

    # Output final locations
    logger.info(f"Final zip archive location: {target_archive.absolute()}")
    logger.info(f"Final extracted data location: {extracted_dir.absolute()}")