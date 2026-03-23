"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: FIRST_STEPS_data_conversion_first_version.py
Author: Tiago TOLOCZKO ROSS

Description:
Mainly human-made code. First loading module


About `AirTS-Forecast` :

A tutored project for ENIT/UTTOP Masters' Program, Semester M1.2

Professor (tutor): M. Raymond HOUÉ NGOUNA

Students: Lucas REINOSO URABAYEN;   Tiago TOLOCZKO ROSS

Goals: explore and analyze the influence of climate variables in air pollution using well-known databases, for educational and training purposes.
"""
import cdsapi
import zipfile as zf
from pathlib import Path
import os

def era5_land_importing_by_date(target_year: int, target_month: int,
                                target_day_start: int, target_day_end: int) -> tuple[Path, Path]:
    """
    Downloads ERA5-Land daily statistics from the Copernicus Climate Data Store
    and extracts the resulting zip file.
    """
    # Convert dates to strings, applying zero-padding for months and days (e.g., '04' instead of '4')
    # This ensures strict compatibility with the CDS API formatting requirements
    year_str = str(target_year)
    month_str = f"{target_month:02d}"
    day_span = [f"{i:02d}" for i in range(target_day_start, target_day_end + 1)]

    # Define the dataset name
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
        "area": [44, 11, 41, 12],
        "data_format": "csv",
    }

    # Define file paths using pathlib for cross-platform compatibility
    target_zip = Path(f"{target_year}_{month_str}_{target_day_start:02d}-{target_day_end:02d}.zip")
    target_directory = Path("../stored_monthly_data") / target_zip.stem

    # Ensure the extraction directory exists
    target_directory.mkdir(parents=True, exist_ok=True)

    print(f"Starting download for {target_zip.name}...")

    try:
        # Initialize the CDS API client.
        # Note: Do not hardcode API keys. The client reads from the ~/.cdsapirc file or environment variables.
        client = cdsapi.Client(url='https://cds.climate.copernicus.eu/api',
                               key='3897f715-96aa-4900-8608-156a4476eae5')

        # Execute the retrieval request and save to the target zip path
        client.retrieve(dataset, request, str(target_zip))
        print("Download completed successfully.")

    except Exception as e:
        # Catch network or API-related errors
        print(f"API Retrieval Error: Failed to download data from CDS. Details: {e}")
        # Return the paths even if failed, or handle the exit as required by your pipeline
        return target_zip, target_directory

    print(f"Extracting files to {target_directory}...")

    try:
        # Open the downloaded zip file in read mode
        with zf.ZipFile(target_zip, 'r') as myzip:
            # Extract all contents to the designated target directory
            myzip.extractall(path=target_directory)
        print("Extraction completed successfully.")

    except zf.BadZipFile:
        # Catch errors related to corrupted or incomplete zip files
        print(f"Zip Extraction Error: The file {target_zip} is corrupted or not a valid zip archive.")
    except Exception as e:
        # Catch any other file I/O exceptions
        print(f"Unexpected error during extraction: {e}")

    # Return both the zip path and extraction directory to match the unpacking in the main block
    return target_zip, target_directory

if __name__ == '__main__':
    # Define time parameters
    year = 2004
    month = 4
    day_start = 1
    day_end = 30

    # Execute the function and unpack the returned tuple
    target, target_directory = era5_land_importing_by_date(year, month, day_start, day_end)

    # Verify outputs
    print(f"Zip archive: {target}")
    print(f"Extracted data location: {target_directory}")
