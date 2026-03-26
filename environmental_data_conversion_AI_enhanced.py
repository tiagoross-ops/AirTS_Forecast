"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: environmental_data_conversion_AI_enhanced.py.py
Author: Tiago TOLOCZKO ROSS

Description:
Data retrieval and exploration tools adapted for the ERA5-Land databases.
Provides utilities to inspect NetCDF (.nc) files and convert multidimensional
climate data arrays into flattened pandas DataFrames for time-series forecasting.
"""

import logging
from pathlib import Path

import h5py
import netCDF4 as cd
import numpy as np
import pandas as pd

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def structure_inspection(file_path: Path, debug_mode: bool = False) -> dict[str, list[str]]:
    """
    Inspects a NetCDF file to extract its variable and dimension names.

    Args:
        file_path (Path): The absolute or relative path to the NetCDF file.
        debug_mode (bool): If True, logs the extracted structure. Defaults to False.

    Returns:
        dict[str, list[str]]: A dictionary containing two keys: 'variable' and 'dimensions',
                              each mapping to a list of strings representing the names.
    """
    if debug_mode:
        logger.debug(f"Analysing structure of {file_path.name}")

    try:
        with cd.Dataset(file_path, mode="r") as ds:
            variable_keys = list(ds.variables.keys())
            dimensions_names = list(ds.dimensions.keys())
    except Exception as e:
        logger.error(f"Failed to open or read {file_path.name}: {e}")
        return {'variable': [], 'dimensions': []}

    if debug_mode:
        logger.debug(f"Variables found: {variable_keys}")
        logger.debug(f"Dimensions found: {dimensions_names}")

    return {'variable': variable_keys, 'dimensions': dimensions_names}


def environmental_data_conversion_era5_to_dfs(
        files_folder: str | Path,
        doc_type: str = '.nc',
        verbose: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Reads ERA5-Land NetCDF files from a directory and flattens them into pandas DataFrames.

    Iterates through all files matching the document type, extracts the primary data
    variable and its coordinate dimensions, generates a Cartesian grid, and maps
    the multidimensional arrays to tabular DataFrame format.

    Args:
        files_folder (str | Path): Directory containing the target data files.
        doc_type (str): File extension to target. Defaults to '.nc'.
        verbose (bool): If True, enables debug-level logging for step-by-step processing. Defaults to False.

    Returns:
        dict[str, pd.DataFrame]: A dictionary mapping the main variable names (str)
                                 to their flattened DataFrames.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    folder_path = Path(files_folder)
    coordinated_grid_dict: dict[str, pd.DataFrame] = {}

    if not folder_path.exists() or not folder_path.is_dir():
        logger.critical(f"Directory not found or invalid: '{folder_path}'")
        return coordinated_grid_dict

    logger.info(f"Initiating data conversion in directory: {folder_path}")

    files = list(folder_path.glob(f"*{doc_type}"))

    if not files:
        logger.warning(f"No files matching '{doc_type}' found in '{folder_path}'.")
        return coordinated_grid_dict

    for file_path in files:
        try:
            # Step 1: Inspect the file structure
            inspection = structure_inspection(file_path, debug_mode=verbose)
            variables = inspection['variable']
            dimensions = inspection['dimensions']

            if not variables:
                logger.warning(f"Skipping {file_path.name} due to unreadable or empty structure.")
                continue

            main_var = variables[0]
            extracted_dims = [dim for dim in variables if dim in dimensions]

            # Step 2: Extract multidimensional arrays
            with cd.Dataset(file_path, mode="r") as ds:
                main_variable_array = np.array(ds[main_var][:])

                dim_arrays = {}
                for dim in extracted_dims:
                    raw_data = ds[dim][:]
                    if isinstance(raw_data, np.ma.MaskedArray):
                        dim_arrays[dim] = raw_data.filled(np.nan)
                    else:
                        dim_arrays[dim] = np.array(raw_data)

                        # Step 3: Compute Cartesian product and construct DataFrame
            grids = np.meshgrid(*dim_arrays.values(), indexing="ij")

            df_data = {dim_name: grid.ravel() for dim_name, grid in zip(dim_arrays.keys(), grids)}
            df_data[main_var] = main_variable_array.ravel()

            df = pd.DataFrame(df_data)
            coordinated_grid_dict[main_var] = df

            logger.debug(f"Successfully converted {file_path.name} to tabular format.")

        except MemoryError:
            logger.error(f"MemoryError: {file_path.name} arrays are too large for RAM during meshgrid generation.")
        except KeyError as e:
            logger.error(f"KeyError: Missing expected variable/dimension in {file_path.name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while processing {file_path.name}: {e}")

    logger.info(f"Data conversion completed. Total variables extracted: {len(coordinated_grid_dict)}")
    return coordinated_grid_dict


def environmental_data_conversion_era5_to_3d(
        files_folder: str | Path,
        doc_type: str = '.nc',
        verbose: bool = False
) -> dict:
    """
    Reads ERA5-Land NetCDF files and extracts them as 3-dimensional NumPy arrays,
    preserving their original spatial and temporal structures.

    Returns:
        dict: A nested dictionary where the main key is the variable name.
              The inner dictionary contains:
              - 'data': The 3D NumPy array.
              - 'coordinates': A dictionary of the 1D axis arrays (e.g., time, lat, lon).
              - 'dimensions': A list of the dimension names in the correct order.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    folder_path = Path(files_folder)
    structured_data_dict = {}

    if not folder_path.exists() or not folder_path.is_dir():
        logger.critical(f"Directory not found: '{folder_path}'")
        return structured_data_dict

    files = list(folder_path.glob(f"*{doc_type}"))

    for file_path in files:
        try:
            with cd.Dataset(file_path, mode="r") as ds:
                # Identify variables vs dimensions
                all_vars = list(ds.variables.keys())
                dimensions = list(ds.dimensions.keys())

                # Assume the first non-coordinate variable is the main data variable
                main_var = [v for v in all_vars if v not in dimensions][0]
                extracted_dims = ds[main_var].dimensions # usually ('time', 'latitude', 'longitude')

                # 1. Extract the 3D data array without flattening
                raw_main_data = ds[main_var][:]
                if isinstance(raw_main_data, np.ma.MaskedArray):
                    main_variable_array = raw_main_data.filled(np.nan)
                else:
                    main_variable_array = np.array(raw_main_data)

                # 2. Extract the 1D coordinate axes
                coords = {}
                for dim in extracted_dims:
                    raw_coord = ds[dim][:]
                    if isinstance(raw_coord, np.ma.MaskedArray):
                        coords[dim] = raw_coord.filled(np.nan)
                    else:
                        coords[dim] = np.array(raw_coord)

                        # Package the 3D structure
            structured_data_dict[main_var] = {
                "data": main_variable_array,
                "coordinates": coords,
                "dimensions": list(extracted_dims)
            }

            logger.debug(f"Successfully extracted 3D variable '{main_var}' with shape {main_variable_array.shape}.")

        except Exception as e:
            logger.error(f"Error extracting 3D data from {file_path.name}: {e}")

    return structured_data_dict


if __name__ == '__main__':
    # Implementation example
    # Define target folder using raw string for Windows paths or Path object
    target_folder = Path(r'C:\Users\Tiago\IdeaProjects\AirTS_Forecast\stored_monthly_data\2004_05_01-31')

    # Execute conversion with verbose logging enabled
    result_data = environmental_data_conversion_era5_to_dfs(target_folder, verbose=True)

    if result_data:
        # Display sample of the first extracted variable
        first_variable = list(result_data.keys())[0]
        logger.info(f"Displaying head for variable: {first_variable}")
        print(result_data[first_variable].head())