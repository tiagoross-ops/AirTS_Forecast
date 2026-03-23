"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: FIRST_STEPS_data_conversion.py
Author: Tiago TOLOCZKO ROSS

Description:
Mainly human-made code to set some ideas for loading step in data conversion
"""
import pandas as pd
import netCDF4 as cd
import numpy as np
from pathlib import Path

def structure_inspection(file_path: Path, print_partials: bool = False) -> dict:
    """Inspects the NetCDF file to extract variable and dimension names."""
    if print_partials:
        # Print a header for the specific file being analyzed
        print(80 * '-', f"\nAnalysing {file_path.name}")

    try:
        # Open the NetCDF dataset in read-only mode using a context manager
        with cd.Dataset(file_path, mode="r") as ds:
            # Extract all variable names into a list
            variable_keys = list(ds.variables.keys())
            # Extract all dimension names into a list
            dimensions_names = list(ds.dimensions.keys())
    except Exception as e:
        # Catch file reading errors, print the error, and return empty lists to prevent crashes
        print(f"Error opening or reading {file_path.name}: {e}")
        return {'variable': [], 'dimensions': []}

    if print_partials:
        # Output the extracted keys for debugging purposes
        print(f"Variables: {variable_keys}")
        print(f"Dimensions: {dimensions_names}")
        print(80 * '-')

    # Return the extracted names as a dictionary
    return {'variable': variable_keys, 'dimensions': dimensions_names}


def environmental_data_conversion_era5_to_dfs(files_folder: str, doc_type: str = '.nc', print_partials: bool = False) -> dict:
    """
    Reads ERA5-Land NetCDF files from a directory and converts them to pandas DataFrames.
    """
    # Convert the input string path to a Pathlib object for cross-platform handling
    folder_path = Path(files_folder)
    coordinated_grid_dict = {}

    # Validate that the provided folder path exists and is a directory
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Critical Error: The specified folder '{folder_path}' does not exist or is not a directory.")
        return coordinated_grid_dict

    print('-' * 30, f'Exploring {folder_path}', '-' * 30)

    # Create a list of all files in the directory matching the specified extension
    files = list(folder_path.glob(f"*{doc_type}"))

    # Check if the file list is empty and exit early if true
    if not files:
        print(f"Warning: No '{doc_type}' files were found in '{folder_path}'.")
        return coordinated_grid_dict

    # Iterate through each found file
    for file_path in files:
        try:
            # Step 1: Inspect the file structure to get variables and dimensions
            inspection = structure_inspection(file_path, print_partials)
            variables = inspection['variable']
            dimensions = inspection['dimensions']

            # Skip the file if structure inspection failed (returned empty lists)
            if not variables:
                print(f"Skipped file: {file_path.name} (Structure empty or unreadable).")
                continue

            # Identify the main variable (assuming it is the first one in the list)
            main_var = variables[0]
            # Filter variables to find which ones act as dimensions
            extracted_dims = [dim for dim in variables if dim in dimensions]

            # Step 2: Extract the actual data from the file
            with cd.Dataset(file_path, mode="r") as ds:
                # Load the main variable's data into a numpy array
                main_variable_array = np.array(ds[main_var][:])

                dim_arrays = {}
                # Iterate through the extracted dimensions to load their coordinate data
                for dim in extracted_dims:
                    raw = ds[dim][:]
                    # Check if the array is masked (contains missing/invalid data points)
                    if isinstance(raw, np.ma.MaskedArray):
                        # Fill masked values with NaN for compatibility with pandas
                        dim_arrays[dim] = raw.filled(np.nan)
                    else:
                        # Store the standard numpy array
                        dim_arrays[dim] = np.array(raw)

                        # Step 3: Create a Cartesian product of dimensions (Meshgrid) and build the DataFrame
            # Generate coordinate grids for all dimensions
            grids = np.meshgrid(*dim_arrays.values(), indexing="ij")

            # Flatten the grids and map them to their corresponding dimension names in a dictionary
            df_data = {dim_name: grid.ravel() for dim_name, grid in zip(dim_arrays.keys(), grids)}
            # Add the flattened main variable data to the dictionary
            df_data[main_var] = main_variable_array.ravel()

            # Convert the dictionary into a pandas DataFrame
            df = pd.DataFrame(df_data)
            # Store the resulting DataFrame in the main dictionary using the variable name as the key
            coordinated_grid_dict[main_var] = df

            if print_partials:
                print(f"Success: {file_path.name} converted to DataFrame.")

        # Catch memory errors specifically (common when flattening large multi-dimensional grids)
        except MemoryError:
            print(f"Memory Error: File {file_path.name} is too large. RAM exhausted during meshgrid creation.")
        # Catch missing keys if the expected variable/dimension is not in the dataset
        except KeyError as e:
            print(f"Key Error: Expected variable or dimension missing in {file_path.name}. Details: {e}")
        # Catch any other unexpected exceptions
        except Exception as e:
            print(f"Unexpected error processing {file_path.name}: {e}")

    print('-' * 30, f'Finished exploring {folder_path}', '-' * 30)

    # Return the final dictionary of DataFrames
    return coordinated_grid_dict

# Test code
if __name__ == '__main__':
    # Define the target folder path
    folder_path = r'C:\Users\Tiago\IdeaProjects\AirTS_Forecast\2004_4_1-30'
    # Execute the main conversion function
    result_dict = environmental_data_conversion_era5_to_dfs(folder_path, print_partials=True)

    # If the dictionary is not empty, print the first key and the head of its DataFrame
    if result_dict:
        first_key = list(result_dict.keys())[0]
        print(f"\nExtracted Variable: {first_key}")
        print(result_dict[first_key].head())