"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: FIRST_STEPS_basic_statistics.py
Author: Tiago TOLOCZKO ROSS

Description:
Mainly human-made code to kickstart the basic statistical analysis and set the framework for the AI-generated code
available in the other files
"""

import numpy as np
from pathlib import Path
import h5py
import logging

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def var_description_by_month(
    month_file: Path,
    granularity: float = .1,
    verbose: bool = False
) -> None:
    try:

        with h5py.File(str(month_file), mode="r") as h5f:
            var_list = list(h5f.keys())
            if verbose: logger.info(f'Found variables {var_list} in {month_file.name}')
            for var in var_list:
                ds = h5f[var]
                coordinates = None

                if type(ds) == h5py.Dataset: df = ds
                else:
                    coordinates = [coord for coord in ds]
                    print(df, f'Coordinates: {coordinates}')
                    array = df[:][:][:]
                    time_mean = np.average(array,axis=0)
                    print(time_mean)



            h5f.close()

    except Exception as e:
        logger.error(f"Error in retrieving data from {month_file.name}: \n{e}")


if __name__ == '__main__':
    test_file = Path(r'/Environmental Data Analysis/era5_monthly_data/era5_3d_2005_03.h5')
    var_description_by_month(test_file, verbose=True)

