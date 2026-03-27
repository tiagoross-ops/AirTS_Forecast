"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: environmental_data_statistics_exporting.py
Author: Tiago TOLOCZKO ROSS

Description:
Export and coordination module. Acts as a universal router that delegates data
extraction to the core engine, passes the data to dynamically injected rendering
engines, and packages the resulting Matplotlib figures into centralized,
multipage PDF reports.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import h5py
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import show

# Import the core extraction engine
from environmental_data_retrieval import (
    file_var_retrieval,
    period_retrieval_function,
    file_name_comprehension,
    monthly_data_directory_exploration
)

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def data_dict_listing(
        data_dict: dict[str, pd.DataFrame]
) -> dict[str, list[pd.DataFrame]]:
    """
    Normalizes a dictionary of DataFrames by wrapping each DataFrame in a list.
    Useful for adapting single-file extraction data to batch-processing plotters.
    """
    return {var: [df] for var, df in data_dict.items()}


def overall_period_data_function(
        period_data_dict: dict[str, list[pd.DataFrame]],
        overall_analysis: Callable[[dict[str, list[pd.DataFrame]]], dict[str, pd.DataFrame]]
) -> dict[str, list[pd.DataFrame]]:
    """
    Applies the mathematical aggregation function (e.g., NaN-safe mean or pd.concat)
    and APPENDS the resulting Overall DataFrame to the END of each variable's list.
    """
    overall_statistic = overall_analysis(period_data_dict)
    for var, ds_list in period_data_dict.items():
        if var in overall_statistic:
            ds_list.append(overall_statistic[var])

    return period_data_dict


def environmental_plotting_function(
        months_list: list[tuple],
        valid_plot_data_dict: dict[str, list[pd.DataFrame]],
        plot_generator_func: Callable[[dict[str, pd.DataFrame], str], plt.Figure],
) -> dict[str, list[plt.Figure]]:
    """
    Delegates the normalized dataset to the dynamically injected rendering engine.
    Returns a dictionary of Matplotlib Figure lists, grouped by variable.
    """
    # 1. Pre-initialize the dictionary keys with empty lists to avoid KeyErrors
    plt_dict: dict[str, list[plt.Figure]] = {var: [] for var in valid_plot_data_dict.keys()}

    # 2. Log once at the start of the batch to keep the console clean
    logger.info(f"Delegating figure generation for {len(valid_plot_data_dict)} variables across {len(months_list)} time steps...")

    for idx, time_tuple in enumerate(months_list):

        # 3. Format title cleanly using isinstance
        if isinstance(time_tuple[0], int):
            plot_title = f"{time_tuple[0]:04d} - {time_tuple[1]:02d}"
        else:
            plot_title = f"{time_tuple[0]} - {time_tuple[1]}"

        for var, data_list in valid_plot_data_dict.items():
            # Isolate the specific DataFrame for this variable and time step
            current_data_dict = {var: data_list[idx]}

            # Delegate to the injected renderer and append
            current_plt = plot_generator_func(current_data_dict, plot_title)
            plt_dict[var].append(current_plt)

    logger.info("Figure generation complete.")
    return plt_dict


def pdf_exporting_function(
        figs_dict: dict[str, list[plt.Figure]],
        months_list: list[tuple],
        target_dir: Path,
        study: str,
        is_period: bool,
        output_filename: str | Path
) -> list[Path]:
    """
    Assembles generated Matplotlib figures into properly paginated PDF files.
    Matches the indices of the generated figures strictly to the time steps
    provided in the months_list.
    """
    exported_paths = []
    clean_study_name = study.replace(" ", "_")

    if not figs_dict:
        logger.warning("No figures provided to the PDF exporter.")
        return exported_paths

    for idx, time_tuple in enumerate(months_list):

        # 1. Determine the exact Output File Name
        if not is_period:
            # Single-month analyses use the explicit output_filename
            output_path = target_dir / Path(output_filename).name
        else:
            # Period analyses dynamically generate file names based on the tuple
            if isinstance(time_tuple[0], int):
                file_name_id = f"month_{time_tuple[0]:04d}-{time_tuple[1]:02d}"
            else:
                # Handles strings like ("OVERALL", "AVERAGE")
                file_name_id = f"{time_tuple[0]}_{time_tuple[1]}".replace(" ", "_")

            output_path = target_dir / f"{clean_study_name}_{file_name_id}.pdf"

        # 2. Open PDF and Save Figures
        try:
            with PdfPages(output_path) as pdf:
                for var, plot_list in figs_dict.items():

                    # Safety check: Ensure the plotting engine actually returned a figure for this index
                    if idx < len(plot_list):
                        current_fig = plot_list[idx]

                        # Save it as a new page in this specific PDF
                        pdf.savefig(current_fig, bbox_inches='tight', pad_inches=0.5)

                        # CRITICAL: Close the figure immediately to prevent massive RAM leaks
                        plt.close(current_fig)

            exported_paths.append(output_path)

        except Exception as e:
            logger.error(f"Failed to save PDF {output_path.name}: {e}")

    logger.info(f"Batch export complete. Generated {len(exported_paths)} individual PDFs.")
    return exported_paths


def visualization_orchestration(
        input_path: Path,
        study: str,
        retrieval_func: Callable[[str, h5py.File | h5py.Group], Optional[pd.DataFrame]],
        plot_generator_func: Callable[[dict[str, pd.DataFrame], str], plt.Figure],
        overall_analysis: Callable[[dict[str, list[pd.DataFrame]]], dict[str, pd.DataFrame]] = None,
        objective: str = 'show',
        output_filename: str | Path = "climate_plots.pdf",
        output_dir: str | Path = "Exported pdf plots",
        verbose: bool = False
) -> Optional[Path | list[Path] | dict[str, list[plt.Figure]]]:
    """
    Acts as the master orchestration router for the environmental data pipeline.
    Dynamically routes execution based on the input path, coordinating data extraction,
    mathematical aggregation, visualization, and output handling.

    Behavioral Routing:
    - Single File Mode: Extracts data for a single month and generates a single set of plots.
    - Period Directory Mode: Extracts chronological data across multiple months, optionally
      calculates overall averages, and generates chronological step-plots.

    Args:
        input_path (Path): The target .h5 file for single-month analysis, or a directory
                           containing chronological .h5 files for period analysis.
        study (str): Master title prefix used for file naming and plot titles.
        retrieval_func (Callable): The injected factory function for extracting data.
        plot_generator_func (Callable): The injected rendering engine.
        overall_analysis (Callable, optional): The mathematical aggregation function. Defaults to None.
        objective (str, optional): Execution intent. 'show' returns the active Matplotlib
                                   figures in RAM. 'export' writes them to disk and clears RAM.
                                   Defaults to 'show'.
        output_filename (str | Path, optional): Explicit filename for single-file exports.
        output_dir (str | Path, optional): The target directory to save the assembled PDFs.
        verbose (bool, optional): If True, enables deep debug logging. Defaults to False.

    Returns:
        Optional[Path | list[Path] | dict[str, list[plt.Figure]]]:
            - dict[str, list[plt.Figure]]: If objective is 'show'.
            - Path: The absolute path to the generated PDF (if exporting a single file).
            - list[Path]: A list of absolute paths to generated PDFs (if exporting a period dir).
            - None: If extraction fails, paths are invalid, or objective is unrecognized.
    """
    if not input_path.exists():
        logger.error(f"Target path does not exist: {input_path.absolute()}")
        return None

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # MAIN ROUTING LOGIC
    # ==========================================
    month_list: list[tuple] = []

    if input_path.is_file():
        logger.info(f"Single file detected. Extracting data from {input_path.name}...")

        # 1. Retrieve Data
        valid_plots_data = file_var_retrieval(input_path, retrieval_func, verbose=verbose)
        if not valid_plots_data:
            logger.warning(f"No valid data extracted for PDF plotting in {input_path.name}.")
            return None

        valid_data = data_dict_listing(valid_plots_data)

        # 2. Generate Tuple metadata
        year, month, _ = file_name_comprehension(input_path)
        month_list = [(year, month)]

        is_period = False

    elif input_path.is_dir():
        logger.info(f"Directory detected. Extracting period data from {input_path.name}...")

        # 1. Retrieve Data
        period_data = period_retrieval_function(input_path, retrieval_func, verbose=verbose)
        if not period_data:
            logger.warning(f"No valid data extracted from {input_path.name}.")
            return None

        # 2. Append overall statistics if aggregation is provided
        if overall_analysis:
            valid_data = overall_period_data_function(period_data, overall_analysis)
        else:
            valid_data = period_data

        # 3. Generate Tuple metadata
        files, _ = monthly_data_directory_exploration(input_path)
        for file in files:
            year, month, _ = file_name_comprehension(file)
            month_list.append((year, month))

        # Add the Overall Tuple identifier to the end of the list
        if overall_analysis:
            month_list.append(("Period", "Overall"))

        is_period = True

    else:
        logger.error("Input path is neither a standard file nor a directory.")
        return None

    # ==========================================
    # EXECUTION PIPELINE
    # ==========================================

    # 1. Execute visualization engine
    figs = environmental_plotting_function(
        months_list=month_list,
        valid_plot_data_dict=valid_data,
        plot_generator_func=plot_generator_func
    )

    # 2. Conditional Branching based on Objective
    if objective == 'show':
        if is_period:
            logger.info("Objective is 'show' for a Period. Isolating Overall figures and clearing monthly steps to prevent window clutter.")
            filtered_figs = {}
            for var, fig_list in figs.items():
                if fig_list:
                    # The Overall Analysis is always appended last
                    overall_fig = fig_list[-1]
                    filtered_figs[var] = [overall_fig]

                    # Safely close all the monthly step figures so they don't consume RAM or pop up
                    for fig in fig_list[:-1]:
                        plt.close(fig)
            return filtered_figs
        else:
            logger.info("Objective is 'show'. Returning active single-month figures to caller.")
            return figs

    elif objective == 'export':
        logger.info("Objective is 'export'. Routing figures to PDF assembly engine...")
        pdf_paths = pdf_exporting_function(
            figs_dict=figs,
            months_list=month_list,
            target_dir=target_dir,
            study=study,
            is_period=is_period,
            output_filename=output_filename
        )
        return pdf_paths if is_period else (pdf_paths[0] if pdf_paths else None)

    else:
        logger.warning(f"Objective not specified: expected 'show' or 'export', got '{objective}'")
        return None