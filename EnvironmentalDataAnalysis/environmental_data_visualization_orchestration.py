"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: environmental_data_visualization_orchestration.py
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

# Import the core extraction engine tools
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
        plot_title: str,
        valid_plot_data_dict: dict[str, list[pd.DataFrame]],
        idx: int,
        plot_generator_func: Callable[..., Optional[plt.Figure]],
) -> dict[str, Optional[plt.Figure]]:
    """
    Delegates a specific chronological step of the dataset to the rendering engine.
    Extracts the DataFrame at the specified index for each variable and generates
    a standalone Matplotlib Figure. Forwards arbitrary arguments to the plotter.
    """
    plot_dict: dict[str, Optional[plt.Figure]] = {var: None for var in valid_plot_data_dict.keys()}

    logger.info(f"Delegating figure generation for {len(valid_plot_data_dict)} variables: [{plot_title}]...")

    for var, var_data_list in valid_plot_data_dict.items():
        if idx >= len(var_data_list):
            logger.warning(f"Index {idx} out of bounds for variable '{var}'. Skipping rendering.")
            continue

        current_data_dict = {var: var_data_list[idx]}

        try:
            # Inject arbitrary *args and **kwargs straight into the factory's generator
            current_fig = plot_generator_func(current_data_dict, plot_title)
            plot_dict[var] = current_fig
        except Exception as exc:
            logger.error(f"Rendering engine failed for '{var}' at step [{plot_title}]: {exc}")
            plot_dict[var] = None

    return plot_dict


def png_exporting_function(
        figs_dict: dict[str, plt.Figure],
        study_header: tuple,
        study: str,
        is_period: bool,
        output_target_dir: Path
) -> list[Path]:
    """
    Saves active Matplotlib figures as high-resolution PNG files.
    Crucially, this function DOES NOT close the figures, allowing them
    to remain active in memory for interactive display (plt.show()).
    """
    exported_pngs: list[Path] = []

    if not figs_dict:
        return exported_pngs

    # Create a dedicated subfolder for PNGs to keep things organized
    png_dir = output_target_dir / "png_dashboards"
    png_dir.mkdir(parents=True, exist_ok=True)

    clean_study_name = study.replace(" ", "_")

    # Format the time identifier
    if not is_period:
        file_name_id = f"month_{study_header[0]:04d}-{study_header[1]:02d}"
    else:
        file_name_id = f"{study_header[0]}_{study_header[1]}".replace(" ", "_")

    # Save each variable's figure
    for var, current_fig in figs_dict.items():
        if current_fig is None:
            continue

        clean_var_name = var.replace(" ", "_")
        output_name = f"{clean_study_name}_{clean_var_name}_{file_name_id}.png"
        output_path = png_dir / output_name

        try:
            # Save as high-res PNG (300 dpi is standard for publications/reports)
            current_fig.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
            exported_pngs.append(output_path)
            logger.debug(f"Saved PNG backup for {var}: {output_path.name}")
        except Exception as exc:
            logger.error(f"Failed to save PNG {output_name}: {exc}")

    if exported_pngs:
        logger.info(f"Saved {len(exported_pngs)} high-res PNG backups to: {png_dir.name}/")

    return exported_pngs


def pdf_exporting_function(
        study_header: tuple,
        figs_dict: dict[str, plt.Figure],
        study: str,
        is_period: bool,
        output_target_dir: Path,
        output_filename: str | Path,
        verbose: bool = False
) -> Path | None:
    """
    Assembles a dictionary of Matplotlib figures into a properly paginated PDF file.
    Designed to process a single chronological step (e.g., one specific month or the overall average).
    """
    if not figs_dict:
        logger.warning("No figures provided to the PDF exporter.")
        return None

    clean_study_name = study.replace(" ", "_")

    if not is_period:
        output_path = output_target_dir / Path(output_filename).name
    else:
        if isinstance(study_header[0], int):
            file_name_id = f"month_{study_header[0]:04d}-{study_header[1]:02d}"
        else:
            file_name_id = f"{study_header[0]}_{study_header[1]}".replace(" ", "_")

        output_path = output_target_dir / f"{clean_study_name}_{file_name_id}.pdf"

    try:
        with PdfPages(output_path) as pdf:
            for var, current_fig in figs_dict.items():
                if current_fig is None:
                    continue

                pdf.savefig(current_fig, bbox_inches='tight', pad_inches=0.5)

                if verbose:
                    logger.info(f"Rendered PDF page for variable: {var}")

                plt.close(current_fig)

        if verbose:
            logger.info(f"Successfully generated PDF: {output_path.name}")

        return output_path

    except Exception as exc:
        logger.error(f"Failed to save PDF {output_path.name}: {exc}")
        return None


def exporting_cycle_engine(
        months_list: list[tuple],
        valid_plot_data_dict: dict[str, list[pd.DataFrame]],
        plot_generator_func: Callable[..., Optional[plt.Figure]],
        study: str,
        is_period: bool,
        output_target_dir: Path,
        output_filename: str | Path,
        verbose: bool = False,
) -> list[Path]:
    """
    Orchestrates the chronological cycle of plotting and PDF exporting.
    Iterates through each time step, delegates rendering to the plotting engine,
    and immediately routes the resulting figures to the paginated PDF builder.
    """
    exported_paths: list[Path] = []

    logger.info(f"Initiating export cycle for {len(valid_plot_data_dict)} variables across {len(months_list)} time steps...")

    for idx, time_tuple in enumerate(months_list):
        if isinstance(time_tuple[0], int):
            plot_title = f"{time_tuple[0]:04d} - {time_tuple[1]:02d}"
        else:
            plot_title = f"{time_tuple[0]} - {time_tuple[1]}"

        fig_dict = environmental_plotting_function(
            plot_title=plot_title,
            valid_plot_data_dict=valid_plot_data_dict,
            idx=idx,
            plot_generator_func=plot_generator_func,
        )

        output_path = pdf_exporting_function(
            study_header=time_tuple,
            figs_dict=fig_dict,
            study=study,
            is_period=is_period,
            output_target_dir=output_target_dir,
            output_filename=output_filename,
            verbose=verbose
        )

        if output_path is not None:
            exported_paths.append(output_path)

    logger.info(f"Export cycle complete. Successfully generated {len(exported_paths)} PDFs.")
    return exported_paths


def visualization_orchestration(
        input_path: Path,
        study: str,
        retrieval_func: Callable[[str, h5py.File | h5py.Group], Optional[pd.DataFrame]],
        plot_generator_func: Callable[..., Optional[plt.Figure]],
        overall_analysis: Callable[[dict[str, list[pd.DataFrame]]], dict[str, pd.DataFrame]] = None,
        objective: str = 'show',
        save_png_backups: bool = True,
        output_filename: str | Path = "climate_plots.pdf",
        output_dir: str | Path = "Exported pdf plots",
        verbose: bool = False,
) -> Optional[Path | list[Path] | dict[str, Optional[plt.Figure]]]:
    """
    Master orchestration router for the environmental data visualization pipeline.

    Dynamically routes execution based on the input path (Single File vs. Period Batch)
    and coordinates data extraction, mathematical aggregation, visualization, and
    output handling according to the specified objective.

    Args:
        input_path (Path): Target .h5 file for single-step analysis, or directory of .h5 files for period analysis.
        study (str): Master title prefix used for file naming and plot titles.
        retrieval_func (Callable): Injected factory function responsible for extracting HDF5 data into DataFrames.
        plot_generator_func (Callable): Injected Matplotlib rendering engine.
        overall_analysis (Callable, optional): Mathematical aggregation engine for period analysis. Defaults to None.
        objective (str, optional): Execution intent ('show' or 'export'). Defaults to 'show'.
        save_png_backups (bool, optional): If True and objective is 'show', saves 300dpi PNG copies. Defaults to True.
        output_filename (str | Path, optional): Explicit PDF filename used ONLY for single-file exports.
        output_dir (str | Path, optional): Target directory for saving PDFs and PNGs. Defaults to "Exported PDF plots".
        verbose (bool, optional): Enables deep debug logging for individual page processing. Defaults to False.

    Returns:
        Optional[Path | list[Path] | dict[str, Optional[plt.Figure]]]:
            - dict[str, Optional[plt.Figure]]: Active Matplotlib figures if objective is 'show'.
            - Path: Absolute path to the generated PDF if exporting a single file.
            - list[Path]: Absolute paths to generated PDFs if exporting a period directory.
            - None: If extraction fails, paths are invalid, or objective is unrecognized.
    """
    if not input_path.exists():
        logger.error(f"Target path does not exist: {input_path.absolute()}")
        return None

    output_target_dir = Path(output_dir)
    output_target_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # MAIN ROUTING LOGIC
    # ==========================================
    month_list: list[tuple] = []

    if input_path.is_file():
        logger.info(f"Single file detected. Extracting data from {input_path.name}...")

        valid_plots_data = file_var_retrieval(input_path, retrieval_func, verbose=verbose)
        if not valid_plots_data:
            logger.warning(f"No valid data extracted for plotting in {input_path.name}.")
            return None

        valid_data = data_dict_listing(valid_plots_data)

        year, month, _ = file_name_comprehension(input_path)
        month_list = [(year, month)]
        is_period = False

    elif input_path.is_dir():
        logger.info(f"Directory detected. Extracting period data from {input_path.name}...")

        period_data = period_retrieval_function(input_path, retrieval_func, verbose=verbose)
        if not period_data:
            logger.warning(f"No valid data extracted from {input_path.name}.")
            return None

        if overall_analysis:
            valid_data = overall_period_data_function(period_data, overall_analysis)
        else:
            valid_data = period_data

        files, _ = monthly_data_directory_exploration(input_path)
        for file in files:
            year, month, _ = file_name_comprehension(file)
            month_list.append((year, month))

        if overall_analysis:
            month_list.append(("Period", "Overall"))

        is_period = True

    else:
        logger.error("Input path is neither a standard file nor a directory.")
        return None

    # ==========================================
    # EXECUTION PIPELINE
    # ==========================================

    if objective == 'show':
        logger.info(f"Objective is 'show'. Generating isolated interactive dashboard for: {'Overall Period' if is_period else 'Single Month'}...")

        time_tuple = month_list[-1]
        plot_title = f"{time_tuple[0]} - {time_tuple[1]}" if is_period else f"{time_tuple[0]:04d} - {time_tuple[1]:02d}"

        # Waterfall *args and **kwargs down to the plotting function
        fig_dict = environmental_plotting_function(
            plot_title=plot_title,
            valid_plot_data_dict=valid_data,
            idx=-1,
            plot_generator_func=plot_generator_func,
        )

        # Trigger PNG backups based on user toggle
        if save_png_backups:
            png_exporting_function(
                figs_dict=fig_dict,
                study_header=time_tuple,
                study=study,
                is_period=is_period,
                output_target_dir=output_target_dir
            )

        return fig_dict

    elif objective == 'export':
        logger.info("Objective is 'export'. Routing data to the chronological PDF cycle engine...")

        # Waterfall *args and **kwargs down to the cycle engine
        pdf_paths = exporting_cycle_engine(
            months_list=month_list,
            valid_plot_data_dict=valid_data,
            plot_generator_func=plot_generator_func,
            study=study,
            is_period=is_period,
            output_target_dir=output_target_dir,
            output_filename=output_filename,
            verbose=verbose,
        )
        return pdf_paths if is_period else (pdf_paths[0] if pdf_paths else None)

    else:
        logger.warning(f"Objective not specified: expected 'show' or 'export', got '{objective}'")
        return None