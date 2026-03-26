"""
AirTS-Forecast Project
Section 1: Data Gathering and Exploration
File: Period_Statistics.py
Author: Tiago TOLOCZKO ROSS

Description:
Period statistical aggregation module. Iterates over a designated time range of
monthly 3D HDF5 climate data, leveraging the monthly aggregation function to
efficiently calculate the overall spatial, zonal, meridional, and grand means
for the entire period without overwhelming system memory.
"""


# --- Import from your existing modules ---
# (Adjust module names if your files are named differently)
from environmental_monthly_average_analysis import *
from environmental_data_retrieval import monthly_data_directory_exploration

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def period_granular_spatial_average_tables(
        data_dir: Path,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, list[pd.DataFrame]]:
    """
    Leverages the directory explorer to find valid files, and routes them
    through the monthly aggregation function to extract and accumulate the
    coarsened spatial grids chronologically.
    """
    # 1. Fetch valid files utilizing the centralized directory auto-discovery
    target_files, period_bounds = monthly_data_directory_exploration(data_dir)

    if not target_files:
        logger.warning(f"No valid files found in {data_dir.name} to process for the period.")
        return {}

    accumulated_grids = {}

    if period_bounds:
        s_m, s_y, e_m, e_y = period_bounds
        logger.info(f"Initiating period extraction across {len(target_files)} months "
                    f"({s_m:02d}/{s_y} to {e_m:02d}/{e_y})...")

    # 2. Iterate through files and delegate directly to the monthly aggregation function!
    for file in target_files:
        logger.debug(f"Processing monthly stats for {file.name}...")

        # This single call natively handles HDF5 extraction, target filtering, AND spatial coarsening
        monthly_stats = monthly_granular_spatial_tables(month_file=file, target_variable=target_variable,
                                                        granularity=granularity)

        for var, stats in monthly_stats.items():
            if var not in accumulated_grids:
                accumulated_grids[var] = []

            # Extract ONLY the formatted 2D spatial grid (ignoring zonal/meridional/grand for now)
            accumulated_grids[var].append(stats["spatial_grid"])

    return accumulated_grids


def period_averages_calculation(
        data_dir: Path,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Coordinates the extraction of monthly grids and calculates the final
    mathematical overall period averages (Spatial, Zonal, Meridional, Grand).
    """
    accumulated_grids = period_granular_spatial_average_tables(data_dir=data_dir, target_variable=target_variable,
                                                               granularity=granularity)

    period_results = {}

    if not accumulated_grids:
        logger.warning("No accumulated grids to calculate period averages.")
        return period_results

    for var, grid_list in accumulated_grids.items():
        logger.info(f"Calculating final period mathematical averages for '{var}'...")

        stacked_grids = np.stack([df.values for df in grid_list], axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            period_mean_values = np.nanmean(stacked_grids, axis=0)

        period_spatial_grid = pd.DataFrame(
            period_mean_values,
            index=grid_list[0].index,
            columns=grid_list[0].columns
        )

        zonal_mean_df = period_spatial_grid.mean(axis=1).to_frame(name="Period_Zonal_Mean")
        meridional_mean_df = period_spatial_grid.mean(axis=0).to_frame(name="Period_Meridional_Mean")
        grand_mean = float(np.nanmean(period_mean_values))

        period_results[var] = {
            "spatial_grid": period_spatial_grid,
            "zonal_mean": zonal_mean_df,
            "meridional_mean": meridional_mean_df,
            "grand_mean": grand_mean
        }

    return period_results


def print_period_spatial_summaries(
        data_dir: Path,
        target_variable: str = None,
        granularity: float = 1.0
) -> dict[str, dict[str, pd.DataFrame | float]]:
    """
    Retrieves aggregated spatial data for the auto-discovered time period
    and prints a formatted statistical summary to the console.
    """
    # Temporarily suppress lower level logs during calculation
    logging.getLogger().setLevel(logging.WARNING)
    stats_dictionary = period_averages_calculation(data_dir, target_variable, granularity)
    logging.getLogger().setLevel(logging.INFO)

    if not stats_dictionary:
        return {}

    # Fetch the bounds dynamically just to print the master label
    _, bounds = monthly_data_directory_exploration(data_dir)
    if bounds:
        s_m, s_y, e_m, e_y = bounds
        period_str = f"{s_m:02d}/{s_y} to {e_m:02d}/{e_y}"
    else:
        period_str = "Unknown Range"

    for var, stats in stats_dictionary.items():
        grand_mean = stats["grand_mean"]
        zonal_mean_df = stats["zonal_mean"]
        meridional_mean_df = stats["meridional_mean"]
        coarsened_grid = stats["spatial_grid"]

        print(f"\n{'='*70}")
        print(f"PERIOD STATISTICAL SUMMARY: {var.upper()} | Range: {period_str} | Granularity: {granularity}°")
        print(f"{'='*70}")
        print(f"Overall Period Grand Mean: {grand_mean:.4f}\n")

        print("--- Period Zonal Averages (By Latitude) ---")
        print(zonal_mean_df.head(5))
        print("...\n")

        print("--- Period Meridional Averages (By Longitude) ---")
        print(meridional_mean_df.head(5))
        print("...\n")

        print("--- Period Spatial Matrix Head (Lat x Lon) ---")
        print(coarsened_grid.iloc[:5, :5])
        print("\n")

    return stats_dictionary


def plot_period_3d_surfaces(
        stats_dictionary: dict[str, dict],
        period_label: str
) -> None:
    """
    Generates interactive 3D surface subplots for all variables, using the
    centralized plotting engine.
    """
    if not stats_dictionary:
        logger.warning("No data provided for period 3D plotting.")
        return

    num_vars = len(stats_dictionary)
    cols = math.ceil(math.sqrt(num_vars))
    rows = math.ceil(num_vars / cols)

    fig = plt.figure(figsize=(7 * cols, 6 * rows))
    fig.suptitle(f"Period Temporal Means: {period_label}", fontsize=16, fontweight='bold')

    for index, (var, stats) in enumerate(stats_dictionary.items()):
        logger.info(f"Rendering 3D surface and bottom contour for period variable: {var}")

        spatial_df = stats["spatial_grid"]
        lats = spatial_df.index.values
        lons = spatial_df.columns.values
        data_2d = spatial_df.values

        ax = fig.add_subplot(rows, cols, index + 1, projection='3d')

        plot_3d_surface_on_axis(
            ax=ax,
            lons=lons,
            lats=lats,
            data_2d=data_2d,
            var_name=f"{var} ({period_label})"
        )

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


def export_period_3d_plots_to_pdf(
        stats_dictionary: dict[str, dict],
        period_label: str,
        output_filename: str | Path = "period_climate_plots.pdf",
        verbose: bool = False
) -> Path:
    """
    Saves each variable's 3D surface plot as a centralized page in a PDF document.
    """
    if not stats_dictionary:
        logger.warning("No data provided for PDF export.")
        return None

    target_dir = Path("Exported pdf plots")
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / Path(output_filename).name

    logger.info(f"Generating Period PDF with {len(stats_dictionary)} pages in '{target_dir.name}'...")

    try:
        with PdfPages(output_path) as pdf:
            for var, stats in stats_dictionary.items():
                fig = plt.figure(figsize=(11.69, 8.27))
                ax = fig.add_subplot(111, projection='3d')

                spatial_df = stats["spatial_grid"]
                plot_3d_surface_on_axis(
                    ax=ax,
                    lons=spatial_df.columns.values,
                    lats=spatial_df.index.values,
                    data_2d=spatial_df.values,
                    var_name=f"{var} ({period_label})"
                )

                fig.suptitle(
                    f"Period Temporal Mean: {var.upper()}",
                    fontsize=16, fontweight='bold', y=0.96
                )

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.5)
                plt.close(fig)

                if verbose:
                    logger.info(f"Rendered centralized PDF page with contour for: {var}")

        logger.info(f"Successfully exported 3D period plots with contours to PDF: {output_path.absolute()}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to construct PDF: {e}")
        return None


def export_monthly_pdfs_in_period(
        data_dir: Path,
        verbose: bool = False
) -> list[Path]:
    """
    Iterates over an auto-discovered period of monthly HDF5 files and exports
    individual, multipage PDF documents containing 3D plots for each month.
    """
    # Utilize the directory explorer directly without dates
    target_files, _ = monthly_data_directory_exploration(data_dir)

    if not target_files:
        logger.warning("No files found to export for the given period.")
        return []

    logger.info(f"Starting batch PDF export for {len(target_files)} months...")
    exported_pdfs = []

    for file in target_files:
        output_name = f"visualizations_{file.stem}.pdf"
        logger.info(f"Delegating PDF export for {file.name}...")

        pdf_path = export_3d_plots_to_pdf(
            month_file=file,
            output_filename=output_name,
            retrieval_func=spatial_mean_granular(),
            verbose=verbose
        )

        if pdf_path:
            exported_pdfs.append(pdf_path)

    logger.info(f"Batch export complete! {len(exported_pdfs)} monthly PDFs generated.")
    return exported_pdfs


if __name__ == '__main__':
    target_directory = Path("era5_monthly_data")

    # Example Period: March 2004 to April 2005
    s_year, s_month = 2004, 3
    e_year, e_month = 2005, 4
    period_label_str = f"{s_year}/{s_month:02d} to {e_year}/{e_month:02d}"

    # 1. Execute calculations
    period_stats = print_period_spatial_summaries(
        data_dir=target_directory,
        granularity=.5 # Set back to 1.0 to prevent massive console output
    )

    if period_stats:
        # 2. Export to Excel
        try:
            excel_filename = f"period_stats_{s_year}_{s_month:02d}_to_{e_year}_{e_month:02d}.xlsx"
            export_stats_to_excel(period_stats, output_filename=excel_filename)
        except Exception as e:
            logger.error(f"Failed to export Excel: {e}")

        # 3. Export to Centralized PDF
        pdf_filename = f"period_visualizations_{s_year}_{s_month:02d}_to_{e_year}_{e_month:02d}.pdf"
        export_period_3d_plots_to_pdf(
            stats_dictionary=period_stats,
            period_label=period_label_str,
            output_filename=pdf_filename,
            verbose=True
        )

        # 4. Show Interactive Window (Optional)
        plot_period_3d_surfaces(
            stats_dictionary=period_stats,
            period_label=period_label_str
        )

        # 5. Batch Export Individual Monthly PDFs
        #logger.info("\n--- Starting Batch Monthly PDF Generation ---")
        #generated_pdfs = export_monthly_pdfs_in_period(
        #    data_dir=target_directory,
        #    start_year=s_year,
        #    start_month=s_month,
        #    end_year=e_year,
        #    end_month=e_month,
        #    verbose=False  # Set to True if you want a line-by-line log of every variable drawn
        #)