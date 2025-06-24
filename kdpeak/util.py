from typing import Dict, Optional
import numpy as np
import pandas as pd
from KDEpy import FFTKDE
import pyBigWig
import multiprocessing
import logging


def setup_logging(log_level="INFO", log_file=None) -> logging.Logger:
    """
    Sets up a logger object for logging runtime information.

    Parameters
    ----------
    log_level : str, optional
        The level of logging. Defaults to "INFO".
    log_file : str, optional
        File to which the log should be written. If None, log is written to standard output.

    Returns
    -------
    logger : logging.Logger
        The logger object.
    """
    level = getattr(logging, log_level)
    log_format = "[%(asctime)s] [%(levelname)-8s] %(message)s"
    logging.basicConfig(level=level, filename=log_file, format=log_format, filemode="w", datefmt="%Y-%m-%d %H:%M:%S")
    return logging.getLogger()


logger = setup_logging()


def read_bed(file_path: str) -> pd.DataFrame:
    """
    Reads a .bed file and returns it as a pandas DataFrame.
    Only reads the first 3 columns (chromosome, start, end) as required for peak calling.

    Parameters
    ----------
    file_path : str
        Path to the .bed file to be read.

    Returns
    -------
    bed_content : pd.DataFrame
        A DataFrame containing the .bed file content with columns: seqname, start, end.
    """
    # Read only the first 3 columns to handle variable column counts in BED files
    # Use string dtype initially to handle any mixed types, then convert
    bed_content = pd.read_csv(
        file_path, 
        delimiter="\t", 
        header=None, 
        usecols=[0, 1, 2],
        names=["seqname", "start", "end"],
        dtype={"seqname": str, "start": str, "end": str},
        low_memory=False
    )
    
    # Convert start and end coordinates to integers
    try:
        bed_content["start"] = pd.to_numeric(bed_content["start"], errors="coerce")
        bed_content["end"] = pd.to_numeric(bed_content["end"], errors="coerce")
        
        # Remove any rows with invalid coordinates
        bed_content = bed_content.dropna(subset=["start", "end"])
        bed_content["start"] = bed_content["start"].astype(int)
        bed_content["end"] = bed_content["end"].astype(int)
        
    except Exception as e:
        logger.error(f"Error converting coordinates to integers: {e}")
        raise
    
    logger.info(f"Read {len(bed_content)} intervals from {file_path}")
    return bed_content


def write_bed(bed_df: pd.DataFrame, out_path: str) -> None:
    """
    Writes a DataFrame to a .bed file.

    Parameters
    ----------
    bed_df : pd.DataFrame
        DataFrame to be written to the .bed file.
    out_path : str
        Path where the .bed file should be written.
    """
    bed_df.to_csv(out_path, sep="\t", header=False, index=False)

def write_bigwig(comb_data: pd.DataFrame, out_path: str, sizes_file: str, span: int) -> None:
    """
    Write a BigWig file from a pandas DataFrame containing genomic data.

    Parameters:
    - comb_data (pd.DataFrame): A DataFrame with columns 'seqname', 'location', and 'density'.
      - 'seqname' (str): Chromosome or sequence name.
      - 'location' (int): Genomic location.
      - 'density' (float): Density value at the given location.
    - sizes_file (str): Path to a file containing chromosome sizes. The file should have two columns: chromosome name and size.
    - span (int): The span (in base pairs) for each data point in the BigWig file. Default is 10.
    - out_path (str): Path to the output BigWig file.
    
    Returns:
    - None

    Raises:
    - ValueError: If 'location' values are negative or exceed the chromosome size.

    Example:
    >>> sizes_file = 'chrom_sizes.txt'
    >>> data = {'seqname': ['chr1', 'chr1', 'chr2'], 'location': [1000, 1010, 2000], 'density': [0.5, 0.8, 0.3]}
    >>> comb_data = pd.DataFrame(data)
    >>> write_bigwig(comb_data, sizes_file, span=10, out_path='output.bw')
    """

    # Read chromosome sizes from file
    chrom_sizes = {}
    try:
        with open(sizes_file, "r") as fl:
            for line in fl:
                seq, size = line.split()
                chrom_sizes[seq] = int(size)
        logger.info(f"Read chromosome sizes from {sizes_file}")
    except Exception as e:
        logger.error(f"Error reading sizes_file: {e}")
        raise

    # Ensure all sequences in comb_data are in the chromosome sizes file
    all_seqs = comb_data["seqname"].unique()
    missing_seqs = [seq for seq in all_seqs if seq not in chrom_sizes]
    if missing_seqs:
        error_message = f"Sequences {missing_seqs} are missing in the chromosome sizes file."
        logger.error(error_message)
        raise ValueError(error_message)

    with pyBigWig.open(out_path, "w") as bw:
        bw.addHeader(list(sorted(chrom_sizes.items())))

        for seqname, data in comb_data.groupby("seqname"):
            if seqname not in chrom_sizes:
                logger.warning(f"Skipping sequence {seqname} not found in chromosome sizes")
                continue

            # Ensure no negative locations and locations do not exceed chromosome size
            mask = (data["location"] >= 0) & (data["location"] <= chrom_sizes[seqname])
            df = data.loc[mask, :]

            if df.empty:
                logger.warning(f"No valid data points for sequence {seqname}")
                continue

            bw.addEntries(
                seqname,
                int(df["location"].min()),
                span=span,
                step=span,
                values=df["density"].values,
            )


def events_from_intervals(interval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms interval DataFrame into an event-based DataFrame.

    Parameters
    ----------
    interval_df : pd.DataFrame
        DataFrame of intervals with start and end columns.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame of events with sequence names and locations.
    """
    id_vars = set(interval_df.columns) - {"start", "end"}
    df = interval_df.melt(id_vars=id_vars, value_name="location")
    return df

def events_dict_from_file(file: str) -> Dict[str, pd.DataFrame]:
    """
    Reads a bed file and transforms it into a dictionary of DataFrames,
    one for each sequence name. Each DataFrame contains events derived
    from intervals present in the bed file.

    Parameters
    ----------
    file : str
        Path to the bed file.

    Returns
    -------
    dict
        A dictionary where keys are sequence names and values are DataFrames
        of events for each sequence name. Each DataFrame contains two columns,
        'variable' and 'location', where 'variable' indicates whether the event
        is the start or end of an interval, and 'location' is the location of the event.

    Examples
    --------
    >>> events = events_dict_from_file("reads.bed")
    >>> events["chr1"]
       variable  location
    0     start      1000
    1     start      2000
    2       end      1500
    3       end      2500
    """
    df = read_bed(file)
    if df.empty:
        logger.warning(f"No events found in {file}")
        return dict()

    events = events_from_intervals(df)
    events_by_seqname = {
        seqname: seq_events for seqname, seq_events in events.groupby("seqname")
    }

    return events_by_seqname



def full_kde_grid(x, xmin=None, xmax=None):
    """
    Computes a grid of points for KDE estimation given an array of data points.

    Parameters
    ----------
    x : array-like
        Data points for KDE estimation.
    xmin : int, optional
        Minimum value of the grid. If None, the grid starts at x.min() - 1.
    xmax : int, optional
        Maximum value of the grid. If None, the grid ends at x.max() + 1.

    Returns
    -------
    grid : np.ndarray
        A grid of points for KDE estimation.
    """
    if xmin is None:
        xmin = np.min(x) - 1
    if xmax is None:
        xmax = np.max(x) + 1
    grid = np.arange(xmin, xmax + 1)
    return grid


def get_kde(
    cut_locations, kde_bw=500, kernel="gaussian", xmin=None, xmax=None, grid=None
):
    """
    Estimates the KDE of the given data points.

    Parameters
    ----------
    cut_locations : array-like
        Locations of data points.
    kde_bw : float, optional
        Bandwidth for KDE estimation. Defaults to 500.
    kernel : str, optional
        Name of the kernel to be used for KDE estimation. Defaults to "gaussian".
    xmin : int, optional
        Minimum value for the grid. If None, defaults to minimum value in cut_locations - 1.
    xmax : int, optional
        Maximum value for the grid. If None, defaults to maximum value in cut_locations + 1.
    grid : array-like, optional
        Grid of points for KDE estimation. If None, a grid is generated using xmin and xmax.

    Returns
    -------
    grid : np.ndarray
        Grid of points used for KDE estimation.
    density : np.ndarray
        Estimated densities at each point in the grid.
    """
    if grid is None:
        grid = full_kde_grid(cut_locations, xmin, xmax)
    kernel = FFTKDE(kernel=kernel, bw=kde_bw)
    kernel = kernel.fit(cut_locations)
    density = kernel.evaluate(grid)
    return grid, density


def read_chrom_sizes_file(chrom_sizes_file: str) -> Dict[str, int]:
    """
    Read chromosome sizes from a file.
    
    Parameters
    ----------
    chrom_sizes_file : str
        Path to chromosome sizes file with format: chromosome_name\tsize
    
    Returns
    -------
    dict
        Dictionary mapping chromosome names to sizes.
    """
    chrom_sizes = {}
    try:
        with open(chrom_sizes_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        chrom_sizes[parts[0]] = int(parts[1])
        logger.info(f"Read sizes for {len(chrom_sizes)} chromosomes from {chrom_sizes_file}")
    except Exception as e:
        logger.warning(f"Could not read chromosome sizes file {chrom_sizes_file}: {e}")
    
    return chrom_sizes


def get_chromosome_size_estimate(events_df: pd.DataFrame) -> int:
    """
    Estimate chromosome size from the maximum coordinate in events.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        DataFrame containing genomic events with 'location' column.
    
    Returns
    -------
    int
        Estimated chromosome size based on maximum coordinate.
    """
    if events_df.empty or 'location' not in events_df.columns:
        return 0
    return int(events_df['location'].max())


def sort_chromosomes_by_size(ebs_c1: Dict[str, pd.DataFrame], 
                           chrom_sizes_file: Optional[str] = None) -> List[str]:
    """
    Sort chromosomes by size (largest first) with robust fallback ordering.
    
    Parameters
    ----------
    ebs_c1 : dict
        Dictionary mapping sequence names to events DataFrames.
    chrom_sizes_file : str, optional
        Path to chromosome sizes file. If provided, uses actual sizes.
        If not provided, estimates sizes from data.
    
    Returns
    -------
    list
        List of chromosome names sorted by size (largest first).
    """
    # Try to read actual chromosome sizes if file is provided
    actual_chrom_sizes = {}
    if chrom_sizes_file:
        actual_chrom_sizes = read_chrom_sizes_file(chrom_sizes_file)
    
    # Calculate size estimates for each chromosome
    chrom_sizes = []
    for seqname in ebs_c1.keys():
        if seqname in actual_chrom_sizes:
            # Use actual chromosome size
            size = actual_chrom_sizes[seqname]
            logger.debug(f"Using actual size for {seqname}: {size:,}")
        else:
            # Fall back to estimation from data
            size = get_chromosome_size_estimate(ebs_c1[seqname])
            logger.debug(f"Estimated size for {seqname}: {size:,}")
        
        chrom_sizes.append((seqname, size))
    
    # Sort by size (descending), then by chromosome name for reproducibility
    # This ensures consistent ordering even when chromosomes have the same size
    sorted_chroms = sorted(chrom_sizes, key=lambda x: (-x[1], x[0]))
    
    # Log the sorting approach used
    if actual_chrom_sizes:
        found_actual = sum(1 for name, _ in sorted_chroms if name in actual_chrom_sizes)
        logger.info(f"Sorted chromosomes using actual sizes for {found_actual}/{len(sorted_chroms)} chromosomes")
    else:
        logger.info("Sorted chromosomes using size estimates from data")
    
    # Extract just the chromosome names
    return [chrom[0] for chrom in sorted_chroms]


def make_kdes(
    ebs_c1,
    step=10,
    kde_bw=200,
    blacklisted=list(),
    chrom_sizes_file=None,
):
    """
    Computes KDEs for given events, with optional blacklist.
    Processes chromosomes in order of decreasing size for better user feedback.

    Parameters
    ----------
    ebs_c1 : dict
        Dictionary mapping sequence names to events.
    step : int, optional
        Step size for KDE grid. Defaults to 10.
    kde_bw : float, optional
        Bandwidth for KDE estimation. Defaults to 200.
    blacklisted : list, optional
        List of sequences to exclude from KDE estimation.
    chrom_sizes_file : str, optional
        Path to chromosome sizes file. If provided, uses actual sizes for sorting.

    Returns
    -------
    comb_data : pd.DataFrame
        DataFrame containing computed KDE for each sequence.
    signal_list_global : list
        List of signal densities for each sequence.
    """

    # Filter out blacklisted sequences
    available_sequences = {k: v for k, v in ebs_c1.items() if k not in blacklisted}
    
    # Sort chromosomes by size (largest first)
    sorted_sequences = sort_chromosomes_by_size(available_sequences, chrom_sizes_file)
    
    result_dfs = list()
    ntotal = len(sorted_sequences)
    signal_list_global = list()
    
    if ntotal == 0:
        logger.warning("No sequences available for processing after filtering")
        return pd.DataFrame(), []
    
    logger.info(f"Processing {ntotal} chromosomes in order of decreasing size")
    
    # Log the processing order for the first few chromosomes
    preview_chroms = sorted_sequences[:min(5, ntotal)]
    logger.info(f"Processing order (first {len(preview_chroms)}): {', '.join(preview_chroms)}")

    for i, seqname in enumerate(sorted_sequences):
        logger.info(f"Making KDE of {seqname} [{i+1}/{ntotal}].")

        events = ebs_c1.get(seqname, pd.DataFrame())
        cuts = events["location"].values

        low_res_cuts = np.round(cuts / step)

        grid = full_kde_grid(low_res_cuts)
        cut_idx = (low_res_cuts - grid.min()).astype(int)

        _, density = get_kde(low_res_cuts, kde_bw=kde_bw / step, grid=grid)
        density *= len(events) * 100 / step

        comb_df = pd.DataFrame(
            {
                "seqname": seqname,
                "interval": seqname,
                "location": grid * step,
                "density": density,
            }
        )

        signal_list_global.append(density[cut_idx])
        result_dfs.append(comb_df)
    comb_data = pd.concat(result_dfs, axis=0)

    return comb_data, signal_list_global


def mark_peaks(
    comb_data,
    signal_list_global,
    fraction,
):
    """
    Marks peaks in the combined data based on a given density values.

    Parameters
    ----------
    comb_data : pd.DataFrame
        DataFrame with combined data.
    signal_list_global : list
        List of signal densities for each cut.
    fraction : float
        Fraction of cuts to consider to be in peak.

    Returns
    -------
    comb_data : pd.DataFrame
        DataFrame with peaks marked.
    """

    bound = np.quantile(np.hstack(signal_list_global), 1 - fraction)
    comb_data["peak"] = comb_data["density"] > bound
    fraction_selected = comb_data["peak"].sum() / len(comb_data["peak"])
    msg = f"{fraction_selected:.2%} of genome covered by peaks."
    logger.info(msg)

    return comb_data


def track_to_interval(job):
    loc_comb_data, step_size, seqname = job

    idx = np.insert(loc_comb_data["peak"].values.astype(int), 0, 0)
    indicator = np.diff(idx)
    loc_comb_data["peak_number"] = np.cumsum(indicator == 1)
    loc_comb_data["peak_name"] = (
        loc_comb_data["interval"]
        + "_"
        + loc_comb_data["peak_number"].astype(str)
        + "_"
        + loc_comb_data["peak"].astype(str)
    )

    peak_indices = loc_comb_data["peak"].values
    peak_location_df = loc_comb_data.loc[peak_indices].set_index("location")

    summit_locations = peak_location_df.groupby("peak_name")["density"].idxmax()
    summit_heights = peak_location_df.loc[summit_locations, "density"]

    peak_groups = peak_location_df.reset_index().groupby("peak_name")

    start = peak_groups["location"].min().values - (step_size / 2)
    end = peak_groups["location"].max().values + (step_size / 2)
    means = peak_groups["density"].mean()

    peaks = pd.DataFrame(
        {
            "seqname": seqname,
            "start": start.astype(int),
            "end": end.astype(int),
            "mean": means,
            "summit": summit_locations.values.astype(int),
            "summit_height": summit_heights.values,
        }
    ).sort_values("start")
    return peaks


def tracks_to_intervals(comb_data, step_size):
    """
    Converts all tracks in the combined data to interval representation.

    Parameters
    ----------
    comb_data : pd.DataFrame
        DataFrame with combined data.
    step_size : int
        Step size used for grid in KDE estimation.

    Returns
    -------
    peaks : pd.DataFrame
        DataFrame with all peak intervals.
    """
    peaks_list = list()
    jobs = [
        (loc_comb_data, step_size, seqname)
        for seqname, loc_comb_data in comb_data.groupby("seqname")
    ]
    with multiprocessing.Pool() as pool:
        for peaks in pool.imap(track_to_interval, jobs, 10):
            peaks_list.append(peaks)
    peaks = pd.concat(peaks_list, axis=0)
    peaks["start"] = peaks["start"].clip(lower=0)
    peaks["length"] = peaks["end"] - peaks["start"]
    return peaks


def call_peaks(
    comb_data,
    signal_list_global,
    fraction_in_peaks=0.3,
    min_peak_size=100,
    span=10,
):
    """
    Identifies and filters peaks in the combined data.

    Parameters
    ----------
    comb_data : pd.DataFrame
        DataFrame with combined data.
    signal_list_global : list
        List of densities for each cut.
    fraction_in_peaks : float, optional
        Fraction of cuts to consider to be in peak.
    min_peak_size : int, optional
        Minimum size of a peak.
    span : int, optional
        Span used to adjust the size of peak intervals.

    Returns
    -------
    peaks : pd.DataFrame
        DataFrame with identified and filtered peaks.
    """
    logger.info("Selecting peak candidates.")
    comb_data = mark_peaks(
        comb_data,
        signal_list_global,
        fraction_in_peaks,
    )
    logger.info("Converting peak signal to intervals.")
    peaks = tracks_to_intervals(comb_data, span)
    peaks = peaks[peaks["length"] > min_peak_size]

    logger.info(f"Peaks: {len(peaks):,}")
    return peaks


def include_auc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Includes an 'auc' (area under curve) column in the peak data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with 'auc' column added.
    """
    df["auc"] = df["mean"] * df["length"]
    return df


def name_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a unique name for each peak in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with a 'name' column added, which contains unique names for each peak.
    """
    name_dat = pd.DataFrame(
        map(lambda x: ("_".join(x[:-2]), x[-2], x[-1]), list(df.index.str.split("_"))),
        columns=["seqname", "interval", "is_peak"],
        index=df.index,
    )
    df["name"] = (
        name_dat["seqname"] + "_" + name_dat["interval"]
    )
    return df
