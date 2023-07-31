from typing import Dict
import numpy as np
import pandas as pd
from KDEpy import FFTKDE
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

    Parameters
    ----------
    file_path : str
        Path to the .bed file to be read.

    Returns
    -------
    bed_content : pd.DataFrame
        A DataFrame containing the .bed file content.
    """
    header = {0: "seqname", 1: "start", 2: "end"}
    bed_content = pd.read_csv(file_path, delimiter="\t", header=None).iloc[:, :3]
    bed_content.rename(columns=header, inplace=True)
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


def make_kdes(
    ebs_c1,
    step=10,
    kde_bw=200,
    blacklisted=list(),
):
    """
    Computes KDEs for given events, with optional blacklist.

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

    Returns
    -------
    comb_data : pd.DataFrame
        DataFrame containing computed KDE for each sequence.
    signal_list_global : list
        List of signal densities for each sequence.
    """

    sequences = set(ebs_c1.keys()) - set(blacklisted)
    result_dfs = list()
    ntotal = len(sequences)
    signal_list_global = list()

    for i, seqname in enumerate(sequences):
        logger.info(f"Making KDE of {seqname} [{i+1}/{ntotal}].")

        events = ebs_c1.get(seqname, pd.DataFrame())
        cuts = events["location"].values

        low_res_cuts = np.round(cuts / step)

        grid = full_kde_grid(low_res_cuts)
        cut_idx = (low_res_cuts - grid.min()).astype(int)

        _, density = get_kde(low_res_cuts, kde_bw=kde_bw, grid=grid)
        density *= len(events)

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
    """
    Converts track data to interval representation.

    Parameters
    ----------
    job : tuple
        A tuple containing the combined data, step size, and sequence name.

    Returns
    -------
    peaks : pd.DataFrame
        DataFrame with peak intervals.
    """
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

    peak_location_df = loc_comb_data[loc_comb_data["peak"].values].set_index("location")
    locations = peak_location_df.groupby("peak_name")["density"].idxmax()
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
            "summit": locations,
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
        list(df.index.str.split("_")),
        columns=["seqname", "interval", "is_peak"],
        index=df.index,
    )
    df["name"] = (
        name_dat["seqname"] + "_" + name_dat["interval"]
    )
    return df
