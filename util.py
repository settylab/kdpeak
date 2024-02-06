from typing import Dict
import numpy as np
import pandas as pd
from KDEpy import FFTKDE, bw_selection
import multiprocessing
import logging
from tqdm.auto import tqdm
import mellon
import pyranges as pr
import numbers

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
    kde_bw : float or str, optional
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

def get_mellon_density(
    cut_locations, cut_counts, xmin=None, xmax=None, grid=None, n_landmarks = 20000
):
    """
    Estimates a function fitting the given data points using sparse Gausian Process regression.

    Parameters
    ----------
    cut_locations : array-like
        Locations of data points.
    cut_counts: array-like
        count values of to the locations in cut_locations.
    xmin : int, optional
        Minimum value for the grid. If None, defaults to minimum value in cut_locations - 1.
    xmax : int, optional
        Maximum value for the grid. If None, defaults to maximum value in cut_locations + 1.
    grid : array-like, optional
        Grid of points for KDE estimation. If None, a grid is generated using xmin and xmax.
    n_landmarks: int, optional
        the number of points to be used for sparse Gaussian Process regression. Default: 20,000
    Returns
    -------
    grid : np.ndarray
        Grid of points used for KDE estimation.
    log_density : np.ndarray
        Estimated log densities at each point in the grid.
    """
    if grid is None:
        grid = full_kde_grid(cut_locations, xmin, xmax)
    model = mellon.FunctionEstimator(n_landmarks = n_landmarks)
    est = model.fit_predict(cut_locations, cut_counts, grid)
    return grid, est

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

    for i, seqname in enumerate(sorted(sequences)):
        #logger.info(f"Making KDE of {seqname} [{i+1}/{ntotal}].")

        events = ebs_c1.get(seqname, pd.DataFrame())
        cuts = events["location"].values

        low_res_cuts = np.round(cuts / step)

        grid = full_kde_grid(low_res_cuts)
        cut_idx = (low_res_cuts - grid.min()).astype(int)

        _, density = get_kde(low_res_cuts, kde_bw=kde_bw, grid=grid)
        density *= len(events) * (1000/step)

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
def write_mellon(
    records,
    chrom_sizes_path,
    file_path,
    step=100,
    sample_frac = .1,
    blacklisted=list(),
    scale_factor = 1,
    predict_point = 'Start'
):
    """
    Computes KDEs for given events, with optional blacklist. Writes KDEs directly to BedGraph format without storing large dataframes in memory.

    Parameters
    ----------
    records: pd.DataFrame
        tabular data object containing fragment locations with columns Chromosome, Start, and End
    chrom_sizes_path: str
        A string of the path to the .chrom.sizes file for the desired genome.
    file_path: string
        Path to output file.
    step : int, optional
        Step size for KDE grid. Defaults to 100.
    sample_frac: float, optional
        percentage of tiles with zero overlap to use for sparse GPR fit. Default: 0.1
    blacklisted : list, optional
        List of sequences to exclude from KDE estimation.
    scale_factor: float or int, optional
        Densities will be multiplied by this value to account for differences in read depth. Defaults to 1 (no scaling).
    predict_point: string, default is ['Start']
        Select whether 'Start' or 'End' of fragments will be used to predict coverage. Default: Start
    """


    assert records.shape[1] == 3, "`records` should have three columns with labels 'Chromosome', 'Start', 'End'."
    assert list(records.columns) == ['Chromosome', 'Start', 'End'], "`records` should have three columns with labels 'Chromosome', 'Start', 'End'."
    sequences = set(records['Chromosome'].values) - set(blacklisted)
    ntotal = len(sequences)
    print(sequences)
    chrom_sizes = pd.read_csv(chrom_sizes_path, sep = '\t', header= None, index_col = 0)
    chrom_sizes = chrom_sizes[~chrom_sizes.index.str.contains('_')][1].to_dict()
    tiles = pr.genomicfeatures.tile_genome(genome = chrom_sizes, tile_size = step, tile_last = True)
    for i, seqname in enumerate(sorted(sequences)):
        logger.info(f"Making KDE of {seqname} [{i+1}/{ntotal}].")
        subset_records = records[records['Chromosome'] == seqname]

        intervals = pr.PyRanges(subset_records)
        overlaps = tiles[tiles.Chromosome == seqname].count_overlaps(intervals).as_df()
        
        data_idx = (overlaps[overlaps['NumberOverlaps'] == 0].sample(frac = sample_frac)).index
        overlaps = overlaps[(overlaps['NumberOverlaps'] > 0) | overlaps.index.isin(data_idx)]
        
        tile_df = tiles[tiles.Chromosome == seqname].as_df()

        _, density = get_mellon_density(overlaps[predict_point], overlaps['NumberOverlaps'], grid =tile_df[predict_point])
        
        if predict_point == 'Start':
            start_coord = tile_df[predict_point]
            end_coord = tile_df[predict_point] + step
        elif predict_point == 'End':
            start_coord = tile_df[predict_point] - step
            end_coord = tile_df[predict_point]
        else:
            raise ValueError("predict point must be 'Start' or 'End'")
            
        start_coord = start_coord.where(start_coord > 0, 0)
        end_coord = end_coord.where(end_coord <= chrom_sizes[seqname], chrom_sizes[seqname])
        if i != 0:
            mode = 'a'
        else:
            mode = 'w'
        pd.DataFrame(
            {
                "seqname": seqname,
                "start": start_coord,
                'end': end_coord,
                "density": density * scale_factor,
            }).to_csv(file_path, sep = "\t", mode = mode, index = False, header = False, chunksize = 100000)

def write_kdes(
    ebs_c1,
    file_path,
    step=10,
    kde_bw=200,
    blacklisted=list(),
    scale_factor = 1,
):
    """
    Computes KDEs for given events, with optional blacklist. Writes KDEs directly to BedGraph format without storing large dataframes in memory.

    Parameters
    ----------
    ebs_c1 : dict
        Dictionary mapping sequence names to events.
    file_path: string
        Path to output file.
    step : int, optional
        Step size for KDE grid. Defaults to 10.
    kde_bw : int or string, optional
        Bandwidth for KDE estimation. Defaults to 200.
    blacklisted : list, optional
        List of sequences to exclude from KDE estimation.
    scale_factor: float or int, optional
        Densities will be multiplied by this value to account for differences in read depth. Defaults to 1 (no scaling).
        
    Returns
    -------
    bandwidths: arraylike
        the bandwidths of the KDE for each chromosome.
    """

    sequences = set(ebs_c1.keys()) - set(blacklisted)
    ntotal = len(sequences)
    all_events = pd.concat(ebs_c1)
    cuts = all_events['location'].values
    low_res_cuts = np.round(cuts/step)
    del all_events ## delete to conserve memory
    # validate user input 
    if (isinstance(kde_bw, numbers.Number) | kde_bw.isnumeric()): #use user specified bandwith
        bandwidth = kde_bw
    elif isinstance(kde_bw, str): #automated bandwidth selection 

        if kde_bw == 'silverman':
            bandwidth = bw_selection.silvermans_rule(low_res_cuts.reshape(-1,1))
        elif kde_bw == 'ISJ':
            bandwidth = bw_selection.improved_sheather_jones(low_res_cuts.reshape(-1,1))
        elif kde_bw.isnumeric():
            bandwidth = int(kde_bw) 
        else:
            raise ValueError("kde_bw must be one of the options: 'ISJ', 'silverman'")

    else:
        bandwidth = int(kde_bw)
    #Compute KDE for each chromosome using bandwidth    
    for i, seqname in enumerate(sorted(sequences)):
        logger.info(f"Making KDE of {seqname} [{i+1}/{ntotal}].")

        events = ebs_c1.get(seqname, pd.DataFrame())
        cuts = events["location"].values

        low_res_cuts = np.round(cuts / step)

        grid = full_kde_grid(low_res_cuts)
        cut_idx = (low_res_cuts - grid.min()).astype(int)
        _, density = get_kde(low_res_cuts, kde_bw=bandwidth, grid=grid)
            
        density *= len(events) * (1000/step)
        
        if i != 0:
            mode = 'a'
        else:
            mode = 'w'
        pd.DataFrame(
            {
                "seqname": seqname,
                "start": grid.astype(int) * step,
                'end': (grid.astype(int) * step) + (step), 
                "density": density * scale_factor,
            }).to_csv(file_path, sep = "\t", mode = mode, index = False, header = False, chunksize = 10000)
    return bandwidth

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
def mark_peaks_prominence(
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
