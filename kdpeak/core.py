#!/usr/bin/env python
"""
Script to identify genomic peaks based on kernel density estimation (KDE) from
genomic reads in bed format.

This script reads a bed file, applies a KDE, identifies peaks based on the KDE
and writes the result into an output directory. It also allows for specification
of various parameters like KDE bandwidth, logging levels, sequence blacklist, etc.
"""

import os
import argparse


from .util import (
    events_dict_from_file,
    make_kdes,
    call_peaks,
    include_auc,
    name_peaks,
    write_bed,
    setup_logging,
)


def parse_arguments():
    """
    Parses command line arguments.

    Returns
    -------
    Parsed command line arguments.
    """
    desc = "A script that uses kernel density estimation (KDE) to identify genomic peaks from bed-formatted genomic reads."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "bed_file",
        metavar="reads.bed",
        type=str,
        help="Path to the bed file containing the genomic reads.",
    )

    parser.add_argument(
        "--out",
        metavar="output_file",
        type=str,
        default="./peaks.bed",
        help="Path to the output file where the results will be saved. Defaults to ./peaks.bed.",
    )

    parser.add_argument(
        "-l",
        "--log",
        dest="logLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level. Options include: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO.",
        metavar="LEVEL",
    )

    parser.add_argument(
        "--logfile",
        metavar="logfile",
        type=str,
        help="Path to the file where a detailed log will be written.",
    )

    parser.add_argument(
        "--blacklisted-seqs",
        nargs="+",
        default=[],
        metavar="chrN",
        type=str,
        help="List of sequences (e.g., chromosomes) to exclude from peak calling. Input as space-separated values.",
    )

    parser.add_argument(
        "--kde-bw",
        metavar="float",
        type=float,
        default=200,
        help="Bandwidth (standard deviation, sigma) for the kernel density estimation (KDE). Default is 200.",
    )

    parser.add_argument(
        "--min-peak-size",
        metavar="int",
        type=int,
        default=100,
        help="Minimal size (in base pairs) for a peak to be considered valid. Default is 100.",
    )

    parser.add_argument(
        "--fraction-in-peaks",
        "--frip",
        metavar="float",
        type=float,
        default=0.3,
        help="Expected fraction of total reads to be located in peaks. Default is 0.3.",
    )

    parser.add_argument(
        "--span",
        metavar="int",
        type=int,
        default=10,
        help="Resolution of the analysis in base pairs, which determines the granularity of the KDE and peak calling. Default is 10.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    logger = setup_logging(args.logLevel, args.logfile)

    logger.info("Reading %s", args.bed_file)
    ebs_c1 = events_dict_from_file(args.bed_file)

    if not ebs_c1:
        logger.error("No events found, aborting process.")
        return

    comb_data, signal_list_global = make_kdes(
        ebs_c1,
        step=args.span,
        kde_bw=args.kde_bw,
        blacklisted=args.blacklisted_seqs,
    )

    logger.info("Calling peaks.")
    peaks = call_peaks(
        comb_data,
        signal_list_global,
        fraction_in_peaks=args.fraction_in_peaks,
        min_peak_size=args.min_peak_size,
        span=args.span,
    )

    bed = include_auc(name_peaks(peaks))

    out_file = args.out
    logger.info("Writing results to %s...", out_file)
    write_bed(bed[["seqname", "start", "end", "name", "auc"]], out_file)
    logger.info("Finished successfully.")


if __name__ == "__main__":
    main()
