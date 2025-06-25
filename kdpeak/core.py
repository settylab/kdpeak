#!/usr/bin/env python
"""
Script to identify genomic peaks based on kernel density estimation (KDE) from
genomic reads in bed format.

This script reads a bed file, applies a KDE, identifies peaks based on the KDE
and writes the result into an output directory. It also allows for specification
of various parameters like KDE bandwidth, logging levels, sequence blacklist, etc.
"""

import os
import sys
import argparse


from .util import (
    events_dict_from_file,
    make_kdes,
    call_peaks,
    include_auc,
    name_peaks,
    write_bed,
    write_bigwig,
    setup_logging,
    handle_error,
    validate_file_exists,
    validate_output_directory,
    safe_file_operation,
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
        metavar="output_file.bed",
        type=str,
        default="./peaks.bed",
        help="""Path to the output file where the results will be saved. \
            Peaks are saved in bed format with the columns: \
            start, end, peak name, AUC (area under the cut density curve \
            where cut-density is in cuts per 100 base pairs). Defaults to peaks.bed.""",
    )

    parser.add_argument(
        "--summits-out",
        metavar="summits_file.bed",
        type=str,
        help="""Path to the output file where the peak summits will be saved.\
        The file will have columns for start, end (start+1), \
        peak name, and summit height (in cuts per 100 base pairs). \
        If nothing is specified the summits will not be saved.""",
    )

    parser.add_argument(
        "--density-out",
        metavar="density_file.bw",
        type=str,
        help="""Path to the output file where the event density will be saved.\
        The event density is the internally computed signal on which the peaks \
        are called based on a cutoff. It will be saved in the bigwig format."""
    )

    parser.add_argument(
        "--chrom-sizes",
        help="""Chromosome sizes file with the two columns: seqname and size.
        This file is only needed if --density-out is specified.
        You can use a script like
        https://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/fetchChromSizes
        to fetch the file.
        """,
        metavar="chrom-sizes-file",
        type=str,
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
        default=20,
        help="Bandwidth (standard deviation, sigma) for the kernel density estimation (KDE). Default is 20.",
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

    parser.add_argument(
        "--exclude-contigs",
        action="store_true",
        help="Exclude contigs/scaffolds containing common keywords: random, Un, alt, patch, hap, scaffold, contig, chrM, chrMT.",
    )

    parser.add_argument(
        "--chromosome-pattern",
        metavar="regex",
        type=str,
        help="Include only chromosomes matching this regex pattern (e.g., '^chr[1-9XY]$|^chr[12][0-9]$|^chr2[0-2]$' for human main chromosomes).",
    )

    return parser.parse_args()


def main():
    """Main entry point for kdpeak with comprehensive error handling."""
    try:
        args = parse_arguments()
        logger = setup_logging(args.logLevel, args.logfile)

        # Validate input file
        try:
            validate_file_exists(args.bed_file, "BED input file")
        except (FileNotFoundError, PermissionError) as e:
            handle_error(e, "Failed to access input BED file", [
                "Check that the file path is correct",
                "Verify the file exists and is readable",
                "Use absolute paths if needed"
            ])
            return 1

        # Validate chromosome sizes file if provided
        if args.chrom_sizes:
            try:
                validate_file_exists(args.chrom_sizes, "chromosome sizes file")
            except (FileNotFoundError, PermissionError) as e:
                handle_error(e, "Failed to access chromosome sizes file", [
                    "Download chromosome sizes with: fetchChromSizes hg38 hg38.chrom.sizes",
                    "Ensure the file format is: chromosome_name<tab>size",
                    "Check file permissions"
                ])
                return 1

        # Validate output directories
        try:
            validate_output_directory(args.out)
            if args.summits_out:
                validate_output_directory(args.summits_out)
            if args.density_out:
                validate_output_directory(args.density_out)
        except PermissionError as e:
            handle_error(e, "Cannot write to output directory", [
                "Create the output directory if it doesn't exist",
                "Check directory permissions",
                "Use a different output directory"
            ])
            return 1

        # Read input data
        logger.info("Reading %s", args.bed_file)
        
        def read_bed_file():
            return events_dict_from_file(args.bed_file)
        
        ebs_c1 = safe_file_operation(
            read_bed_file,
            "Failed to read BED file",
            [
                "Ensure the file is in valid BED format (tab-separated)",
                "Check that the file contains at least 3 columns: chromosome, start, end",
                "Verify the file is not corrupted"
            ]
        )
        
        if ebs_c1 is None:
            return 1

        if not ebs_c1:
            handle_error(
                ValueError("No genomic intervals found in input file"),
                "Empty or invalid BED file",
                [
                    "Check that the BED file contains data",
                    "Verify the file format: chromosome<tab>start<tab>end",
                    "Ensure coordinates are valid (start < end, non-negative)"
                ]
            )
            return 1

        # Process KDE and peak calling
        try:
            logger.info("Computing kernel density estimation...")
            comb_data, signal_list_global = make_kdes(
                ebs_c1,
                step=args.span,
                kde_bw=args.kde_bw,
                blacklisted=args.blacklisted_seqs,
                chrom_sizes_file=args.chrom_sizes,
                exclude_contigs=args.exclude_contigs,
                chromosome_pattern=args.chromosome_pattern,
            )

            if comb_data.empty:
                handle_error(
                    ValueError("No data remaining after filtering"),
                    "All chromosomes were filtered out",
                    [
                        "Check chromosome filtering settings (--exclude-contigs, --chromosome-pattern)",
                        "Verify blacklisted sequences (--blacklisted-seqs)",
                        "Ensure your BED file contains valid chromosome names"
                    ]
                )
                return 1

            logger.info("Calling peaks...")
            peaks = call_peaks(
                comb_data,
                signal_list_global,
                fraction_in_peaks=args.fraction_in_peaks,
                min_peak_size=args.min_peak_size,
                span=args.span,
            )

            if peaks.empty:
                handle_error(
                    ValueError("No peaks found"),
                    "Peak calling did not identify any peaks",
                    [
                        f"Try reducing --frip (currently {args.fraction_in_peaks})",
                        f"Try reducing --min-peak-size (currently {args.min_peak_size})",
                        f"Try adjusting --kde-bw (currently {args.kde_bw})",
                        "Check if your data has sufficient signal"
                    ]
                )
                return 1

            bed = include_auc(name_peaks(peaks))

        except Exception as e:
            handle_error(e, "Failed during peak calling analysis", [
                "Try adjusting parameters (--kde-bw, --frip, --min-peak-size)",
                "Check input data quality",
                "Ensure sufficient memory is available"
            ])
            return 1

        # Write output files
        def write_main_output():
            write_bed(bed[["seqname", "start", "end", "name", "auc"]], args.out)
            return True  # Return success indicator
        
        logger.info("Writing results to %s...", args.out)
        if safe_file_operation(
            write_main_output,
            f"Failed to write main output to {args.out}",
            [
                "Check output directory permissions",
                "Ensure sufficient disk space",
                "Verify the output path is valid"
            ]
        ) is None:
            return 1
        
        # Write optional outputs
        if args.summits_out:
            def write_summits():
                bed_summits = bed.copy()
                bed_summits["start"] = bed_summits["summit"]
                bed_summits["end"] = bed_summits["summit"] + 1
                write_bed(bed_summits[["seqname", "start", "end", "name", "summit_height"]], args.summits_out)
                return True  # Return success indicator
            
            logger.info("Writing summits to %s...", args.summits_out)
            if safe_file_operation(
                write_summits,
                f"Failed to write summits to {args.summits_out}"
            ) is None:
                return 1

        if args.density_out:
            if not args.chrom_sizes:
                handle_error(
                    ValueError("Chromosome sizes file required for BigWig output"),
                    "Missing required chromosome sizes file",
                    [
                        "Add --chrom-sizes argument with a chromosome sizes file",
                        "Download with: fetchChromSizes hg38 hg38.chrom.sizes",
                        "Or use a different output format"
                    ]
                )
                return 1
            
            def write_density():
                write_bigwig(comb_data, args.density_out, sizes_file=args.chrom_sizes, span=args.span)
                return True  # Return success indicator
            
            logger.info("Writing density to %s...", args.density_out)
            if safe_file_operation(
                write_density,
                f"Failed to write density BigWig to {args.density_out}",
                [
                    "Ensure pyBigWig is installed correctly",
                    "Check chromosome sizes file format",
                    "Verify sufficient disk space"
                ]
            ) is None:
                return 1
        
        logger.info("Finished successfully.")
        print(f"\nSUCCESS: Peak calling completed!")
        print(f"Results written to: {args.out}")
        if args.summits_out:
            print(f"Summits written to: {args.summits_out}")
        if args.density_out:
            print(f"Density written to: {args.density_out}")
        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        handle_error(e, "Unexpected error during peak calling", [
            "Run with --log DEBUG for detailed information",
            "Check input file format and parameters",
            "Report this error if it persists"
        ])
        return 1

if __name__ == "__main__":
    sys.exit(main())
