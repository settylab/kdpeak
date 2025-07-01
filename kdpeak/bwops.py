#!/usr/bin/env python
"""
BigWig Operations Tool (bwops)

A utility for processing and analyzing BigWig files with mathematical operations,
regression analysis, and multiple output formats.
"""

import argparse
import json
import logging
import os
import re
import sys
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyBigWig
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score

from .util import (
    handle_error,
    safe_file_operation,
    setup_logging,
    validate_file_exists,
    validate_output_directory,
)


def parse_arguments(args=None):
    """Parse command line arguments for bwops."""
    parser = argparse.ArgumentParser(
        description="BigWig Operations Tool - perform mathematical operations and regression analysis on BigWig files"
    )

    subparsers = parser.add_subparsers(dest="operation", help="Available operations")

    # Add operation
    add_parser = subparsers.add_parser("add", help="Add multiple BigWig files")
    add_parser.add_argument("input_files", nargs="+", help="Input BigWig files to add")
    add_parser.add_argument("--out", required=True, help="Output file")
    add_parser.add_argument(
        "--format",
        choices=["bigwig", "csv", "bed", "tsv", "json"],
        default="bigwig",
        help="Output format (default: bigwig)",
    )
    add_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each BigWig file so that its mean equals 1 before adding",
    )

    # Multiply operation
    mult_parser = subparsers.add_parser(
        "multiply", help="Multiply multiple BigWig files"
    )
    mult_parser.add_argument(
        "input_files", nargs="+", help="Input BigWig files to multiply"
    )
    mult_parser.add_argument("--out", required=True, help="Output file")
    mult_parser.add_argument(
        "--format",
        choices=["bigwig", "csv", "bed", "tsv", "json"],
        default="bigwig",
        help="Output format (default: bigwig)",
    )
    mult_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each BigWig file so that its mean equals 1 before multiplying",
    )

    # Regression operation
    regress_parser = subparsers.add_parser(
        "regress", help="Perform regression analysis"
    )
    regress_parser.add_argument(
        "--formula",
        help='Formula using variable names, e.g., "target ~ pred1 + pred2 + pred1*pred2" (default: auto-generate)',
    )
    regress_parser.add_argument(
        "--target",
        required=True,
        help='Target variable file and optional name, e.g., "target=file.bw" or just "file.bw" (uses "target")',
    )
    regress_parser.add_argument(
        "--predictors",
        nargs="+",
        required=True,
        help='Predictor variables as name=file pairs, e.g., "pred1=file1.bw pred2=file2.bw" or just files (uses a,b,c...)',
    )
    regress_parser.add_argument(
        "--type",
        choices=["linear", "logistic"],
        default="linear",
        help="Regression type (default: linear)",
    )
    regress_parser.add_argument("--out-prediction", help="Output file for predictions")
    regress_parser.add_argument("--out-residuals", help="Output file for residuals")
    regress_parser.add_argument(
        "--out-stats", help="Output file for detailed statistics"
    )
    regress_parser.add_argument(
        "--format",
        choices=["bigwig", "csv", "bed", "tsv", "json"],
        default="bigwig",
        help="Output format (default: bigwig)",
    )
    regress_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each BigWig file so that its mean equals 1 before regression",
    )

    # Stats operation
    stats_parser = subparsers.add_parser(
        "stats", help="Compute descriptive statistics for a single BigWig file"
    )
    stats_parser.add_argument("input_file", help="Input BigWig file")
    stats_parser.add_argument("--out", help="Output file for statistics (optional)")
    stats_parser.add_argument(
        "--format",
        choices=["json", "csv", "tsv"],
        default="json",
        help="Output format (default: json)",
    )
    stats_parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=[25, 50, 75, 90, 95, 99],
        help="Percentiles to compute (default: 25 50 75 90 95 99)",
    )

    # Correlation operation
    corr_parser = subparsers.add_parser(
        "correlate", help="Compute pairwise correlation matrix"
    )
    corr_parser.add_argument(
        "input_files", nargs="+", help="Input BigWig files for correlation analysis"
    )
    corr_parser.add_argument(
        "--out", required=True, help="Output file for correlation matrix"
    )
    corr_parser.add_argument(
        "--method",
        choices=["pearson", "spearman", "kendall"],
        default="pearson",
        help="Correlation method (default: pearson)",
    )
    corr_parser.add_argument(
        "--scope",
        choices=["global", "per-chromosome"],
        default="global",
        help="Compute global correlation or per-chromosome correlations (default: global)",
    )
    corr_parser.add_argument(
        "--min-overlap",
        type=int,
        default=1000,
        help="Minimum number of overlapping non-zero values required (default: 1000)",
    )
    corr_parser.add_argument(
        "--format",
        choices=["csv", "tsv", "json"],
        default="csv",
        help="Output format (default: csv)",
    )
    corr_parser.add_argument(
        "--include-stats",
        action="store_true",
        help="Include additional statistics (mean, std, coverage)",
    )
    corr_parser.add_argument(
        "--heatmap", help="Output heatmap plot (requires matplotlib)"
    )
    corr_parser.add_argument(
        "--scatter-plots", help="Directory for scatter plots of all pairs"
    )

    # Common arguments for all operations
    for subparser in [add_parser, mult_parser, regress_parser, stats_parser, corr_parser]:
        subparser.add_argument(
            "--chrom-sizes", help="Chromosome sizes file (required for BigWig output)"
        )
        subparser.add_argument(
            "--region", help="Limit analysis to genomic region (chr:start-end)"
        )
        subparser.add_argument(
            "--chromosomes", nargs="+", help="Limit analysis to specific chromosomes"
        )
        subparser.add_argument(
            "--span",
            type=int,
            default=None,
            help="Resolution for analysis in base pairs (default: auto-detect from BigWig files)",
        )

        # Chromosome filtering parameters (similar to kdpeak)
        subparser.add_argument(
            "--blacklisted-seqs",
            nargs="+",
            default=[],
            help="List of sequences to exclude from analysis",
        )
        subparser.add_argument(
            "--exclude-contigs",
            action="store_true",
            help="Exclude contigs/scaffolds with common keywords",
        )
        subparser.add_argument(
            "--chromosome-pattern",
            type=str,
            help="Regex pattern - only include chromosomes matching this pattern",
        )

        subparser.add_argument(
            "-l",
            "--log",
            dest="logLevel",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Set logging level",
        )
        subparser.add_argument("--logfile", help="Write log to file")

    return parser.parse_args(args)


def read_chrom_sizes(sizes_file: str) -> Dict[str, int]:
    """Read chromosome sizes from file."""
    chrom_sizes = {}
    with open(sizes_file, "r") as f:
        for line in f:
            chrom, size = line.strip().split()
            chrom_sizes[chrom] = int(size)
    return chrom_sizes


def parse_region(region_str: str) -> Tuple[str, int, int]:
    """Parse region string like 'chr1:1000-2000'."""
    if ":" not in region_str:
        raise ValueError("Region must be in format 'chr:start-end'")

    chrom, coords = region_str.split(":")
    start, end = coords.split("-")
    return chrom, int(start), int(end)


def get_common_chromosomes(
    bw_files: List[str],
    blacklisted_seqs: List[str] = None,
    exclude_contigs: bool = False,
    chromosome_pattern: str = None,
) -> List[str]:
    """Find chromosomes common to all BigWig files with filtering options."""
    from .util import filter_chromosomes

    common_chroms = None

    for bw_file in bw_files:
        with pyBigWig.open(bw_file) as bw:
            chroms = set(bw.chroms().keys())
            if common_chroms is None:
                common_chroms = chroms
            else:
                common_chroms = common_chroms.intersection(chroms)

    if common_chroms is None:
        return []

    # Convert to sorted list
    common_chroms = sorted(list(common_chroms))

    # Apply filtering using the same logic as kdpeak
    if blacklisted_seqs or exclude_contigs or chromosome_pattern:
        # Filter out blacklisted sequences first
        if blacklisted_seqs:
            common_chroms = [
                chrom for chrom in common_chroms if chrom not in blacklisted_seqs
            ]

        # Apply chromosome filtering
        common_chroms = filter_chromosomes(
            common_chroms, exclude_contigs, chromosome_pattern
        )

    return common_chroms


def get_native_resolution(
    bw_files: List[str], chromosomes: List[str]
) -> Tuple[int, bool]:
    """
    Determine the native resolution of BigWig files to avoid interpolation.
    Returns (resolution, needs_interpolation_warning)
    """
    import math
    from collections import Counter

    logger = logging.getLogger()

    all_spans = []
    file_resolutions = {}

    for bw_file in bw_files:
        with pyBigWig.open(bw_file) as bw:
            file_spans = []
            # Sample intervals from multiple chromosomes to get better estimate
            for chrom in chromosomes[:5]:  # Check first few chromosomes
                if chrom in bw.chroms():
                    # Get intervals from different parts of the chromosome
                    chrom_size = bw.chroms()[chrom]
                    sample_regions = [
                        (0, min(50000, chrom_size)),
                        (chrom_size // 2, min(chrom_size // 2 + 50000, chrom_size)),
                        (max(0, chrom_size - 50000), chrom_size),
                    ]

                    for start, end in sample_regions:
                        intervals = bw.intervals(chrom, start, end)
                        if intervals:
                            for interval_start, interval_end, _ in intervals[:20]:
                                span = interval_end - interval_start
                                if span > 0:
                                    file_spans.append(span)
                            if len(file_spans) >= 50:  # Enough samples
                                break
                    if len(file_spans) >= 50:
                        break

            if file_spans:
                # Use the most common span as the native resolution
                span_counts = Counter(file_spans)
                native_resolution = span_counts.most_common(1)[0][0]
                file_resolutions[bw_file] = native_resolution
                all_spans.extend(file_spans)
                logger.debug(
                    f"Native resolution for {os.path.basename(bw_file)}: {native_resolution} bp"
                )

    if not all_spans:
        logger.warning("Could not determine native resolution, using default 10 bp")
        return 10, True

    # Find the GCD of all detected resolutions for compatibility
    unique_resolutions = list(file_resolutions.values())
    if len(set(unique_resolutions)) == 1:
        # All files have the same resolution - perfect!
        resolution = unique_resolutions[0]
        needs_warning = False
        logger.info(f"All files have matching native resolution: {resolution} bp")
    else:
        # Files have different resolutions - need to find common divisor
        resolution = unique_resolutions[0]
        for res in unique_resolutions[1:]:
            resolution = math.gcd(resolution, res)

        needs_warning = resolution != max(unique_resolutions)
        if needs_warning:
            logger.warning(
                f"Files have different native resolutions: {unique_resolutions}"
            )
            logger.warning(
                f"Using GCD resolution {resolution} bp - this may require interpolation"
            )
            logger.warning(
                "For best performance, use BigWig files with matching resolutions"
            )

    return max(resolution, 1), needs_warning


def read_bigwig_data(
    bw_files: List[str],
    chromosomes: List[str],
    span: int,
    region: Optional[Tuple[str, int, int]] = None,
) -> pd.DataFrame:
    """Read and align data from multiple BigWig files efficiently at native resolution."""
    logger = logging.getLogger()

    all_data = []

    target_chroms = chromosomes
    if region:
        target_chroms = [region[0]] if region[0] in chromosomes else []

    # Check if we can use fast reading for all files (explicit detection)
    can_use_fast_read = {}
    for bw_file in bw_files:
        with pyBigWig.open(bw_file) as bw:
            # Sample a chromosome to check native resolution
            test_chrom = target_chroms[0] if target_chroms else None
            if test_chrom and test_chrom in bw.chroms():
                sample_intervals = bw.intervals(
                    test_chrom, 0, min(100000, bw.chroms()[test_chrom])
                )
                if sample_intervals:
                    native_spans = {
                        end - start for start, end, _ in sample_intervals[:50]
                    }
                    # Check if target span is present in the file
                    spans_match = span in native_spans
                    can_use_fast_read[bw_file] = spans_match
                    logger.debug(
                        f"File {os.path.basename(bw_file)}: found spans {sorted(native_spans)}, target={span}, can_use_fast={spans_match}"
                    )
                else:
                    can_use_fast_read[bw_file] = False
                    logger.debug(
                        f"File {os.path.basename(bw_file)}: no intervals found, can_use_fast=False"
                    )
            else:
                can_use_fast_read[bw_file] = False
                logger.debug(
                    f"File {os.path.basename(bw_file)}: test_chrom={test_chrom} not available, can_use_fast=False"
                )

    fast_files = [f for f, can_fast in can_use_fast_read.items() if can_fast]
    slow_files = [f for f, can_fast in can_use_fast_read.items() if not can_fast]

    if fast_files:
        logger.info(f"Fast reading enabled for {len(fast_files)}/{len(bw_files)} files")
        logger.debug(f"Fast reading files: {[os.path.basename(f) for f in fast_files]}")
    if slow_files:
        logger.info(
            f"Stats reading required for {len(slow_files)}/{len(bw_files)} files"
        )
        logger.debug(
            f"Stats reading files: {[os.path.basename(f) for f in slow_files]}"
        )

    # Progress tracking
    total_chroms = len(target_chroms)
    logger.info(f"Reading data from {total_chroms} chromosomes...")

    for chrom_idx, chrom in enumerate(target_chroms):
        # Simple progress indicator
        progress_pct = (chrom_idx / total_chroms) * 100
        logger.info(
            f"[{progress_pct:5.1f}%] Processing {chrom} ({chrom_idx+1}/{total_chroms})"
        )

        # Determine chromosome bounds
        chrom_start = 0
        chrom_end = None

        if region and region[0] == chrom:
            chrom_start = region[1]
            chrom_end = region[2]
        else:
            # Get chromosome size from first BigWig file
            with pyBigWig.open(bw_files[0]) as bw:
                if chrom in bw.chroms():
                    chrom_end = bw.chroms()[chrom]

        if chrom_end is None:
            logger.warning(f"Cannot determine size for chromosome {chrom}")
            continue

        # Use efficient grid-based approach but with native span
        coords = np.arange(chrom_start, chrom_end, span)
        if len(coords) == 0:
            continue

        # Warn if this will create a very large number of intervals AND we're using slow stats reading
        slow_files_for_chrom = [
            f for f in bw_files if not can_use_fast_read.get(f, False)
        ]
        if (
            len(coords) > 1_000_000 and slow_files_for_chrom
        ):  # 1M intervals AND slow files
            logger.warning(
                f"Chromosome {chrom} has {len(coords):,} intervals - will be slow for {len(slow_files_for_chrom)} files using stats reading"
            )
            logger.warning(
                "Consider using --span with a larger value or --region to limit analysis"
            )

        logger.debug(
            f"Chromosome {chrom}: {len(coords):,} intervals at {span}bp resolution"
        )

        # Create DataFrame for this chromosome
        chrom_data = {"chromosome": chrom, "start": coords, "end": coords + span}

        # Read data from each BigWig file efficiently
        for i, bw_file in enumerate(bw_files):
            col_name = f"bw_{i}_{os.path.basename(bw_file).replace('.bw', '').replace('.bigwig', '')}"

            with pyBigWig.open(bw_file) as bw:
                if chrom not in bw.chroms():
                    chrom_data[col_name] = np.zeros(len(coords))
                    continue

                # Use fast or slow method based on pre-computed check
                if can_use_fast_read.get(bw_file, False):
                    try:
                        logger.debug(
                            f"Using fast interval read for {os.path.basename(bw_file)} on {chrom}"
                        )
                        # Fast direct reading using intervals
                        intervals = bw.intervals(chrom, chrom_start, chrom_end)
                        if intervals:
                            # Simple approach: create array and fill values directly
                            values = np.zeros(len(coords))

                            for start, end, value in intervals:
                                # Find overlapping coordinates
                                start_idx = max(0, (start - chrom_start) // span)
                                end_idx = min(
                                    len(coords), (end - chrom_start) // span + 1
                                )

                                # Fill the overlapping range
                                if start_idx < len(coords) and end_idx > 0:
                                    actual_start = max(0, start_idx)
                                    actual_end = min(len(coords), end_idx)
                                    values[actual_start:actual_end] = value

                            chrom_data[col_name] = values
                            logger.debug(
                                f"Fast interval read completed for {os.path.basename(bw_file)}: {len(intervals)} intervals processed"
                            )
                        else:
                            logger.debug(
                                f"No intervals found for {os.path.basename(bw_file)} on {chrom}"
                            )
                            chrom_data[col_name] = np.zeros(len(coords))
                        continue  # Skip stats method
                    except Exception as e:
                        logger.warning(
                            f"Fast reading failed for {os.path.basename(bw_file)}: {e}, falling back to stats"
                        )

                # Stats method for averaging/interpolation
                # Process in chunks with progress reporting
                chunk_size = 50000  # Larger chunks for better performance
                values = []
                num_chunks = (len(coords) + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(coords))
                    chunk_coords = coords[start_idx:end_idx]

                    # Progress reporting for large chromosomes (less verbose)
                    if len(coords) > 1000000 and chunk_idx % 50 == 0:
                        progress = (chunk_idx / num_chunks) * 100
                        logger.info(
                            f"  {os.path.basename(bw_file)}: {progress:.1f}% complete"
                        )

                    # Process chunk
                    chunk_values = []
                    for coord in chunk_coords:
                        end_coord = min(coord + span, bw.chroms()[chrom])
                        try:
                            val = bw.stats(chrom, coord, end_coord, type="mean")[0]
                            chunk_values.append(val if val is not None else 0.0)
                        except:
                            chunk_values.append(0.0)

                    values.extend(chunk_values)

                chrom_data[col_name] = values

        # Convert to DataFrame and add to results
        if len(coords) > 0:
            all_data.append(pd.DataFrame(chrom_data))

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def parse_variable_mapping(
    target_arg: str, predictors_args: List[str]
) -> Tuple[Dict[str, str], str, List[str]]:
    """
    Parse target and predictor arguments to create variable name to file mapping.

    Returns:
        mapping: Dict of variable_name -> file_path
        target_var: Target variable name
        predictor_vars: List of predictor variable names
    """
    mapping = {}

    # Parse target
    if "=" in target_arg:
        target_var, target_file = target_arg.split("=", 1)
        target_var = target_var.strip()
        target_file = target_file.strip()
    else:
        target_var = "target"
        target_file = target_arg.strip()

    mapping[target_var] = target_file

    # Parse predictors
    predictor_vars = []
    default_names = [chr(ord("a") + i) for i in range(26)]  # a, b, c, ..., z
    default_idx = 0

    for pred_arg in predictors_args:
        if "=" in pred_arg:
            pred_var, pred_file = pred_arg.split("=", 1)
            pred_var = pred_var.strip()
            pred_file = pred_file.strip()
        else:
            pred_var = (
                default_names[default_idx]
                if default_idx < len(default_names)
                else f"pred{default_idx}"
            )
            pred_file = pred_arg.strip()
            default_idx += 1

        mapping[pred_var] = pred_file
        predictor_vars.append(pred_var)

    return mapping, target_var, predictor_vars


def generate_default_formula(target_var: str, predictor_vars: List[str]) -> str:
    """Generate default formula with all monomial terms."""
    predictors_str = " + ".join(predictor_vars)
    return f"{target_var} ~ {predictors_str}"


def parse_formula(formula: str) -> Tuple[str, List[str]]:
    """Parse R-style formula and return target and predictor variables."""
    if "~" not in formula:
        raise ValueError("Formula must contain '~' separator")

    target, predictors = formula.split("~", 1)
    target = target.strip()

    # Parse predictors (handle +, *, :, etc.)
    # For now, simple parsing - split by + and handle interactions
    predictor_terms = []
    for term in predictors.split("+"):
        term = term.strip()
        if term:
            predictor_terms.append(term)

    return target, predictor_terms


def perform_regression(
    data: pd.DataFrame,
    formula: str,
    regression_type: str = "linear",
    var_mapping: Dict[str, str] = None,
) -> Dict:
    """Perform regression analysis on the data using variable mapping."""
    logger = logging.getLogger()

    target_var, predictor_terms = parse_formula(formula)

    # Get BigWig columns from data
    file_columns = [col for col in data.columns if col.startswith("bw_")]

    # Create mapping from variable names to actual data columns
    if var_mapping is None:
        # Fallback to old logic if no variable mapping provided
        logger.warning("No variable mapping provided, using filename-based matching")
        var_to_col = {}
        for var_name in [target_var] + predictor_terms:
            # Extract base variable name (remove interaction symbols)
            base_var = re.sub(r"[*:].*", "", var_name.strip())
            var_basename = (
                os.path.basename(base_var).replace(".bw", "").replace(".bigwig", "")
            )

            for col in file_columns:
                if var_basename in col:
                    var_to_col[base_var] = col
                    break
    else:
        # Use provided variable mapping to find columns
        var_to_col = {}
        for var_name, file_path in var_mapping.items():
            file_basename = (
                os.path.basename(file_path).replace(".bw", "").replace(".bigwig", "")
            )

            # Find the column that corresponds to this file
            for col in file_columns:
                if file_basename in col:
                    var_to_col[var_name] = col
                    break

            if var_name not in var_to_col:
                raise ValueError(
                    f"Could not find data column for variable '{var_name}' (file: {file_path})"
                )

    # Find target column
    if target_var not in var_to_col:
        raise ValueError(
            f"Target variable '{target_var}' not found in variable mapping"
        )
    target_col = var_to_col[target_var]

    # Build design matrix
    X_cols = []
    X_names = []

    for term in predictor_terms:
        if "*" in term:
            # Interaction term
            factors = [f.strip() for f in term.split("*")]
            factor_cols = []

            for factor in factors:
                if factor not in var_to_col:
                    raise ValueError(
                        f"Predictor variable '{factor}' not found in variable mapping"
                    )
                factor_cols.append(var_to_col[factor])

            # Create interaction column
            interaction_data = data[factor_cols[0]].copy()
            interaction_name = factors[0]
            for i in range(1, len(factor_cols)):
                interaction_data *= data[factor_cols[i]]
                interaction_name += f"*{factors[i]}"

            X_cols.append(interaction_data)
            X_names.append(interaction_name)

        else:
            # Simple term
            term_clean = term.strip()
            if term_clean not in var_to_col:
                raise ValueError(
                    f"Predictor variable '{term_clean}' not found in variable mapping"
                )

            X_cols.append(data[var_to_col[term_clean]])
            X_names.append(term_clean)

    if not X_cols:
        raise ValueError("No predictor variables found in data")

    # Create design matrix
    X = np.column_stack(X_cols)
    y = data[target_col].values

    # Remove NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    if len(X) == 0:
        raise ValueError("No valid data points after removing NaN values")

    logger.info(f"Regression analysis: {len(y)} observations, {X.shape[1]} predictors")
    logger.debug(f"Target: {target_var} -> {target_col}")
    for i, name in enumerate(X_names):
        logger.debug(f"Predictor {i+1}: {name}")

    # Perform regression
    if regression_type == "linear":
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        residuals = y - predictions

        # Calculate statistics
        r2 = r2_score(y, predictions)
        n = len(y)
        p = X.shape[1]

        # Calculate p-values for coefficients (approximation)
        mse = np.sum(residuals**2) / (n - p - 1)
        try:
            var_coef = mse * np.diag(np.linalg.inv(X.T @ X))
            se_coef = np.sqrt(var_coef)
            t_stats = model.coef_ / se_coef
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
        except np.linalg.LinAlgError:
            logger.warning(
                "Singular matrix encountered - cannot compute p-values (likely due to multicollinearity)"
            )
            p_values = np.full(len(model.coef_), np.nan)

        results = {
            "model": model,
            "predictions": predictions,
            "residuals": residuals,
            "r2": r2,
            "coefficients": model.coef_,
            "intercept": model.intercept_,
            "p_values": p_values,
            "variable_names": X_names,
            "n_obs": n,
            "mse": mse,
            "mask": mask,
        }

    elif regression_type == "logistic":
        model = LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        residuals = y - predictions

        results = {
            "model": model,
            "predictions": predictions,
            "residuals": residuals,
            "coefficients": model.coef_[0],
            "intercept": model.intercept_[0],
            "variable_names": X_names,
            "n_obs": len(y),
            "mask": mask,
        }

    return results


def write_output(
    data: pd.DataFrame,
    output_file: str,
    format_type: str,
    chrom_sizes: Optional[Dict[str, int]] = None,
    span: int = 10,
):
    """Write data to specified output format."""
    logger = logging.getLogger()

    if format_type == "bigwig":
        if chrom_sizes is None:
            raise ValueError("Chromosome sizes required for BigWig output")
        write_bigwig_output(data, output_file, chrom_sizes, span)
    elif format_type == "csv":
        data.to_csv(output_file, index=False)
    elif format_type == "tsv":
        data.to_csv(output_file, sep="\t", index=False)
    elif format_type == "bed":
        # For BED format, use first 3 columns as chr, start, end
        bed_data = data[["chromosome", "start", "end"]].copy()
        # Add additional columns as BED scores/names
        for col in data.columns:
            if col not in ["chromosome", "start", "end"]:
                bed_data[col] = data[col]
        bed_data.to_csv(output_file, sep="\t", header=False, index=False)
    elif format_type == "json":
        data.to_json(output_file, orient="records", indent=2)

    logger.info(f"Output written to {output_file}")


def write_bigwig_output(
    data: pd.DataFrame, output_file: str, chrom_sizes: Dict[str, int], span: int
):
    """Write data to BigWig format."""
    # Assume the last column contains the values to write
    value_col = [
        col for col in data.columns if col not in ["chromosome", "start", "end"]
    ][-1]

    with pyBigWig.open(output_file, "w") as bw:
        bw.addHeader(list(sorted(chrom_sizes.items())))

        for chrom, chrom_data in data.groupby("chromosome"):
            if chrom not in chrom_sizes:
                continue

            bw.addEntries(
                chrom,
                int(chrom_data["start"].min()),
                span=span,
                step=span,
                values=chrom_data[value_col].values,
            )


def normalize_bigwig_data(data: pd.DataFrame, logger: logging.Logger = None) -> pd.DataFrame:
    """
    Normalize BigWig data so that the mean of each file equals 1.
    
    This is useful for comparing and combining signals from different experiments
    by putting them on the same scale.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with columns ['chromosome', 'start', 'end'] and BigWig value columns (bw_*)
    logger : logging.Logger, optional
        Logger for progress messages
        
    Returns
    -------
    pd.DataFrame
        Normalized data where each BigWig column has mean = 1
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create a copy to avoid modifying the original data
    normalized_data = data.copy()
    
    # Find all BigWig value columns
    bw_cols = [col for col in data.columns if col.startswith("bw_")]
    
    if not bw_cols:
        logger.warning("No BigWig value columns found for normalization")
        return normalized_data
    
    logger.info(f"Normalizing {len(bw_cols)} BigWig files to mean = 1...")
    
    normalization_factors = {}
    
    for col in bw_cols:
        # Calculate mean, excluding zeros for more meaningful normalization
        non_zero_values = data[col][data[col] != 0]
        
        if len(non_zero_values) == 0:
            logger.warning(f"Column {col} contains only zeros - cannot normalize")
            normalization_factors[col] = 1.0
            continue
            
        # Use mean of non-zero values for normalization
        mean_val = non_zero_values.mean()
        
        if mean_val == 0:
            logger.warning(f"Column {col} has zero mean - cannot normalize")
            normalization_factors[col] = 1.0
            continue
            
        normalization_factor = 1.0 / mean_val
        normalization_factors[col] = normalization_factor
        
        # Apply normalization
        normalized_data[col] = data[col] * normalization_factor
        
        # Verify normalization (check non-zero mean)
        new_mean = normalized_data[col][normalized_data[col] != 0].mean()
        
        logger.info(f"  {col}: original mean = {mean_val:.6f}, "
                   f"normalization factor = {normalization_factor:.6f}, "
                   f"new mean = {new_mean:.6f}")
    
    # Log overall statistics
    logger.info("Normalization completed:")
    for col in bw_cols:
        original_mean = data[col].mean()
        normalized_mean = normalized_data[col].mean() 
        logger.info(f"  {col}: {original_mean:.6f} → {normalized_mean:.6f} "
                   f"(factor: {normalization_factors[col]:.6f})")
    
    return normalized_data


def compute_statistics(
    data: pd.DataFrame, input_file: str, percentiles: List[float] = None
) -> Dict:
    """
    Compute comprehensive descriptive statistics for a single BigWig file.

    Parameters
    ----------
    data : pd.DataFrame
        Data with columns ['chromosome', 'start', 'end', 'bw_0'] 
    input_file : str
        Path to the input BigWig file
    percentiles : List[float]
        List of percentiles to compute (0-100)

    Returns
    -------
    Dict
        Dictionary containing comprehensive statistics
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95, 99]

    # Get the value column (should be 'bw_0' for single file)
    value_col = [col for col in data.columns if col.startswith("bw_")][0]
    values = data[value_col]

    # Basic statistics
    stats_dict = {
        "file": os.path.basename(input_file),
        "file_path": input_file,
        "total_bins": len(data),
        "non_zero_bins": int((values != 0).sum()),
        "zero_bins": int((values == 0).sum()),
        "coverage": float((values != 0).mean()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "variance": float(values.var()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(values.median()),
        "sum": float(values.sum()),
    }

    # Add percentiles
    stats_dict["percentiles"] = {}
    for p in percentiles:
        stats_dict["percentiles"][f"p{p}"] = float(np.percentile(values, p))

    # Additional statistics for non-zero values only
    non_zero_values = values[values != 0]
    if len(non_zero_values) > 0:
        stats_dict["non_zero_stats"] = {
            "mean": float(non_zero_values.mean()),
            "std": float(non_zero_values.std()),
            "min": float(non_zero_values.min()),
            "max": float(non_zero_values.max()),
            "median": float(non_zero_values.median()),
        }
    else:
        stats_dict["non_zero_stats"] = None

    # Per-chromosome statistics
    stats_dict["per_chromosome"] = {}
    for chrom in data["chromosome"].unique():
        chrom_data = data[data["chromosome"] == chrom]
        chrom_values = chrom_data[value_col]
        
        stats_dict["per_chromosome"][chrom] = {
            "bins": len(chrom_data),
            "non_zero_bins": int((chrom_values != 0).sum()),
            "coverage": float((chrom_values != 0).mean()),
            "mean": float(chrom_values.mean()),
            "std": float(chrom_values.std()),
            "min": float(chrom_values.min()),
            "max": float(chrom_values.max()),
            "median": float(chrom_values.median()),
            "sum": float(chrom_values.sum()),
        }

    return stats_dict


def write_statistics_output(stats_dict: Dict, output_file: str, format_type: str):
    """
    Write statistics to output file in specified format.

    Parameters
    ----------
    stats_dict : Dict
        Statistics dictionary from compute_statistics
    output_file : str
        Output file path
    format_type : str
        Output format: 'json', 'csv', or 'tsv'
    """
    if format_type == "json":
        with open(output_file, "w") as f:
            json.dump(stats_dict, f, indent=2)
    
    elif format_type in ["csv", "tsv"]:
        delimiter = "," if format_type == "csv" else "\t"
        
        # Flatten the statistics for tabular output
        rows = []
        
        # Overall statistics
        for key, value in stats_dict.items():
            if key not in ["percentiles", "per_chromosome", "non_zero_stats"]:
                rows.append({"category": "overall", "statistic": key, "value": value})
        
        # Percentiles
        if "percentiles" in stats_dict and stats_dict["percentiles"]:
            for p_name, p_value in stats_dict["percentiles"].items():
                rows.append({"category": "percentile", "statistic": p_name, "value": p_value})
        
        # Non-zero statistics
        if "non_zero_stats" in stats_dict and stats_dict["non_zero_stats"]:
            for stat_name, stat_value in stats_dict["non_zero_stats"].items():
                rows.append({"category": "non_zero", "statistic": stat_name, "value": stat_value})
        
        # Per-chromosome statistics (top 10 chromosomes by coverage)
        if "per_chromosome" in stats_dict:
            chrom_stats = stats_dict["per_chromosome"]
            # Sort chromosomes by coverage
            sorted_chroms = sorted(
                chrom_stats.items(), 
                key=lambda x: x[1]["coverage"], 
                reverse=True
            )[:10]  # Top 10 chromosomes
            
            for chrom, chrom_data in sorted_chroms:
                for stat_name, stat_value in chrom_data.items():
                    rows.append({
                        "category": f"chromosome_{chrom}", 
                        "statistic": stat_name, 
                        "value": stat_value
                    })
        
        # Write to file
        df = pd.DataFrame(rows)
        df.to_csv(output_file, sep=delimiter, index=False)


def compute_correlation_matrix(
    data: pd.DataFrame, method: str = "pearson", min_overlap: int = 1000
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute pairwise correlation matrix for BigWig data.

    Parameters
    ----------
    data : pd.DataFrame
        Data with columns ['chromosome', 'start', 'end'] and BigWig value columns
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'
    min_overlap : int
        Minimum number of overlapping non-zero values required

    Returns
    -------
    corr_matrix : pd.DataFrame
        Correlation matrix
    stats_dict : dict
        Additional statistics for each file
    """
    logger = logging.getLogger()

    # Get BigWig value columns
    value_cols = [col for col in data.columns if col.startswith("bw_")]
    n_files = len(value_cols)

    if n_files < 2:
        raise ValueError("Need at least 2 BigWig files for correlation analysis")

    logger.info(f"Computing {method} correlations for {n_files} files")

    # Clean column names for output
    clean_names = [col.replace("bw_", "").replace("_", ".") for col in value_cols]

    # Initialize correlation matrix
    corr_matrix = pd.DataFrame(index=clean_names, columns=clean_names, dtype=float)
    np.fill_diagonal(corr_matrix.values, 1.0)

    # Compute statistics for each file
    stats_dict = {}
    for i, (col, name) in enumerate(zip(value_cols, clean_names)):
        values = data[col].values
        non_zero_mask = values != 0
        stats_dict[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "mean_nonzero": (
                float(np.mean(values[non_zero_mask])) if non_zero_mask.any() else 0.0
            ),
            "coverage": float(np.sum(non_zero_mask) / len(values)),
            "n_points": len(values),
            "n_nonzero": int(np.sum(non_zero_mask)),
        }

    # Compute pairwise correlations
    n_pairs = len(list(combinations(range(n_files), 2)))
    logger.info(f"Computing {n_pairs} pairwise correlations...")

    for i, j in combinations(range(n_files), 2):
        col1, col2 = value_cols[i], value_cols[j]
        name1, name2 = clean_names[i], clean_names[j]

        x = data[col1].values
        y = data[col2].values

        # Remove NaN values and apply minimum overlap filter
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]

        if len(x_clean) < min_overlap:
            logger.warning(
                f"Insufficient overlap between {name1} and {name2}: {len(x_clean)} < {min_overlap}"
            )
            corr_matrix.loc[name1, name2] = np.nan
            corr_matrix.loc[name2, name1] = np.nan
            continue

        # Compute correlation
        try:
            if method == "pearson":
                corr_coef, p_value = stats.pearsonr(x_clean, y_clean)
            elif method == "spearman":
                corr_coef, p_value = stats.spearmanr(x_clean, y_clean)
            elif method == "kendall":
                corr_coef, p_value = stats.kendalltau(x_clean, y_clean)
            else:
                raise ValueError(f"Unknown correlation method: {method}")

            corr_matrix.loc[name1, name2] = corr_coef
            corr_matrix.loc[name2, name1] = corr_coef

            logger.debug(
                f"{name1} vs {name2}: r={corr_coef:.3f}, p={p_value:.2e}, n={len(x_clean)}"
            )

        except Exception as e:
            logger.warning(
                f"Failed to compute correlation between {name1} and {name2}: {e}"
            )
            corr_matrix.loc[name1, name2] = np.nan
            corr_matrix.loc[name2, name1] = np.nan

    return corr_matrix, stats_dict


def compute_per_chromosome_correlations(
    data: pd.DataFrame, method: str = "pearson", min_overlap: int = 1000
) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
    """
    Compute correlation matrices for each chromosome separately.

    Parameters
    ----------
    data : pd.DataFrame
        Data with columns ['chromosome', 'start', 'end'] and BigWig value columns
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'
    min_overlap : int
        Minimum number of overlapping non-zero values required

    Returns
    -------
    dict
        Dictionary mapping chromosome names to (correlation_matrix, stats_dict) tuples
    """
    logger = logging.getLogger()

    chromosomes = data["chromosome"].unique()
    logger.info(
        f"Computing per-chromosome correlations for {len(chromosomes)} chromosomes"
    )

    results = {}
    for chrom in chromosomes:
        logger.debug(f"Computing correlations for {chrom}")
        chrom_data = data[data["chromosome"] == chrom].copy()

        try:
            corr_matrix, stats_dict = compute_correlation_matrix(
                chrom_data, method=method, min_overlap=min_overlap
            )
            results[chrom] = (corr_matrix, stats_dict)
        except Exception as e:
            logger.warning(f"Failed to compute correlations for {chrom}: {e}")
            results[chrom] = (None, None)

    return results


def write_correlation_output(
    corr_matrix: pd.DataFrame,
    stats_dict: Dict,
    output_file: str,
    format_type: str = "csv",
    include_stats: bool = False,
    method: str = "pearson",
):
    """Write correlation results to file."""
    logger = logging.getLogger()

    if format_type == "csv":
        corr_matrix.to_csv(output_file)
        if include_stats:
            stats_file = output_file.replace(".csv", "_stats.csv")
            pd.DataFrame(stats_dict).T.to_csv(stats_file)
            logger.info(f"Statistics written to {stats_file}")
    elif format_type == "tsv":
        corr_matrix.to_csv(output_file, sep="\t")
        if include_stats:
            stats_file = output_file.replace(".tsv", "_stats.tsv")
            pd.DataFrame(stats_dict).T.to_csv(stats_file, sep="\t")
            logger.info(f"Statistics written to {stats_file}")
    elif format_type == "json":
        result = {"correlation_matrix": corr_matrix.to_dict(), "method": method}
        if include_stats:
            result["statistics"] = stats_dict

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

    logger.info(f"Correlation matrix written to {output_file}")


def create_correlation_heatmap(
    corr_matrix: pd.DataFrame, output_file: str, method: str = "pearson"
):
    """Create a heatmap visualization of the correlation matrix."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "matplotlib and seaborn are required for heatmap generation. Install with: pip install matplotlib seaborn"
        )

    logger = logging.getLogger()

    plt.figure(figsize=(10, 8))

    # Create heatmap
    mask = corr_matrix.isnull()
    sns.heatmap(
        corr_matrix.astype(float),
        annot=True,
        cmap="RdBu_r",
        center=0,
        square=True,
        mask=mask,
        cbar_kws={"label": f"{method.title()} Correlation"},
    )

    plt.title(f"{method.title()} Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Heatmap saved to {output_file}")


def create_scatter_plots(data: pd.DataFrame, output_dir: str, method: str = "pearson"):
    """Create scatter plots for all pairwise comparisons."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for scatter plot generation. Install with: pip install matplotlib"
        )

    logger = logging.getLogger()
    os.makedirs(output_dir, exist_ok=True)

    value_cols = [col for col in data.columns if col.startswith("bw_")]
    clean_names = [col.replace("bw_", "").replace("_", ".") for col in value_cols]

    n_pairs = len(list(combinations(range(len(value_cols)), 2)))
    logger.info(f"Creating {n_pairs} scatter plots...")

    for i, j in combinations(range(len(value_cols)), 2):
        col1, col2 = value_cols[i], value_cols[j]
        name1, name2 = clean_names[i], clean_names[j]

        x = data[col1].values
        y = data[col2].values

        # Remove NaN values and sample for plotting if too many points
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]

        if len(x_clean) > 50000:  # Sample for performance
            idx = np.random.choice(len(x_clean), 50000, replace=False)
            x_clean, y_clean = x_clean[idx], y_clean[idx]

        # Compute correlation for plot
        try:
            if method == "pearson":
                corr_coef, _ = stats.pearsonr(x_clean, y_clean)
            elif method == "spearman":
                corr_coef, _ = stats.spearmanr(x_clean, y_clean)
            elif method == "kendall":
                corr_coef, _ = stats.kendalltau(x_clean, y_clean)
        except:
            corr_coef = np.nan

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x_clean, y_clean, alpha=0.5, s=1)
        plt.xlabel(name1)
        plt.ylabel(name2)
        plt.title(f"{name1} vs {name2}\n{method.title()} r = {corr_coef:.3f}")

        # Add trend line
        if not np.isnan(corr_coef):
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)
            plt.plot(x_clean, p(x_clean), "r--", alpha=0.8)

        plt.tight_layout()

        safe_name1 = name1.replace("/", "_").replace(" ", "_")
        safe_name2 = name2.replace("/", "_").replace(" ", "_")
        plot_file = os.path.join(output_dir, f"{safe_name1}_vs_{safe_name2}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Scatter plots saved to {output_dir}")


def main():
    """Main function for bwops tool with comprehensive error handling."""
    try:
        args = parse_arguments()

        if args.operation is None:
            print("Please specify an operation. Use --help for available operations.")
            return 1

        logger = setup_logging(args.logLevel, args.logfile)

        # Validate input files early
        input_files = getattr(args, "input_files", [])
        if args.operation == "regress":
            # For regression, get files from target and predictors arguments
            if hasattr(args, "target") and args.target:
                target_file = args.target.split("=")[-1].strip()  # Get file part
                input_files.append(target_file)
            if hasattr(args, "predictors") and args.predictors:
                for pred in args.predictors:
                    pred_file = pred.split("=")[-1].strip()  # Get file part
                    input_files.append(pred_file)

        for input_file in input_files:
            try:
                validate_file_exists(input_file, "BigWig input file")
            except (FileNotFoundError, PermissionError) as e:
                handle_error(
                    e,
                    f"Cannot access input file '{input_file}'",
                    [
                        "Check that the BigWig file path is correct",
                        "Verify the file exists and is readable",
                        "Ensure the file is in BigWig format (.bw or .bigwig)",
                    ],
                )
                return 1

        # Validate output directory
        if hasattr(args, "out") and args.out:
            try:
                validate_output_directory(args.out)
            except PermissionError as e:
                handle_error(
                    e,
                    "Cannot write to output directory",
                    [
                        "Create the output directory if needed",
                        "Check directory permissions",
                        "Use a different output location",
                    ],
                )
                return 1

        # Read chromosome sizes if needed
        chrom_sizes = None
        if args.format == "bigwig" or any(
            getattr(args, attr, None) and attr.startswith("out_") for attr in vars(args)
        ):
            if not args.chrom_sizes:
                handle_error(
                    ValueError("Chromosome sizes file required for BigWig output"),
                    "Missing required chromosome sizes file",
                    [
                        "Add --chrom-sizes argument with a chromosome sizes file",
                        "Download with: fetchChromSizes hg38 hg38.chrom.sizes",
                        "Or use a different output format (csv, tsv, json)",
                    ],
                )
                return 1

            try:
                validate_file_exists(args.chrom_sizes, "chromosome sizes file")
                chrom_sizes = read_chrom_sizes(args.chrom_sizes)
            except (FileNotFoundError, PermissionError) as e:
                handle_error(
                    e,
                    "Failed to read chromosome sizes file",
                    [
                        "Check file path and permissions",
                        "Ensure file format: chromosome_name<tab>size",
                        "Download with fetchChromSizes if needed",
                    ],
                )
                return 1
            except Exception as e:
                handle_error(
                    e,
                    "Invalid chromosome sizes file format",
                    [
                        "Ensure file format: chromosome_name<tab>size",
                        "Check for proper tab separation",
                        "Verify numeric sizes in second column",
                    ],
                )
                return 1

        # Parse region if specified
        region = None
        if args.region:
            region = parse_region(args.region)

        # Get input files based on operation
        if args.operation in ["add", "multiply", "correlate"]:
            input_files = args.input_files
        elif args.operation == "stats":
            input_files = [args.input_file]
        elif args.operation == "regress":
            # Parse variable mappings from target and predictors arguments
            var_mapping, target_var, predictor_vars = parse_variable_mapping(
                args.target, args.predictors
            )

            # Generate default formula if not provided
            if not args.formula:
                args.formula = generate_default_formula(target_var, predictor_vars)
                logger.info(f"Using auto-generated formula: {args.formula}")

            # Extract all input files from variable mapping
            input_files = list(var_mapping.values())

        # Find common chromosomes with filtering
        common_chroms = get_common_chromosomes(
            input_files,
            args.blacklisted_seqs,
            args.exclude_contigs,
            args.chromosome_pattern,
        )

        # Apply additional chromosome selection if specified
        if args.chromosomes:
            common_chroms = [c for c in common_chroms if c in args.chromosomes]

        if not common_chroms:
            logger.error("No common chromosomes found after filtering")
            return

        # Less verbose chromosome listing - just count and first few
        if len(common_chroms) <= 5:
            logger.info(f"Processing {len(common_chroms)} chromosomes: {common_chroms}")
        else:
            logger.info(
                f"Processing {len(common_chroms)} chromosomes (showing first 5): {common_chroms[:5]}..."
            )

        # Determine span using native resolution to avoid interpolation
        if hasattr(args, "span") and args.span is not None:
            span = args.span
            logger.info(f"Using user-specified span: {span} bp")
            if span < 1000:
                logger.warning(
                    "Small span values may result in slow processing for large chromosomes"
                )
        else:
            span, needs_warning = get_native_resolution(input_files, common_chroms)
            if needs_warning:
                logger.warning(
                    "Consider using BigWig files with matching resolutions for optimal performance"
                )
            else:
                logger.info(
                    f"Using native resolution: {span} bp (no interpolation needed)"
                )

            # Suggest larger span for performance if native resolution is very small
            if span < 50:
                total_genome_size = sum(
                    250_000_000 for _ in common_chroms
                )  # Rough estimate
                estimated_intervals = total_genome_size // span
                if estimated_intervals > 10_000_000:  # 10M intervals
                    logger.warning(
                        f"Native resolution of {span}bp will create ~{estimated_intervals//1_000_000}M intervals"
                    )
                    logger.warning(
                        "For faster processing, consider using --span 1000 or larger"
                    )
                    logger.warning(
                        "This will require interpolation but may complete much faster"
                    )

        # Read data
        logger.info("Reading BigWig data...")
        data = read_bigwig_data(input_files, common_chroms, span, region)

        if data.empty:
            logger.error("No data read from input files")
            return

        logger.info(f"Read {len(data)} data points")

        # Apply normalization if requested
        if hasattr(args, 'normalize') and args.normalize:
            data = normalize_bigwig_data(data, logger)

        # Perform operation
        if args.operation == "add":
            value_cols = [col for col in data.columns if col.startswith("bw_")]
            data["result"] = data[value_cols].sum(axis=1)
            write_output(
                data[["chromosome", "start", "end", "result"]],
                args.out,
                args.format,
                chrom_sizes,
                span,
            )

        elif args.operation == "multiply":
            value_cols = [col for col in data.columns if col.startswith("bw_")]
            data["result"] = data[value_cols].prod(axis=1)
            write_output(
                data[["chromosome", "start", "end", "result"]],
                args.out,
                args.format,
                chrom_sizes,
                span,
            )

        elif args.operation == "stats":
            logger.info("Computing statistics...")
            
            # Compute comprehensive statistics
            stats_dict = compute_statistics(data, args.input_file, args.percentiles)
            
            # Print summary to console
            print("\n" + "=" * 50)
            print("BIGWIG STATISTICS")
            print("=" * 50)
            print(f"File: {stats_dict['file']}")
            print(f"Total bins: {stats_dict['total_bins']:,}")
            print(f"Non-zero bins: {stats_dict['non_zero_bins']:,}")
            print(f"Coverage: {stats_dict['coverage']:.4f}")
            print(f"Mean: {stats_dict['mean']:.4f}")
            print(f"Standard deviation: {stats_dict['std']:.4f}")
            print(f"Min: {stats_dict['min']:.4f}")
            print(f"Max: {stats_dict['max']:.4f}")
            print(f"Median: {stats_dict['median']:.4f}")
            print(f"Sum: {stats_dict['sum']:.4f}")
            
            # Print percentiles
            print("\nPercentiles:")
            for p_name, p_value in stats_dict['percentiles'].items():
                print(f"  {p_name}: {p_value:.4f}")
            
            # Print non-zero statistics if available
            if stats_dict['non_zero_stats']:
                nz_stats = stats_dict['non_zero_stats']
                print(f"\nNon-zero values only:")
                print(f"  Mean: {nz_stats['mean']:.4f}")
                print(f"  Std: {nz_stats['std']:.4f}")
                print(f"  Min: {nz_stats['min']:.4f}")
                print(f"  Max: {nz_stats['max']:.4f}")
            
            # Print top chromosomes by coverage
            chrom_stats = stats_dict['per_chromosome']
            sorted_chroms = sorted(
                chrom_stats.items(), 
                key=lambda x: x[1]['coverage'], 
                reverse=True
            )[:5]  # Top 5 chromosomes
            
            print(f"\nTop chromosomes by coverage:")
            for chrom, chrom_data in sorted_chroms:
                print(f"  {chrom}: coverage={chrom_data['coverage']:.4f}, mean={chrom_data['mean']:.4f}")
            
            # Write output file if specified
            if args.out:
                write_statistics_output(stats_dict, args.out, args.format)
                logger.info(f"Statistics written to {args.out}")
                print(f"\nDetailed statistics written to: {args.out}")

        elif args.operation == "regress":
            logger.info("Performing regression analysis...")
            results = perform_regression(data, args.formula, args.type, var_mapping)

            # Print summary statistics
            print("\n" + "=" * 50)
            print("REGRESSION RESULTS")
            print("=" * 50)
            print(f"Formula: {args.formula}")
            print(f"Regression type: {args.type}")
            print(f"Number of observations: {results['n_obs']}")

            if args.type == "linear":
                print(f"R-squared: {results['r2']:.4f}")
                print(f"MSE: {results['mse']:.4f}")

            print(f"\nIntercept: {results['intercept']:.4f}")
            print("\nCoefficients:")
            for name, coef in zip(results["variable_names"], results["coefficients"]):
                if args.type == "linear" and "p_values" in results:
                    pval_idx = results["variable_names"].index(name)
                    print(
                        f"  {name}: {coef:.4f} (p-value: {results['p_values'][pval_idx]:.4f})"
                    )
                else:
                    print(f"  {name}: {coef:.4f}")

            # Write outputs if specified
            if args.out_prediction:
                pred_data = data.copy()
                pred_data["prediction"] = np.nan
                pred_data.loc[results["mask"], "prediction"] = results["predictions"]
                write_output(
                    pred_data[["chromosome", "start", "end", "prediction"]],
                    args.out_prediction,
                    args.format,
                    chrom_sizes,
                    span,
                )

            if args.out_residuals:
                resid_data = data.copy()
                resid_data["residuals"] = np.nan
                resid_data.loc[results["mask"], "residuals"] = results["residuals"]
                write_output(
                    resid_data[["chromosome", "start", "end", "residuals"]],
                    args.out_residuals,
                    args.format,
                    chrom_sizes,
                    span,
                )

            if args.out_stats:
                # Write detailed statistics
                stats_dict = {
                    "formula": args.formula,
                    "regression_type": args.type,
                    "n_observations": results["n_obs"],
                    "intercept": float(results["intercept"]),
                    "coefficients": {
                        name: float(coef)
                        for name, coef in zip(
                            results["variable_names"], results["coefficients"]
                        )
                    },
                }

                if args.type == "linear":
                    stats_dict["r_squared"] = float(results["r2"])
                    stats_dict["mse"] = float(results["mse"])
                    stats_dict["p_values"] = {
                        name: float(pval)
                        for name, pval in zip(
                            results["variable_names"], results["p_values"]
                        )
                    }

                with open(args.out_stats, "w") as f:
                    json.dump(stats_dict, f, indent=2)

        elif args.operation == "correlate":
            logger.info("Performing correlation analysis...")

            if args.scope == "global":
                # Global correlation across all chromosomes
                corr_matrix, stats_dict = compute_correlation_matrix(
                    data, method=args.method, min_overlap=args.min_overlap
                )

                # Print summary
                print("\n" + "=" * 50)
                print("CORRELATION ANALYSIS RESULTS")
                print("=" * 50)
                print(f"Method: {args.method}")
                print(f"Scope: {args.scope}")
                print(f"Number of files: {len(input_files)}")
                print(f"Total data points: {len(data)}")
                print(f"Minimum overlap required: {args.min_overlap}")
                print("\nCorrelation Matrix:")
                print(corr_matrix.round(3))

                # Write output
                write_correlation_output(
                    corr_matrix,
                    stats_dict,
                    args.out,
                    args.format,
                    args.include_stats,
                    args.method,
                )

            elif args.scope == "per-chromosome":
                # Per-chromosome correlations
                chrom_results = compute_per_chromosome_correlations(
                    data, method=args.method, min_overlap=args.min_overlap
                )

                # Print summary
                print("\n" + "=" * 50)
                print("PER-CHROMOSOME CORRELATION RESULTS")
                print("=" * 50)
                print(f"Method: {args.method}")
                print(f"Number of chromosomes: {len(chrom_results)}")

                # Write per-chromosome results
                for chrom, (corr_matrix, stats_dict) in chrom_results.items():
                    if corr_matrix is not None:
                        chrom_output = args.out.replace(".", f"_{chrom}.")
                        write_correlation_output(
                            corr_matrix,
                            stats_dict,
                            chrom_output,
                            args.format,
                            args.include_stats,
                            args.method,
                        )
                        print(f"\n{chrom}:")
                        print(corr_matrix.round(3))

            # Generate optional visualizations
            if args.heatmap:
                if args.scope == "global":
                    create_correlation_heatmap(corr_matrix, args.heatmap, args.method)
                else:
                    logger.warning(
                        "Heatmap generation only supported for global correlations"
                    )

            if args.scatter_plots:
                create_scatter_plots(data, args.scatter_plots, args.method)

            logger.info("Analysis completed successfully")
            print(f"\nSUCCESS: {args.operation} operation completed!")
            return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        handle_error(
            e,
            f"Unexpected error during {getattr(args, 'operation', 'bwops')} operation",
            [
                "Run with --log DEBUG for detailed information",
                "Check input file formats and parameters",
                "Ensure sufficient memory and disk space",
                "Report this error if it persists",
            ],
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
