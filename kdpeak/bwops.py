#!/usr/bin/env python
"""
BigWig Operations Tool (bwops)

A utility for processing and analyzing BigWig files with mathematical operations,
regression analysis, and multiple output formats.
"""

import os
import argparse
import re
import numpy as np
import pandas as pd
import pyBigWig
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
import logging
from typing import Dict, List, Tuple, Optional, Union
import json

from .util import setup_logging


def parse_arguments(args=None):
    """Parse command line arguments for bwops."""
    parser = argparse.ArgumentParser(
        description="BigWig Operations Tool - perform mathematical operations and regression analysis on BigWig files"
    )
    
    subparsers = parser.add_subparsers(dest='operation', help='Available operations')
    
    # Add operation
    add_parser = subparsers.add_parser('add', help='Add multiple BigWig files')
    add_parser.add_argument('input_files', nargs='+', help='Input BigWig files to add')
    add_parser.add_argument('--out', required=True, help='Output file')
    add_parser.add_argument('--format', choices=['bigwig', 'csv', 'bed', 'tsv', 'json'], 
                           default='bigwig', help='Output format (default: bigwig)')
    
    # Multiply operation
    mult_parser = subparsers.add_parser('multiply', help='Multiply multiple BigWig files')
    mult_parser.add_argument('input_files', nargs='+', help='Input BigWig files to multiply')
    mult_parser.add_argument('--out', required=True, help='Output file')
    mult_parser.add_argument('--format', choices=['bigwig', 'csv', 'bed', 'tsv', 'json'], 
                            default='bigwig', help='Output format (default: bigwig)')
    
    # Regression operation
    regress_parser = subparsers.add_parser('regress', help='Perform regression analysis')
    regress_parser.add_argument('--formula', required=True, 
                               help='R-style formula, e.g., "target.bw ~ predictor1.bw + predictor2.bw"')
    regress_parser.add_argument('--type', choices=['linear', 'logistic'], default='linear',
                               help='Regression type (default: linear)')
    regress_parser.add_argument('--out-prediction', help='Output file for predictions')
    regress_parser.add_argument('--out-residuals', help='Output file for residuals')
    regress_parser.add_argument('--out-stats', help='Output file for detailed statistics')
    regress_parser.add_argument('--format', choices=['bigwig', 'csv', 'bed', 'tsv', 'json'], 
                               default='bigwig', help='Output format (default: bigwig)')
    
    # Common arguments for all operations
    for subparser in [add_parser, mult_parser, regress_parser]:
        subparser.add_argument('--chrom-sizes', help='Chromosome sizes file (required for BigWig output)')
        subparser.add_argument('--region', help='Limit analysis to genomic region (chr:start-end)')
        subparser.add_argument('--chromosomes', nargs='+', help='Limit analysis to specific chromosomes')
        subparser.add_argument('--span', type=int, default=None, 
                             help='Resolution for analysis in base pairs (default: auto-detect from BigWig files)')
        
        # Chromosome filtering parameters (similar to kdpeak)
        subparser.add_argument('--blacklisted-seqs', nargs='+', default=[], 
                             help='List of sequences to exclude from analysis')
        subparser.add_argument('--exclude-contigs', action='store_true',
                             help='Exclude contigs/scaffolds with common keywords')
        subparser.add_argument('--chromosome-pattern', type=str,
                             help='Regex pattern - only include chromosomes matching this pattern')
        
        subparser.add_argument('-l', '--log', dest='logLevel',
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                             default='INFO', help='Set logging level')
        subparser.add_argument('--logfile', help='Write log to file')
    
    return parser.parse_args(args)


def read_chrom_sizes(sizes_file: str) -> Dict[str, int]:
    """Read chromosome sizes from file."""
    chrom_sizes = {}
    with open(sizes_file, 'r') as f:
        for line in f:
            chrom, size = line.strip().split()
            chrom_sizes[chrom] = int(size)
    return chrom_sizes


def parse_region(region_str: str) -> Tuple[str, int, int]:
    """Parse region string like 'chr1:1000-2000'."""
    if ':' not in region_str:
        raise ValueError("Region must be in format 'chr:start-end'")
    
    chrom, coords = region_str.split(':')
    start, end = coords.split('-')
    return chrom, int(start), int(end)


def get_common_chromosomes(bw_files: List[str], blacklisted_seqs: List[str] = None, 
                          exclude_contigs: bool = False, chromosome_pattern: str = None) -> List[str]:
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
            common_chroms = [chrom for chrom in common_chroms if chrom not in blacklisted_seqs]
        
        # Apply chromosome filtering
        common_chroms = filter_chromosomes(
            common_chroms, 
            exclude_contigs, 
            chromosome_pattern
        )
    
    return common_chroms


def get_native_resolution(bw_files: List[str], chromosomes: List[str]) -> Tuple[int, bool]:
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
                        (max(0, chrom_size - 50000), chrom_size)
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
                logger.debug(f"Native resolution for {os.path.basename(bw_file)}: {native_resolution} bp")
    
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
            logger.warning(f"Files have different native resolutions: {unique_resolutions}")
            logger.warning(f"Using GCD resolution {resolution} bp - this may require interpolation")
            logger.warning("For best performance, use BigWig files with matching resolutions")
    
    return max(resolution, 1), needs_warning


def read_bigwig_data(bw_files: List[str], chromosomes: List[str], 
                    span: int, region: Optional[Tuple[str, int, int]] = None) -> pd.DataFrame:
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
                sample_intervals = bw.intervals(test_chrom, 0, min(100000, bw.chroms()[test_chrom]))
                if sample_intervals:
                    native_spans = {end - start for start, end, _ in sample_intervals[:50]}
                    # Check if target span is present in the file
                    spans_match = span in native_spans
                    can_use_fast_read[bw_file] = spans_match
                    logger.info(f"File {os.path.basename(bw_file)}: found spans {sorted(native_spans)}, target={span}, can_use_fast={spans_match}")
                else:
                    can_use_fast_read[bw_file] = False
                    logger.info(f"File {os.path.basename(bw_file)}: no intervals found, can_use_fast=False")
            else:
                can_use_fast_read[bw_file] = False
                logger.info(f"File {os.path.basename(bw_file)}: test_chrom={test_chrom} not available, can_use_fast=False")
    
    fast_files = [f for f, can_fast in can_use_fast_read.items() if can_fast]
    slow_files = [f for f, can_fast in can_use_fast_read.items() if not can_fast]
    
    if fast_files:
        logger.info(f"Using fast direct reading for {len(fast_files)} files: {[os.path.basename(f) for f in fast_files]}")
    if slow_files:
        logger.info(f"Using stats-based reading for {len(slow_files)} files: {[os.path.basename(f) for f in slow_files]}")
    
    for chrom_idx, chrom in enumerate(target_chroms):
        logger.info(f"Processing chromosome {chrom} ({chrom_idx+1}/{len(target_chroms)})")
        
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
        slow_files_for_chrom = [f for f in bw_files if not can_use_fast_read.get(f, False)]
        if len(coords) > 1_000_000 and slow_files_for_chrom:  # 1M intervals AND slow files
            logger.warning(f"Chromosome {chrom} has {len(coords):,} intervals at {span}bp resolution")
            logger.warning(f"Will be slow for {len(slow_files_for_chrom)} files using stats reading: {[os.path.basename(f) for f in slow_files_for_chrom]}")
            logger.warning("Consider using --span with a larger value or --region to limit analysis")
        elif len(coords) > 1_000_000:
            logger.info(f"Chromosome {chrom} has {len(coords):,} intervals at {span}bp resolution (fast reading enabled)")
            
        # Create DataFrame for this chromosome
        chrom_data = {
            'chromosome': chrom,
            'start': coords, 
            'end': coords + span
        }
        
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
                        logger.info(f"Using fast interval read for {os.path.basename(bw_file)} on {chrom}")
                        # Fast direct reading using intervals
                        intervals = bw.intervals(chrom, chrom_start, chrom_end)
                        if intervals:
                            # Simple approach: create array and fill values directly
                            values = np.zeros(len(coords))
                            
                            for start, end, value in intervals:
                                # Find overlapping coordinates
                                start_idx = max(0, (start - chrom_start) // span)
                                end_idx = min(len(coords), (end - chrom_start) // span + 1)
                                
                                # Fill the overlapping range
                                if start_idx < len(coords) and end_idx > 0:
                                    actual_start = max(0, start_idx)
                                    actual_end = min(len(coords), end_idx)
                                    values[actual_start:actual_end] = value
                            
                            chrom_data[col_name] = values
                            logger.info(f"Fast interval read completed for {os.path.basename(bw_file)}: {len(intervals)} intervals processed")
                        else:
                            logger.debug(f"No intervals found for {os.path.basename(bw_file)} on {chrom}")
                            chrom_data[col_name] = np.zeros(len(coords))
                        continue  # Skip stats method
                    except Exception as e:
                        logger.warning(f"Fast reading failed for {os.path.basename(bw_file)}: {e}, falling back to stats")
                
                # Stats method for averaging/interpolation
                # Process in chunks with progress reporting
                chunk_size = 50000  # Larger chunks for better performance
                values = []
                num_chunks = (len(coords) + chunk_size - 1) // chunk_size
                
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(coords))
                    chunk_coords = coords[start_idx:end_idx]
                    
                    # Progress reporting for large chromosomes
                    if len(coords) > 500000 and chunk_idx % 10 == 0:
                        progress = (chunk_idx / num_chunks) * 100
                        logger.info(f"  {os.path.basename(bw_file)}: {progress:.1f}% complete")
                    
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


def parse_formula(formula: str) -> Tuple[str, List[str]]:
    """Parse R-style formula and return target and predictor variables."""
    if '~' not in formula:
        raise ValueError("Formula must contain '~' separator")
    
    target, predictors = formula.split('~', 1)
    target = target.strip()
    
    # Parse predictors (handle +, *, :, etc.)
    # For now, simple parsing - split by + and handle interactions
    predictor_terms = []
    for term in predictors.split('+'):
        term = term.strip()
        if term:
            predictor_terms.append(term)
    
    return target, predictor_terms


def perform_regression(data: pd.DataFrame, formula: str, regression_type: str = 'linear') -> Dict:
    """Perform regression analysis on the data."""
    logger = logging.getLogger()
    
    target_var, predictor_terms = parse_formula(formula)
    
    # Map formula variables to actual column names
    file_columns = [col for col in data.columns if col.startswith('bw_')]
    
    # Find target column
    target_col = None
    for col in file_columns:
        if target_var.replace('.bw', '').replace('.bigwig', '') in col:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"Target variable '{target_var}' not found in data")
    
    # Build design matrix
    X_cols = []
    X_names = []
    
    for term in predictor_terms:
        if '*' in term:
            # Interaction term
            factors = [f.strip() for f in term.split('*')]
            factor_cols = []
            for factor in factors:
                for col in file_columns:
                    if factor.replace('.bw', '').replace('.bigwig', '') in col:
                        factor_cols.append(col)
                        break
            
            if len(factor_cols) == len(factors):
                # Create interaction column
                interaction_data = data[factor_cols[0]].copy()
                interaction_name = factors[0]
                for i in range(1, len(factor_cols)):
                    interaction_data *= data[factor_cols[i]]
                    interaction_name += f"*{factors[i]}"
                
                X_cols.append(interaction_data)
                X_names.append(interaction_name)
                
                # Individual terms should be explicitly specified in the formula, not auto-added
                # This prevents duplication when interaction terms are used
        else:
            # Simple term
            for col in file_columns:
                if term.strip().replace('.bw', '').replace('.bigwig', '') in col:
                    X_cols.append(data[col])
                    X_names.append(term.strip())
                    break
    
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
    
    # Perform regression
    if regression_type == 'linear':
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
            logger.warning("Singular matrix encountered - cannot compute p-values (likely due to multicollinearity)")
            p_values = np.full(len(model.coef_), np.nan)
        
        results = {
            'model': model,
            'predictions': predictions,
            'residuals': residuals,
            'r2': r2,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'p_values': p_values,
            'variable_names': X_names,
            'n_obs': n,
            'mse': mse,
            'mask': mask
        }
        
    elif regression_type == 'logistic':
        model = LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        residuals = y - predictions
        
        results = {
            'model': model,
            'predictions': predictions,
            'residuals': residuals,
            'coefficients': model.coef_[0],
            'intercept': model.intercept_[0],
            'variable_names': X_names,
            'n_obs': len(y),
            'mask': mask
        }
    
    return results


def write_output(data: pd.DataFrame, output_file: str, format_type: str, 
                chrom_sizes: Optional[Dict[str, int]] = None, span: int = 10):
    """Write data to specified output format."""
    logger = logging.getLogger()
    
    if format_type == 'bigwig':
        if chrom_sizes is None:
            raise ValueError("Chromosome sizes required for BigWig output")
        write_bigwig_output(data, output_file, chrom_sizes, span)
    elif format_type == 'csv':
        data.to_csv(output_file, index=False)
    elif format_type == 'tsv':
        data.to_csv(output_file, sep='\t', index=False)
    elif format_type == 'bed':
        # For BED format, use first 3 columns as chr, start, end
        bed_data = data[['chromosome', 'start', 'end']].copy()
        # Add additional columns as BED scores/names
        for col in data.columns:
            if col not in ['chromosome', 'start', 'end']:
                bed_data[col] = data[col]
        bed_data.to_csv(output_file, sep='\t', header=False, index=False)
    elif format_type == 'json':
        data.to_json(output_file, orient='records', indent=2)
    
    logger.info(f"Output written to {output_file}")


def write_bigwig_output(data: pd.DataFrame, output_file: str, 
                       chrom_sizes: Dict[str, int], span: int):
    """Write data to BigWig format."""
    # Assume the last column contains the values to write
    value_col = [col for col in data.columns if col not in ['chromosome', 'start', 'end']][-1]
    
    with pyBigWig.open(output_file, 'w') as bw:
        bw.addHeader(list(sorted(chrom_sizes.items())))
        
        for chrom, chrom_data in data.groupby('chromosome'):
            if chrom not in chrom_sizes:
                continue
                
            bw.addEntries(
                chrom,
                int(chrom_data['start'].min()),
                span=span,
                step=span,
                values=chrom_data[value_col].values
            )


def main():
    """Main function for bwops tool."""
    args = parse_arguments()
    
    if args.operation is None:
        print("Please specify an operation. Use --help for available operations.")
        import sys
        sys.exit(1)
    
    logger = setup_logging(args.logLevel, args.logfile)
    
    # Read chromosome sizes if needed
    chrom_sizes = None
    if args.format == 'bigwig' or any(getattr(args, attr, None) and attr.startswith('out_') 
                                     for attr in vars(args)):
        if not args.chrom_sizes:
            logger.error("Chromosome sizes file required for BigWig output")
            raise ValueError("--chrom-sizes required for BigWig output")
        chrom_sizes = read_chrom_sizes(args.chrom_sizes)
    
    # Parse region if specified
    region = None
    if args.region:
        region = parse_region(args.region)
    
    # Get input files based on operation
    if args.operation in ['add', 'multiply']:
        input_files = args.input_files
    elif args.operation == 'regress':
        # Extract file names from formula
        target_var, predictor_terms = parse_formula(args.formula)
        input_files = []
        for var in [target_var] + predictor_terms:
            # Extract filename from variable (remove interaction symbols)
            clean_var = re.sub(r'[*:].*', '', var.strip())
            if clean_var.endswith('.bw') or clean_var.endswith('.bigwig'):
                input_files.append(clean_var)
        input_files = list(set(input_files))  # Remove duplicates
    
    # Find common chromosomes with filtering
    common_chroms = get_common_chromosomes(
        input_files, 
        args.blacklisted_seqs, 
        args.exclude_contigs, 
        args.chromosome_pattern
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
        logger.info(f"Processing {len(common_chroms)} chromosomes (showing first 5): {common_chroms[:5]}...")
    
    # Determine span using native resolution to avoid interpolation
    if hasattr(args, 'span') and args.span is not None:
        span = args.span
        logger.info(f"Using user-specified span: {span} bp")
        if span < 1000:
            logger.warning("Small span values may result in slow processing for large chromosomes")
    else:
        span, needs_warning = get_native_resolution(input_files, common_chroms)
        if needs_warning:
            logger.warning("Consider using BigWig files with matching resolutions for optimal performance")
        else:
            logger.info(f"Using native resolution: {span} bp (no interpolation needed)")
        
        # Suggest larger span for performance if native resolution is very small
        if span < 50:
            total_genome_size = sum(250_000_000 for _ in common_chroms)  # Rough estimate
            estimated_intervals = total_genome_size // span
            if estimated_intervals > 10_000_000:  # 10M intervals
                logger.warning(f"Native resolution of {span}bp will create ~{estimated_intervals//1_000_000}M intervals")
                logger.warning("For faster processing, consider using --span 1000 or larger")
                logger.warning("This will require interpolation but may complete much faster")
    
    # Read data
    logger.info("Reading BigWig data...")
    data = read_bigwig_data(input_files, common_chroms, span, region)
    
    if data.empty:
        logger.error("No data read from input files")
        return
    
    logger.info(f"Read {len(data)} data points")
    
    # Perform operation
    if args.operation == 'add':
        value_cols = [col for col in data.columns if col.startswith('bw_')]
        data['result'] = data[value_cols].sum(axis=1)
        write_output(data[['chromosome', 'start', 'end', 'result']], 
                    args.out, args.format, chrom_sizes, span)
        
    elif args.operation == 'multiply':
        value_cols = [col for col in data.columns if col.startswith('bw_')]
        data['result'] = data[value_cols].prod(axis=1)
        write_output(data[['chromosome', 'start', 'end', 'result']], 
                    args.out, args.format, chrom_sizes, span)
        
    elif args.operation == 'regress':
        logger.info("Performing regression analysis...")
        results = perform_regression(data, args.formula, args.type)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("REGRESSION RESULTS")
        print("="*50)
        print(f"Formula: {args.formula}")
        print(f"Regression type: {args.type}")
        print(f"Number of observations: {results['n_obs']}")
        
        if args.type == 'linear':
            print(f"R-squared: {results['r2']:.4f}")
            print(f"MSE: {results['mse']:.4f}")
        
        print(f"\nIntercept: {results['intercept']:.4f}")
        print("\nCoefficients:")
        for name, coef in zip(results['variable_names'], results['coefficients']):
            if args.type == 'linear' and 'p_values' in results:
                pval_idx = results['variable_names'].index(name)
                print(f"  {name}: {coef:.4f} (p-value: {results['p_values'][pval_idx]:.4f})")
            else:
                print(f"  {name}: {coef:.4f}")
        
        # Write outputs if specified
        if args.out_prediction:
            pred_data = data.copy()
            pred_data['prediction'] = np.nan
            pred_data.loc[results['mask'], 'prediction'] = results['predictions']
            write_output(pred_data[['chromosome', 'start', 'end', 'prediction']], 
                        args.out_prediction, args.format, chrom_sizes, span)
        
        if args.out_residuals:
            resid_data = data.copy()
            resid_data['residuals'] = np.nan
            resid_data.loc[results['mask'], 'residuals'] = results['residuals']
            write_output(resid_data[['chromosome', 'start', 'end', 'residuals']], 
                        args.out_residuals, args.format, chrom_sizes, span)
        
        if args.out_stats:
            # Write detailed statistics
            stats_dict = {
                'formula': args.formula,
                'regression_type': args.type,
                'n_observations': results['n_obs'],
                'intercept': float(results['intercept']),
                'coefficients': {name: float(coef) for name, coef in 
                               zip(results['variable_names'], results['coefficients'])}
            }
            
            if args.type == 'linear':
                stats_dict['r_squared'] = float(results['r2'])
                stats_dict['mse'] = float(results['mse'])
                stats_dict['p_values'] = {name: float(pval) for name, pval in 
                                        zip(results['variable_names'], results['p_values'])}
            
            with open(args.out_stats, 'w') as f:
                json.dump(stats_dict, f, indent=2)
    
    logger.info("Analysis completed successfully")


if __name__ == "__main__":
    main()