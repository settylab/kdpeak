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


def parse_arguments():
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
        subparser.add_argument('--span', type=int, default=10, 
                             help='Resolution for analysis in base pairs (default: 10)')
        subparser.add_argument('-l', '--log', dest='logLevel',
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                             default='INFO', help='Set logging level')
        subparser.add_argument('--logfile', help='Write log to file')
    
    return parser.parse_args()


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


def get_common_chromosomes(bw_files: List[str]) -> List[str]:
    """Find chromosomes common to all BigWig files."""
    common_chroms = None
    
    for bw_file in bw_files:
        with pyBigWig.open(bw_file) as bw:
            chroms = set(bw.chroms().keys())
            if common_chroms is None:
                common_chroms = chroms
            else:
                common_chroms = common_chroms.intersection(chroms)
    
    return sorted(list(common_chroms))


def get_common_span(bw_files: List[str], chromosomes: List[str]) -> int:
    """Determine common span (GCD of all spans) across BigWig files."""
    import math
    
    spans = []
    for bw_file in bw_files:
        with pyBigWig.open(bw_file) as bw:
            # Sample a few intervals to estimate span
            for chrom in chromosomes[:3]:  # Check first few chromosomes
                if chrom in bw.chroms():
                    intervals = bw.intervals(chrom, 0, min(10000, bw.chroms()[chrom]))
                    if intervals:
                        for start, end, _ in intervals[:10]:  # Check first 10 intervals
                            spans.append(end - start)
                        break
    
    if not spans:
        return 10  # Default span
    
    # Find GCD of all spans
    result = spans[0]
    for span in spans[1:]:
        result = math.gcd(result, span)
    
    return max(result, 1)


def read_bigwig_data(bw_files: List[str], chromosomes: List[str], 
                    span: int, region: Optional[Tuple[str, int, int]] = None) -> pd.DataFrame:
    """Read and align data from multiple BigWig files."""
    logger = logging.getLogger()
    
    all_data = []
    
    target_chroms = chromosomes
    if region:
        target_chroms = [region[0]] if region[0] in chromosomes else []
    
    for chrom in target_chroms:
        logger.info(f"Processing chromosome {chrom}")
        
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
        
        # Create coordinate grid
        coords = np.arange(chrom_start, chrom_end, span)
        
        # Read data from each BigWig file
        chrom_data = {'chromosome': chrom, 'start': coords, 'end': coords + span}
        
        for i, bw_file in enumerate(bw_files):
            col_name = f"bw_{i}_{os.path.basename(bw_file).replace('.bw', '').replace('.bigwig', '')}"
            
            with pyBigWig.open(bw_file) as bw:
                if chrom not in bw.chroms():
                    logger.warning(f"Chromosome {chrom} not found in {bw_file}")
                    chrom_data[col_name] = np.zeros(len(coords))
                    continue
                
                # Get values for each coordinate
                values = []
                for coord in coords:
                    end_coord = min(coord + span, bw.chroms()[chrom])
                    try:
                        val = bw.stats(chrom, coord, end_coord, type="mean")[0]
                        values.append(val if val is not None else 0.0)
                    except:
                        values.append(0.0)
                
                chrom_data[col_name] = values
        
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
                
                # Also add individual terms if not already present
                for i, factor_col in enumerate(factor_cols):
                    if factor_col not in X_names:
                        X_cols.append(data[factor_col])
                        X_names.append(factors[i])
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
        var_coef = mse * np.diag(np.linalg.inv(X.T @ X))
        se_coef = np.sqrt(var_coef)
        t_stats = model.coef_ / se_coef
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
        
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
        return
    
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
    
    # Find common chromosomes
    common_chroms = get_common_chromosomes(input_files)
    if len(common_chroms) != sum(len(pyBigWig.open(f).chroms()) for f in input_files) / len(input_files):
        logger.warning("Not all chromosomes are present in all input files")
    
    # Filter chromosomes if specified
    if args.chromosomes:
        common_chroms = [c for c in common_chroms if c in args.chromosomes]
    
    if not common_chroms:
        logger.error("No common chromosomes found")
        return
    
    logger.info(f"Processing {len(common_chroms)} chromosomes: {common_chroms}")
    
    # Determine span
    if hasattr(args, 'span') and args.span:
        span = args.span
    else:
        span = get_common_span(input_files, common_chroms)
    
    logger.info(f"Using span: {span}")
    
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