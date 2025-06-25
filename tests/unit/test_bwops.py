"""
Unit tests for bwops (BigWig operations) functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path

from kdpeak.bwops import (
    parse_formula, perform_regression, read_chrom_sizes,
    get_common_chromosomes, write_output, parse_arguments
)
from kdpeak.util import filter_chromosomes


class TestFormulaHandling:
    """Test R-style formula parsing."""
    
    def test_parse_simple_formula(self):
        """Test parsing simple linear formula."""
        formula = "target.bw ~ predictor1.bw + predictor2.bw"
        target, predictors = parse_formula(formula)
        
        assert target == "target.bw"
        assert predictors == ["predictor1.bw", "predictor2.bw"]
    
    def test_parse_interaction_formula(self):
        """Test parsing formula with interaction terms."""
        formula = "y.bw ~ x1.bw + x2.bw + x1.bw*x2.bw"
        target, predictors = parse_formula(formula)
        
        assert target == "y.bw"
        assert "x1.bw*x2.bw" in predictors
        assert "x1.bw" in predictors
        assert "x2.bw" in predictors
    
    def test_parse_formula_no_tilde(self):
        """Test parsing formula without tilde separator."""
        with pytest.raises(ValueError, match="Formula must contain '~'"):
            parse_formula("invalid formula")
    
    def test_parse_formula_whitespace(self):
        """Test parsing formula with extra whitespace."""
        formula = "  target.bw  ~  predictor1.bw  +  predictor2.bw  "
        target, predictors = parse_formula(formula)
        
        assert target == "target.bw"
        assert predictors == ["predictor1.bw", "predictor2.bw"]


class TestRegressionAnalysis:
    """Test regression functionality."""
    
    @pytest.fixture
    def regression_data(self):
        """Create sample data for regression testing."""
        np.random.seed(42)
        n = 1000
        
        # Create correlated predictors with known relationships
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # True relationship: y = 2*x1 + 1.5*x2 + 0.5*x1*x2 + noise
        noise = np.random.normal(0, 0.1, n)
        y = 2*x1 + 1.5*x2 + 0.5*x1*x2 + noise
        
        data = pd.DataFrame({
            'chromosome': ['chr1'] * n,
            'start': range(0, n*100, 100),
            'end': range(100, (n+1)*100, 100),
            'bw_0_predictor1': x1,
            'bw_1_predictor2': x2,
            'bw_2_target': y
        })
        
        return data
    
    def test_linear_regression_simple(self, regression_data):
        """Test simple linear regression."""
        formula = "target.bw ~ predictor1.bw + predictor2.bw"
        
        results = perform_regression(regression_data, formula, 'linear')
        
        assert 'coefficients' in results
        assert 'intercept' in results
        assert 'predictions' in results
        assert 'residuals' in results
        assert 'r2' in results
        assert 'p_values' in results
        
        # Check that coefficients are approximately correct
        coeffs = results['coefficients']
        assert len(coeffs) == 2  # Two predictors
        assert abs(coeffs[0] - 2.0) < 0.1  # Should be close to 2
        assert abs(coeffs[1] - 1.5) < 0.1  # Should be close to 1.5
        
        # R² should be high for this synthetic data
        assert results['r2'] > 0.8
    
    def test_linear_regression_interaction(self, regression_data):
        """Test linear regression with interaction terms."""
        formula = "target.bw ~ predictor1.bw + predictor2.bw + predictor1.bw*predictor2.bw"
        
        results = perform_regression(regression_data, formula, 'linear')
        
        # Should have 3 coefficients: x1, x2, and x1*x2
        assert len(results['coefficients']) == 3
        assert len(results['variable_names']) == 3
        
        # R² should be very high with interaction term
        assert results['r2'] > 0.95
    
    def test_logistic_regression(self):
        """Test logistic regression."""
        np.random.seed(42)
        n = 500
        
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # Binary outcome with logistic relationship
        logit = 0.5*x1 + 0.3*x2
        prob = 1 / (1 + np.exp(-logit))
        y = np.random.binomial(1, prob)
        
        data = pd.DataFrame({
            'chromosome': ['chr1'] * n,
            'start': range(0, n*100, 100),
            'end': range(100, (n+1)*100, 100),
            'bw_0_predictor1': x1,
            'bw_1_predictor2': x2,
            'bw_2_target': y
        })
        
        formula = "target.bw ~ predictor1.bw + predictor2.bw"
        results = perform_regression(data, formula, 'logistic')
        
        assert 'coefficients' in results
        assert 'predictions' in results
        assert len(results['coefficients']) == 2
    
    def test_regression_with_missing_data(self, regression_data):
        """Test regression handling of missing data."""
        # Introduce some NaN values
        regression_data.loc[0:10, 'bw_0_predictor1'] = np.nan
        regression_data.loc[20:25, 'bw_2_target'] = np.nan
        
        formula = "target.bw ~ predictor1.bw + predictor2.bw"
        results = perform_regression(regression_data, formula, 'linear')
        
        # Should still work, just with fewer observations
        assert results['n_obs'] < len(regression_data)
        assert results['n_obs'] > 900  # Most data should still be valid
    
    def test_regression_invalid_formula(self, regression_data):
        """Test regression with invalid formula."""
        formula = "nonexistent.bw ~ predictor1.bw"
        
        with pytest.raises(ValueError, match="Target variable.*not found"):
            perform_regression(regression_data, formula, 'linear')


class TestDataHandling:
    """Test data handling functions for bwops."""
    
    def test_read_chrom_sizes(self, chrom_sizes_file):
        """Test reading chromosome sizes."""
        sizes = read_chrom_sizes(str(chrom_sizes_file))
        
        assert isinstance(sizes, dict)
        assert sizes['chr1'] == 200000
        assert sizes['chr2'] == 150000
        assert len(sizes) == 5
    
    def test_get_common_chromosomes(self, temp_dir):
        """Test finding common chromosomes across BigWig files."""
        # This would require actual BigWig files, so we'll mock this
        # In a real implementation, you'd create test BigWig files
        pytest.skip("Requires actual BigWig files - implement with test BigWig creation")
    
    def test_write_output_csv(self, temp_dir, mock_bigwig_data):
        """Test writing output to CSV format."""
        output_file = temp_dir / "test_output.csv"
        
        write_output(mock_bigwig_data, str(output_file), 'csv')
        
        assert output_file.exists()
        
        # Read back and verify
        df = pd.read_csv(output_file)
        assert len(df) == len(mock_bigwig_data)
        assert 'chromosome' in df.columns
    
    def test_write_output_json(self, temp_dir, mock_bigwig_data):
        """Test writing output to JSON format."""
        output_file = temp_dir / "test_output.json"
        
        write_output(mock_bigwig_data, str(output_file), 'json')
        
        assert output_file.exists()
        
        # Read back and verify
        with open(output_file) as f:
            data = json.load(f)
        
        assert isinstance(data, list)
        assert len(data) == len(mock_bigwig_data)
    
    def test_write_output_bed(self, temp_dir, mock_bigwig_data):
        """Test writing output to BED format."""
        output_file = temp_dir / "test_output.bed"
        
        write_output(mock_bigwig_data, str(output_file), 'bed')
        
        assert output_file.exists()
        
        # Read back and verify basic BED format
        with open(output_file) as f:
            lines = f.readlines()
        
        assert len(lines) == len(mock_bigwig_data)
        
        # Check first line has correct format
        first_line = lines[0].strip().split('\t')
        assert first_line[0] in ['chr1', 'chr2']  # chromosome
        assert first_line[1].isdigit()  # start
        assert first_line[2].isdigit()  # end


class TestCommandLineInterface:
    """Test command-line argument parsing."""
    
    def test_parse_add_arguments(self):
        """Test parsing add operation arguments."""
        args = parse_arguments(['add', 'file1.bw', 'file2.bw', '--out', 'sum.bw'])
        
        assert args.operation == 'add'
        assert args.input_files == ['file1.bw', 'file2.bw']
        assert args.out == 'sum.bw'
    
    def test_parse_multiply_arguments(self):
        """Test parsing multiply operation arguments."""
        args = parse_arguments(['multiply', 'file1.bw', 'file2.bw', '--out', 'product.bw', '--format', 'csv'])
        
        assert args.operation == 'multiply'
        assert args.input_files == ['file1.bw', 'file2.bw']
        assert args.out == 'product.bw'
        assert args.format == 'csv'
    
    def test_parse_regress_arguments(self):
        """Test parsing regression operation arguments."""
        args = parse_arguments([
            'regress',
            '--formula', 'target.bw ~ pred1.bw + pred2.bw',
            '--out-prediction', 'pred.bw',
            '--out-residuals', 'resid.bw',
            '--type', 'linear'
        ])
        
        assert args.operation == 'regress'
        assert args.formula == 'target.bw ~ pred1.bw + pred2.bw'
        assert args.out_prediction == 'pred.bw'
        assert args.out_residuals == 'resid.bw'
        assert args.type == 'linear'
    
    def test_parse_region_arguments(self):
        """Test parsing region-specific arguments."""
        args = parse_arguments([
            'add', 'file1.bw', 'file2.bw',
            '--out', 'sum.bw',
            '--region', 'chr1:1000-2000',
            '--chromosomes', 'chr1', 'chr2'
        ])
        
        assert args.region == 'chr1:1000-2000'
        assert args.chromosomes == ['chr1', 'chr2']
    
    def test_parse_filtering_arguments(self):
        """Test parsing chromosome filtering arguments."""
        args = parse_arguments([
            'regress',
            '--formula', 'target.bw ~ pred.bw',
            '--blacklisted-seqs', 'chrM', 'chrY',
            '--exclude-contigs',
            '--chromosome-pattern', 'chr[0-9]+$'
        ])
        
        assert args.blacklisted_seqs == ['chrM', 'chrY']
        assert args.exclude_contigs is True
        assert args.chromosome_pattern == 'chr[0-9]+$'
        assert args.span is None  # Default should be None for auto-detection


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data_regression(self):
        """Test regression with empty data."""
        empty_data = pd.DataFrame(columns=['chromosome', 'start', 'end'])
        
        with pytest.raises(ValueError):
            perform_regression(empty_data, "y.bw ~ x.bw", 'linear')
    
    def test_single_predictor_regression(self):
        """Test regression with single predictor."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        y = 2*x + np.random.normal(0, 0.1, n)
        
        data = pd.DataFrame({
            'chromosome': ['chr1'] * n,
            'start': range(0, n*100, 100),
            'end': range(100, (n+1)*100, 100),
            'bw_0_predictor': x,
            'bw_1_target': y
        })
        
        formula = "target.bw ~ predictor.bw"
        results = perform_regression(data, formula, 'linear')
        
        assert len(results['coefficients']) == 1
        assert abs(results['coefficients'][0] - 2.0) < 0.2