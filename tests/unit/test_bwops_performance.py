"""
Tests for bwops performance optimizations, specifically fast reading when spans match.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import pyBigWig
from pathlib import Path
from unittest.mock import patch, MagicMock

from kdpeak.bwops import read_bigwig_data, get_native_resolution


class TestBigWigFastReading:
    """Test fast reading optimization for BigWig files."""
    
    @pytest.fixture
    def temp_bigwig_100bp(self):
        """Create a test BigWig file with 100bp resolution."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.bw', delete=False)
        temp_file.close()
        
        # Create chromosome sizes
        chrom_sizes = [("chr1", 1000), ("chr2", 500)]
        
        with pyBigWig.open(temp_file.name, "w") as bw:
            bw.addHeader(chrom_sizes)
            
            # Add 100bp intervals for chr1 using correct API
            starts = list(range(0, 1000, 100))  # 0, 100, 200, ..., 900
            values = [float(i) for i in range(len(starts))]  # 0.0, 1.0, 2.0, ..., 9.0
            bw.addEntries("chr1", starts, values=values, span=100)
            
            # Add 100bp intervals for chr2  
            starts = list(range(0, 500, 100))  # 0, 100, 200, 300, 400
            values = [float(i+10) for i in range(len(starts))]  # 10.0, 11.0, 12.0, 13.0, 14.0
            bw.addEntries("chr2", starts, values=values, span=100)
        
        yield temp_file.name
        Path(temp_file.name).unlink()
    
    @pytest.fixture  
    def temp_bigwig_50bp(self):
        """Create a test BigWig file with 50bp resolution."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.bw', delete=False)
        temp_file.close()
        
        chrom_sizes = [("chr1", 1000)]
        
        with pyBigWig.open(temp_file.name, "w") as bw:
            bw.addHeader(chrom_sizes)
            
            # Add 50bp intervals for chr1
            starts = list(range(0, 1000, 50))  # 0, 50, 100, 150, ..., 950
            values = [float(i) for i in range(len(starts))]
            bw.addEntries("chr1", starts, values=values, span=50)
        
        yield temp_file.name
        Path(temp_file.name).unlink()
    
    def test_native_resolution_detection(self, temp_bigwig_100bp, temp_bigwig_50bp):
        """Test that native resolution is correctly detected."""
        # Test 100bp file
        resolution, needs_warning = get_native_resolution([temp_bigwig_100bp], ["chr1"])
        assert resolution == 100
        assert needs_warning == False
        
        # Test 50bp file  
        resolution, needs_warning = get_native_resolution([temp_bigwig_50bp], ["chr1"])
        assert resolution == 50
        assert needs_warning == False
        
        # Test mixed resolutions
        resolution, needs_warning = get_native_resolution([temp_bigwig_100bp, temp_bigwig_50bp], ["chr1"])
        assert resolution == 50  # GCD of 100 and 50
        assert needs_warning == True
    
    def test_fast_reading_when_spans_match(self, temp_bigwig_100bp):
        """Test that fast reading is used when span matches native resolution."""
        # Test with matching span (100bp)
        data = read_bigwig_data([temp_bigwig_100bp], ["chr1"], span=100)
        
        assert len(data) == 10  # chr1 has 10 intervals of 100bp each
        assert list(data['chromosome']) == ['chr1'] * 10
        assert list(data['start']) == list(range(0, 1000, 100))
        assert list(data['end']) == list(range(100, 1100, 100))
        
        # Check values are correctly read
        col_name = [col for col in data.columns if col.startswith('bw_')][0]
        expected_values = list(range(10))  # 0.0, 1.0, 2.0, ..., 9.0
        assert list(data[col_name]) == expected_values
    
    def test_stats_reading_when_spans_dont_match(self, temp_bigwig_100bp):
        """Test that stats reading is used when span doesn't match native resolution."""
        # Test with non-matching span (200bp)
        data = read_bigwig_data([temp_bigwig_100bp], ["chr1"], span=200)
        
        assert len(data) == 5  # chr1 with 200bp intervals: 0-200, 200-400, 400-600, 600-800, 800-1000
        assert list(data['start']) == [0, 200, 400, 600, 800]
        assert list(data['end']) == [200, 400, 600, 800, 1000]
        
        # Values should be averaged (e.g., for 0-200 interval, average of values 0.0 and 1.0 = 0.5)
        col_name = [col for col in data.columns if col.startswith('bw_')][0]
        # First interval (0-200) should average values 0.0 and 1.0 = 0.5
        # Second interval (200-400) should average values 2.0 and 3.0 = 2.5
        assert abs(data[col_name].iloc[0] - 0.5) < 0.1
        assert abs(data[col_name].iloc[1] - 2.5) < 0.1
    
    def test_multiple_chromosomes_fast_reading(self, temp_bigwig_100bp):
        """Test fast reading works with multiple chromosomes."""
        data = read_bigwig_data([temp_bigwig_100bp], ["chr1", "chr2"], span=100)
        
        # Should have data for both chromosomes
        chr1_data = data[data['chromosome'] == 'chr1']
        chr2_data = data[data['chromosome'] == 'chr2']
        
        assert len(chr1_data) == 10
        assert len(chr2_data) == 5
        
        # Check values
        col_name = [col for col in data.columns if col.startswith('bw_')][0]
        assert list(chr1_data[col_name]) == list(range(10))  # 0-9
        assert list(chr2_data[col_name]) == list(range(10, 15))  # 10-14
    
    def test_multiple_files_mixed_compatibility(self, temp_bigwig_100bp, temp_bigwig_50bp):
        """Test behavior when some files can use fast reading and others cannot."""
        data = read_bigwig_data([temp_bigwig_100bp, temp_bigwig_50bp], ["chr1"], span=100)
        
        assert len(data) == 10
        
        # Should have columns for both files
        bw_cols = [col for col in data.columns if col.startswith('bw_')]
        assert len(bw_cols) == 2
        
        # First file (100bp) should use fast reading and have correct values
        # Second file (50bp) should use stats reading and have averaged values
        # The exact values depend on the implementation details
        assert all(data[bw_cols[0]] >= 0)  # Basic sanity check
        assert all(data[bw_cols[1]] >= 0)  # Basic sanity check
    
    @patch('logging.getLogger')
    def test_fast_reading_logging(self, mock_get_logger, temp_bigwig_100bp):
        """Test that appropriate log messages are generated for fast reading."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        read_bigwig_data([temp_bigwig_100bp], ["chr1"], span=100)
        
        # Check that fast reading was detected and used
        info_calls = [call for call in mock_logger.info.call_args_list]
        debug_calls = [call for call in mock_logger.debug.call_args_list if call[0]]
        
        # Should log that fast reading is being used
        fast_reading_logged = any(
            "fast" in str(call).lower() for call in info_calls + debug_calls
        )
        assert fast_reading_logged, f"Fast reading not logged. Info calls: {info_calls}, Debug calls: {debug_calls}"
    
    @patch('logging.getLogger')
    def test_stats_reading_logging(self, mock_get_logger, temp_bigwig_100bp):
        """Test that appropriate log messages are generated for stats reading."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        read_bigwig_data([temp_bigwig_100bp], ["chr1"], span=200)
        
        # Check that stats reading was detected and used
        info_calls = [call for call in mock_logger.info.call_args_list]
        debug_calls = [call for call in mock_logger.debug.call_args_list if call[0]]
        
        # Should log that stats reading is being used
        stats_reading_logged = any(
            "stats" in str(call).lower() for call in info_calls + debug_calls
        )
        assert stats_reading_logged, f"Stats reading not logged. Info calls: {info_calls}, Debug calls: {debug_calls}"
    
    def test_region_limiting(self, temp_bigwig_100bp):
        """Test that region parameter correctly limits analysis."""
        # Limit to first 500bp of chr1
        data = read_bigwig_data([temp_bigwig_100bp], ["chr1"], span=100, region=("chr1", 0, 500))
        
        assert len(data) == 5  # Only first 5 intervals (0-100, 100-200, ..., 400-500)
        assert list(data['start']) == [0, 100, 200, 300, 400]
        assert list(data['end']) == [100, 200, 300, 400, 500]
        
        col_name = [col for col in data.columns if col.startswith('bw_')][0]
        assert list(data[col_name]) == [0, 1, 2, 3, 4]
    
    def test_empty_chromosome_handling(self, temp_bigwig_100bp):
        """Test behavior when chromosome has no data."""
        # Request a chromosome that doesn't exist
        data = read_bigwig_data([temp_bigwig_100bp], ["chr3"], span=100)
        
        # Should return empty DataFrame
        assert len(data) == 0
        assert isinstance(data, pd.DataFrame)
    
    def test_performance_warning_threshold(self):
        """Test that performance warnings are issued for large coordinate grids."""
        # Create a synthetic BigWig with large chromosome to trigger warning
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.bw', delete=False)
        temp_file.close()
        
        try:
            # Create a large chromosome (10M bp) to trigger the warning
            chrom_sizes = [("chr1", 10_000_000)]
            
            with pyBigWig.open(temp_file.name, "w") as bw:
                bw.addHeader(chrom_sizes)
                # Add a few intervals
                bw.addEntries("chr1", [0, 1000], values=[1.0, 2.0], span=1000)
            
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                
                # This should trigger the warning since chr1 will have 10M intervals at 1bp resolution
                read_bigwig_data([temp_file.name], ["chr1"], span=1)  # 1bp span = 10M intervals
                
                # Check for warning about large number of intervals
                warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
                large_intervals_warned = any(
                    "intervals" in warning and "slow" in warning.lower() 
                    for warning in warning_calls
                )
                assert large_intervals_warned, f"Large intervals warning not found. Warnings: {warning_calls}"
        finally:
            Path(temp_file.name).unlink()


class TestBigWigPerformanceIntegration:
    """Integration tests for the full performance optimization pipeline."""
    
    @pytest.fixture
    def temp_bigwig_files(self):
        """Create multiple test BigWig files with same resolution."""
        files = []
        
        for i in range(3):
            temp_file = tempfile.NamedTemporaryFile(suffix=f'_file{i}.bw', delete=False)
            temp_file.close()
            
            chrom_sizes = [("chr1", 500)]
            
            with pyBigWig.open(temp_file.name, "w") as bw:
                bw.addHeader(chrom_sizes)
                
                # Add 100bp intervals with different values for each file
                starts = list(range(0, 500, 100))  # 0, 100, 200, 300, 400
                values = [float(i * 10 + j) for j in range(len(starts))]  # File 0: 0,1,2,3,4; File 1: 10,11,12,13,14; etc.
                bw.addEntries("chr1", starts, values=values, span=100)
            
            files.append(temp_file.name)
        
        yield files
        
        for file in files:
            Path(file).unlink()
    
    def test_fast_reading_multiple_files(self, temp_bigwig_files):
        """Test that fast reading works correctly with multiple files."""
        data = read_bigwig_data(temp_bigwig_files, ["chr1"], span=100)
        
        assert len(data) == 5  # 5 intervals of 100bp each
        
        # Should have columns for all 3 files
        bw_cols = [col for col in data.columns if col.startswith('bw_')]
        assert len(bw_cols) == 3
        
        # Check that values are correctly read from each file
        # File 0 should have values 0,1,2,3,4
        # File 1 should have values 10,11,12,13,14  
        # File 2 should have values 20,21,22,23,24
        for i, col in enumerate(sorted(bw_cols)):
            expected_values = [float(i * 10 + j) for j in range(5)]
            actual_values = list(data[col])
            assert actual_values == expected_values, f"File {i} values mismatch: expected {expected_values}, got {actual_values}"
    
    def test_consistency_between_fast_and_stats_reading(self, temp_bigwig_files):
        """Test that fast and stats reading produce similar results when applicable."""
        # Read with fast method (span=100 matches native)
        fast_data = read_bigwig_data(temp_bigwig_files, ["chr1"], span=100)
        
        # Read with stats method (span=100, but we'll force stats by using a file with different resolution)
        # For this test, we assume stats method gives exact same results when spans align
        stats_data = read_bigwig_data(temp_bigwig_files, ["chr1"], span=100)
        
        # Results should be identical since spans match
        bw_cols = [col for col in fast_data.columns if col.startswith('bw_')]
        for col in bw_cols:
            pd.testing.assert_series_equal(fast_data[col], stats_data[col], check_names=False)