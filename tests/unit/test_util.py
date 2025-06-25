"""
Unit tests for kdpeak utility functions.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from kdpeak.util import (
    read_bed, events_from_intervals, events_dict_from_file,
    read_chrom_sizes_file, get_chromosome_size_estimate,
    filter_chromosomes, sort_chromosomes_by_size,
    get_kde, make_kdes, setup_logging
)


class TestReadBed:
    """Test BED file reading functionality."""
    
    def test_read_simple_bed(self, small_bed_file):
        """Test reading a simple 3-column BED file."""
        df = read_bed(str(small_bed_file))
        
        assert len(df) == 10
        assert list(df.columns) == ['seqname', 'start', 'end']
        assert df['seqname'].iloc[0] == 'chr1'
        assert df['start'].dtype == 'int64'
        assert df['end'].dtype == 'int64'
    
    def test_read_multi_column_bed(self, multi_column_bed_file):
        """Test reading BED file with extra columns (should ignore them)."""
        df = read_bed(str(multi_column_bed_file))
        
        assert len(df) == 8
        assert list(df.columns) == ['seqname', 'start', 'end']
        assert df['seqname'].iloc[0] == 'chr1'
        assert df['start'].iloc[0] == 1000
        assert df['end'].iloc[0] == 1500
    
    def test_read_bed_with_mixed_types(self, temp_dir):
        """Test reading BED file with mixed data types."""
        bed_file = temp_dir / "mixed_types.bed"
        
        # Create BED file with some string coordinates that should be converted
        with open(bed_file, 'w') as f:
            f.write("chr1\t1000\t1500\n")
            f.write("chr2\t2000\t2500\n")
        
        df = read_bed(str(bed_file))
        assert len(df) == 2
        assert df['start'].dtype == 'int64'
        assert df['end'].dtype == 'int64'
    
    def test_read_empty_bed(self, temp_dir):
        """Test reading empty BED file."""
        bed_file = temp_dir / "empty.bed"
        bed_file.touch()
        
        df = read_bed(str(bed_file))
        assert len(df) == 0
        assert list(df.columns) == ['seqname', 'start', 'end']


class TestEventsProcessing:
    """Test event processing functions."""
    
    def test_events_from_intervals(self):
        """Test conversion from intervals to events."""
        intervals = pd.DataFrame({
            'seqname': ['chr1', 'chr1'],
            'start': [1000, 2000],
            'end': [1500, 2500]
        })
        
        events = events_from_intervals(intervals)
        
        assert len(events) == 4  # 2 intervals × 2 events each
        assert 'seqname' in events.columns
        assert 'location' in events.columns
        assert 'variable' in events.columns
        assert set(events['variable']) == {'start', 'end'}
        assert set(events['location']) == {1000, 1500, 2000, 2500}
    
    def test_events_dict_from_file(self, small_bed_file):
        """Test creating events dictionary from BED file."""
        events_dict = events_dict_from_file(str(small_bed_file))
        
        assert isinstance(events_dict, dict)
        assert 'chr1' in events_dict
        assert 'chr2' in events_dict
        assert 'chrX' in events_dict
        
        # Check chr1 events
        chr1_events = events_dict['chr1']
        assert len(chr1_events) == 10  # 5 intervals × 2 events each
        assert 'location' in chr1_events.columns


class TestChromosomeHandling:
    """Test chromosome size and filtering functions."""
    
    def test_read_chrom_sizes_file(self, chrom_sizes_file):
        """Test reading chromosome sizes from file."""
        sizes = read_chrom_sizes_file(str(chrom_sizes_file))
        
        assert isinstance(sizes, dict)
        assert sizes['chr1'] == 200000
        assert sizes['chr2'] == 150000
        assert sizes['chrX'] == 100000
    
    def test_read_chrom_sizes_invalid_file(self):
        """Test reading from non-existent file."""
        sizes = read_chrom_sizes_file('/nonexistent/file.txt')
        assert sizes == {}
    
    def test_get_chromosome_size_estimate(self):
        """Test chromosome size estimation from data."""
        events_df = pd.DataFrame({
            'location': [1000, 5000, 10000, 25000]
        })
        
        size = get_chromosome_size_estimate(events_df)
        assert size == 25000
    
    def test_get_chromosome_size_estimate_empty(self):
        """Test size estimation with empty DataFrame."""
        empty_df = pd.DataFrame()
        size = get_chromosome_size_estimate(empty_df)
        assert size == 0
    
    def test_filter_chromosomes_exclude_contigs(self):
        """Test contig exclusion filtering."""
        chromosomes = ['chr1', 'chr2', 'chrX', 'chr1_random', 'chrUn_scaffold123', 'chrM']
        
        filtered = filter_chromosomes(chromosomes, exclude_contigs=True)
        
        assert 'chr1' in filtered
        assert 'chr2' in filtered
        assert 'chrX' in filtered
        assert 'chr1_random' not in filtered
        assert 'chrUn_scaffold123' not in filtered
        assert 'chrM' not in filtered
    
    def test_filter_chromosomes_pattern(self):
        """Test regex pattern filtering."""
        chromosomes = ['chr1', 'chr2', 'chrX', 'chrY', 'chr22', 'chr1_random']
        
        # Pattern for main human chromosomes (exact matches)
        pattern = r'^chr[1-9XY]$|^chr[12][0-9]$|^chr2[0-2]$'
        filtered = filter_chromosomes(chromosomes, chromosome_pattern=pattern)
        
        assert 'chr1' in filtered
        assert 'chr2' in filtered
        assert 'chrX' in filtered
        assert 'chrY' in filtered
        assert 'chr22' in filtered
        assert 'chr1_random' not in filtered
    
    def test_filter_chromosomes_both_filters(self):
        """Test applying both exclusion and pattern filters."""
        chromosomes = ['chr1', 'chr2', 'chrX', 'chr1_random', 'chrUn_scaffold123']
        
        filtered = filter_chromosomes(
            chromosomes, 
            exclude_contigs=True, 
            chromosome_pattern=r'chr[1-9XY]+'
        )
        
        assert filtered == ['chr1', 'chr2', 'chrX']
    
    def test_filter_chromosomes_invalid_pattern(self):
        """Test invalid regex pattern handling."""
        chromosomes = ['chr1', 'chr2']
        
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            filter_chromosomes(chromosomes, chromosome_pattern='[invalid')


class TestSortChromosomes:
    """Test chromosome sorting functionality."""
    
    def test_sort_chromosomes_by_size_with_file(self, sample_events_dict, chrom_sizes_file):
        """Test sorting using actual chromosome sizes."""
        sorted_chroms = sort_chromosomes_by_size(
            sample_events_dict,
            chrom_sizes_file=str(chrom_sizes_file)
        )
        
        # Should be sorted by actual size: chr1 (200k) > chr2 (150k) > chrX (100k)
        assert sorted_chroms[0] == 'chr1'
        assert sorted_chroms[1] == 'chr2'
        assert sorted_chroms[2] == 'chrX'
    
    def test_sort_chromosomes_by_size_estimate(self, sample_events_dict):
        """Test sorting using size estimates from data."""
        sorted_chroms = sort_chromosomes_by_size(sample_events_dict)
        
        # Should be sorted by estimated size from max coordinate
        assert len(sorted_chroms) == 3
        assert 'chr1' in sorted_chroms
        assert 'chr2' in sorted_chroms
        assert 'chrX' in sorted_chroms
    
    def test_sort_chromosomes_with_filtering(self, sample_events_dict):
        """Test sorting with chromosome filtering."""
        # Add some contigs to the events dict
        sample_events_dict['chr1_random'] = pd.DataFrame({
            'location': [1000, 2000]
        })
        
        sorted_chroms = sort_chromosomes_by_size(
            sample_events_dict,
            exclude_contigs=True
        )
        
        assert 'chr1_random' not in sorted_chroms
        assert len(sorted_chroms) == 3  # Only main chromosomes


class TestKDEFunctions:
    """Test KDE-related functions."""
    
    def test_get_kde_basic(self):
        """Test basic KDE computation."""
        # Simple test data with known distribution
        cut_locations = np.array([1000, 1010, 1020, 2000, 2010])
        
        grid, density = get_kde(cut_locations, kde_bw=50)
        
        assert len(grid) == len(density)
        assert grid.min() <= cut_locations.min()
        assert grid.max() >= cut_locations.max()
        assert np.all(density >= 0)  # Density should be non-negative
        assert np.sum(density) > 0   # Should have some signal
    
    def test_make_kdes_basic(self, sample_events_dict):
        """Test KDE computation for multiple chromosomes."""
        comb_data, signal_list = make_kdes(
            sample_events_dict,
            step=100,
            kde_bw=500
        )
        
        assert isinstance(comb_data, pd.DataFrame)
        assert isinstance(signal_list, list)
        assert len(comb_data) > 0
        assert len(signal_list) == 3  # One per chromosome
        
        # Check required columns
        required_cols = ['seqname', 'interval', 'location', 'density']
        assert all(col in comb_data.columns for col in required_cols)
        
        # Check that we have data for all chromosomes
        assert set(comb_data['seqname'].unique()) == {'chr1', 'chr2', 'chrX'}
    
    def test_make_kdes_with_blacklist(self, sample_events_dict):
        """Test KDE computation with blacklisted chromosomes."""
        comb_data, signal_list = make_kdes(
            sample_events_dict,
            blacklisted=['chrX']
        )
        
        assert 'chrX' not in comb_data['seqname'].unique()
        assert len(signal_list) == 2  # chr1 and chr2 only
    
    def test_make_kdes_with_filtering(self, sample_events_dict):
        """Test KDE computation with chromosome filtering."""
        # Add a contig to test exclusion
        sample_events_dict['chr1_random'] = pd.DataFrame({
            'variable': ['start'] * 10,
            'location': np.random.randint(1000, 5000, 10)
        })
        
        comb_data, signal_list = make_kdes(
            sample_events_dict,
            exclude_contigs=True
        )
        
        assert 'chr1_random' not in comb_data['seqname'].unique()


class TestUtilityFunctions:
    """Test other utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging(log_level="DEBUG")
        
        assert logger is not None
        assert logger.level <= 10  # DEBUG level
    
    def test_setup_logging_with_file(self, temp_dir):
        """Test logging setup with file output."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(log_level="INFO", log_file=str(log_file))
        
        logger.info("Test message")
        
        assert log_file.exists()
        assert log_file.stat().st_size > 0