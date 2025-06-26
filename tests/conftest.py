"""
pytest configuration and shared fixtures for kdpeak tests.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def small_bed_file(test_data_dir):
    """Path to small test BED file."""
    return test_data_dir / "test_small.bed"


@pytest.fixture
def multi_column_bed_file(test_data_dir):
    """Path to multi-column BED file."""
    return test_data_dir / "test_multi_column.bed"


@pytest.fixture
def chrom_sizes_file(test_data_dir):
    """Path to chromosome sizes file."""
    return test_data_dir / "test_chrom_sizes.txt"


@pytest.fixture
def sample_events_dict():
    """Sample events dictionary for testing."""
    np.random.seed(42)  # For reproducible tests

    # Create synthetic genomic events
    events = {}

    # chr1: Large chromosome with many events
    chr1_locations = np.concatenate(
        [
            np.random.normal(10000, 1000, 50),  # Peak around 10kb
            np.random.normal(50000, 1500, 30),  # Peak around 50kb
            np.random.uniform(1000, 100000, 20),  # Background noise
        ]
    )
    chr1_locations = np.clip(chr1_locations, 0, 200000).astype(int)
    events["chr1"] = pd.DataFrame(
        {"variable": ["start"] * len(chr1_locations), "location": chr1_locations}
    )

    # chr2: Medium chromosome
    chr2_locations = np.concatenate(
        [
            np.random.normal(25000, 800, 30),  # Single peak
            np.random.uniform(5000, 80000, 15),  # Background
        ]
    )
    chr2_locations = np.clip(chr2_locations, 0, 150000).astype(int)
    events["chr2"] = pd.DataFrame(
        {"variable": ["start"] * len(chr2_locations), "location": chr2_locations}
    )

    # chrX: Smaller chromosome
    chrX_locations = np.random.normal(15000, 500, 20)
    chrX_locations = np.clip(chrX_locations, 0, 100000).astype(int)
    events["chrX"] = pd.DataFrame(
        {"variable": ["start"] * len(chrX_locations), "location": chrX_locations}
    )

    return events


@pytest.fixture
def sample_chrom_sizes():
    """Sample chromosome sizes for testing."""
    return {
        "chr1": 200000,
        "chr2": 150000,
        "chrX": 100000,
        "chr1_random": 5000,
        "chrUn_scaffold123": 2000,
    }


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Set up logging for tests."""
    import logging

    logging.getLogger("kdpeak").setLevel(logging.WARNING)  # Reduce noise in tests


@pytest.fixture
def mock_bigwig_data():
    """Mock BigWig-like data for testing bwops."""
    np.random.seed(42)

    data = []
    for chrom in ["chr1", "chr2"]:
        for i in range(0, 10000, 100):  # 100bp resolution
            data.append(
                {
                    "chromosome": chrom,
                    "start": i,
                    "end": i + 100,
                    "density1": np.random.exponential(2),
                    "density2": np.random.exponential(1.5),
                    "density3": np.random.normal(5, 1),
                }
            )

    return pd.DataFrame(data)
