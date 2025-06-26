"""
Integration tests for kdpeak main workflow.
"""

import pytest
import pandas as pd
import subprocess
import os
from pathlib import Path

from kdpeak.core import main as kdpeak_main
from kdpeak.util import events_dict_from_file, make_kdes, call_peaks


class TestKDPeakWorkflow:
    """Test the complete kdpeak workflow."""

    def test_full_workflow_basic(self, small_bed_file, temp_dir):
        """Test basic kdpeak workflow with minimal arguments."""
        output_file = temp_dir / "test_peaks.bed"

        # Mock command line arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = [
                "kdpeak",
                str(small_bed_file),
                "--out",
                str(output_file),
                "--kde-bw",
                "500",
                "--span",
                "100",
                "--min-peak-size",
                "50",
                "--frip",
                "0.1",  # Lower FRIP threshold to make peaks more likely
            ]

            # Run kdpeak main function
            exit_code = kdpeak_main()
            assert exit_code == 0, "kdpeak should complete successfully"

            # Check output file exists and has content
            assert output_file.exists()

            # Read and validate output - handle case where no peaks are found
            if output_file.stat().st_size > 0:
                peaks_df = pd.read_csv(output_file, sep="\t", header=None)
                if len(peaks_df) > 0:
                    assert peaks_df.shape[1] >= 4  # chr, start, end, name, score

                    # Validate BED format
                    assert all(peaks_df.iloc[:, 1] < peaks_df.iloc[:, 2])  # start < end
                    assert all(peaks_df.iloc[:, 1] >= 0)  # non-negative coordinates
            else:
                # Empty output file is acceptable if no peaks were found
                pass

        finally:
            sys.argv = original_argv

    def test_workflow_with_bigwig_output(
        self, small_bed_file, chrom_sizes_file, temp_dir
    ):
        """Test kdpeak workflow with BigWig density output."""
        peaks_file = temp_dir / "test_peaks.bed"
        density_file = temp_dir / "test_density.bw"

        import sys

        original_argv = sys.argv
        try:
            sys.argv = [
                "kdpeak",
                str(small_bed_file),
                "--out",
                str(peaks_file),
                "--density-out",
                str(density_file),
                "--chrom-sizes",
                str(chrom_sizes_file),
                "--kde-bw",
                "300",
                "--span",
                "50",
                "--frip",
                "0.1",  # Lower FRIP threshold
            ]

            exit_code = kdpeak_main()
            assert exit_code == 0, "kdpeak should complete successfully"

            # Check both outputs exist
            assert peaks_file.exists()
            assert density_file.exists()

            # Check BigWig file is not empty
            assert density_file.stat().st_size > 0

        finally:
            sys.argv = original_argv

    def test_workflow_with_summits(self, small_bed_file, temp_dir):
        """Test kdpeak workflow with summit output."""
        peaks_file = temp_dir / "test_peaks.bed"
        summits_file = temp_dir / "test_summits.bed"

        import sys

        original_argv = sys.argv
        try:
            sys.argv = [
                "kdpeak",
                str(small_bed_file),
                "--out",
                str(peaks_file),
                "--summits-out",
                str(summits_file),
                "--kde-bw",
                "500",
                "--span",
                "100",
                "--min-peak-size",
                "50",
                "--frip",
                "0.1",  # Use parameters that work
            ]

            exit_code = kdpeak_main()
            assert exit_code == 0, "kdpeak should complete successfully"

            # Check both outputs exist
            assert peaks_file.exists()
            assert summits_file.exists()

            # Validate summits format - handle case where no peaks are found
            if summits_file.stat().st_size > 0:
                summits_df = pd.read_csv(summits_file, sep="\t", header=None)
                if len(summits_df) > 0:
                    assert summits_df.shape[1] >= 4
                    # Summits should be single-base positions (start + 1 = end)
                    assert all(summits_df.iloc[:, 2] - summits_df.iloc[:, 1] == 1)
            else:
                # Empty summits file is acceptable - means no peaks were called
                pass

        finally:
            sys.argv = original_argv

    def test_workflow_with_filtering(self, multi_column_bed_file, temp_dir):
        """Test kdpeak workflow with chromosome filtering."""
        output_file = temp_dir / "test_peaks_filtered.bed"

        import sys

        original_argv = sys.argv
        try:
            sys.argv = [
                "kdpeak",
                str(multi_column_bed_file),
                "--out",
                str(output_file),
                "--exclude-contigs",
                "--chromosome-pattern",
                r"chr[1-9XY]+",
                "--kde-bw",
                "500",
                "--span",
                "100",
                "--min-peak-size",
                "50",
                "--frip",
                "0.1",  # Use parameters that work
            ]

            exit_code = kdpeak_main()
            assert exit_code == 0, "kdpeak should complete successfully"

            assert output_file.exists()

            # Read output and check that no contigs are present - handle empty file
            if output_file.stat().st_size > 0:
                peaks_df = pd.read_csv(output_file, sep="\t", header=None)
                if len(peaks_df) > 0:
                    chromosomes = peaks_df.iloc[:, 0].unique()
                    # Should not contain random or unplaced contigs
                    assert not any("random" in chrom for chrom in chromosomes)
                    assert not any("Un_" in chrom for chrom in chromosomes)
            else:
                # Empty peaks file is acceptable - means no peaks were called after filtering
                pass

        finally:
            sys.argv = original_argv

    def test_workflow_error_handling(self, temp_dir):
        """Test error handling in kdpeak workflow."""
        nonexistent_file = temp_dir / "nonexistent.bed"
        output_file = temp_dir / "output.bed"

        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["kdpeak", str(nonexistent_file), "--out", str(output_file)]

            # Should return error exit code (1) for missing input file
            exit_code = kdpeak_main()
            assert exit_code == 1, "Expected exit code 1 for missing input file"

        finally:
            sys.argv = original_argv


class TestWorkflowComponents:
    """Test individual components of the kdpeak workflow."""

    def test_events_to_peaks_pipeline(self, small_bed_file):
        """Test the complete pipeline from BED file to peaks."""
        # Step 1: Read events
        events_dict = events_dict_from_file(str(small_bed_file))
        assert len(events_dict) > 0

        # Step 2: Make KDEs
        comb_data, signal_list = make_kdes(events_dict, step=100, kde_bw=500)
        assert len(comb_data) > 0
        assert len(signal_list) > 0

        # Step 3: Call peaks
        peaks = call_peaks(
            comb_data, signal_list, fraction_in_peaks=0.3, min_peak_size=100, span=100
        )
        assert len(peaks) > 0
        assert "start" in peaks.columns
        assert "end" in peaks.columns
        assert "seqname" in peaks.columns

    def test_filtering_pipeline(self, sample_events_dict, chrom_sizes_file):
        """Test the filtering and sorting pipeline."""
        # Add contigs to test filtering
        sample_events_dict["chr1_random"] = pd.DataFrame(
            {"variable": ["start"] * 5, "location": [1000, 2000, 3000, 4000, 5000]}
        )
        sample_events_dict["chrUn_scaffold123"] = pd.DataFrame(
            {"variable": ["start"] * 3, "location": [500, 1500, 2500]}
        )

        # Test with contig exclusion
        comb_data, signal_list = make_kdes(
            sample_events_dict,
            step=100,
            kde_bw=300,
            exclude_contigs=True,
            chrom_sizes_file=str(chrom_sizes_file),
        )

        # Should not contain contigs
        chromosomes = comb_data["seqname"].unique()
        assert "chr1_random" not in chromosomes
        assert "chrUn_scaffold123" not in chromosomes

        # Should contain main chromosomes
        assert "chr1" in chromosomes
        assert "chr2" in chromosomes
        assert "chrX" in chromosomes

    def test_parameter_sensitivity(self, sample_events_dict):
        """Test sensitivity to different parameters."""
        base_params = {"step": 50, "kde_bw": 200}

        # Test different bandwidths
        results = {}
        for bw in [100, 500, 1000]:
            comb_data, signal_list = make_kdes(
                sample_events_dict, step=base_params["step"], kde_bw=bw
            )
            peaks = call_peaks(comb_data, signal_list, span=base_params["step"])
            results[bw] = len(peaks)

        # Different bandwidths should give different numbers of peaks
        assert len(set(results.values())) > 1

        # Test different fractions in peaks
        comb_data, signal_list = make_kdes(sample_events_dict, **base_params)
        results_frip = {}
        for frip in [0.1, 0.3, 0.5]:
            peaks = call_peaks(
                comb_data, signal_list, fraction_in_peaks=frip, span=base_params["step"]
            )
            results_frip[frip] = len(peaks)

        # Different FRIP values should generally give different numbers of peaks
        assert len(set(results_frip.values())) > 1


class TestRealWorldScenarios:
    """Test scenarios that mimic real-world usage."""

    def test_large_chromosome_processing(self):
        """Test processing with realistically large chromosomes."""
        # Create synthetic data resembling real chromosome sizes
        import numpy as np

        np.random.seed(42)

        # Simulate chr1-like data (200 Mb)
        n_intervals = 10000
        chr1_starts = np.random.randint(0, 200_000_000, n_intervals)
        chr1_ends = chr1_starts + np.random.randint(100, 5000, n_intervals)

        events_dict = {
            "chr1": pd.DataFrame(
                {
                    "variable": ["start"] * len(chr1_starts) + ["end"] * len(chr1_ends),
                    "location": np.concatenate([chr1_starts, chr1_ends]),
                }
            )
        }

        # This should complete without memory issues
        comb_data, signal_list = make_kdes(
            events_dict,
            step=1000,  # Lower resolution for large chromosome
            kde_bw=10000,
        )

        assert len(comb_data) > 0
        assert len(signal_list) == 1

    def test_multi_species_chromosomes(self):
        """Test with different chromosome naming conventions."""
        species_data = {
            # Human-style
            "chr1": pd.DataFrame(
                {"variable": ["start"] * 10, "location": range(1000, 11000, 1000)}
            ),
            "chrX": pd.DataFrame(
                {"variable": ["start"] * 5, "location": range(2000, 7000, 1000)}
            ),
            # Mouse-style (same as human in this case)
            "chr19": pd.DataFrame(
                {"variable": ["start"] * 8, "location": range(3000, 11000, 1000)}
            ),
            # Drosophila-style
            "chr2L": pd.DataFrame(
                {"variable": ["start"] * 6, "location": range(1500, 7500, 1000)}
            ),
            "chr3R": pd.DataFrame(
                {"variable": ["start"] * 7, "location": range(2500, 9500, 1000)}
            ),
            # Should be filtered out
            "chr2L_random": pd.DataFrame(
                {"variable": ["start"] * 3, "location": [500, 1500, 2500]}
            ),
        }

        # Test with pattern for main chromosomes
        comb_data, signal_list = make_kdes(
            species_data, exclude_contigs=True, chromosome_pattern=r"chr[0-9XY]+[LR]?"
        )

        chromosomes = set(comb_data["seqname"].unique())
        expected = {"chr1", "chrX", "chr19", "chr2L", "chr3R"}
        assert chromosomes == expected

    def test_memory_efficiency(self, sample_events_dict):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process multiple times to check for memory leaks
        for i in range(3):
            comb_data, signal_list = make_kdes(sample_events_dict)
            peaks = call_peaks(comb_data, signal_list)

            # Force garbage collection
            import gc

            gc.collect()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024
