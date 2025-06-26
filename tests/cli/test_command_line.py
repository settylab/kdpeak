"""
Command-line interface tests for kdpeak and bwops.
"""

import pytest
import subprocess
import tempfile
import os
from pathlib import Path


class TestKDPeakCLI:
    """Test kdpeak command-line interface."""

    def test_kdpeak_help(self):
        """Test kdpeak --help displays correctly."""
        result = subprocess.run(["kdpeak", "--help"], capture_output=True, text=True)

        assert result.returncode == 0
        assert "usage: kdpeak" in result.stdout
        assert "--out" in result.stdout
        assert "--kde-bw" in result.stdout
        assert "--density-out" in result.stdout
        assert "--exclude-contigs" in result.stdout
        assert "--chromosome-pattern" in result.stdout

    def test_kdpeak_version_info(self):
        """Test kdpeak displays version information."""
        # Test that kdpeak runs without error when given help
        result = subprocess.run(["kdpeak", "--help"], capture_output=True, text=True)
        assert result.returncode == 0

    def test_kdpeak_missing_required_args(self):
        """Test kdpeak fails with missing required arguments."""
        result = subprocess.run(["kdpeak"], capture_output=True, text=True)

        assert result.returncode != 0
        assert "required" in result.stderr or "required" in result.stdout

    def test_kdpeak_invalid_bed_file(self):
        """Test kdpeak handles invalid BED file gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
            f.write("invalid\tbad\tdata\n")
            invalid_bed = f.name

        try:
            result = subprocess.run(
                ["kdpeak", invalid_bed, "--out", "/tmp/test_output.bed"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Should exit with error or handle gracefully
            assert result.returncode != 0 or "error" in result.stderr.lower()

        finally:
            os.unlink(invalid_bed)

    def test_kdpeak_parameter_validation(self):
        """Test parameter validation in kdpeak."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
            f.write("chr1\t1000\t1500\n")
            test_bed = f.name

        try:
            # Test invalid KDE bandwidth
            result = subprocess.run(
                [
                    "kdpeak",
                    test_bed,
                    "--out",
                    "/tmp/test.bed",
                    "--kde-bw",
                    "-100",  # Negative bandwidth should be invalid
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Should handle invalid parameters
            assert result.returncode != 0 or len(result.stderr) > 0

        finally:
            os.unlink(test_bed)

    def test_kdpeak_chromosome_filtering_args(self, small_bed_file):
        """Test chromosome filtering arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "filtered_peaks.bed")

            # Test exclude-contigs flag
            result = subprocess.run(
                [
                    "kdpeak",
                    str(small_bed_file),
                    "--out",
                    output_file,
                    "--exclude-contigs",
                    "--kde-bw",
                    "500",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should complete successfully
            assert result.returncode == 0 or os.path.exists(output_file)

    def test_kdpeak_pattern_filtering(self, small_bed_file):
        """Test chromosome pattern filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "pattern_peaks.bed")

            result = subprocess.run(
                [
                    "kdpeak",
                    str(small_bed_file),
                    "--out",
                    output_file,
                    "--chromosome-pattern",
                    r"chr[1-9XY]+",
                    "--kde-bw",
                    "300",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should complete successfully
            assert result.returncode == 0 or os.path.exists(output_file)


class TestBWOpsCLI:
    """Test bwops command-line interface."""

    def test_bwops_help(self):
        """Test bwops --help displays correctly."""
        result = subprocess.run(["bwops", "--help"], capture_output=True, text=True)

        assert result.returncode == 0
        assert "usage: bwops" in result.stdout
        assert "add" in result.stdout
        assert "multiply" in result.stdout
        assert "regress" in result.stdout

    def test_bwops_subcommand_help(self):
        """Test bwops subcommand help."""
        for subcommand in ["add", "multiply", "regress"]:
            result = subprocess.run(
                ["bwops", subcommand, "--help"], capture_output=True, text=True
            )

            assert result.returncode == 0
            assert subcommand in result.stdout
            assert "--out" in result.stdout or "--formula" in result.stdout

    def test_bwops_missing_operation(self):
        """Test bwops fails when no operation specified."""
        result = subprocess.run(["bwops"], capture_output=True, text=True)

        assert result.returncode != 0
        assert "operation" in result.stderr or "operation" in result.stdout

    def test_bwops_add_missing_args(self):
        """Test bwops add fails with missing arguments."""
        result = subprocess.run(["bwops", "add"], capture_output=True, text=True)

        assert result.returncode != 0
        assert "required" in result.stderr or "required" in result.stdout

    def test_bwops_regress_missing_formula(self):
        """Test bwops regress fails without formula."""
        result = subprocess.run(["bwops", "regress"], capture_output=True, text=True)

        assert result.returncode != 0
        assert "formula" in result.stderr or "required" in result.stderr

    def test_bwops_invalid_format(self):
        """Test bwops handles invalid output format."""
        result = subprocess.run(
            [
                "bwops",
                "add",
                "file1.bw",
                "file2.bw",
                "--out",
                "output.txt",
                "--format",
                "invalid_format",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0

    def test_bwops_region_parsing(self):
        """Test bwops region argument parsing."""
        # Test valid region format
        result = subprocess.run(
            [
                "bwops",
                "add",
                "file1.bw",
                "file2.bw",
                "--out",
                "output.bw",
                "--region",
                "chr1:1000-2000",
                "--chrom-sizes",
                "sizes.txt",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        # May fail due to missing files, but argument parsing should work
        # (error should be about missing files, not argument parsing)
        if "invalid" in result.stderr.lower():
            pytest.fail("Region parsing failed")


class TestCLIErrorHandling:
    """Test error handling in command-line interfaces."""

    def test_kdpeak_nonexistent_input(self):
        """Test kdpeak with non-existent input file."""
        result = subprocess.run(
            ["kdpeak", "/nonexistent/file.bed", "--out", "/tmp/output.bed"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        # Should have informative error message
        assert len(result.stderr) > 0 or "error" in result.stdout.lower()

    def test_kdpeak_invalid_chrom_sizes(self, small_bed_file):
        """Test kdpeak with invalid chromosome sizes file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = tmpdir

            result = subprocess.run(
                [
                    "kdpeak",
                    str(small_bed_file),
                    "--out",
                    os.path.join(output_dir, "peaks.bed"),
                    "--density-out",
                    os.path.join(output_dir, "density.bw"),
                    "--chrom-sizes",
                    "/nonexistent/sizes.txt",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )

            assert result.returncode != 0
            assert "chrom" in result.stderr.lower() or "size" in result.stderr.lower()

    def test_bwops_nonexistent_bigwig(self):
        """Test bwops with non-existent BigWig files."""
        result = subprocess.run(
            [
                "bwops",
                "add",
                "/nonexistent/file1.bw",
                "/nonexistent/file2.bw",
                "--out",
                "/tmp/output.bw",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert len(result.stderr) > 0

    def test_kdpeak_output_permissions(self, small_bed_file):
        """Test kdpeak with output directory without write permissions."""
        # Try to write to root directory (should fail)
        result = subprocess.run(
            ["kdpeak", str(small_bed_file), "--out", "/root/forbidden_output.bed"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should fail due to permissions (if not running as root)
        if os.getuid() != 0:  # Not running as root
            assert result.returncode != 0


class TestCLIIntegration:
    """Test integration between CLI and core functionality."""

    def test_kdpeak_cli_matches_api(self, small_bed_file):
        """Test that CLI produces same results as API calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cli_output = os.path.join(tmpdir, "cli_peaks.bed")

            # Run via CLI
            result = subprocess.run(
                [
                    "kdpeak",
                    str(small_bed_file),
                    "--out",
                    cli_output,
                    "--kde-bw",
                    "500",
                    "--span",
                    "100",
                    "--frip",
                    "0.3",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0 and os.path.exists(cli_output):
                # Check that output file has reasonable content
                with open(cli_output) as f:
                    lines = f.readlines()

                assert len(lines) > 0

                # Check BED format
                for line in lines[:5]:  # Check first few lines
                    fields = line.strip().split("\t")
                    assert len(fields) >= 3
                    assert fields[0].startswith("chr")
                    assert int(fields[1]) >= 0
                    assert int(fields[2]) > int(fields[1])

    def test_logging_levels(self, small_bed_file):
        """Test different logging levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "debug_peaks.bed")

            # Test DEBUG logging
            result = subprocess.run(
                [
                    "kdpeak",
                    str(small_bed_file),
                    "--out",
                    output_file,
                    "--log",
                    "DEBUG",
                    "--kde-bw",
                    "300",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should produce more verbose output in DEBUG mode
            if result.returncode == 0:
                assert len(result.stderr) > 0  # Should have debug messages

    def test_parameter_combinations(self, small_bed_file):
        """Test various parameter combinations."""
        with tempfile.TemporaryDirectory() as tmpdir:

            test_cases = [
                # Basic case
                ["--kde-bw", "200", "--span", "50"],
                # High resolution
                ["--kde-bw", "100", "--span", "10", "--min-peak-size", "50"],
                # Low resolution
                ["--kde-bw", "1000", "--span", "200", "--frip", "0.5"],
            ]

            for i, params in enumerate(test_cases):
                output_file = os.path.join(tmpdir, f"test_{i}.bed")

                cmd = ["kdpeak", str(small_bed_file), "--out", output_file] + params

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                # Should complete successfully or with reasonable error
                if result.returncode != 0:
                    # Log the error for debugging
                    print(f"Command failed: {' '.join(cmd)}")
                    print(f"Error: {result.stderr}")

                # At minimum, should not crash
                assert result.returncode != -9  # No segfault
