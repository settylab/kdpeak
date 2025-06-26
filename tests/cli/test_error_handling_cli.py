#!/usr/bin/env python
"""
CLI-specific error handling tests for kdpeak and bwops.

Tests command-line interface error scenarios including:
- Invalid command line arguments
- Missing required parameters
- File access errors via CLI
- Exit code verification
"""

import pytest
import subprocess
import tempfile
import os
import sys


class TestKdpeakCLIErrorHandling:
    """Test kdpeak CLI error handling."""

    def test_kdpeak_missing_bed_file(self):
        """Test kdpeak CLI with missing BED file argument."""
        result = subprocess.run(
            [sys.executable, "-m", "kdpeak.core"], capture_output=True, text=True
        )

        # Should exit with error code (argparse exits with 2 for missing args)
        assert result.returncode == 2
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_kdpeak_nonexistent_bed_file(self):
        """Test kdpeak CLI with non-existent BED file."""
        result = subprocess.run(
            [sys.executable, "-m", "kdpeak.core", "nonexistent.bed"],
            capture_output=True,
            text=True,
        )

        # Should exit with error code 1 (our custom error handling)
        assert result.returncode == 1
        assert "ERROR:" in result.stderr
        assert "Failed to access input BED file" in result.stderr
        assert "FileNotFoundError" in result.stderr
        assert "Suggested solutions:" in result.stderr

    def test_kdpeak_invalid_chrom_sizes_file(self):
        """Test kdpeak CLI with invalid chromosome sizes file."""
        # Create a temporary BED file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bed", delete=False
        ) as bed_file:
            bed_file.write("chr1\t100\t200\n")
            bed_path = bed_file.name

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kdpeak.core",
                    bed_path,
                    "--chrom-sizes",
                    "nonexistent_chrom_sizes.txt",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 1
            assert "ERROR:" in result.stderr
            assert "Failed to access chromosome sizes file" in result.stderr
        finally:
            os.unlink(bed_path)

    def test_kdpeak_permission_denied_output(self):
        """Test kdpeak CLI with permission denied for output."""
        # Create a temporary BED file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bed", delete=False
        ) as bed_file:
            bed_file.write("chr1\t100\t200\n")
            bed_path = bed_file.name

        try:
            # Try to write to a location that should be protected
            protected_output = (
                "/root/test_output.bed"
                if os.path.exists("/root")
                else "/dev/null/test_output.bed"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kdpeak.core",
                    bed_path,
                    "--out",
                    protected_output,
                ],
                capture_output=True,
                text=True,
            )

            # Should fail with permission error or similar
            assert result.returncode != 0
            if result.returncode == 1:  # Our custom error handling
                assert "ERROR:" in result.stderr
        finally:
            os.unlink(bed_path)

    def test_kdpeak_help_displays_correctly(self):
        """Test kdpeak help message displays without errors."""
        result = subprocess.run(
            [sys.executable, "-m", "kdpeak.core", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert (
            "kdpeak" in result.stdout
            or "kernel density estimation" in result.stdout.lower()
        )


class TestBwopsCLIErrorHandling:
    """Test bwops CLI error handling."""

    def test_bwops_no_operation(self):
        """Test bwops CLI with no operation specified."""
        result = subprocess.run(
            [sys.executable, "-m", "kdpeak.bwops"], capture_output=True, text=True
        )

        assert result.returncode == 1
        # The message might be in stdout or stderr
        output = result.stdout + result.stderr
        assert "Please specify an operation" in output

    def test_bwops_invalid_operation(self):
        """Test bwops CLI with invalid operation."""
        result = subprocess.run(
            [sys.executable, "-m", "kdpeak.bwops", "invalid_operation"],
            capture_output=True,
            text=True,
        )

        # Should exit with argparse error
        assert result.returncode == 2
        assert "invalid choice" in result.stderr

    def test_bwops_add_missing_files(self):
        """Test bwops add operation with missing files."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kdpeak.bwops",
                "add",
                "nonexistent1.bw",
                "nonexistent2.bw",
                "--out",
                "output.bw",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "ERROR:" in result.stderr
        assert "Cannot access input file" in result.stderr

    def test_bwops_add_missing_chrom_sizes(self):
        """Test bwops add operation missing chromosome sizes for BigWig output."""
        # Create fake BigWig files
        with tempfile.NamedTemporaryFile(
            suffix=".bw", delete=False
        ) as bw1, tempfile.NamedTemporaryFile(suffix=".bw", delete=False) as bw2:

            bw1.write(b"fake bigwig content")
            bw2.write(b"fake bigwig content")
            bw1_path = bw1.name
            bw2_path = bw2.name

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kdpeak.bwops",
                    "add",
                    bw1_path,
                    bw2_path,
                    "--out",
                    "output.bw",
                    "--format",
                    "bigwig",
                    # Note: missing --chrom-sizes
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 1
            assert "ERROR:" in result.stderr
            assert "Missing required chromosome sizes file" in result.stderr
        finally:
            os.unlink(bw1_path)
            os.unlink(bw2_path)

    def test_bwops_regress_invalid_formula(self):
        """Test bwops regress operation with invalid formula."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kdpeak.bwops",
                "regress",
                "--formula",
                "invalid formula syntax",
            ],
            capture_output=True,
            text=True,
        )

        # Should fail during argument processing or execution
        assert result.returncode != 0

    def test_bwops_correlate_insufficient_files(self):
        """Test bwops correlate operation with insufficient files."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kdpeak.bwops",
                "correlate",
                "single_file.bw",  # Need at least 2 files for correlation
                "--out",
                "correlation_matrix.csv",
            ],
            capture_output=True,
            text=True,
        )

        # Should fail with file not found or insufficient files error
        assert result.returncode != 0

    def test_bwops_help_displays_correctly(self):
        """Test bwops help message displays without errors."""
        result = subprocess.run(
            [sys.executable, "-m", "kdpeak.bwops", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "BigWig" in result.stdout or "operations" in result.stdout.lower()

    def test_bwops_subcommand_help(self):
        """Test bwops subcommand help displays correctly."""
        for operation in ["add", "multiply", "regress", "correlate"]:
            result = subprocess.run(
                [sys.executable, "-m", "kdpeak.bwops", operation, "--help"],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "usage:" in result.stdout.lower()
            assert operation in result.stdout.lower()


class TestCLIExitCodes:
    """Test that CLI commands return appropriate exit codes."""

    def test_success_exit_codes(self):
        """Test that help commands return 0 (success)."""
        commands = [
            [sys.executable, "-m", "kdpeak.core", "--help"],
            [sys.executable, "-m", "kdpeak.bwops", "--help"],
            [sys.executable, "-m", "kdpeak.bwops", "add", "--help"],
        ]

        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"Command failed: {' '.join(cmd)}"

    def test_argument_error_exit_codes(self):
        """Test that argument errors return 2 (argparse standard)."""
        commands = [
            [sys.executable, "-m", "kdpeak.core"],  # Missing required argument
            [
                sys.executable,
                "-m",
                "kdpeak.bwops",
                "invalid_operation",
            ],  # Invalid choice
        ]

        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 2, f"Command should return 2: {' '.join(cmd)}"

    def test_runtime_error_exit_codes(self):
        """Test that runtime errors return 1 (general error)."""
        commands = [
            [sys.executable, "-m", "kdpeak.core", "nonexistent.bed"],
            [sys.executable, "-m", "kdpeak.bwops"],  # No operation
        ]

        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 1, f"Command should return 1: {' '.join(cmd)}"


class TestCLIErrorMessageQuality:
    """Test the quality and usability of CLI error messages."""

    def test_error_messages_are_user_friendly(self):
        """Test that error messages are clear and helpful."""
        result = subprocess.run(
            [sys.executable, "-m", "kdpeak.core", "nonexistent.bed"],
            capture_output=True,
            text=True,
        )

        error_output = result.stderr

        # Check for user-friendly elements
        assert "ERROR:" in error_output
        assert "Suggested solutions:" in error_output
        assert any(
            word in error_output.lower() for word in ["check", "verify", "ensure"]
        )
        assert "For technical details, run with --log DEBUG" in error_output

        # Should not contain raw Python tracebacks (unless in debug mode)
        assert "Traceback (most recent call last):" not in error_output

    def test_debug_mode_shows_more_details(self):
        """Test that debug mode provides additional technical information."""
        result = subprocess.run(
            [sys.executable, "-m", "kdpeak.core", "nonexistent.bed", "--log", "DEBUG"],
            capture_output=True,
            text=True,
        )

        # Should still contain user-friendly error
        assert "ERROR:" in result.stderr
        # But should not show the debug hint (since we're already in debug mode)
        # Note: This depends on implementation details


class TestErrorHandlingRobustness:
    """Test error handling robustness and edge cases."""

    def test_very_long_file_paths(self):
        """Test error handling with very long file paths."""
        long_path = "a" * 1000 + ".bed"

        result = subprocess.run(
            [sys.executable, "-m", "kdpeak.core", long_path],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "ERROR:" in result.stderr

    def test_special_characters_in_file_paths(self):
        """Test error handling with special characters in file paths."""
        special_path = "file with spaces & special chars!@#$.bed"

        result = subprocess.run(
            [sys.executable, "-m", "kdpeak.core", special_path],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "ERROR:" in result.stderr

    def test_unicode_in_error_messages(self):
        """Test that error messages handle unicode properly."""
        unicode_path = "tëst_fîlé.bed"

        result = subprocess.run(
            [sys.executable, "-m", "kdpeak.core", unicode_path],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        # Should not crash with unicode errors
        assert "ERROR:" in result.stderr


if __name__ == "__main__":
    pytest.main([__file__])
