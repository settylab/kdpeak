#!/usr/bin/env python
"""
Test suite for error handling functionality in kdpeak and bwops.

Tests comprehensive error handling including:
- Utility error functions
- File validation
- User-friendly error messages
- Proper exit codes
- Debug information preservation
"""

import pytest
import tempfile
import os
import sys
import logging
from unittest.mock import patch, MagicMock
from io import StringIO

# Import the modules we're testing
from kdpeak.util import (
    handle_error, 
    validate_file_exists, 
    validate_output_directory,
    safe_file_operation
)


class TestUtilityErrorFunctions:
    """Test utility error handling functions."""
    
    def test_handle_error_basic(self, capfd):
        """Test basic error handling with simple exception."""
        error = FileNotFoundError("Test file not found")
        
        with patch('logging.getLogger') as mock_logger_getter:
            mock_logger = MagicMock()
            mock_logger.getEffectiveLevel.return_value = logging.INFO
            mock_logger_getter.return_value = mock_logger
            
            handle_error(error, "Test operation failed")
            
            # Check stderr output
            captured = capfd.readouterr()
            assert "ERROR: Test operation failed" in captured.err
            assert "FileNotFoundError" in captured.err
            assert "Test file not found" in captured.err
            assert "For technical details, run with --log DEBUG" in captured.err
            
            # Check logger calls
            mock_logger.error.assert_called_once()
            mock_logger.debug.assert_called_once()

    def test_handle_error_with_suggestions(self, capfd):
        """Test error handling with suggestions."""
        error = PermissionError("Permission denied")
        suggestions = [
            "Check file permissions",
            "Run with sudo if needed",
            "Verify file ownership"
        ]
        
        with patch('logging.getLogger') as mock_logger_getter:
            mock_logger = MagicMock()
            mock_logger.getEffectiveLevel.return_value = logging.INFO
            mock_logger_getter.return_value = mock_logger
            
            handle_error(error, "Cannot write to file", suggestions)
            
            captured = capfd.readouterr()
            assert "Suggested solutions:" in captured.err
            assert "1. Check file permissions" in captured.err
            assert "2. Run with sudo if needed" in captured.err
            assert "3. Verify file ownership" in captured.err

    def test_handle_error_debug_level(self, capfd):
        """Test error handling at debug level (no debug hint)."""
        error = ValueError("Invalid value")
        
        with patch('logging.getLogger') as mock_logger_getter:
            mock_logger = MagicMock()
            mock_logger.getEffectiveLevel.return_value = logging.DEBUG
            mock_logger_getter.return_value = mock_logger
            
            handle_error(error, "Validation failed")
            
            captured = capfd.readouterr()
            # At debug level, should not show the debug hint
            assert "For technical details, run with --log DEBUG" not in captured.err

    def test_validate_file_exists_success(self):
        """Test successful file validation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            # Should not raise any exception
            validate_file_exists(tmp_path, "test file")
        finally:
            os.unlink(tmp_path)

    def test_validate_file_exists_not_found(self):
        """Test file validation with non-existent file."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            validate_file_exists("/nonexistent/file.txt", "test file")

    def test_validate_file_exists_permission_denied(self):
        """Test file validation with permission issues."""
        # Create a file and make it unreadable
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            # Remove read permissions
            os.chmod(tmp_path, 0o000)
            
            with pytest.raises(PermissionError):
                validate_file_exists(tmp_path, "test file")
        finally:
            # Restore permissions and cleanup
            os.chmod(tmp_path, 0o644)
            os.unlink(tmp_path)

    def test_validate_output_directory_success(self):
        """Test successful output directory validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = os.path.join(tmp_dir, "output.txt")
            # Should not raise any exception
            validate_output_directory(test_file)

    def test_validate_output_directory_permission_denied(self):
        """Test output directory validation with permission issues."""
        # Test with a directory that doesn't allow writing
        if os.path.exists("/root") and not os.access("/root", os.W_OK):
            with pytest.raises(PermissionError):
                validate_output_directory("/root/test_output.txt")
        else:
            pytest.skip("Cannot test permission denied without inaccessible directory")

    def test_safe_file_operation_success(self):
        """Test successful safe file operation."""
        def dummy_operation():
            return "success"
        
        result = safe_file_operation(
            dummy_operation,
            "Test operation failed"
        )
        
        assert result == "success"

    def test_safe_file_operation_with_error(self, capfd):
        """Test safe file operation with exception."""
        def failing_operation():
            raise IOError("Disk full")
        
        with patch('logging.getLogger') as mock_logger_getter:
            mock_logger = MagicMock()
            mock_logger.getEffectiveLevel.return_value = logging.INFO
            mock_logger_getter.return_value = mock_logger
            
            result = safe_file_operation(
                failing_operation,
                "File operation failed",
                ["Check disk space", "Try different location"]
            )
            
            assert result is None
            captured = capfd.readouterr()
            assert "File operation failed" in captured.err
            assert "Check disk space" in captured.err


class TestKdpeakErrorHandling:
    """Test kdpeak-specific error handling scenarios."""
    
    def test_main_with_nonexistent_bed_file(self):
        """Test kdpeak main function with non-existent BED file."""
        from kdpeak.core import main
        
        # Mock sys.argv to simulate command line
        test_args = ["kdpeak", "nonexistent.bed", "--out", "test_output.bed"]
        
        with patch('sys.argv', test_args), \
             patch('kdpeak.core.parse_arguments') as mock_parse:
            
            # Create mock arguments
            mock_args = MagicMock()
            mock_args.bed_file = "nonexistent.bed"
            mock_args.out = "test_output.bed"
            mock_args.chrom_sizes = None
            mock_args.summits_out = None
            mock_args.density_out = None
            mock_args.logLevel = "INFO"
            mock_args.logfile = None
            mock_parse.return_value = mock_args
            
            # Should return 1 (error exit code)
            exit_code = main()
            assert exit_code == 1

    def test_main_with_invalid_chrom_sizes(self):
        """Test kdpeak main function with invalid chromosome sizes file."""
        from kdpeak.core import main
        
        # Create a temporary BED file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as bed_file:
            bed_file.write("chr1\t100\t200\n")
            bed_file.write("chr1\t300\t400\n")
            bed_path = bed_file.name
        
        try:
            test_args = ["kdpeak", bed_path, "--out", "test_output.bed", 
                        "--chrom-sizes", "nonexistent_chrom_sizes.txt"]
            
            with patch('sys.argv', test_args), \
                 patch('kdpeak.core.parse_arguments') as mock_parse:
                
                mock_args = MagicMock()
                mock_args.bed_file = bed_path
                mock_args.out = "test_output.bed"
                mock_args.chrom_sizes = "nonexistent_chrom_sizes.txt"
                mock_args.summits_out = None
                mock_args.density_out = None
                mock_args.logLevel = "INFO"
                mock_args.logfile = None
                mock_parse.return_value = mock_args
                
                exit_code = main()
                assert exit_code == 1
        finally:
            os.unlink(bed_path)

    def test_main_keyboard_interrupt(self):
        """Test kdpeak main function with keyboard interrupt."""
        from kdpeak.core import main
        
        with patch('kdpeak.core.parse_arguments') as mock_parse:
            mock_parse.side_effect = KeyboardInterrupt()
            
            exit_code = main()
            assert exit_code == 130  # Standard exit code for SIGINT


class TestBwopsErrorHandling:  
    """Test bwops-specific error handling scenarios."""
    
    def test_main_no_operation(self):
        """Test bwops main function with no operation specified."""
        from kdpeak.bwops import main
        
        with patch('kdpeak.bwops.parse_arguments') as mock_parse:
            mock_args = MagicMock()
            mock_args.operation = None
            mock_parse.return_value = mock_args
            
            exit_code = main()
            assert exit_code == 1

    def test_main_with_nonexistent_bigwig(self):
        """Test bwops main function with non-existent BigWig file."""
        from kdpeak.bwops import main
        
        with patch('kdpeak.bwops.parse_arguments') as mock_parse:
            mock_args = MagicMock()
            mock_args.operation = 'add'
            mock_args.input_files = ['nonexistent.bw']
            mock_args.out = 'output.bw'
            mock_args.format = 'bigwig'
            mock_args.chrom_sizes = None
            mock_args.logLevel = 'INFO'
            mock_args.logfile = None
            mock_parse.return_value = mock_args
            
            exit_code = main()
            assert exit_code == 1

    def test_main_missing_chrom_sizes_for_bigwig_output(self):
        """Test bwops main function missing chromosome sizes for BigWig output."""
        from kdpeak.bwops import main
        
        # Create a temporary BigWig-like file (just for file existence check)
        with tempfile.NamedTemporaryFile(suffix='.bw', delete=False) as bw_file:
            bw_file.write(b"fake bigwig content")
            bw_path = bw_file.name
        
        try:
            with patch('kdpeak.bwops.parse_arguments') as mock_parse:
                mock_args = MagicMock()
                mock_args.operation = 'add'
                mock_args.input_files = [bw_path]
                mock_args.out = 'output.bw'
                mock_args.format = 'bigwig'
                mock_args.chrom_sizes = None  # Missing chrom sizes
                mock_args.logLevel = 'INFO'
                mock_args.logfile = None
                mock_parse.return_value = mock_args
                
                # Mock the validate_file_exists to pass for our fake BigWig
                with patch('kdpeak.bwops.validate_file_exists'):
                    exit_code = main()
                    assert exit_code == 1
        finally:
            os.unlink(bw_path)

    def test_main_keyboard_interrupt(self):
        """Test bwops main function with keyboard interrupt."""
        from kdpeak.bwops import main
        
        with patch('kdpeak.bwops.parse_arguments') as mock_parse:
            mock_parse.side_effect = KeyboardInterrupt()
            
            exit_code = main()
            assert exit_code == 130


class TestErrorMessageFormat:
    """Test error message formatting and content."""
    
    def test_error_message_contains_required_elements(self, capfd):
        """Test that error messages contain all required elements."""
        error = FileNotFoundError("Test file missing")
        suggestions = ["Check the path", "Verify permissions"]
        
        with patch('logging.getLogger') as mock_logger_getter:
            mock_logger = MagicMock()
            mock_logger.getEffectiveLevel.return_value = logging.INFO
            mock_logger_getter.return_value = mock_logger
            
            handle_error(error, "File operation failed", suggestions)
            
            captured = capfd.readouterr()
            
            # Check all required elements are present
            assert "ERROR:" in captured.err
            assert "File operation failed" in captured.err
            assert "FileNotFoundError" in captured.err
            assert "Test file missing" in captured.err
            assert "Suggested solutions:" in captured.err
            assert "1. Check the path" in captured.err
            assert "2. Verify permissions" in captured.err
            assert "For technical details, run with --log DEBUG" in captured.err

    def test_error_message_without_suggestions(self, capfd):
        """Test error message format without suggestions."""
        error = ValueError("Invalid input")
        
        with patch('logging.getLogger') as mock_logger_getter:
            mock_logger = MagicMock()
            mock_logger.getEffectiveLevel.return_value = logging.INFO
            mock_logger_getter.return_value = mock_logger
            
            handle_error(error, "Validation failed")
            
            captured = capfd.readouterr()
            
            # Should not contain suggestions section
            assert "Suggested solutions:" not in captured.err
            assert "ERROR: Validation failed" in captured.err
            assert "Invalid input" in captured.err


class TestIntegrationErrorHandling:
    """Integration tests for error handling across the system."""
    
    def test_end_to_end_kdpeak_error(self, capfd):
        """Test end-to-end error handling in kdpeak."""
        from kdpeak.core import main
        
        # Test with completely invalid arguments
        with patch('sys.argv', ['kdpeak']):  # Missing required argument
            with pytest.raises(SystemExit):  # argparse will cause SystemExit
                from kdpeak.core import parse_arguments
                parse_arguments()

    def test_end_to_end_bwops_error(self, capfd):
        """Test end-to-end error handling in bwops."""
        from kdpeak.bwops import main
        
        # Test with no operation specified - should return exit code 1
        with patch('kdpeak.bwops.parse_arguments') as mock_parse:
            mock_args = MagicMock()
            mock_args.operation = None
            mock_parse.return_value = mock_args
            
            exit_code = main()
            assert exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__])