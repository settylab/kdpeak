# kdpeak Test Suite

This directory contains comprehensive tests for the kdpeak package.

## Test Organization

```
tests/
├── conftest.py              # pytest configuration and shared fixtures
├── data/                    # Test data files
├── unit/                    # Unit tests for individual functions
├── integration/             # Integration tests for complete workflows
├── cli/                     # Command-line interface tests
└── performance/             # Performance and stress tests
```

## Running Tests

### Quick Start
```bash
# Run all fast tests
make test-fast

# Run specific test categories
make test-unit
make test-integration
make test-cli
make test-performance
```

### Using pytest directly
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/cli/          # CLI tests only
pytest tests/performance/  # Performance tests only

# Exclude slow tests
pytest -m "not slow"

# Run with coverage
pytest --cov=kdpeak --cov-report=html
```

### Using tox (multiple Python versions)
```bash
tox                    # Run tests on all Python versions
tox -e py39           # Run tests on Python 3.9 only
tox -e flake8         # Run linting only
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **test_util.py**: Tests for utility functions (BED reading, KDE computation, etc.)
- **test_bwops.py**: Tests for BigWig operations functionality

**Coverage**: Individual functions and methods
**Runtime**: Fast (~10-30 seconds)

### Integration Tests (`tests/integration/`)
- **test_kdpeak_workflow.py**: End-to-end kdpeak pipeline tests

**Coverage**: Complete workflows from input to output
**Runtime**: Medium (~30-120 seconds)

### CLI Tests (`tests/cli/`)
- **test_command_line.py**: Command-line interface and argument parsing

**Coverage**: Command-line interfaces, error handling, parameter validation
**Runtime**: Medium (~30-60 seconds)

### Performance Tests (`tests/performance/`)
- **test_large_datasets.py**: Performance, memory usage, and stress tests

**Coverage**: Large datasets, edge cases, resource usage
**Runtime**: Slow (~60-300 seconds)

## Test Data

Test data is located in `tests/data/`:
- `test_small.bed`: Small BED file for basic testing
- `test_multi_column.bed`: BED file with extra columns
- `test_chrom_sizes.txt`: Chromosome sizes file

## Test Markers

Tests are marked with categories:
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.cli`: CLI tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.bigwig`: Tests requiring BigWig files

## Continuous Integration

Tests run automatically on:
- **GitHub Actions**: On push/PR to main branch
- **Multiple platforms**: Ubuntu and macOS
- **Multiple Python versions**: 3.8, 3.9, 3.10, 3.11

## Writing New Tests

### Test Structure
```python
class TestFeatureName:
    """Test specific feature or functionality."""
    
    def test_basic_functionality(self):
        """Test basic case."""
        # Arrange
        input_data = create_test_data()
        
        # Act  
        result = function_under_test(input_data)
        
        # Assert
        assert result is not None
        assert len(result) > 0
    
    def test_edge_case(self):
        """Test edge case or error condition."""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

### Using Fixtures
```python
def test_with_fixtures(self, small_bed_file, temp_dir):
    """Test using shared fixtures."""
    output_file = temp_dir / "output.bed"
    result = process_bed_file(small_bed_file, output_file)
    assert output_file.exists()
```

### Performance Tests
```python
@pytest.mark.slow
def test_large_dataset_performance(self):
    """Test performance with large dataset."""
    start_time = time.time()
    result = process_large_dataset()
    elapsed = time.time() - start_time
    
    assert elapsed < 60  # Should complete in under 1 minute
    assert len(result) > 0
```

## Coverage Goals

- **Unit tests**: >90% code coverage
- **Integration tests**: All major workflows
- **CLI tests**: All command-line options and error cases
- **Performance tests**: Memory usage and timing bounds

## Debugging Tests

```bash
# Run tests with verbose output
pytest -v -s

# Run specific test
pytest tests/unit/test_util.py::TestReadBed::test_read_simple_bed -v

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l
```

## Test Dependencies

Required for testing:
- pytest>=6.0
- pytest-cov>=2.0 (for coverage)
- psutil>=5.0 (for memory monitoring)
- numpy, pandas, scipy (core dependencies)

Install test dependencies:
```bash
pip install -e ".[test]"    # Test dependencies only
pip install -e ".[dev]"     # All development dependencies
```