.PHONY: test test-unit test-integration test-cli test-performance test-all clean lint format install dev-install help

# Default target
help:
	@echo "Available targets:"
	@echo "  test-unit        Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-cli         Run CLI tests"
	@echo "  test-performance Run performance tests"
	@echo "  test-all         Run all tests"
	@echo "  test-fast        Run fast tests only (exclude slow performance tests)"
	@echo "  lint             Run code linting"
	@echo "  format           Format code with black and isort"
	@echo "  install          Install package"
	@echo "  dev-install      Install package in development mode"
	@echo "  clean            Clean build artifacts"
	@echo "  coverage         Run tests with coverage report"

# Test targets
test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cli:
	pytest tests/cli/ -v

test-performance:
	pytest tests/performance/ -v

test-fast:
	pytest tests/unit/ tests/integration/ tests/cli/ tests/performance/ -v -m "not slow"

test-all:
	pytest tests/ -v

# Test with coverage
coverage:
	pytest tests/unit/ tests/integration/ tests/cli/ -v --cov=kdpeak --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 kdpeak tests
	black --check kdpeak tests
	isort --check-only kdpeak tests

format:
	black kdpeak tests
	isort kdpeak tests

# Installation
install:
	pip install .

dev-install:
	pip install -e .
	pip install pytest pytest-cov pytest-xdist psutil flake8 black isort

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	@echo "Documentation targets would go here"
	@echo "Consider adding sphinx documentation"

# CI simulation
ci-test:
	@echo "Running CI-like test suite..."
	make lint
	make test-fast
	@echo "CI tests completed successfully!"

# Performance benchmarks
benchmark:
	pytest tests/performance/test_large_datasets.py::TestBenchmarks -v -s