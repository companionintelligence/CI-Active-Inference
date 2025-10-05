#!/bin/bash

# CI Active Inference Slopbox - Test Runner
# This script runs the test suite

set -e

echo "=================================================="
echo "CI Active Inference Slopbox - Test Suite"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if tests directory exists
if [ ! -d "tests" ]; then
    echo "Creating tests directory..."
    mkdir -p tests
fi

# Run tests
echo "Running test suite..."
echo ""

# Run pytest with coverage if available
if pip list | grep -q pytest-cov; then
    pytest tests/ -v --cov=ci_active_inference --cov-report=term-missing
else
    pytest tests/ -v
fi

echo ""
echo "=================================================="
echo "Test suite completed!"
echo "=================================================="