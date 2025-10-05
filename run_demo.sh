#!/bin/bash

# CI Active Inference Slopbox - Demo Runner
# This script runs the Active Inference demonstration

set -e

echo "=================================================="
echo "CI Active Inference Slopbox - Demo"
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

# Run demo
echo "Running Active Inference demonstration..."
echo ""
python main.py demo

echo ""
echo "=================================================="
echo "Demo completed successfully!"
echo "=================================================="
echo ""
echo "Other available commands:"
echo "  ./run_api.sh              # Start API server"
echo "  python main.py status     # Show system status"
echo "  python main.py train      # Run training loop"
echo ""