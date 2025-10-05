#!/bin/bash

# CI Active Inference Slopbox - Setup Script for Linux
# This script sets up the development environment and installs all dependencies

set -e  # Exit on any error

echo "=================================================="
echo "CI Active Inference Slopbox - Setup Script"
echo "=================================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Found Python version: $python_version"

# Check if Python version is >= 3.8
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8 or higher is required"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not available"
    echo "Please install pip3 and try again"
    exit 1
fi

echo "✓ Python and pip are available"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✓ Dependencies installed successfully"

# Create .env file from template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo "Please edit .env file to configure your database and other settings"
else
    echo "✓ .env file already exists"
fi

# Initialize database
echo "Initializing database..."
python main.py init-db

# Create sample data
echo "Creating sample data..."
python main.py sample-data

echo "=================================================="
echo "Setup completed successfully!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file to configure your settings"
echo "2. Run './run_demo.sh' to test the system"
echo "3. Run './run_api.sh' to start the API server"
echo ""
echo "To activate the virtual environment manually:"
echo "  source venv/bin/activate"
echo ""
echo "Available commands:"
echo "  python main.py --help    # Show available CLI commands"
echo "  python main.py demo      # Run Active Inference demonstration"
echo "  python main.py status    # Show system status"
echo "  python main.py train     # Run training loop"
echo ""