#!/bin/bash

# CI Active Inference Slopbox - API Server Runner
# This script starts the FastAPI server

set -e

echo "=================================================="
echo "CI Active Inference Slopbox - API Server"
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

# Default configuration
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}
RELOAD=${API_RELOAD:-true}

echo "Starting API server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Auto-reload: $RELOAD"
echo ""
echo "API will be available at: http://$HOST:$PORT"
echo "API documentation at: http://$HOST:$PORT/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the API server
if [ "$RELOAD" = "true" ]; then
    uvicorn ci_active_inference.api:app --host $HOST --port $PORT --reload
else
    uvicorn ci_active_inference.api:app --host $HOST --port $PORT
fi