# CI Active Inference Slopbox - Setup Instructions

## System Requirements

- **Operating System**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **Memory**: 2GB RAM minimum
- **Storage**: 1GB free disk space
- **Network**: Internet access for dependency installation

## Quick Setup

### Automated Setup (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/companionintelligence/CI-Active-Inference-Slopbox.git
   cd CI-Active-Inference-Slopbox
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
   
   This script will:
   - Check Python version compatibility
   - Create a virtual environment
   - Install all dependencies
   - Initialize the database
   - Create sample data
   - Generate configuration files

3. **Verify the installation:**
   ```bash
   ./run_demo.sh
   ```

### Manual Setup

If you prefer manual control:

```bash
# 1. Check Python version
python3 --version  # Should be 3.8+

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env file as needed

# 5. Initialize system
python main.py init-db
python main.py sample-data

# 6. Test the system
python main.py demo
```

## Available Commands

### Command Line Interface

```bash
# System management
python main.py init-db         # Initialize database
python main.py sample-data     # Create sample data
python main.py status          # Show system status

# Active Inference operations
python main.py demo            # Run demonstration
python main.py train           # Run training loop

# Help
python main.py --help          # Show all commands
python main.py --debug demo    # Run with debug output
```

### Quick Scripts

```bash
# Demo and testing
./run_demo.sh                  # Run Active Inference demo
./run_tests.sh                 # Run test suite

# API server
./run_api.sh                   # Start FastAPI server
# API available at: http://localhost:8000
# Documentation at: http://localhost:8000/docs
```

## Configuration

### Environment Variables (.env file)

```bash
# Database configuration
DATABASE_PATH=ci_database.db
DATABASE_URL=postgresql://user:pass@localhost/ci_db  # Optional PostgreSQL

# Active Inference parameters
INFERENCE_UPDATE_INTERVAL=60
PREDICTION_HORIZON=10
LEARNING_RATE=0.01

# API server settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Logging
LOG_LEVEL=INFO
```

### Database Options

**SQLite (Default)**: Automatic setup, no external dependencies
```bash
DATABASE_PATH=ci_database.db
```

**PostgreSQL (Optional)**: For production use
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/ci_database
```

## Testing

### Run All Tests
```bash
./run_tests.sh
# or
pytest tests/ -v --cov=ci_active_inference
```

### Test Categories
```bash
# Database functionality
pytest tests/test_database.py -v

# Active Inference agents
pytest tests/test_agents.py -v

# Command line interface
pytest tests/test_main.py -v
```

### Test Coverage
```bash
pytest tests/ --cov=ci_active_inference --cov-report=html
# View coverage: open htmlcov/index.html
```

## Troubleshooting

### Common Issues

**Python version error:**
```bash
# Install Python 3.8+
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev
```

**Permission denied on scripts:**
```bash
chmod +x *.sh
```

**Package installation issues:**
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Database connection issues:**
```bash
# Remove existing database and reinitialize
rm -f ci_database.db
python main.py init-db
python main.py sample-data
```

**Port already in use (API server):**
```bash
# Use different port
API_PORT=8001 ./run_api.sh
# or
uvicorn ci_active_inference.api:app --port 8001
```

### Debug Mode

```bash
# Enable debug logging
python main.py --debug demo

# Or set environment variable
DEBUG=True python main.py demo
```

### Logging

Application logs are written to:
- Console output (INFO level and above)
- `ci_active_inference.log` file (when DEBUG=False)

## Development

### Project Structure
```
CI-Active-Inference-Slopbox/
├── ci_active_inference/     # Main package
│   ├── agents/             # Active Inference agents
│   ├── api/                # FastAPI endpoints
│   ├── database/           # Database and DAOs
│   ├── models/             # Data models
│   └── utils/              # Utilities
├── tests/                  # Test suite
├── main.py                 # CLI entry point
├── setup.sh               # Setup script
├── run_*.sh               # Operation scripts
└── requirements.txt       # Dependencies
```

### Adding Features

1. **New agents**: Add to `ci_active_inference/agents/`
2. **API endpoints**: Add to `ci_active_inference/api/`
3. **Database models**: Add to `ci_active_inference/database/`
4. **Tests**: Add to `tests/` directory

### Code Style

```bash
# Install development dependencies
pip install black flake8 mypy

# Format code
black ci_active_inference/ tests/

# Check style
flake8 ci_active_inference/ tests/

# Type checking
mypy ci_active_inference/
```

## Production Deployment

### Environment Setup
```bash
# Use production database
DATABASE_URL=postgresql://username:password@host:port/database

# Configure for production
DEBUG=False
LOG_LEVEL=WARNING
API_HOST=0.0.0.0
API_PORT=8000
```

### Performance Tuning
- Use PostgreSQL for better performance
- Enable connection pooling
- Configure appropriate logging levels
- Monitor memory usage during training

### Security
- Use environment variables for sensitive data
- Enable HTTPS for API endpoints
- Implement authentication if needed
- Regular database backups

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review test output: `./run_tests.sh`
3. Enable debug mode: `--debug` flag
4. Check application logs
5. Consult the project documentation

---

**Ready to start?** Run `./setup.sh` and then `./run_demo.sh` to see the Active Inference system in action!