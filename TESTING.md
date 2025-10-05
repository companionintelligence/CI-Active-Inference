# CI Active Inference Slopbox - Testing Instructions

## Quick Test

Run the complete test suite:
```bash
./run_tests.sh
```

Or manually:
```bash
pytest tests/ -v --cov=ci_active_inference --cov-report=term-missing
```

## Test Categories

### 1. Database Tests (`tests/test_database.py`)
Tests database functionality including:
- Database connection and table creation
- User DAO operations (create, read)
- Person, Place, Event DAO operations
- Data analyzer system state generation

### 2. Agent Tests (`tests/test_agents.py`)
Tests Active Inference functionality:
- Agent initialization
- Future state prediction
- Action recommendations
- Belief updating
- Behavior optimization

### 3. CLI Tests (`tests/test_main.py`)
Tests command-line interface:
- Help command functionality
- Database initialization
- Sample data creation
- System status reporting
- Shell script existence and permissions

## Running Specific Tests

### Run specific test categories:
```bash
# Database tests only
pytest tests/test_database.py -v

# Agent tests only
pytest tests/test_agents.py -v

# CLI tests only
pytest tests/test_main.py -v
```

### Run specific test classes:
```bash
# User DAO tests
pytest tests/test_database.py::TestUserDAO -v

# Agent functionality tests
pytest tests/test_agents.py::TestCIActiveInferenceAgent -v
```

### Run specific tests:
```bash
# Test database connection
pytest tests/test_database.py::TestDatabase::test_database_connection -v

# Test agent initialization
pytest tests/test_agents.py::TestCIActiveInferenceAgent::test_agent_initialization -v
```

## Test Coverage

Generate test coverage report:
```bash
pytest tests/ --cov=ci_active_inference --cov-report=html
# Open htmlcov/index.html to view detailed coverage report
```

## Test Configuration

Tests use isolated temporary databases to avoid interfering with the main application. Each test session:

1. Creates a temporary SQLite database
2. Initializes required tables
3. Runs tests in isolation
4. Cleans up temporary files

## Adding New Tests

### Test Structure
```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_feature_functionality(self, test_user):
        """Test that new feature works correctly."""
        # Test implementation
        assert expected_result == actual_result
```

### Using Fixtures
Available fixtures:
- `test_db`: Temporary test database
- `test_user`: Test user account

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

## Continuous Integration

The test suite is designed for CI/CD integration. All tests should:
- Run independently
- Clean up after themselves
- Provide clear error messages
- Have deterministic outcomes

## Troubleshooting

### Common Issues

**Database connection errors:**
```bash
# Ensure clean test environment
rm -f ci_database.db
pytest tests/test_database.py -v
```

**Import errors:**
```bash
# Install dependencies
pip install -r requirements.txt
```

**Permission errors:**
```bash
# Ensure scripts are executable
chmod +x *.sh
```

### Debug Mode
```bash
# Run with debug output
pytest tests/ -v -s --tb=long
```

### Logging
Test runs with INFO level logging. To see debug logs:
```bash
pytest tests/ -v -s --log-cli-level=DEBUG
```