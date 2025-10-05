# CI Active Inference System

A sophisticated Active Inference system for predicting and optimizing behavior in CI (Continuous Integration) environments using the Free Energy Principle. This system integrates with PostgreSQL databases to analyze patterns in people, places, things, content, and events.

## Features

- **Active Inference Engine**: Implements the Free Energy Principle using the pymdp library
- **PostgreSQL Integration**: Comprehensive database models for CI server data
- **Behavior Prediction**: Predicts user behavior and system states
- **Optimization Recommendations**: Provides actionable insights for resource allocation and scheduling
- **RESTful API**: Full REST API for system interaction
- **Real-time Learning**: Continuous learning from system observations

## Architecture

The system is built with the following components:

- **Database Layer**: PostgreSQL with SQLAlchemy ORM
- **Active Inference Agents**: pymdp-based agents for prediction and optimization
- **API Layer**: FastAPI-based REST endpoints
- **Data Analysis**: Pattern recognition and feature extraction
- **Utilities**: Configuration management and helper functions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/companionintelligence/CI-Active-Inference-Slopbox.git
cd CI-Active-Inference-Slopbox
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your database configuration
```

4. Initialize the database:
```bash
python main.py init-db
```

5. Create sample data (optional):
```bash
python main.py sample-data
```

## Usage

### Starting the API Server

```bash
python main.py server
```

The API server will start on `http://localhost:8000`. You can access the interactive API documentation at `http://localhost:8000/docs`.

### Running a Demo

```bash
python main.py demo
```

This runs a demonstration of the Active Inference capabilities with sample data.

### Training the Agent

```bash
python main.py train
```

Runs a continuous training loop for the Active Inference agent.

## API Endpoints

### Data Management
- `POST /persons/` - Create a person
- `GET /persons/` - List persons
- `GET /persons/{id}/patterns` - Get activity patterns for a person
- `POST /places/` - Create a place
- `GET /places/` - List places
- `GET /places/{id}/utilization` - Get place utilization
- `POST /things/` - Create a thing
- `GET /things/available` - List available things
- `POST /content/` - Create content
- `GET /content/recent` - Get recent content
- `POST /events/` - Create an event
- `GET /events/upcoming` - Get upcoming events

### Active Inference
- `GET /inference/system-state` - Get current system state
- `POST /inference/predict` - Predict future states
- `POST /inference/recommend-actions` - Get action recommendations
- `POST /inference/update-beliefs` - Update agent beliefs
- `GET /inference/learning-progress` - Get learning progress

### Optimization
- `POST /optimization/user-schedule/{person_id}` - Optimize user schedule
- `POST /optimization/resource-allocation` - Optimize resource allocation

## Database Schema

The system uses the following main entities:

- **Person**: Users in the CI system (name, email, role, department)
- **Place**: Locations (meeting rooms, offices, labs)
- **Thing**: Resources (equipment, software, tools)
- **Content**: Documents, reports, artifacts
- **Event**: Activities, meetings, tasks

All entities support flexible metadata through JSON fields.

## Active Inference Implementation

The system implements Active Inference using:

- **Generative Models**: A-matrices (observation model) and B-matrices (transition model)
- **Belief Updates**: Variational Message Passing for state inference
- **Action Selection**: Policy optimization based on expected free energy
- **Learning**: Online parameter updates from experience

### Key Components

- `CIActiveInferenceAgent`: Main agent implementing Active Inference
- `CIBehaviorOptimizer`: High-level optimization using the agent
- `DataAnalyzer`: Feature extraction and pattern analysis

## Configuration

Configuration is managed through environment variables:

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/ci_db

# Active Inference
INFERENCE_UPDATE_INTERVAL=60
PREDICTION_HORIZON=10
LEARNING_RATE=0.01

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black ci_active_inference/
flake8 ci_active_inference/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Pezzulo, G., Friston, K. J., & Parr, T. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*
- pymdp library: https://github.com/infer-actively/pymdp
- The Free Energy Principle and Active Inference research