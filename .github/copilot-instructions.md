# GitHub Copilot Instructions for CI-Active-Inference-Slopbox

## Project Overview

This repository implements a personalized Active Inference agent using `pymdp` that leverages personal life-log data stored in PostgreSQL to predict behavior and suggest optimal actions for health and happiness.

## Architecture Guidelines

### Core Technologies
- **Active Inference Framework**: `pymdp` for discrete POMDP implementations
- **Database**: PostgreSQL for life-log data storage
- **Backend**: FastAPI with Python 3.8+
- **Frontend**: React/Next.js with TypeScript
- **Data Processing**: NumPy, Pandas for matrix operations
- **Validation**: Pydantic for data schemas

### Project Structure
```
ci_active_inference/
├── agents/          # Active Inference agent implementations
├── api/             # FastAPI web service endpoints
├── database/        # PostgreSQL schemas and DAOs
├── models/          # Data models and Pydantic schemas
└── utils/           # Utility functions and helpers
```

## Coding Standards

### Python Code
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Prefer composition over inheritance
- Use dataclasses or Pydantic models for data structures
- Include comprehensive docstrings with examples

### Example Active Inference Agent Pattern:
```python
from pymdp.agent import Agent
from pymdp import utils
import numpy as np
from typing import Dict, List, Any, Optional

class PersonalLifeAgent:
    def __init__(self, user_id: str, db_connection):
        self.user_id = user_id
        self.db = db_connection
        
        # Define state space based on life-log data
        self.num_states = [3, 3, 5]  # [Energy, Mood, Context]
        self.num_obs = [4, 5, 4, 3]  # [Digital, Location, Social, Content]
        self.num_actions = [9]       # Available actions
        
        # Build matrices from database
        self.A = self._build_A_matrices()  # Observation likelihood
        self.B = self._build_B_matrices()  # State transitions
        self.C = self._build_C_vectors()   # Preferences
        self.D = self._build_D_vectors()   # Prior beliefs
        
        # Initialize pymdp agent
        self.agent = Agent(A=self.A, B=self.B, C=self.C, D=self.D)
    
    def predict_and_suggest(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Main prediction and suggestion pipeline"""
        obs_indices = self._observations_to_indices(observations)
        qs = self.agent.infer_states(obs_indices)
        q_pi, neg_G = self.agent.infer_policies()
        action = self.agent.sample_action()
        
        return {
            'current_beliefs': qs,
            'suggested_action': self._action_to_description(action),
            'confidence': float(np.max(q_pi)),
            'efe_breakdown': self._decompose_efe(neg_G)
        }
```

### Database Integration
- Use connection pooling for PostgreSQL
- Implement proper transaction management
- Create reusable DAO patterns for data access
- Use SQL queries for complex temporal aggregations

### Example Database Pattern:
```python
from typing import List, Dict, Any
import psycopg2
import pandas as pd

class DataExtractor:
    def __init__(self, db_connection, user_id: str):
        self.db = db_connection
        self.user_id = user_id
    
    def extract_temporal_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract time-series features for matrix construction"""
        query = """
        WITH hourly_aggregates AS (
          SELECT 
            DATE_TRUNC('hour', e.start_time) as time_bucket,
            CASE 
              WHEN e.name ~* 'work|meeting' THEN 'work'
              WHEN e.name ~* 'exercise|gym' THEN 'health'
              WHEN e.name ~* 'social|dinner' THEN 'social'
              ELSE 'other'
            END as context_state,
            COUNT(*) as frequency
          FROM events e
          WHERE e.user_id = %s 
            AND e.start_time BETWEEN %s AND %s
          GROUP BY 1, 2
        )
        SELECT * FROM hourly_aggregates ORDER BY time_bucket
        """
        return pd.read_sql(query, self.db, params=[self.user_id, start_date, end_date])
```

## Active Inference Specific Guidelines

### Matrix Construction
- Use Laplace smoothing for sparse data: `A = np.ones(shape) * 0.01`
- Normalize properly: observations sum to 1 for each hidden state
- Implement Dirichlet priors for online learning
- Use object arrays for multi-factor state spaces

### State Space Design
- Keep states interpretable (Energy: low/med/high)
- Limit complexity: start with 3-5 states per factor
- Map database entities to discrete categories
- Design actions as concrete interventions

### Expected Free Energy (EFE)
- Decompose into Risk (preference mismatch) + Ambiguity (uncertainty)
- Use for both action selection and explanation generation
- Track EFE components for debugging and user understanding

## API Design Patterns

### FastAPI Endpoints
```python
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import Dict, Any

class ObservationInput(BaseModel):
    digital_behavior: str
    location_context: str
    social_context: str
    timestamp: str

class AgentResponse(BaseModel):
    predicted_default: str
    suggested_action: str
    confidence: float
    explanation: str

@app.post("/agent/{user_id}/predict", response_model=AgentResponse)
async def predict_and_suggest(user_id: str, observations: ObservationInput):
    agent = get_user_agent(user_id)
    result = agent.predict_and_suggest(observations.dict())
    return AgentResponse(**result)
```

### Error Handling
- Use proper HTTP status codes
- Provide meaningful error messages
- Log errors with context for debugging
- Implement graceful degradation for model failures

## Frontend Guidelines

### React Components
- Use functional components with hooks
- Implement proper loading and error states
- Create reusable UI components for agent interactions
- Display confidence levels and explanations clearly

### State Management
- Use React Query for API state management
- Implement proper caching for predictions
- Handle real-time updates efficiently

## Testing Strategy

### Unit Tests
- Test matrix construction and normalization
- Validate state space mappings
- Test database query logic
- Mock external dependencies

### Integration Tests
- Test full prediction pipeline
- Validate API endpoints
- Test database transactions
- Check frontend-backend integration

### Performance Tests
- Benchmark matrix operations
- Test with realistic data volumes
- Monitor memory usage during inference
- Validate response times

## Documentation Requirements

### Code Comments
- Explain Active Inference concepts for non-experts
- Document matrix shapes and dimensions
- Clarify state space mappings
- Include mathematical formulations where helpful

### API Documentation
- Use OpenAPI/Swagger for automatic documentation
- Include example requests and responses
- Document error conditions
- Provide integration examples

## Common Patterns to Follow

### Matrix Operations
```python
import numpy as np
from pymdp import utils

# Initialize with prior
A = np.ones((num_obs, num_states)) * 0.01

# Fill with empirical counts
for observation, state in data:
    A[observation, state] += 1

# Normalize columns
A = utils.normalize(A, axis=0)
```

### Database Queries
```python
# Use parameterized queries
query = """
SELECT state_factor, observation_type, COUNT(*) as frequency
FROM user_data 
WHERE user_id = %s AND date_range = %s
GROUP BY state_factor, observation_type
"""
df = pd.read_sql(query, db_conn, params=[user_id, date_range])
```

### Error Handling
```python
try:
    result = agent.predict_and_suggest(observations)
    return result
except Exception as e:
    logger.error(f"Prediction failed for user {user_id}: {e}")
    # Return fallback prediction
    return get_default_suggestion()
```

## Performance Considerations

- Use sparse matrices for large state spaces
- Cache frequently accessed matrix computations
- Implement batch processing for multiple users
- Consider approximate inference for real-time constraints

## Security Guidelines

- Validate all user inputs
- Use parameterized database queries
- Implement proper authentication
- Don't log sensitive user data
- Use environment variables for secrets

## Debugging Tips

- Log matrix shapes and properties during construction
- Track EFE components to understand agent decisions
- Validate probability distributions sum to 1
- Monitor convergence during belief updates
- Use synthetic data for controlled testing

## Development Workflow

1. Start with minimal working examples
2. Build incrementally with frequent testing
3. Use type hints and validation extensively
4. Document assumptions and design decisions
5. Implement comprehensive error handling
6. Add performance monitoring early

For detailed implementation examples, see [README.md](./README.md) and [development-plan.md](./development-plan.md).