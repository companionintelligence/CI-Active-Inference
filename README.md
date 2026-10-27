Experimental 
 
<h1 align="center">CI Personalized Active Inference Agent</h1>

<p align="center"><strong>Active Inference Agent</strong></p>

<p align="center"><a href="#quick-setup-linux">Get started</a></p>

## Quick Setup (Linux)

1. **Clone and setup the environment:**
   ```bash
   git clone https://github.com/companionintelligence/CI-Active-Inference-Slopbox.git
   cd CI-Active-Inference-Slopbox
   ./setup.sh
   ```

2. **Run the demonstration:**
   ```bash
   ./run_demo.sh
   ```

3. **Start the API server:**
   ```bash
   ./run_api.sh
   ```

4. **Run tests:**
   ```bash
   ./run_tests.sh
   ```

### Manual Setup

If you prefer manual setup:

```bash
# Install Python 3.8+ and pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Initialize database and sample data
python main.py init-db
python main.py sample-data

# Test the system
python main.py demo
```

### Available Commands

```bash
python main.py --help          # Show all available commands
python main.py init-db         # Initialize database
python main.py sample-data     # Create sample data
python main.py demo            # Run Active Inference demonstration
python main.py train           # Run training loop
python main.py status          # Show system status
```

### API Server

Start the FastAPI server:
```bash
./run_api.sh
# or manually:
uvicorn ci_active_inference.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs

### Testing

Run the comprehensive test suite:
```bash
./run_tests.sh
# or manually:
pytest tests/ -v --cov=ci_active_inference --cov-report=term-missing
```

## Requirements

- Python 3.8+
- Linux (tested on Ubuntu 20.04+)
- 2GB RAM minimum
- 1GB disk space

## Dependencies

Core dependencies:
- **pymdp**: Active Inference framework
- **FastAPI**: Web API framework
- **SQLAlchemy**: Database ORM
- **pandas/numpy**: Data processing
- **psycopg2**: PostgreSQL adapter (optional)

See `requirements.txt` for complete list.

---

This repository provides a comprehensive development plan for implementing a personalized Active Inference agent using `pymdp` that leverages your personal data to predict behavior and suggest optimal actions for health and happiness.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/companionintelligence/CI-Active-Inference-Slopbox.git
cd CI-Active-Inference-Slopbox

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your database configuration

# Initialize database
python main.py init-db

# Create sample data
python main.py sample-data

# Run Active Inference demonstration
python main.py demo

# Check system status
python main.py status
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- PostgreSQL 12+ 
- Node.js 16+ (for frontend development)

### Environment Setup
1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure PostgreSQL:**
   ```bash
   # Create database
   createdb ci_active_inference
   
   # Set environment variables in .env
   DATABASE_URL=postgresql://user:password@localhost/ci_active_inference
   ```

3. **Initialize database schema:**
   ```bash
   python main.py init-db
   ```

4. **Create sample data:**
   ```bash
   python main.py sample-data
   ```

For detailed development setup, see [development-plan.md](./development-plan.md).

---

## Table of Contents

1. [What pymdp Gives You](#1-what-pymdp-gives-you)
2. [Mapping Life-Log Data to Discrete POMDP](#2-mapping-life-log-data-to-discrete-pomdp)
3. [Database Integration Strategy](#3-database-integration-strategy)
4. [Building A/B/C/D Matrices from PostgreSQL Data](#4-building-abcd-matrices-from-postgresql-data)
5. [Minimal Working Example](#5-minimal-working-example-mwe-with-pymdp)
6. [Learning and Model Updates](#6-learning-from-your-data-updating-a--b)
7. [Daily Predict & Suggest Loop](#7-daily-predict--suggest-loop)
8. [Data Pipeline: Postgres to Arrays](#8-from-postgres-to-arrays-practical-recipe)
9. [Service Architecture](#9-service-layering-python--typescript)
10. [Development Phases](#10-development-phases-and-implementation-roadmap)
11. [Validation Strategy](#11-validation)
12. [Scaling and Advanced Features](#12-scope-horizons-and-scaling-tips)

---

## 1. What pymdp Gives You

`pymdp` is a NumPy-centric library for **Active Inference in discrete POMDPs**. You specify a generative model with:

- **A** (likelihood matrices): `P(observation | hidden_state)`
- **B** (transition matrices): `P(next_state | current_state, action)` 
- **C** (preference vectors): Prior preferences over observations
- **D** (prior beliefs): Prior distribution over initial states

Then you use the high-level **`Agent`** API to:
1. `infer_states(obs)` → Update beliefs about hidden states
2. `infer_policies()` → Plan by minimizing Expected Free Energy (EFE)
3. `sample_action()` → Select action (or pick argmin-EFE)

**Key principle**: The agent minimizes **Expected Free Energy** = **Risk** (preference mismatch) + **Ambiguity** (uncertainty), providing both exploitation and exploration without separate reward engineering.

> **Reference**: [pymdp documentation](https://pymdp-rtd.readthedocs.io/) - Agent API and Fundamentals

---

## 2. Mapping Life-Log Data to Discrete POMDP

### 2.1 Time Resolution
Choose decision points aligned with your data: hourly, or key moments (morning, midday, evening). Start with daily granularity.

### 2.2 Hidden State Factors (Internal states we want to track)
Based on the database schema, we can extract:

- **Energy Level** ∈ {low, medium, high}
- **Mood State** ∈ {negative, neutral, positive}  
- **Context** ∈ {home, work, social, outdoors, travel}
- **Health Status** ∈ {poor, fair, good, excellent}
- **Social Engagement** ∈ {isolated, minimal, moderate, high}

*Note: pymdp uses factorized distributions in NumPy object arrays for efficient multi-factor state spaces.*

### 2.3 Observation Modalities (What we can measure)
Map database entities to observation categories:

- **Digital Behavior** (from `events`, `content`): {work-focus, entertainment, social-media, communication}
- **Location Context** (from `locations`, `places`): {home, office, gym, restaurant, outdoors}
- **Social Signals** (from `people`, `contacts`, `events`): {alone, small-group, large-group, family}
- **Content Engagement** (from `content`, `things`): {productive, recreational, educational, health}
- **Physiological Proxies** (if available): {calm, elevated, restorative}

### 2.4 Action Space (What the agent can suggest)
- **Health Actions**: {exercise, meditate, sleep-early, hydrate, walk}
- **Social Actions**: {call-friend, family-time, social-event, alone-time}  
- **Work Actions**: {deep-focus, break, delegate, prioritize}
- **Recreation**: {read, music, hobby, nature, rest}

---

## 3. Database Integration Strategy

### 3.1 Key Database Entities for Active Inference

From `database-schema.puml`, the most relevant entities are:

**Temporal Data Sources:**
- `events` → Activities and their temporal boundaries
- `locations` → Spatial context over time  
- `content` → Information consumption patterns
- `people` + `contacts` → Social interaction data

**Metadata for Context:**
- `tags` → Categorical labels for all entities
- `embeddings` → Semantic similarity for content
- `connections` → Data source integrations

### 3.2 Data Aggregation Strategy

**Temporal Bucketing:**
```sql
-- Daily aggregation example
SELECT 
  DATE(start_time) as day,
  EXTRACT(hour FROM start_time) as hour_bucket,
  -- Derive context from events
  CASE 
    WHEN name ILIKE '%work%' OR name ILIKE '%meeting%' THEN 'work'
    WHEN name ILIKE '%exercise%' OR name ILIKE '%gym%' THEN 'health'
    WHEN name ILIKE '%social%' OR name ILIKE '%dinner%' THEN 'social'
    ELSE 'other'
  END as context_category,
  -- Add location context
  places.name as location_context
FROM events e
LEFT JOIN locations l ON e.user_id = l.user_id 
  AND l.datetime BETWEEN e.start_time AND e.end_time
LEFT JOIN places ON l.place_id = places.id
WHERE e.user_id = $user_id
```

---

## 4. Building A/B/C/D Matrices from PostgreSQL Data

### 4.1 A Matrices (Observation Likelihood)

For each observation modality, build `P(observation | hidden_state_combination)`:

```python
# Example: Digital Behavior observations given Energy × Mood × Context
# Shape: (num_digital_behaviors, num_energy, num_mood, num_context)

# SQL to get co-occurrence counts
query = """
SELECT 
  energy_bucket, mood_bucket, context_bucket,
  digital_behavior_type,
  COUNT(*) as frequency
FROM daily_state_observations 
GROUP BY energy_bucket, mood_bucket, context_bucket, digital_behavior_type
"""

# Convert to A matrix with Laplace smoothing
A_digital = np.ones((4, 3, 3, 5)) * 0.1  # Small prior
for row in query_results:
    A_digital[
        behavior_to_idx[row.digital_behavior_type],
        row.energy_bucket, 
        row.mood_bucket, 
        row.context_bucket
    ] += row.frequency

# Normalize columns (each hidden state sums to 1)
A_digital = normalize(A_digital, axis=0)
```

### 4.2 B Matrices (State Transitions)

For each state factor and action, build `P(next_state | current_state, action)`:

```python
# Example: How actions affect Energy transitions
# Shape: (num_energy_next, num_energy_current, num_actions)

query = """
SELECT 
  current_energy, next_energy, action_taken,
  COUNT(*) as frequency
FROM (
  SELECT 
    energy_bucket as current_energy,
    LEAD(energy_bucket) OVER (PARTITION BY user_id ORDER BY day) as next_energy,
    action_taken
  FROM daily_state_observations
  WHERE action_taken IS NOT NULL
) transitions
GROUP BY current_energy, next_energy, action_taken
"""

B_energy = np.ones((3, 3, num_actions)) * 0.1  # Prior
# ... fill from query results and normalize
```

### 4.3 C Vectors (Preferences)

Encode preferences over observations. Start with domain knowledge, then personalize:

```python
# Prefer observations indicating positive outcomes
C_digital = np.array([0.1, 0.2, 0.3, 0.4])  # [social-media, entertainment, communication, work-focus]
C_location = np.array([0.3, 0.2, 0.4, 0.1])  # [home, office, gym, restaurant] - prefer gym/home
C_social = np.array([0.1, 0.3, 0.4, 0.2])    # [alone, small-group, large-group, family] - prefer social
```

### 4.4 D Vectors (Prior Beliefs)

Initial state distribution, typically from historical start-of-day data:

```python
query = """
SELECT energy_bucket, COUNT(*) as frequency
FROM daily_state_observations 
WHERE hour_bucket = 8  -- Morning states
GROUP BY energy_bucket
"""
# Convert to normalized probability distribution
```

---

## 5. Minimal Working Example (MWE) with pymdp

```python
import numpy as np
from pymdp.agent import Agent
from pymdp import utils
import psycopg2
import pandas as pd

class PersonalLifeAgent:
    def __init__(self, user_id, db_connection):
        self.user_id = user_id
        self.db = db_connection
        
        # Define state space dimensions based on DB analysis
        self.num_states = [3, 3, 5]  # [Energy, Mood, Context]
        self.num_obs = [4, 5, 4, 3]  # [Digital, Location, Social, Content]
        self.num_actions = [1, 1, 9]  # Control over Context factor
        
        # Build matrices from database
        self.A = self._build_A_matrices()
        self.B = self._build_B_matrices() 
        self.C = self._build_C_vectors()
        self.D = self._build_D_vectors()
        
        # Initialize agent
        self.agent = Agent(A=self.A, B=self.B, C=self.C, D=self.D)
    
    def _build_A_matrices(self):
        """Build observation likelihood matrices from database"""
        A = utils.obj_array(len(self.num_obs))
        
        # Digital behavior observations
        A[0] = self._query_observation_likelihood('digital_behavior')
        # Location observations  
        A[1] = self._query_observation_likelihood('location_context')
        # Social observations
        A[2] = self._query_observation_likelihood('social_context')
        # Content observations
        A[3] = self._query_observation_likelihood('content_type')
        
        return A
    
    def _query_observation_likelihood(self, obs_type):
        """Query database for P(observation | state) counts"""
        query = f"""
        SELECT energy_bucket, mood_bucket, context_bucket, 
               {obs_type}_category, COUNT(*) as freq
        FROM user_daily_states 
        WHERE user_id = %s
        GROUP BY energy_bucket, mood_bucket, context_bucket, {obs_type}_category
        """
        
        df = pd.read_sql(query, self.db, params=[self.user_id])
        
        # Convert to properly shaped matrix with smoothing
        matrix = np.ones(self._get_A_shape(obs_type)) * 0.01
        
        for _, row in df.iterrows():
            matrix[
                self._obs_to_idx(obs_type, row[f'{obs_type}_category']),
                row.energy_bucket,
                row.mood_bucket, 
                row.context_bucket
            ] += row.freq
            
        return utils.normalize(matrix, axis=0)
    
    def predict_and_suggest(self, current_observations):
        """Main prediction and suggestion pipeline"""
        # Convert observations to indices
        obs_indices = self._observations_to_indices(current_observations)
        
        # Infer current states
        qs = self.agent.infer_states(obs_indices)
        
        # Plan actions (minimize expected free energy)
        q_pi, neg_G = self.agent.infer_policies()
        
        # Sample action or take minimum EFE
        suggested_action = self.agent.sample_action()
        
        # Predict default behavior (most likely observation given current state)
        predicted_behavior = self._predict_default_behavior(qs)
        
        return {
            'current_state_beliefs': qs,
            'suggested_action': self._action_to_description(suggested_action),
            'predicted_default': predicted_behavior,
            'efe_breakdown': self._decompose_efe(neg_G),
            'confidence': float(np.max(q_pi))
        }
    
    def update_from_outcome(self, executed_action, next_observations):
        """Update model based on actual outcomes (online learning)"""
        # This would increment Dirichlet concentration parameters
        # for A and B matrices based on observed transitions
        pass

# Usage example
agent = PersonalLifeAgent(user_id="some-uuid", db_connection=db_conn)

current_obs = {
    'digital_behavior': 'social-media',
    'location': 'home', 
    'social_context': 'alone',
    'content_type': 'entertainment'
}

result = agent.predict_and_suggest(current_obs)
print(f"Suggested action: {result['suggested_action']}")
print(f"Predicted default: {result['predicted_default']}")
```

---

## 6. Learning from Your Data (Updating A & B)

Active Inference uses **Dirichlet priors** for online learning:

1. **Initialize** concentration parameters with small values (α = 1)
2. **Add empirical counts** from historical PostgreSQL data  
3. **Update online** as new observations come in:
   - **A updates**: When observing `o_t` with inferred state `qs_t`, increment `A[o_t, :]` weighted by `qs_t`
   - **B updates**: When taking action `u_t` and inferring `qs_{t+1}`, increment `B[:, s_t, u_t]` weighted by posteriors

This keeps the model **continually personalized** without heavyweight retraining.

---

## 7. Daily "Predict & Suggest" Loop

### 7.1 Core Pipeline
1. **Ingest** latest observations for decision point
2. **Belief Update**: `qs = agent.infer_states(obs)`  
3. **Planning**: `q_pi, neg_G = agent.infer_policies()`
4. **Action Selection**: `action = argmin(neg_G)` (minimum Expected Free Energy)
5. **Prediction**: Most likely default behavior from current state beliefs
6. **Explanation**: Decompose EFE into **risk** (preference mismatch) and **ambiguity** (uncertainty)
7. **Logging**: Record choice and outcome for model updates

### 7.2 Database Logging Schema

Extend the existing schema with Active Inference specific tables:

```sql
-- Store daily state inferences and suggestions
CREATE TABLE daily_agent_states (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  timestamp TIMESTAMPTZ,
  
  -- Inferred hidden states (posterior means)
  energy_belief DECIMAL(3,2),
  mood_belief DECIMAL(3,2), 
  context_belief DECIMAL(3,2),
  
  -- Observations that led to this inference
  observations JSONB,
  
  -- Agent suggestions and predictions
  suggested_action TEXT,
  predicted_default_behavior TEXT,
  efe_score DECIMAL(10,6),
  confidence DECIMAL(3,2),
  
  -- Actual outcome (filled later)
  executed_action TEXT,
  outcome_observations JSONB,
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Store model parameters for versioning
CREATE TABLE agent_model_versions (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  version INTEGER,
  
  -- Serialized model matrices
  A_matrices JSONB,
  B_matrices JSONB, 
  C_vectors JSONB,
  D_vectors JSONB,
  
  -- Performance metrics
  prediction_accuracy DECIMAL(3,2),
  user_satisfaction_score DECIMAL(3,2),
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 8. From Postgres to Arrays (Practical Recipe)

### 8.1 Data Extraction Pipeline

```python
class DataExtractor:
    def __init__(self, db_connection, user_id):
        self.db = db_connection
        self.user_id = user_id
    
    def extract_temporal_features(self, start_date, end_date):
        """Extract time-series features for POMDP construction"""
        
        query = """
        WITH hourly_aggregates AS (
          SELECT 
            DATE_TRUNC('hour', e.start_time) as time_bucket,
            
            -- Context inference from event names and locations
            CASE 
              WHEN e.name ~* 'work|meeting|call' THEN 'work'
              WHEN e.name ~* 'exercise|gym|run|bike' THEN 'health'  
              WHEN e.name ~* 'social|dinner|friend' THEN 'social'
              WHEN p.name ~* 'home' THEN 'home'
              ELSE 'other'
            END as context_state,
            
            -- Digital behavior from content interactions
            CASE
              WHEN c.type = 'communication' THEN 'communication'
              WHEN c.type = 'entertainment' THEN 'entertainment'
              WHEN c.type = 'work' THEN 'work-focus'
              ELSE 'other'
            END as digital_behavior,
            
            -- Social context from people associations
            CASE 
              WHEN COUNT(ep.person_id) = 0 THEN 'alone'
              WHEN COUNT(ep.person_id) <= 2 THEN 'small-group'
              ELSE 'large-group'
            END as social_context,
            
            -- Aggregate location context
            STRING_AGG(DISTINCT p.name, ',') as locations
            
          FROM events e
          LEFT JOIN locations l ON e.user_id = l.user_id 
            AND l.datetime BETWEEN e.start_time AND e.end_time
          LEFT JOIN places p ON l.place_id = p.id
          LEFT JOIN content c ON e.id = c.event_id
          LEFT JOIN event_person ep ON e.id = ep.event_id
          
          WHERE e.user_id = %s 
            AND e.start_time BETWEEN %s AND %s
          GROUP BY 1, 2, 3, 4
        )
        
        SELECT * FROM hourly_aggregates
        ORDER BY time_bucket
        """
        
        return pd.read_sql(query, self.db, params=[
            self.user_id, start_date, end_date
        ])
    
    def discretize_features(self, df):
        """Convert continuous/categorical data to discrete state indices"""
        
        # Map contexts to discrete states
        context_mapping = {
            'home': 0, 'work': 1, 'social': 2, 'health': 3, 'other': 4
        }
        
        digital_mapping = {
            'work-focus': 0, 'communication': 1, 'entertainment': 2, 'other': 3
        }
        
        social_mapping = {
            'alone': 0, 'small-group': 1, 'large-group': 2
        }
        
        df['context_idx'] = df['context_state'].map(context_mapping).fillna(4)
        df['digital_idx'] = df['digital_behavior'].map(digital_mapping).fillna(3)
        df['social_idx'] = df['social_context'].map(social_mapping).fillna(0)
        
        return df
```

### 8.2 Matrix Construction

```python
def build_A_matrix_from_data(df, state_cols, obs_col):
    """Build observation likelihood matrix from co-occurrence data"""
    
    # Get unique values for dimensions
    state_values = [df[col].unique() for col in state_cols]
    obs_values = df[obs_col].unique()
    
    # Initialize with small prior
    A = np.ones([len(obs_values)] + [len(sv) for sv in state_values]) * 0.01
    
    # Fill with empirical counts
    for _, row in df.iterrows():
        if pd.notna(row[obs_col]):
            obs_idx = np.where(obs_values == row[obs_col])[0][0]
            state_indices = [
                np.where(state_values[i] == row[state_cols[i]])[0][0] 
                for i in range(len(state_cols))
            ]
            A[obs_idx][tuple(state_indices)] += 1
    
    # Normalize columns (each hidden state combination sums to 1)
    return normalize_columns(A)

def build_B_matrix_from_transitions(df, state_col, action_col):
    """Build state transition matrix from temporal data"""
    
    # Create next-state column
    df[f'next_{state_col}'] = df[state_col].shift(-1)
    
    # Filter valid transitions
    transitions = df.dropna(subset=[state_col, f'next_{state_col}', action_col])
    
    state_values = df[state_col].unique()
    action_values = df[action_col].unique()
    
    # Initialize B matrix: P(next_state | current_state, action)
    B = np.ones((len(state_values), len(state_values), len(action_values))) * 0.01
    
    for _, row in transitions.iterrows():
        current_idx = np.where(state_values == row[state_col])[0][0]
        next_idx = np.where(state_values == row[f'next_{state_col}'])[0][0]
        action_idx = np.where(action_values == row[action_col])[0][0]
        
        B[next_idx, current_idx, action_idx] += 1
    
    # Normalize each (current_state, action) column
    return normalize_columns(B, axis=0)
```

---

## 9. Service Layering (Python + TypeScript)

### 9.1 Python Backend (FastAPI)

```python
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio

app = FastAPI(title="Personal Active Inference Agent")

class ObservationInput(BaseModel):
    digital_behavior: str
    location_context: str  
    social_context: str
    content_type: str
    timestamp: str

class AgentResponse(BaseModel):
    predicted_default: str
    suggested_action: str
    confidence: float
    efe_breakdown: Dict[str, float]
    explanation: str

class FeedbackInput(BaseModel):
    executed_action: str
    outcome_observations: Dict[str, Any]

# Global agent instance per user
user_agents: Dict[str, PersonalLifeAgent] = {}

@app.post("/agent/{user_id}/predict", response_model=AgentResponse)
async def predict_and_suggest(
    user_id: str, 
    observations: ObservationInput
):
    """Main prediction and suggestion endpoint"""
    
    agent = get_user_agent(user_id)
    
    # Convert to agent format
    obs_dict = observations.dict()
    result = agent.predict_and_suggest(obs_dict)
    
    # Store prediction in database for later validation
    await store_prediction(user_id, observations, result)
    
    return AgentResponse(
        predicted_default=result['predicted_default'],
        suggested_action=result['suggested_action'],
        confidence=result['confidence'],
        efe_breakdown=result['efe_breakdown'],
        explanation=generate_explanation(result)
    )

@app.post("/agent/{user_id}/feedback")
async def provide_feedback(
    user_id: str,
    feedback: FeedbackInput
):
    """Update agent model based on actual outcomes"""
    
    agent = get_user_agent(user_id)
    agent.update_from_outcome(
        feedback.executed_action,
        feedback.outcome_observations
    )
    
    return {"status": "updated"}

@app.get("/agent/{user_id}/model-stats")
async def get_model_statistics(user_id: str):
    """Return model performance and uncertainty metrics"""
    
    agent = get_user_agent(user_id)
    
    return {
        "total_predictions": agent.prediction_count,
        "accuracy_last_week": agent.recent_accuracy(),
        "uncertainty_level": agent.average_uncertainty(),
        "last_updated": agent.last_update_time
    }

def get_user_agent(user_id: str) -> PersonalLifeAgent:
    """Get or create agent instance for user"""
    if user_id not in user_agents:
        # Load from database or create new
        user_agents[user_id] = PersonalLifeAgent.load_or_create(user_id)
    return user_agents[user_id]
```

### 9.2 TypeScript Frontend (React/Next.js)

```typescript
// components/AgentDashboard.tsx
import React, { useState, useEffect } from 'react';
import { Card, Button, Progress, Alert } from '@/components/ui';

interface AgentPrediction {
  predicted_default: string;
  suggested_action: string;
  confidence: number;
  efe_breakdown: Record<string, number>;
  explanation: string;
}

export const AgentDashboard: React.FC = () => {
  const [prediction, setPrediction] = useState<AgentPrediction | null>(null);
  const [loading, setLoading] = useState(false);

  const getCurrentObservations = async () => {
    // Get current context from various sources
    return {
      digital_behavior: await detectCurrentDigitalBehavior(),
      location_context: await getCurrentLocation(),
      social_context: await detectSocialContext(),
      content_type: await getRecentContentType(),
      timestamp: new Date().toISOString()
    };
  };

  const fetchPrediction = async () => {
    setLoading(true);
    try {
      const observations = await getCurrentObservations();
      const response = await fetch('/api/agent/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(observations)
      });
      
      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const executeSuggestion = async (action: string) => {
    // Integrate with calendar, notifications, etc.
    await scheduleAction(action);
    
    // Provide feedback to agent
    await fetch('/api/agent/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        executed_action: action,
        outcome_observations: await getCurrentObservations()
      })
    });
  };

  useEffect(() => {
    fetchPrediction();
    // Set up periodic refresh
    const interval = setInterval(fetchPrediction, 3600000); // Hourly
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div>Analyzing current state...</div>;

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h2 className="text-2xl font-bold mb-4">Behavioral Insight</h2>
        
        {prediction && (
          <>
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div>
                <h3 className="font-semibold text-gray-600">Likely Next Action</h3>
                <p className="text-lg">{prediction.predicted_default}</p>
              </div>
              
              <div>
                <h3 className="font-semibold text-green-600">Suggested Alternative</h3>
                <p className="text-lg font-medium">{prediction.suggested_action}</p>
                <Button 
                  onClick={() => executeSuggestion(prediction.suggested_action)}
                  className="mt-2"
                >
                  Apply Suggestion
                </Button>
              </div>
            </div>

            <div className="mb-4">
              <h4 className="font-semibold mb-2">Confidence</h4>
              <Progress value={prediction.confidence * 100} className="h-2" />
              <span className="text-sm text-gray-500">
                {(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>

            <Alert>
              <h4 className="font-semibold">Why this suggestion?</h4>
              <p>{prediction.explanation}</p>
            </Alert>

            {/* EFE Breakdown Visualization */}
            <div className="mt-4">
              <h4 className="font-semibold mb-2">Decision Factors</h4>
              {Object.entries(prediction.efe_breakdown).map(([factor, value]) => (
                <div key={factor} className="flex justify-between">
                  <span className="capitalize">{factor.replace('_', ' ')}</span>
                  <span>{value.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </Card>
    </div>
  );
};
```

---

## 10. Development Phases and Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up development environment with `pymdp`, PostgreSQL, FastAPI
- [ ] Create database views for temporal feature extraction
- [ ] Implement basic data discretization pipeline
- [ ] Build minimal A/B/C/D matrices from sample data
- [ ] Validate pymdp Agent initialization and basic inference

### Phase 2: Core Agent (Weeks 3-4)  
- [ ] Implement full PersonalLifeAgent class with database integration
- [ ] Create observation likelihood construction from real data
- [ ] Build state transition matrices with action dependencies
- [ ] Design and validate preference vectors (C matrices)
- [ ] Test predict_and_suggest pipeline end-to-end

### Phase 3: Learning & Adaptation (Weeks 5-6)
- [ ] Implement online learning with Dirichlet updates
- [ ] Create model versioning and rollback system
- [ ] Add uncertainty quantification and confidence metrics
- [ ] Build performance tracking and validation pipeline
- [ ] Test model adaptation over time with synthetic data

### Phase 4: Service Integration (Weeks 7-8)
- [ ] Develop FastAPI backend with agent endpoints
- [ ] Create database schemas for storing predictions and feedback
- [ ] Implement user authentication and multi-user support
- [ ] Build basic React frontend for agent interaction
- [ ] Add real-time prediction and suggestion display

### Phase 5: Advanced Features (Weeks 9-12)
- [ ] Implement multi-step planning with longer horizons
- [ ] Add explanation generation for agent suggestions
- [ ] Create A/B testing framework for different preference settings
- [ ] Build data quality monitoring and anomaly detection
- [ ] Add integration with external APIs (calendar, notifications)

### Phase 6: Validation & Optimization (Weeks 13-16)
- [ ] Conduct backtesting on historical user data
- [ ] Implement model performance benchmarking
- [ ] Create user feedback collection system
- [ ] Optimize matrix construction and inference speed
- [ ] Prepare documentation and deployment guides

---

## 11. Validation

### 11.1 Quantitative Validation
- **Prediction Accuracy**: Compare predicted vs. actual next behaviors
- **Suggestion Effectiveness**: Track user adoption rate of suggestions
- **Model Calibration**: Validate that confidence scores match actual accuracy
- **Temporal Consistency**: Test model stability over different time periods

### 11.2 Validation Metrics
```python
def evaluate_agent_performance(agent, test_data):
    """Comprehensive agent evaluation suite"""
    
    metrics = {
        'prediction_accuracy': 0.0,
        'suggestion_adoption_rate': 0.0, 
        'efe_calibration': 0.0,
        'temporal_stability': 0.0
    }
    
    for day_data in test_data:
        # Prediction accuracy
        predicted = agent.predict_default_behavior(day_data.observations)
        actual = day_data.actual_behavior
        metrics['prediction_accuracy'] += int(predicted == actual)
        
        # EFE calibration (confidence vs accuracy correlation)
        confidence = agent.get_prediction_confidence(day_data.observations)
        accuracy = int(predicted == actual)
        metrics['efe_calibration'] += abs(confidence - accuracy)
    
    return {k: v / len(test_data) for k, v in metrics.items()}
```

### 11.3 A/B Testing Framework
- **Control Days**: User follows natural behavior patterns
- **Intervention Days**: User receives and follows agent suggestions  
- **Metrics**: Compare end-of-day well-being, productivity, health markers
- **Statistical Testing**: Use paired t-tests or Bayesian comparison

---

## 12. Scope, Horizons, and Scaling Tips

### 12.1 Starting Scope
- **Temporal**: Daily or 3x daily decision points initially
- **State Factors**: 3-4 factors with 3-5 states each (Energy, Mood, Context, Social)
- **Observations**: 4-5 modalities with 3-4 categories each
- **Actions**: 8-12 discrete intervention options
- **Horizon**: 1-3 time steps for planning

### 12.2 Scaling Strategies
- **Gradual Expansion**: Add one new factor/modality per month
- **Hierarchical Decomposition**: Use different agents for different life domains
- **Temporal Multi-scale**: Combine daily, weekly, and monthly planning agents
- **Continuous States**: Gradually introduce continuous observations through discretization

### 12.3 Performance Optimization
- **Matrix Sparsity**: Use sparse representations for large state spaces
- **Approximate Inference**: Implement mean-field or sampling-based belief updates
- **Caching**: Store frequently accessed matrix computations
- **Batch Processing**: Update multiple users' models in parallel

### 12.4 Data Quality & Robustness
- **Missing Data Handling**: Add "unknown" observation categories
- **Outlier Detection**: Monitor for unusual observation patterns
- **Model Uncertainty**: Track and display epistemic uncertainty
- **Graceful Degradation**: Provide reasonable defaults when models fail

---

## Final Notes

This blueprint provides a comprehensive roadmap for implementing a personalized Active Inference agent using `pymdp` and your existing PostgreSQL life-log database. The approach:

1. **Leverages existing data** through thoughtful discretization and feature engineering
2. **Provides principled decision-making** via Active Inference's Expected Free Energy minimization
3. **Enables continuous learning** through online Bayesian updates
4. **Scales incrementally** from simple to sophisticated models
5. **Integrates seamlessly** with modern web architectures

The key insight is that Active Inference provides both **prediction** and **optimization** in a single framework, eliminating the need for separate recommendation systems while providing intrinsic motivation for both exploitation and exploration.

Start with Phase 1 and iterate rapidly through the validation cycle to build confidence in each component before moving to the next phase.

---

## Additional Resources

- **[Development Plan](./development-plan.md)**: Detailed 16-week implementation roadmap
- **[Copilot Instructions](./.github/copilot-instructions.md)**: AI assistant guidelines for development
- **[Database Schema](./database-schema.puml)**: Complete database design documentation

### External References

- [pymdp Documentation](https://pymdp-rtd.readthedocs.io/): Official Active Inference framework docs
- [Active Inference: A Process Theory](https://www.fil.ion.ucl.ac.uk/~karl/A%20tutorial%20on%20the%20free-energy%20framework.pdf): Theoretical foundations
- [FastAPI Documentation](https://fastapi.tiangolo.com/): Backend framework guide
- [React Documentation](https://react.dev/learn): Frontend framework guide

<!-- companion-hub-readme-deep-review -->

## Runtime requirements and configuration

| Area | Supported setup |
|---|---|
| Operating system | Linux; setup documentation targets Ubuntu 20.04 or later |
| Python | 3.8 or later |
| Resources | At least 2 GB RAM and 1 GB free disk space |
| Database | SQLite by default; PostgreSQL can be selected with `DATABASE_URL` |
| API | FastAPI runner; interactive API docs at `/docs` |

The main configuration knobs are read from `.env`: `INFERENCE_UPDATE_INTERVAL`, `PREDICTION_HORIZON`, `LEARNING_RATE`, `API_HOST`, `API_PORT`, and `LOG_LEVEL`. Keep `.env` local and use `.env.example` as the template. See [SETUP.md](SETUP.md) and [TESTING.md](TESTING.md) for the full setup and verification steps.

```mermaid
flowchart LR
  Env[.env and database] --> App[Active Inference service]
  App --> Agents[Agents and learning loop]
  Agents --> Store[(SQLite or PostgreSQL)]
  App --> API[FastAPI /api]
  API --> Client[Local client or demo]
```

