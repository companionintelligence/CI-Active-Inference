"""
FastAPI endpoints for PersonalLifeAgent.

This module provides REST API endpoints for the PersonalLifeAgent
as specified in the README Phase 2 requirements.
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from ..agents.personal_life_agent import PersonalLifeAgent
from ..database import get_db_session

logger = logging.getLogger(__name__)

# Global agent cache
user_agents: Dict[str, PersonalLifeAgent] = {}

app = FastAPI(title="PersonalLifeAgent API", version="1.0.0")


class ObservationInput(BaseModel):
    """Input schema for observations."""
    digital_behavior: str
    location_context: str
    social_context: str
    content_type: str
    timestamp: Optional[str] = None


class ActionOutcome(BaseModel):
    """Input schema for action outcomes."""
    executed_action: str
    next_observations: ObservationInput


class MultiStepPlanRequest(BaseModel):
    """Request for multi-step planning."""
    digital_behavior: str
    location_context: str  
    social_context: str
    timestamp: Optional[str] = None
    horizon: int = 10


class ExplanationRequest(BaseModel):
    """Request for detailed explanation."""
    digital_behavior: str
    location_context: str
    social_context: str
    suggested_action: str
    predicted_default: str
    confidence: float
    timestamp: Optional[str] = None


class DataQualityRequest(BaseModel):
    """Request for data quality assessment."""
    data_batch: list
    batch_metadata: Optional[dict] = None
    """Response schema for agent predictions."""
    predicted_default: str
    suggested_action: str
    confidence: float
    explanation: str
    efe_breakdown: Dict[str, float]
    current_state_beliefs: list


class ModelStats(BaseModel):
    """Model statistics response schema."""
    total_predictions: int
    accuracy_last_week: float
    uncertainty_level: float
    last_updated: Optional[str]


def get_user_agent(user_id: str) -> PersonalLifeAgent:
    """
    Get or create agent instance for user.
    
    Args:
        user_id: User ID
        
    Returns:
        PersonalLifeAgent instance
    """
    if user_id not in user_agents:
        # Load from database or create new
        user_agents[user_id] = PersonalLifeAgent.load_or_create(user_id)
    return user_agents[user_id]


@app.post("/agent/{user_id}/predict", response_model=AgentResponse)
async def predict_and_suggest(user_id: str, observations: ObservationInput):
    """
    Get predictions and action suggestions for user.
    
    This endpoint implements the main predict_and_suggest pipeline
    as specified in the README.
    """
    try:
        agent = get_user_agent(user_id)
        
        # Convert Pydantic model to dict
        obs_dict = observations.dict()
        if 'timestamp' in obs_dict:
            del obs_dict['timestamp']  # Remove timestamp for processing
        
        # Get prediction
        result = agent.predict_and_suggest(obs_dict)
        
        # Handle error cases
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Format response
        explanation = _generate_explanation(result)
        
        return AgentResponse(
            predicted_default=result['predicted_default'],
            suggested_action=result['suggested_action'],
            confidence=result['confidence'],
            explanation=explanation,
            efe_breakdown=result['efe_breakdown'],
            current_state_beliefs=result['current_state_beliefs']
        )
        
    except Exception as e:
        logger.error(f"Error in predict_and_suggest for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/{user_id}/update")
async def update_agent(user_id: str, outcome: ActionOutcome):
    """
    Update agent with action outcome for learning.
    
    This endpoint allows the agent to learn from actual outcomes
    and improve its predictions over time.
    """
    try:
        agent = get_user_agent(user_id)
        
        # Convert observations to dict
        next_obs_dict = outcome.next_observations.dict()
        if 'timestamp' in next_obs_dict:
            del next_obs_dict['timestamp']
        
        # Update agent
        agent.update_from_outcome(outcome.executed_action, next_obs_dict)
        
        return {"status": "success", "message": "Agent updated with outcome"}
        
    except Exception as e:
        logger.error(f"Error updating agent for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/{user_id}/model-stats", response_model=ModelStats)
async def get_model_statistics(user_id: str):
    """
    Return model performance and uncertainty metrics.
    
    This endpoint provides insights into the agent's performance
    and confidence levels.
    """
    try:
        agent = get_user_agent(user_id)
        stats = agent.get_model_statistics()
        
        return ModelStats(
            total_predictions=stats['total_predictions'],
            accuracy_last_week=stats['accuracy_last_week'],
            uncertainty_level=stats['uncertainty_level'],
            last_updated=stats['last_updated']
        )
        
    except Exception as e:
        logger.error(f"Error getting model stats for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/{user_id}/reinitialize")
async def reinitialize_agent(user_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Reinitialize agent with fresh data.
    
    This endpoint allows rebuilding the agent's matrices
    with updated data from the database.
    """
    try:
        # Remove from cache to force recreation
        if user_id in user_agents:
            del user_agents[user_id]
        
        # Create new agent
        agent = PersonalLifeAgent(user_id)
        
        # Initialize with specified date range
        success = agent.initialize_from_data(start_date, end_date)
        
        if success:
            user_agents[user_id] = agent
            return {"status": "success", "message": "Agent reinitialized successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize agent from data")
            
    except Exception as e:
        logger.error(f"Error reinitializing agent for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/{user_id}/state")
async def get_agent_state(user_id: str):
    """
    Get current agent state and configuration.
    
    Returns information about the agent's current state,
    matrix dimensions, and configuration.
    """
    try:
        agent = get_user_agent(user_id)
        
        return {
            "user_id": agent.user_id,
            "initialized": agent.agent is not None,
            "last_update": agent.last_update.isoformat() if agent.last_update else None,
            "prediction_count": agent.prediction_count,
            "matrix_dimensions": {
                "num_states": agent.num_states,
                "num_obs": agent.num_obs,
                "num_actions": agent.num_actions
            },
            "matrices_built": {
                "A": agent.A is not None,
                "B": agent.B is not None,
                "C": agent.C is not None,
                "D": agent.D is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting agent state for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_explanation(result: Dict[str, Any]) -> str:
    """
    Generate human-readable explanation for prediction.
    
    Args:
        result: Prediction result from agent
        
    Returns:
        Human-readable explanation string
    """
    try:
        confidence = result.get('confidence', 0.5)
        suggested_action = result.get('suggested_action', 'no_action')
        predicted_default = result.get('predicted_default', 'unknown')
        efe = result.get('efe_breakdown', {})
        
        # Generate confidence description
        if confidence > 0.8:
            conf_desc = "high confidence"
        elif confidence > 0.6:
            conf_desc = "moderate confidence"
        else:
            conf_desc = "low confidence"
        
        # Generate EFE description
        risk = efe.get('risk', 0.5)
        ambiguity = efe.get('ambiguity', 0.5)
        
        efe_desc = ""
        if risk > 0.6:
            efe_desc += "High preference mismatch risk. "
        if ambiguity > 0.6:
            efe_desc += "High uncertainty in current state. "
        
        explanation = (
            f"Based on current observations, you would likely default to '{predicted_default}'. "
            f"However, I suggest '{suggested_action}' for better outcomes "
            f"(predicted with {conf_desc}). {efe_desc}"
        )
        
        return explanation.strip()
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return f"Suggested action: {result.get('suggested_action', 'no_action')}"


# Additional request models for advanced features
class MultiStepPlanRequest(BaseModel):
    """Request for multi-step planning."""
    digital_behavior: str
    location_context: str  
    social_context: str
    timestamp: Optional[str] = None
    horizon: int = 10


class ExplanationRequest(BaseModel):
    """Request for detailed explanation."""
    digital_behavior: str
    location_context: str
    social_context: str
    suggested_action: str
    predicted_default: str
    confidence: float
    timestamp: Optional[str] = None


class DataQualityRequest(BaseModel):
    """Request for data quality assessment."""
    data_batch: list
    batch_metadata: Optional[dict] = None


@app.post("/agent/{user_id}/multi-step-plan")
async def get_multi_step_plan(user_id: str, request: MultiStepPlanRequest):
    """
    Get sophisticated multi-step plan for user.
    
    This endpoint uses advanced planning algorithms to generate
    longer-horizon plans with policy trees.
    """
    try:
        agent = get_user_agent(user_id)
        
        # Convert request to state dict
        current_state = {
            "digital_behavior": request.digital_behavior,
            "location_context": request.location_context,
            "social_context": request.social_context,
            "timestamp": request.timestamp or datetime.now().isoformat()
        }
        
        # Get multi-step plan
        plan = agent.get_multi_step_plan(current_state, request.horizon)
        
        return plan
    
    except Exception as e:
        logger.error(f"Error in multi-step planning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/{user_id}/detailed-explanation")
async def get_detailed_explanation(user_id: str, request: ExplanationRequest):
    """
    Get detailed explanation for agent recommendation.
    
    This endpoint provides comprehensive explanations including
    EFE decomposition and causal analysis.
    """
    try:
        agent = get_user_agent(user_id)
        
        # Convert request to state dict
        current_state = {
            "digital_behavior": request.digital_behavior,
            "location_context": request.location_context,
            "social_context": request.social_context,
            "timestamp": request.timestamp or datetime.now().isoformat()
        }
        
        # Create suggestion result for explanation
        suggestion_result = {
            "suggested_action": request.suggested_action,
            "predicted_default": request.predicted_default,
            "confidence": request.confidence,
            "efe_breakdown": {"risk": 0.5, "ambiguity": 0.5}  # Would come from actual agent
        }
        
        # Get detailed explanation
        explanation = agent.get_detailed_explanation(current_state, suggestion_result)
        
        return explanation
    
    except Exception as e:
        logger.error(f"Error in detailed explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitoring/data-quality")
async def assess_data_quality(request: DataQualityRequest):
    """
    Assess data quality for a batch of data.
    
    This endpoint provides comprehensive data quality assessment
    including completeness, consistency, accuracy, and timeliness.
    """
    try:
        from ..monitoring.data_quality_monitor import DataQualityMonitor
        monitor = DataQualityMonitor()
        
        # Assess data quality
        assessment = monitor.assess_data_quality(
            request.data_batch, 
            request.batch_metadata
        )
        
        return assessment
    
    except Exception as e:
        logger.error(f"Error in data quality assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitoring/anomaly-detection")
async def detect_anomalies(request: DataQualityRequest):
    """
    Detect anomalies in data batch.
    
    This endpoint uses statistical and pattern-based methods
    to identify unusual data patterns.
    """
    try:
        from ..monitoring.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector()
        
        # Detect anomalies
        anomalies = detector.detect_anomalies(request.data_batch)
        
        # Convert anomalies to serializable format
        anomaly_data = []
        for anomaly in anomalies:
            anomaly_data.append({
                "timestamp": anomaly.timestamp.isoformat(),
                "anomaly_type": anomaly.anomaly_type,
                "severity": anomaly.severity,
                "description": anomaly.description,
                "confidence": anomaly.confidence,
                "affected_entity": anomaly.affected_entity,
                "data": anomaly.data
            })
        
        return {
            "anomalies": anomaly_data,
            "total_anomalies": len(anomaly_data),
            "assessment_time": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ab-testing/experiments")
async def list_experiments():
    """
    List active and completed A/B testing experiments.
    """
    try:
        from ..ab_testing.ab_test_runner import ABTestRunner
        runner = ABTestRunner()
        experiments = runner.experiment_manager.list_experiments()
        
        return {
            "experiments": experiments,
            "total_count": len(experiments)
        }
    
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ab-testing/experiments")
async def create_experiment(experiment_config: dict):
    """
    Create a new A/B testing experiment for preference variations.
    """
    try:
        from ..ab_testing.ab_test_runner import ABTestRunner
        runner = ABTestRunner()
        
        experiment_id = runner.create_preference_experiment(
            experiment_name=experiment_config.get("name", "New Experiment"),
            duration_days=experiment_config.get("duration_days", 7),
            variant_types=experiment_config.get("variant_types", ["health_focused", "productivity_focused"]),
            success_metrics=experiment_config.get("success_metrics", ["user_satisfaction", "recommendation_adoption"])
        )
        
        return {
            "experiment_id": experiment_id,
            "status": "created",
            "created_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ab-testing/experiments/{experiment_id}")
async def get_experiment_status(experiment_id: str):
    """
    Get status and results of an A/B testing experiment.
    """
    try:
        from ..ab_testing.ab_test_runner import ABTestRunner
        runner = ABTestRunner()
        status = runner.get_experiment_summary(experiment_id)
        
        return status
    
    except Exception as e:
        logger.error(f"Error getting experiment status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "PersonalLifeAgent API"}