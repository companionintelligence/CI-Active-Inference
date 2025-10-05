"""FastAPI application for CI Active Inference system."""

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ci_active_inference.database import get_db, create_tables
from ci_active_inference.database.dao import (
    PersonDAO, PlaceDAO, ThingDAO, ContentDAO, EventDAO, DataAnalyzer
)
from ci_active_inference.agents import CIActiveInferenceAgent, CIBehaviorOptimizer
from ci_active_inference.models import (
    PersonSchema, PlaceSchema, ThingSchema, ContentSchema, EventSchema
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CI Active Inference System",
    description="Active Inference system for predicting and optimizing behavior in CI environments",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance (in production, use proper dependency injection)
agent = None
optimizer = None


def get_agent() -> CIActiveInferenceAgent:
    """Get or create the Active Inference agent."""
    global agent
    if agent is None:
        agent = CIActiveInferenceAgent()
    return agent


def get_optimizer(db: Session = Depends(get_db)) -> CIBehaviorOptimizer:
    """Get or create the behavior optimizer."""
    global optimizer
    if optimizer is None:
        data_analyzer = DataAnalyzer(db)
        optimizer = CIBehaviorOptimizer(get_agent(), data_analyzer)
    return optimizer


@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    try:
        create_tables()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CI Active Inference System",
        "version": "0.1.0",
        "status": "running"
    }


# Person endpoints
@app.post("/persons/", response_model=PersonSchema)
async def create_person(person: PersonSchema, db: Session = Depends(get_db)):
    """Create a new person."""
    dao = PersonDAO(db)
    return dao.create(person.dict(exclude={"id"}))


@app.get("/persons/", response_model=List[PersonSchema])
async def list_persons(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List persons."""
    dao = PersonDAO(db)
    return dao.get_all(skip=skip, limit=limit)


@app.get("/persons/{person_id}", response_model=PersonSchema)
async def get_person(person_id: int, db: Session = Depends(get_db)):
    """Get person by ID."""
    dao = PersonDAO(db)
    person = dao.get_by_id(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@app.get("/persons/{person_id}/patterns")
async def get_person_patterns(person_id: int, days: int = 30, db: Session = Depends(get_db)):
    """Get activity patterns for a person."""
    analyzer = DataAnalyzer(db)
    patterns = analyzer.get_user_activity_patterns(person_id, days)
    return patterns


# Place endpoints
@app.post("/places/", response_model=PlaceSchema)
async def create_place(place: PlaceSchema, db: Session = Depends(get_db)):
    """Create a new place."""
    dao = PlaceDAO(db)
    return dao.create(place.dict(exclude={"id"}))


@app.get("/places/", response_model=List[PlaceSchema])
async def list_places(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List places."""
    dao = PlaceDAO(db)
    return dao.get_all(skip=skip, limit=limit)


@app.get("/places/{place_id}/utilization")
async def get_place_utilization(place_id: int, days: int = 30, db: Session = Depends(get_db)):
    """Get utilization patterns for a place."""
    analyzer = DataAnalyzer(db)
    utilization = analyzer.get_place_utilization(place_id, days)
    return utilization


# Thing endpoints
@app.post("/things/", response_model=ThingSchema)
async def create_thing(thing: ThingSchema, db: Session = Depends(get_db)):
    """Create a new thing."""
    dao = ThingDAO(db)
    return dao.create(thing.dict(exclude={"id"}))


@app.get("/things/", response_model=List[ThingSchema])
async def list_things(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List things."""
    dao = ThingDAO(db)
    return dao.get_all(skip=skip, limit=limit)


@app.get("/things/available", response_model=List[ThingSchema])
async def list_available_things(db: Session = Depends(get_db)):
    """List available things."""
    dao = ThingDAO(db)
    return dao.get_available()


# Content endpoints
@app.post("/content/", response_model=ContentSchema)
async def create_content(content: ContentSchema, db: Session = Depends(get_db)):
    """Create new content."""
    dao = ContentDAO(db)
    return dao.create(content.dict(exclude={"id"}))


@app.get("/content/", response_model=List[ContentSchema])
async def list_content(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List content."""
    dao = ContentDAO(db)
    return dao.get_all(skip=skip, limit=limit)


@app.get("/content/recent", response_model=List[ContentSchema])
async def list_recent_content(days: int = 7, db: Session = Depends(get_db)):
    """List recent content."""
    dao = ContentDAO(db)
    return dao.get_recent(days=days)


# Event endpoints
@app.post("/events/", response_model=EventSchema)
async def create_event(event: EventSchema, db: Session = Depends(get_db)):
    """Create a new event."""
    dao = EventDAO(db)
    return dao.create(event.dict(exclude={"id"}))


@app.get("/events/", response_model=List[EventSchema])
async def list_events(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List events."""
    dao = EventDAO(db)
    return dao.get_all(skip=skip, limit=limit)


@app.get("/events/upcoming", response_model=List[EventSchema])
async def list_upcoming_events(days: int = 7, db: Session = Depends(get_db)):
    """List upcoming events."""
    dao = EventDAO(db)
    return dao.get_upcoming(days=days)


@app.get("/events/current", response_model=List[EventSchema])
async def list_current_events(db: Session = Depends(get_db)):
    """List currently active events."""
    dao = EventDAO(db)
    return dao.get_current()


# Active Inference endpoints
@app.get("/inference/system-state")
async def get_system_state(db: Session = Depends(get_db)):
    """Get current system state vector."""
    analyzer = DataAnalyzer(db)
    state = analyzer.get_system_state_vector()
    return state


@app.post("/inference/predict")
async def predict_future_states(
    time_steps: int = 5,
    db: Session = Depends(get_db),
    agent: CIActiveInferenceAgent = Depends(get_agent)
):
    """Predict future system states."""
    analyzer = DataAnalyzer(db)
    current_state = analyzer.get_system_state_vector()
    
    predictions = agent.predict_future_state(current_state, time_steps)
    
    return {
        "current_state": current_state,
        "predictions": predictions,
        "time_steps": time_steps,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/inference/recommend-actions")
async def recommend_actions(
    db: Session = Depends(get_db),
    agent: CIActiveInferenceAgent = Depends(get_agent)
):
    """Get action recommendations based on current state."""
    analyzer = DataAnalyzer(db)
    current_state = analyzer.get_system_state_vector()
    
    recommendations = agent.recommend_actions(current_state)
    
    return recommendations


@app.post("/inference/update-beliefs")
async def update_agent_beliefs(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    agent: CIActiveInferenceAgent = Depends(get_agent)
):
    """Update agent beliefs with current observations."""
    analyzer = DataAnalyzer(db)
    current_state = analyzer.get_system_state_vector()
    observations = agent.encode_observations(current_state)
    
    background_tasks.add_task(agent.update_beliefs, observations)
    
    return {
        "message": "Belief update scheduled",
        "observations": observations,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/inference/learning-progress")
async def get_learning_progress(agent: CIActiveInferenceAgent = Depends(get_agent)):
    """Get agent learning progress metrics."""
    progress = agent.get_learning_progress()
    return progress


# Optimization endpoints
@app.post("/optimization/user-schedule/{person_id}")
async def optimize_user_schedule(
    person_id: int,
    db: Session = Depends(get_db),
    optimizer: CIBehaviorOptimizer = Depends(get_optimizer)
):
    """Optimize schedule for a specific user."""
    result = optimizer.optimize_user_schedule(person_id)
    return result


@app.post("/optimization/resource-allocation")
async def optimize_resource_allocation(
    db: Session = Depends(get_db),
    optimizer: CIBehaviorOptimizer = Depends(get_optimizer)
):
    """Optimize system-wide resource allocation."""
    result = optimizer.optimize_resource_allocation()
    return result


# System health endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0"
    }


@app.get("/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    """Get system metrics."""
    analyzer = DataAnalyzer(db)
    state = analyzer.get_system_state_vector()
    
    # Add some computed metrics
    metrics = {
        "system_state": state,
        "api_status": "operational",
        "database_status": "connected",
        "agent_status": "active" if agent else "inactive"
    }
    
    return metrics


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)