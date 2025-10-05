#!/usr/bin/env python3
"""
Main application entry point for CI Active Inference System.

This script provides command-line interface and application startup functionality.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

import uvicorn
from sqlalchemy.orm import Session

from ci_active_inference.utils import Config, setup_logging, generate_sample_data
from ci_active_inference.database import create_tables, get_db_session
from ci_active_inference.database.dao import (
    PersonDAO, PlaceDAO, ThingDAO, ContentDAO, EventDAO, DataAnalyzer
)
from ci_active_inference.agents import CIActiveInferenceAgent, CIBehaviorOptimizer
from ci_active_inference.api import app

logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data in the database for testing."""
    try:
        with get_db_session() as db:
            # Check if data already exists
            person_dao = PersonDAO(db)
            if len(person_dao.get_all()) > 0:
                logger.info("Sample data already exists, skipping creation")
                return
            
            sample_data = generate_sample_data()
            
            # Create persons
            for person_data in sample_data["persons"]:
                person_dao.create(person_data)
            
            # Create places
            place_dao = PlaceDAO(db)
            for place_data in sample_data["places"]:
                place_dao.create(place_data)
            
            # Create things
            thing_dao = ThingDAO(db)
            for thing_data in sample_data["things"]:
                thing_dao.create(thing_data)
            
            # Create content
            content_dao = ContentDAO(db)
            for content_data in sample_data["content"]:
                content_dao.create(content_data)
            
            # Create events
            event_dao = EventDAO(db)
            for event_data in sample_data["events"]:
                event_dao.create(event_data)
            
            logger.info("Sample data created successfully")
            
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        raise


def run_inference_demo():
    """Run a demonstration of the Active Inference capabilities."""
    try:
        logger.info("Starting Active Inference demonstration...")
        
        # Initialize agent
        agent = CIActiveInferenceAgent()
        
        # Get system state
        with get_db_session() as db:
            analyzer = DataAnalyzer(db)
            current_state = analyzer.get_system_state_vector()
            
            logger.info(f"Current system state: {current_state}")
            
            # Predict future states
            predictions = agent.predict_future_state(current_state, time_steps=5)
            logger.info("Future state predictions:")
            for i, pred in enumerate(predictions):
                logger.info(f"  Step {i+1}: {pred}")
            
            # Get action recommendations
            recommendations = agent.recommend_actions(current_state)
            logger.info(f"Action recommendations: {recommendations}")
            
            # Initialize optimizer
            optimizer = CIBehaviorOptimizer(agent, analyzer)
            
            # Test user optimization (if users exist)
            persons = PersonDAO(db).get_all(limit=1)
            if persons:
                person_id = persons[0].id
                optimization = optimizer.optimize_user_schedule(person_id)
                logger.info(f"User optimization for person {person_id}: {optimization}")
            
            # Test resource optimization
            resource_opt = optimizer.optimize_resource_allocation()
            logger.info(f"Resource optimization: {resource_opt}")
            
        logger.info("Active Inference demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to run inference demo: {e}")
        raise


def run_api_server():
    """Run the FastAPI server."""
    logger.info(f"Starting API server on {Config.API_HOST}:{Config.API_PORT}")
    uvicorn.run(
        "ci_active_inference.api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.DEBUG,
        log_level=Config.LOG_LEVEL.lower()
    )


def run_training_loop():
    """Run continuous training loop for the Active Inference agent."""
    logger.info("Starting training loop...")
    
    try:
        agent = CIActiveInferenceAgent()
        
        with get_db_session() as db:
            analyzer = DataAnalyzer(db)
            
            # Training loop
            for epoch in range(100):  # Run for 100 iterations
                # Get current system state
                current_state = analyzer.get_system_state_vector()
                observations = agent.encode_observations(current_state)
                
                # Update agent beliefs
                agent.update_beliefs(observations)
                
                # Log progress
                if epoch % 10 == 0:
                    progress = agent.get_learning_progress()
                    logger.info(f"Epoch {epoch}: {progress}")
                
                # Simulate some delay between observations
                import time
                time.sleep(1)
        
        logger.info("Training loop completed")
        
    except KeyboardInterrupt:
        logger.info("Training loop interrupted by user")
    except Exception as e:
        logger.error(f"Training loop failed: {e}")
        raise


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="CI Active Inference System")
    parser.add_argument(
        "command",
        choices=["server", "demo", "train", "init-db", "sample-data"],
        help="Command to run"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.debug:
        Config.DEBUG = True
    Config.LOG_LEVEL = args.log_level
    
    # Setup logging
    setup_logging()
    
    try:
        if args.command == "init-db":
            logger.info("Initializing database...")
            create_tables()
            logger.info("Database initialized successfully")
            
        elif args.command == "sample-data":
            logger.info("Creating sample data...")
            create_tables()  # Ensure tables exist
            create_sample_data()
            logger.info("Sample data created successfully")
            
        elif args.command == "server":
            # Ensure database is initialized
            create_tables()
            run_api_server()
            
        elif args.command == "demo":
            # Ensure database and sample data exist
            create_tables()
            create_sample_data()
            run_inference_demo()
            
        elif args.command == "train":
            # Ensure database and sample data exist
            create_tables()
            create_sample_data()
            run_training_loop()
            
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()