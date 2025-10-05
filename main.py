#!/usr/bin/env python3
"""
Main application entry point for CI Active Inference System.

This script provides command-line interface and application startup functionality.
"""

import argparse
import logging
import sys
import json
from typing import Optional
from datetime import datetime, timedelta

from ci_active_inference.database import create_tables, get_db_session
from ci_active_inference.database.dao import (
    PersonDAO, PlaceDAO, ThingDAO, ContentDAO, EventDAO, DataAnalyzer
)
from ci_active_inference.agents import CIActiveInferenceAgent, CIBehaviorOptimizer

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data in the database for testing."""
    try:
        with get_db_session() as conn:
            # Check if data already exists
            person_dao = PersonDAO(conn)
            if len(person_dao.get_all()) > 0:
                logger.info("Sample data already exists, skipping creation")
                return
            
            # Create persons
            persons_data = [
                {
                    "name": "Alice Johnson",
                    "email": "alice@company.com",
                    "role": "Data Scientist",
                    "department": "AI Research"
                },
                {
                    "name": "Bob Smith", 
                    "email": "bob@company.com",
                    "role": "Software Engineer",
                    "department": "Engineering"
                },
                {
                    "name": "Carol Wilson",
                    "email": "carol@company.com", 
                    "role": "Project Manager",
                    "department": "Operations"
                }
            ]
            
            for person_data in persons_data:
                person_dao.create(person_data)
            
            # Create places
            place_dao = PlaceDAO(conn)
            places_data = [
                {
                    "name": "Conference Room A",
                    "place_type": "meeting_room",
                    "capacity": 10
                },
                {
                    "name": "Main Office",
                    "place_type": "office",
                    "capacity": 50
                },
                {
                    "name": "Lab Space",
                    "place_type": "laboratory", 
                    "capacity": 8
                }
            ]
            
            for place_data in places_data:
                place_dao.create(place_data)
            
            # Create things
            thing_dao = ThingDAO(conn)
            things_data = [
                {
                    "name": "Laptop Dell XPS",
                    "thing_type": "equipment",
                    "status": "available"
                },
                {
                    "name": "Projector Sony",
                    "thing_type": "equipment", 
                    "status": "in_use"
                },
                {
                    "name": "AI Development Server",
                    "thing_type": "software",
                    "status": "available"
                }
            ]
            
            for thing_data in things_data:
                thing_dao.create(thing_data)
            
            # Create content
            content_dao = ContentDAO(conn)
            content_data = [
                {
                    "title": "Q4 Progress Report",
                    "content_type": "document",
                    "creator_id": 1
                },
                {
                    "title": "Team Meeting Notes", 
                    "content_type": "document",
                    "creator_id": 2
                }
            ]
            
            for content_item in content_data:
                content_dao.create(content_item)
            
            # Create events
            event_dao = EventDAO(conn)
            now = datetime.now()
            events_data = [
                {
                    "name": "Daily Standup",
                    "event_type": "meeting",
                    "start_time": (now + timedelta(hours=1)).isoformat(),
                    "end_time": (now + timedelta(hours=1.5)).isoformat(),
                    "person_id": 1,
                    "place_id": 1
                },
                {
                    "name": "Code Review",
                    "event_type": "task", 
                    "start_time": (now + timedelta(hours=2)).isoformat(),
                    "end_time": (now + timedelta(hours=3)).isoformat(),
                    "person_id": 2
                }
            ]
            
            for event_data in events_data:
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
        with get_db_session() as conn:
            analyzer = DataAnalyzer(conn)
            current_state = analyzer.get_system_state_vector()
            
            logger.info(f"Current system state: {json.dumps(current_state, indent=2)}")
            
            # Predict future states
            predictions = agent.predict_future_state(current_state, time_steps=5)
            logger.info("Future state predictions:")
            for i, pred in enumerate(predictions):
                logger.info(f"  Step {i+1}: {json.dumps(pred, indent=2)}")
            
            # Get action recommendations
            recommendations = agent.recommend_actions(current_state)
            logger.info(f"Action recommendations: {json.dumps(recommendations, indent=2)}")
            
            # Initialize optimizer
            optimizer = CIBehaviorOptimizer(agent, analyzer)
            
            # Test user optimization (if users exist)
            persons = PersonDAO(conn).get_all(limit=1)
            if persons:
                person_id = persons[0]['id']
                optimization = optimizer.optimize_user_schedule(person_id)
                logger.info(f"User optimization for person {person_id}:")
                logger.info(json.dumps(optimization, indent=2))
            
            # Test resource optimization
            resource_opt = optimizer.optimize_resource_allocation()
            logger.info("Resource optimization:")
            logger.info(json.dumps(resource_opt, indent=2))
            
        logger.info("Active Inference demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to run inference demo: {e}")
        raise


def run_training_loop():
    """Run continuous training loop for the Active Inference agent."""
    logger.info("Starting training loop...")
    
    try:
        agent = CIActiveInferenceAgent()
        
        with get_db_session() as conn:
            analyzer = DataAnalyzer(conn)
            
            # Training loop
            for epoch in range(10):  # Run for 10 iterations for demo
                # Get current system state
                current_state = analyzer.get_system_state_vector()
                observations = agent.encode_observations(current_state)
                
                # Update agent beliefs
                agent.update_beliefs(observations)
                
                # Log progress
                progress = agent.get_learning_progress()
                logger.info(f"Epoch {epoch}: {progress}")
                
                # Simulate some delay between observations
                import time
                time.sleep(0.5)
        
        logger.info("Training loop completed")
        
    except KeyboardInterrupt:
        logger.info("Training loop interrupted by user")
    except Exception as e:
        logger.error(f"Training loop failed: {e}")
        raise


def show_status():
    """Show system status and data."""
    try:
        logger.info("CI Active Inference System Status")
        logger.info("=" * 40)
        
        with get_db_session() as conn:
            # Count data
            person_dao = PersonDAO(conn)
            place_dao = PlaceDAO(conn)
            thing_dao = ThingDAO(conn)
            content_dao = ContentDAO(conn)
            event_dao = EventDAO(conn)
            
            logger.info(f"Persons: {len(person_dao.get_all())}")
            logger.info(f"Places: {len(place_dao.get_all())}")
            logger.info(f"Things: {len(thing_dao.get_all())}")
            logger.info(f"Content: {len(content_dao.get_all())}")
            logger.info(f"Events: {len(event_dao.get_all())}")
            
            # Show system state
            analyzer = DataAnalyzer(conn)
            state = analyzer.get_system_state_vector()
            logger.info("\nCurrent System State:")
            logger.info(json.dumps(state, indent=2))
            
    except Exception as e:
        logger.error(f"Failed to show status: {e}")
        raise


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="CI Active Inference System")
    parser.add_argument(
        "command",
        choices=["demo", "train", "init-db", "sample-data", "status"],
        help="Command to run"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
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
            
        elif args.command == "status":
            show_status()
            
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()