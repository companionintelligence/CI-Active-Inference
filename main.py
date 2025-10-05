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
    UserDAO, PersonDAO, PlaceDAO, ThingDAO, ContentDAO, EventDAO, DataAnalyzer
)
from ci_active_inference.agents import CIActiveInferenceAgent, CIBehaviorOptimizer

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data in the database for testing."""
    try:
        with get_db_session() as conn:
            # Create or get default user
            user_dao = UserDAO(conn)
            users = user_dao.get_all()
            if len(users) == 0:
                user = user_dao.create_default_user()
                user_id = user['id']
                logger.info(f"Created default user: {user['email']}")
            else:
                user = users[0]
                user_id = user['id']
                logger.info(f"Using existing user: {user['email']}")
            
            # Check if data already exists
            person_dao = PersonDAO(conn)
            if len(person_dao.get_all(user_id=user_id)) > 0:
                logger.info("Sample data already exists, skipping creation")
                return user_id
            
            # Create people
            people_data = [
                {
                    "first_name": "Alice",
                    "last_name": "Johnson",
                    "type": "colleague"
                },
                {
                    "first_name": "Bob",
                    "last_name": "Smith",
                    "type": "colleague"
                },
                {
                    "first_name": "Carol",
                    "last_name": "Wilson",
                    "type": "manager"
                }
            ]
            
            created_people = []
            for person_data in people_data:
                person = person_dao.create(person_data, user_id)
                created_people.append(person)
            
            # Create places
            place_dao = PlaceDAO(conn)
            places_data = [
                {
                    "name": "Conference Room A",
                    "type": "meeting_room"
                },
                {
                    "name": "Main Office",
                    "type": "office"
                },
                {
                    "name": "Lab Space",
                    "type": "laboratory"
                }
            ]
            
            created_places = []
            for place_data in places_data:
                place = place_dao.create(place_data, user_id)
                created_places.append(place)
            
            # Create things
            thing_dao = ThingDAO(conn)
            things_data = [
                {
                    "name": "Laptop Dell XPS",
                    "type": "equipment",
                    "description": "Development laptop"
                },
                {
                    "name": "Projector Sony",
                    "type": "equipment",
                    "description": "Meeting room projector"
                },
                {
                    "name": "AI Development Server",
                    "type": "software",
                    "description": "Server for AI model training"
                }
            ]
            
            created_things = []
            for thing_data in things_data:
                thing = thing_dao.create(thing_data, user_id)
                created_things.append(thing)
            
            # Create content
            content_dao = ContentDAO(conn)
            content_data = [
                {
                    "title": "Q4 Progress Report",
                    "type": "document",
                    "text": "Quarterly progress report content"
                },
                {
                    "title": "Team Meeting Notes",
                    "type": "document", 
                    "text": "Notes from team meeting"
                }
            ]
            
            created_content = []
            for content_item in content_data:
                content = content_dao.create(content_item, user_id)
                created_content.append(content)
            
            # Create events
            event_dao = EventDAO(conn)
            now = datetime.now()
            events_data = [
                {
                    "name": "Daily Standup",
                    "type": "meeting",
                    "description": "Daily team standup meeting",
                    "start_time": (now + timedelta(hours=1)).isoformat(),
                    "end_time": (now + timedelta(hours=1.5)).isoformat()
                },
                {
                    "name": "Code Review",
                    "type": "task",
                    "description": "Review code changes",
                    "start_time": (now + timedelta(hours=2)).isoformat(),
                    "end_time": (now + timedelta(hours=3)).isoformat()
                }
            ]
            
            for event_data in events_data:
                event_dao.create(event_data, user_id)
            
            logger.info("Sample data created successfully")
            return user_id
            
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
            # Get or create default user
            user_dao = UserDAO(conn)
            users = user_dao.get_all()
            if users:
                user_id = users[0]['id']
            else:
                user = user_dao.create_default_user()
                user_id = user['id']
            
            analyzer = DataAnalyzer(conn, user_id)
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
            
            # Test user optimization (if people exist)
            person_dao = PersonDAO(conn)
            people = person_dao.get_all(user_id=user_id, limit=1)
            if people:
                person_id = people[0]['id']
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
            # Get or create default user
            user_dao = UserDAO(conn)
            users = user_dao.get_all()
            if users:
                user_id = users[0]['id']
            else:
                user = user_dao.create_default_user()
                user_id = user['id']
            
            analyzer = DataAnalyzer(conn, user_id)
            
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
            # Get or create default user
            user_dao = UserDAO(conn)
            users = user_dao.get_all()
            if users:
                user_id = users[0]['id']
                user_email = users[0]['email']
            else:
                user = user_dao.create_default_user()
                user_id = user['id']
                user_email = user['email']
            
            logger.info(f"Current User: {user_email}")
            
            # Count data
            person_dao = PersonDAO(conn)
            place_dao = PlaceDAO(conn)
            thing_dao = ThingDAO(conn)
            content_dao = ContentDAO(conn)
            event_dao = EventDAO(conn)
            
            logger.info(f"People: {len(person_dao.get_all(user_id=user_id))}")
            logger.info(f"Places: {len(place_dao.get_all(user_id=user_id))}")
            logger.info(f"Things: {len(thing_dao.get_all(user_id=user_id))}")
            logger.info(f"Content: {len(content_dao.get_all(user_id=user_id))}")
            logger.info(f"Events: {len(event_dao.get_all(user_id=user_id))}")
            
            # Show system state
            analyzer = DataAnalyzer(conn, user_id)
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
            user_id = create_sample_data()
            logger.info(f"Sample data created successfully for user: {user_id}")
            
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