"""
Tests for database functionality
"""

import pytest
from ci_active_inference.database import get_db_session, create_tables, drop_tables
from ci_active_inference.database.dao import (
    UserDAO, PersonDAO, PlaceDAO, ThingDAO, ContentDAO, EventDAO, DataAnalyzer
)


class TestDatabase:
    """Test database operations."""
    
    def test_database_connection(self, test_db):
        """Test that we can connect to the database."""
        with get_db_session() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_table_creation(self, test_db):
        """Test that all required tables are created."""
        with get_db_session() as conn:
            cursor = conn.cursor()
            
            # Check that core tables exist
            tables = [
                'users', 'tags', 'people', 'places', 'things', 
                'content', 'events', 'people_content', 'people_events'
            ]
            
            for table in tables:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,)
                )
                result = cursor.fetchone()
                assert result is not None, f"Table {table} should exist"


class TestUserDAO:
    """Test User DAO operations."""
    
    def test_create_default_user(self, test_db):
        """Test creating a default user."""
        with get_db_session() as conn:
            user_dao = UserDAO(conn)
            user = user_dao.create_default_user()
            
            assert user['name'] == 'Demo User'
            assert user['email'] == 'demo@ci-active-inference.com'
            assert user['onboarded'] is False
    
    def test_get_all_users(self, test_user):
        """Test getting all users."""
        with get_db_session() as conn:
            user_dao = UserDAO(conn)
            users = user_dao.get_all()
            
            assert len(users) >= 1
            assert any(user['email'] == 'demo@ci-active-inference.com' for user in users)
    
    def test_get_user_by_id(self, test_user):
        """Test getting a user by ID."""
        with get_db_session() as conn:
            user_dao = UserDAO(conn)
            user = user_dao.get_by_id(test_user['id'])
            
            assert user is not None
            assert user['id'] == test_user['id']
            assert user['email'] == test_user['email']


class TestPersonDAO:
    """Test Person DAO operations."""
    
    def test_create_person(self, test_user):
        """Test creating a person."""
        with get_db_session() as conn:
            person_dao = PersonDAO(conn)
            person_data = {
                'first_name': 'Test',
                'last_name': 'Person',
                'type': 'colleague'
            }
            
            person = person_dao.create(person_data, test_user['id'])
            
            assert person['first_name'] == 'Test'
            assert person['last_name'] == 'Person'
            assert person['type'] == 'colleague'
            assert person['user_id'] == test_user['id']
    
    def test_get_all_people(self, test_user):
        """Test getting all people for a user."""
        with get_db_session() as conn:
            person_dao = PersonDAO(conn)
            
            # Create a test person
            person_data = {'first_name': 'Test', 'last_name': 'Person', 'type': 'friend'}
            person_dao.create(person_data, test_user['id'])
            
            # Get all people
            people = person_dao.get_all(user_id=test_user['id'])
            
            assert len(people) >= 1
            assert any(person['first_name'] == 'Test' for person in people)


class TestPlaceDAO:
    """Test Place DAO operations."""
    
    def test_create_place(self, test_user):
        """Test creating a place."""
        with get_db_session() as conn:
            place_dao = PlaceDAO(conn)
            place_data = {
                'name': 'Test Location',
                'type': 'office',
                'tracked': True
            }
            
            place = place_dao.create(place_data, test_user['id'])
            
            assert place['name'] == 'Test Location'
            assert place['type'] == 'office'
            assert place['tracked'] is True
            assert place['user_id'] == test_user['id']


class TestEventDAO:
    """Test Event DAO operations."""
    
    def test_create_event(self, test_user):
        """Test creating an event."""
        from datetime import datetime, timedelta
        
        with get_db_session() as conn:
            event_dao = EventDAO(conn)
            now = datetime.now()
            event_data = {
                'name': 'Test Meeting',
                'type': 'meeting',
                'description': 'A test meeting',
                'start_time': (now + timedelta(hours=1)).isoformat(),
                'end_time': (now + timedelta(hours=2)).isoformat()
            }
            
            event = event_dao.create(event_data, test_user['id'])
            
            assert event['name'] == 'Test Meeting'
            assert event['type'] == 'meeting'
            assert event['description'] == 'A test meeting'
            assert event['user_id'] == test_user['id']


class TestDataAnalyzer:
    """Test Data Analyzer functionality."""
    
    def test_get_system_state_vector(self, test_user):
        """Test getting system state vector."""
        with get_db_session() as conn:
            analyzer = DataAnalyzer(conn, test_user['id'])
            state = analyzer.get_system_state_vector()
            
            # Check that state contains expected fields
            required_fields = [
                'timestamp', 'user_id', 'active_events', 'available_resources',
                'recent_content_items', 'total_people', 'total_places', 
                'total_things', 'hour_of_day', 'day_of_week', 'week_of_year'
            ]
            
            for field in required_fields:
                assert field in state, f"State should contain {field}"
            
            assert state['user_id'] == test_user['id']
            assert isinstance(state['hour_of_day'], int)
            assert 0 <= state['hour_of_day'] <= 23
            assert isinstance(state['day_of_week'], int)
            assert 0 <= state['day_of_week'] <= 6