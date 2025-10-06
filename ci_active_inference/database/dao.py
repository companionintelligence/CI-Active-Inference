"""Data Access Object (DAO) classes for database operations using CI-Server schema."""

import json
import psycopg2
import psycopg2.extras
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from ci_active_inference.database import generate_uuid

logger = logging.getLogger(__name__)


class BaseDAO:
    """Base Data Access Object with common operations."""
    
    def __init__(self, conn: psycopg2.extensions.connection, table_name: str):
        self.conn = conn
        self.table_name = table_name
        self.cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    def create(self, obj_data: dict, user_id: str = None) -> Dict[str, Any]:
        """Create a new object."""
        # Generate UUID for primary key
        if 'id' not in obj_data or obj_data['id'] is None:
            obj_data['id'] = generate_uuid()
        
        # Add user_id if provided and not already set
        if user_id and 'user_id' not in obj_data:
            obj_data['user_id'] = user_id
        
        # Remove None values
        obj_data = {k: v for k, v in obj_data.items() if v is not None}
        
        # Convert JSON fields to JSONB if present
        json_fields = ['source_data', 'geolocation', 'location_boundary']
        for field in json_fields:
            if field in obj_data and isinstance(obj_data[field], (dict, list)):
                # PostgreSQL will handle JSON serialization automatically
                pass
        
        # Build INSERT query with placeholders for PostgreSQL
        columns = list(obj_data.keys())
        placeholders = ['%s' for _ in columns]
        query = f"INSERT INTO {self.table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)}) RETURNING *"
        
        self.cursor.execute(query, list(obj_data.values()))
        result = self.cursor.fetchone()
        return dict(result) if result else None
    
    def get_by_id(self, obj_id: str) -> Optional[Dict[str, Any]]:
        """Get object by ID."""
        self.cursor.execute(f"SELECT * FROM {self.table_name} WHERE id = %s", (obj_id,))
        row = self.cursor.fetchone()
        return dict(row) if row else None
    
    def get_all(self, user_id: str = None, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all objects with pagination, optionally filtered by user."""
        if user_id:
            self.cursor.execute(
                f"SELECT * FROM {self.table_name} WHERE user_id = %s LIMIT %s OFFSET %s", 
                (user_id, limit, skip)
            )
        else:
            self.cursor.execute(f"SELECT * FROM {self.table_name} LIMIT %s OFFSET %s", (limit, skip))
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    
    def update(self, obj_id: str, obj_data: dict) -> Optional[Dict[str, Any]]:
        """Update an object."""
        # Remove None values and id
        obj_data = {k: v for k, v in obj_data.items() if v is not None and k != 'id'}
        
        if not obj_data:
            return self.get_by_id(obj_id)
        
        # Convert JSON fields if present (PostgreSQL handles JSONB automatically)
        json_fields = ['source_data', 'geolocation', 'location_boundary']
        for field in json_fields:
            if field in obj_data and isinstance(obj_data[field], (dict, list)):
                # PostgreSQL will handle JSON serialization automatically
                pass
        
        # Add updated_at timestamp
        obj_data['updated_at'] = datetime.now()
        
        # Build UPDATE query with PostgreSQL placeholders
        set_clauses = [f"{key} = %s" for key in obj_data.keys()]
        query = f"UPDATE {self.table_name} SET {', '.join(set_clauses)} WHERE id = %s RETURNING *"
        
        self.cursor.execute(query, list(obj_data.values()) + [obj_id])
        result = self.cursor.fetchone()
        return dict(result) if result else None
    
    def delete(self, obj_id: str) -> bool:
        """Delete an object."""
        self.cursor.execute(f"DELETE FROM {self.table_name} WHERE id = %s", (obj_id,))
        return self.cursor.rowcount > 0
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert PostgreSQL row to dictionary (not needed with RealDictCursor but kept for compatibility)."""
        if not row:
            return None
        
        # RealDictCursor already returns dict-like objects, so we just convert to regular dict
        return dict(row)


class UserDAO(BaseDAO):
    """Data Access Object for User operations."""
    
    def __init__(self, conn: psycopg2.extensions.connection):
        super().__init__(conn, 'users')
    
    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email."""
        self.cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = self.cursor.fetchone()
        return dict(row) if row else None
    
    def create_default_user(self) -> Dict[str, Any]:
        """Create a default user for demo purposes."""
        user_data = {
            'name': 'Demo User',
            'email': 'demo@ci-active-inference.com',
            'password': 'demo_password',
            'onboarded': True
        }
        return self.create(user_data)


class PersonDAO(BaseDAO):
    """Data Access Object for Person operations."""
    
    def __init__(self, conn: psycopg2.extensions.connection):
        super().__init__(conn, 'people')
    
    def get_by_name(self, first_name: str, last_name: str = None, user_id: str = None) -> List[Dict[str, Any]]:
        """Get people by name."""
        if last_name:
            query = "SELECT * FROM people WHERE first_name = %s AND last_name = %s"
            params = [first_name, last_name]
        else:
            query = "SELECT * FROM people WHERE first_name = %s"
            params = [first_name]
        
        if user_id:
            query += " AND user_id = %s"
            params.append(user_id)
        
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]


class PlaceDAO(BaseDAO):
    """Data Access Object for Place operations."""
    
    def __init__(self, conn: psycopg2.extensions.connection):
        super().__init__(conn, 'places')
    
    def get_by_type(self, place_type: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get places by type."""
        if user_id:
            self.cursor.execute("SELECT * FROM places WHERE type = %s AND user_id = %s", (place_type, user_id))
        else:
            self.cursor.execute("SELECT * FROM places WHERE type = %s", (place_type,))
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_by_name(self, name: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get places by name."""
        if user_id:
            self.cursor.execute("SELECT * FROM places WHERE name ILIKE %s AND user_id = %s", (f"%{name}%", user_id))
        else:
            self.cursor.execute("SELECT * FROM places WHERE name ILIKE %s", (f"%{name}%",))
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]


class ThingDAO(BaseDAO):
    """Data Access Object for Thing operations."""
    
    def __init__(self, conn: psycopg2.extensions.connection):
        super().__init__(conn, 'things')
    
    def get_by_type(self, thing_type: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get things by type."""
        if user_id:
            self.cursor.execute("SELECT * FROM things WHERE type = %s AND user_id = %s", (thing_type, user_id))
        else:
            self.cursor.execute("SELECT * FROM things WHERE type = %s", (thing_type,))
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_by_event(self, event_id: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get things associated with an event."""
        if user_id:
            self.cursor.execute(
                "SELECT * FROM things WHERE event_id = %s AND user_id = %s", 
                (event_id, user_id)
            )
        else:
            self.cursor.execute("SELECT * FROM things WHERE event_id = %s", (event_id,))
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]


class ContentDAO(BaseDAO):
    """Data Access Object for Content operations."""
    
    def __init__(self, conn: psycopg2.extensions.connection):
        super().__init__(conn, 'content')
    
    def get_by_type(self, content_type: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get content by type."""
        if user_id:
            self.cursor.execute("SELECT * FROM content WHERE type = %s AND user_id = %s", (content_type, user_id))
        else:
            self.cursor.execute("SELECT * FROM content WHERE type = %s", (content_type,))
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    
    def search_by_title(self, title_part: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Search content by title."""
        if user_id:
            self.cursor.execute(
                "SELECT * FROM content WHERE title ILIKE %s AND user_id = %s", 
                (f"%{title_part}%", user_id)
            )
        else:
            self.cursor.execute("SELECT * FROM content WHERE title ILIKE %s", (f"%{title_part}%",))
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_recent(self, days: int = 7, user_id: str = None) -> List[Dict[str, Any]]:
        """Get recent content."""
        since = datetime.now() - timedelta(days=days)
        
        if user_id:
            self.cursor.execute(
                "SELECT * FROM content WHERE created_at >= %s AND user_id = %s ORDER BY created_at DESC", 
                (since, user_id)
            )
        else:
            self.cursor.execute(
                "SELECT * FROM content WHERE created_at >= %s ORDER BY created_at DESC", 
                (since,)
            )
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]


class EventDAO(BaseDAO):
    """Data Access Object for Event operations."""
    
    def __init__(self, conn: psycopg2.extensions.connection):
        super().__init__(conn, 'events')
    
    def get_by_type(self, event_type: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get events by type."""
        if user_id:
            self.cursor.execute("SELECT * FROM events WHERE type = %s AND user_id = %s", (event_type, user_id))
        else:
            self.cursor.execute("SELECT * FROM events WHERE type = %s", (event_type,))
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_upcoming(self, days: int = 7, user_id: str = None) -> List[Dict[str, Any]]:
        """Get upcoming events."""
        now = datetime.now()
        until = now + timedelta(days=days)
        
        if user_id:
            self.cursor.execute(
                "SELECT * FROM events WHERE start_time >= %s AND start_time <= %s AND user_id = %s ORDER BY start_time ASC",
                (now, until, user_id)
            )
        else:
            self.cursor.execute(
                "SELECT * FROM events WHERE start_time >= %s AND start_time <= %s ORDER BY start_time ASC",
                (now, until)
            )
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_past(self, days: int = 30, user_id: str = None) -> List[Dict[str, Any]]:
        """Get past events."""
        since = datetime.now() - timedelta(days=days)
        now = datetime.now()
        
        if user_id:
            self.cursor.execute(
                "SELECT * FROM events WHERE end_time >= %s AND end_time <= %s AND user_id = %s ORDER BY end_time DESC",
                (since, now, user_id)
            )
        else:
            self.cursor.execute(
                "SELECT * FROM events WHERE end_time >= %s AND end_time <= %s ORDER BY end_time DESC",
                (since, now)
            )
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_current(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get currently active events."""
        now = datetime.now()
        
        if user_id:
            self.cursor.execute(
                "SELECT * FROM events WHERE start_time <= %s AND end_time >= %s AND user_id = %s",
                (now, now, user_id)
            )
        else:
            self.cursor.execute(
                "SELECT * FROM events WHERE start_time <= %s AND end_time >= %s",
                (now, now)
            )
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]


class DataAnalyzer:
    """Utility class for data analysis and preparation for Active Inference."""
    
    def __init__(self, conn: psycopg2.extensions.connection, user_id: str = None):
        self.conn = conn
        self.user_id = user_id
        self.user_dao = UserDAO(conn)
        self.person_dao = PersonDAO(conn)
        self.place_dao = PlaceDAO(conn)
        self.thing_dao = ThingDAO(conn)
        self.content_dao = ContentDAO(conn)
        self.event_dao = EventDAO(conn)
    
    def get_user_activity_patterns(self, person_id: str = None, days: int = 30) -> Dict[str, Any]:
        """Analyze user activity patterns for Active Inference."""
        # If person_id provided, analyze that person's events, otherwise analyze user's events
        if person_id:
            # Get events where person is involved (would need event_person junction table)
            events = self.event_dao.get_all(user_id=self.user_id)
            # Filter events related to this person (simplified)
            events = [e for e in events if e.get('name', '').lower().find('person') >= 0]
        else:
            events = self.event_dao.get_all(user_id=self.user_id)
        
        # Filter recent events
        cutoff = datetime.now() - timedelta(days=days)
        recent_events = [e for e in events if e.get('created_at') and e.get('created_at') >= cutoff]
        
        # Analyze patterns
        event_types = {}
        hourly_activity = [0] * 24
        daily_activity = [0] * 7
        
        for event in recent_events:
            # Event type frequency
            event_type = event.get('type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            # Time patterns (simplified - would need proper datetime parsing in production)
            if event.get('start_time'):
                try:
                    # Simple hour extraction (assuming ISO format)
                    hour = 12  # Default hour
                    day_of_week = 1  # Default day
                    hourly_activity[hour] += 1
                    daily_activity[day_of_week] += 1
                except:
                    pass
        
        return {
            "event_types": event_types,
            "hourly_activity": hourly_activity,
            "daily_activity": daily_activity,
            "total_events": len(recent_events)
        }
    
    def get_place_utilization(self, place_id: str, days: int = 30) -> Dict[str, Any]:
        """Analyze place utilization patterns."""
        # Get events at this place (would need proper relationship)
        events = self.event_dao.get_all(user_id=self.user_id)
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_events = [e for e in events if e.get('created_at') and e.get('created_at') >= cutoff]
        
        utilization = {
            "total_bookings": len(recent_events),
            "unique_users": 1,  # Simplified - just the current user
            "avg_duration": 2.0  # Simplified - would calculate from start/end times
        }
        
        return utilization
    
    def get_system_state_vector(self) -> Dict[str, Any]:
        """Get current system state for Active Inference agent."""
        now = datetime.now()
        
        # Current state indicators
        active_events = len(self.event_dao.get_current(user_id=self.user_id))
        available_things = len(self.thing_dao.get_all(user_id=self.user_id))
        recent_content = len(self.content_dao.get_recent(days=1, user_id=self.user_id))
        
        # Get some distribution statistics
        total_people = len(self.person_dao.get_all(user_id=self.user_id))
        total_places = len(self.place_dao.get_all(user_id=self.user_id))
        total_things = len(self.thing_dao.get_all(user_id=self.user_id))
        
        return {
            "timestamp": now.isoformat(),
            "user_id": self.user_id,
            "active_events": active_events,
            "available_resources": available_things,
            "recent_content_items": recent_content,
            "total_people": total_people,
            "total_places": total_places,
            "total_things": total_things,
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "week_of_year": now.isocalendar()[1]
        }