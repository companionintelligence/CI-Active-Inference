"""Data Access Object (DAO) classes for database operations using CI-Server schema."""

import json
import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from ci_active_inference.database import generate_uuid

logger = logging.getLogger(__name__)


class BaseDAO:
    """Base Data Access Object with common operations."""
    
    def __init__(self, conn: sqlite3.Connection, table_name: str):
        self.conn = conn
        self.table_name = table_name
        self.cursor = conn.cursor()
    
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
        
        # Convert JSON fields to strings if present
        json_fields = ['source_data', 'geolocation', 'location_boundary']
        for field in json_fields:
            if field in obj_data and isinstance(obj_data[field], (dict, list)):
                obj_data[field] = json.dumps(obj_data[field])
        
        # Build INSERT query
        columns = list(obj_data.keys())
        placeholders = ['?' for _ in columns]
        query = f"INSERT INTO {self.table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        
        self.cursor.execute(query, list(obj_data.values()))
        
        # Get the created object
        return self.get_by_id(obj_data['id'])
    
    def get_by_id(self, obj_id: str) -> Optional[Dict[str, Any]]:
        """Get object by ID."""
        self.cursor.execute(f"SELECT * FROM {self.table_name} WHERE id = ?", (obj_id,))
        row = self.cursor.fetchone()
        return self._row_to_dict(row) if row else None
    
    def get_all(self, user_id: str = None, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all objects with pagination, optionally filtered by user."""
        if user_id:
            self.cursor.execute(
                f"SELECT * FROM {self.table_name} WHERE user_id = ? LIMIT ? OFFSET ?", 
                (user_id, limit, skip)
            )
        else:
            self.cursor.execute(f"SELECT * FROM {self.table_name} LIMIT ? OFFSET ?", (limit, skip))
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def update(self, obj_id: str, obj_data: dict) -> Optional[Dict[str, Any]]:
        """Update an object."""
        # Remove None values and id
        obj_data = {k: v for k, v in obj_data.items() if v is not None and k != 'id'}
        
        if not obj_data:
            return self.get_by_id(obj_id)
        
        # Convert JSON fields to strings if present
        json_fields = ['source_data', 'geolocation', 'location_boundary']
        for field in json_fields:
            if field in obj_data and isinstance(obj_data[field], (dict, list)):
                obj_data[field] = json.dumps(obj_data[field])
        
        # Add updated_at timestamp
        obj_data['updated_at'] = datetime.now().isoformat()
        
        # Build UPDATE query
        set_clauses = [f"{key} = ?" for key in obj_data.keys()]
        query = f"UPDATE {self.table_name} SET {', '.join(set_clauses)} WHERE id = ?"
        
        self.cursor.execute(query, list(obj_data.values()) + [obj_id])
        return self.get_by_id(obj_id)
    
    def delete(self, obj_id: str) -> bool:
        """Delete an object."""
        self.cursor.execute(f"DELETE FROM {self.table_name} WHERE id = ?", (obj_id,))
        return self.cursor.rowcount > 0
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary."""
        if not row:
            return None
        
        result = dict(row)
        
        # Parse JSON fields if present
        json_fields = ['source_data', 'geolocation', 'location_boundary']
        for field in json_fields:
            if field in result and result[field]:
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    result[field] = {}
        
        return result


class UserDAO(BaseDAO):
    """Data Access Object for User operations."""
    
    def __init__(self, conn: sqlite3.Connection):
        super().__init__(conn, 'users')
    
    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email."""
        self.cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = self.cursor.fetchone()
        return self._row_to_dict(row) if row else None
    
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
    
    def __init__(self, conn: sqlite3.Connection):
        super().__init__(conn, 'people')
    
    def get_by_name(self, first_name: str, last_name: str = None, user_id: str = None) -> List[Dict[str, Any]]:
        """Get people by name."""
        if last_name:
            query = "SELECT * FROM people WHERE first_name = ? AND last_name = ?"
            params = [first_name, last_name]
        else:
            query = "SELECT * FROM people WHERE first_name = ?"
            params = [first_name]
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]


class PlaceDAO(BaseDAO):
    """Data Access Object for Place operations."""
    
    def __init__(self, conn: sqlite3.Connection):
        super().__init__(conn, 'places')
    
    def get_by_type(self, place_type: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get places by type."""
        if user_id:
            self.cursor.execute("SELECT * FROM places WHERE type = ? AND user_id = ?", (place_type, user_id))
        else:
            self.cursor.execute("SELECT * FROM places WHERE type = ?", (place_type,))
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_by_name(self, name: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get places by name."""
        if user_id:
            self.cursor.execute("SELECT * FROM places WHERE name LIKE ? AND user_id = ?", (f"%{name}%", user_id))
        else:
            self.cursor.execute("SELECT * FROM places WHERE name LIKE ?", (f"%{name}%",))
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]


class ThingDAO(BaseDAO):
    """Data Access Object for Thing operations."""
    
    def __init__(self, conn: sqlite3.Connection):
        super().__init__(conn, 'things')
    
    def get_by_type(self, thing_type: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get things by type."""
        if user_id:
            self.cursor.execute("SELECT * FROM things WHERE type = ? AND user_id = ?", (thing_type, user_id))
        else:
            self.cursor.execute("SELECT * FROM things WHERE type = ?", (thing_type,))
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_by_event(self, event_id: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get things associated with an event."""
        if user_id:
            self.cursor.execute(
                "SELECT * FROM things WHERE event_id = ? AND user_id = ?", 
                (event_id, user_id)
            )
        else:
            self.cursor.execute("SELECT * FROM things WHERE event_id = ?", (event_id,))
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]


class ContentDAO(BaseDAO):
    """Data Access Object for Content operations."""
    
    def __init__(self, conn: sqlite3.Connection):
        super().__init__(conn, 'content')
    
    def get_by_type(self, content_type: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get content by type."""
        if user_id:
            self.cursor.execute("SELECT * FROM content WHERE type = ? AND user_id = ?", (content_type, user_id))
        else:
            self.cursor.execute("SELECT * FROM content WHERE type = ?", (content_type,))
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def search_by_title(self, title_part: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Search content by title."""
        if user_id:
            self.cursor.execute(
                "SELECT * FROM content WHERE title LIKE ? AND user_id = ?", 
                (f"%{title_part}%", user_id)
            )
        else:
            self.cursor.execute("SELECT * FROM content WHERE title LIKE ?", (f"%{title_part}%",))
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_recent(self, days: int = 7, user_id: str = None) -> List[Dict[str, Any]]:
        """Get recent content."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        
        if user_id:
            self.cursor.execute(
                "SELECT * FROM content WHERE created_at >= ? AND user_id = ? ORDER BY created_at DESC", 
                (since, user_id)
            )
        else:
            self.cursor.execute(
                "SELECT * FROM content WHERE created_at >= ? ORDER BY created_at DESC", 
                (since,)
            )
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]


class EventDAO(BaseDAO):
    """Data Access Object for Event operations."""
    
    def __init__(self, conn: sqlite3.Connection):
        super().__init__(conn, 'events')
    
    def get_by_type(self, event_type: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get events by type."""
        if user_id:
            self.cursor.execute("SELECT * FROM events WHERE type = ? AND user_id = ?", (event_type, user_id))
        else:
            self.cursor.execute("SELECT * FROM events WHERE type = ?", (event_type,))
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_upcoming(self, days: int = 7, user_id: str = None) -> List[Dict[str, Any]]:
        """Get upcoming events."""
        now = datetime.now().isoformat()
        until = (datetime.now() + timedelta(days=days)).isoformat()
        
        if user_id:
            self.cursor.execute(
                "SELECT * FROM events WHERE start_time >= ? AND start_time <= ? AND user_id = ? ORDER BY start_time ASC",
                (now, until, user_id)
            )
        else:
            self.cursor.execute(
                "SELECT * FROM events WHERE start_time >= ? AND start_time <= ? ORDER BY start_time ASC",
                (now, until)
            )
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_past(self, days: int = 30, user_id: str = None) -> List[Dict[str, Any]]:
        """Get past events."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        now = datetime.now().isoformat()
        
        if user_id:
            self.cursor.execute(
                "SELECT * FROM events WHERE end_time >= ? AND end_time <= ? AND user_id = ? ORDER BY end_time DESC",
                (since, now, user_id)
            )
        else:
            self.cursor.execute(
                "SELECT * FROM events WHERE end_time >= ? AND end_time <= ? ORDER BY end_time DESC",
                (since, now)
            )
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_current(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get currently active events."""
        now = datetime.now().isoformat()
        
        if user_id:
            self.cursor.execute(
                "SELECT * FROM events WHERE start_time <= ? AND end_time >= ? AND user_id = ?",
                (now, now, user_id)
            )
        else:
            self.cursor.execute(
                "SELECT * FROM events WHERE start_time <= ? AND end_time >= ?",
                (now, now)
            )
        
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]


class DataAnalyzer:
    """Utility class for data analysis and preparation for Active Inference."""
    
    def __init__(self, conn: sqlite3.Connection, user_id: str = None):
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
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent_events = [e for e in events if e.get('created_at', '') >= cutoff]
        
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
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent_events = [e for e in events if e.get('created_at', '') >= cutoff]
        
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