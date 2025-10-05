"""Data Access Object (DAO) classes for database operations using SQLite."""

import json
import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BaseDAO:
    """Base Data Access Object with common operations."""
    
    def __init__(self, conn: sqlite3.Connection, table_name: str):
        self.conn = conn
        self.table_name = table_name
        self.cursor = conn.cursor()
    
    def create(self, obj_data: dict) -> Dict[str, Any]:
        """Create a new object."""
        # Remove None values and id if present
        obj_data = {k: v for k, v in obj_data.items() if v is not None and k != 'id'}
        
        # Convert metadata to JSON string if present
        if 'metadata' in obj_data:
            obj_data['metadata'] = json.dumps(obj_data['metadata'])
        
        # Build INSERT query
        columns = list(obj_data.keys())
        placeholders = ['?' for _ in columns]
        query = f"INSERT INTO {self.table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        
        self.cursor.execute(query, list(obj_data.values()))
        
        # Get the created object
        obj_id = self.cursor.lastrowid
        return self.get_by_id(obj_id)
    
    def get_by_id(self, obj_id: int) -> Optional[Dict[str, Any]]:
        """Get object by ID."""
        self.cursor.execute(f"SELECT * FROM {self.table_name} WHERE id = ?", (obj_id,))
        row = self.cursor.fetchone()
        return self._row_to_dict(row) if row else None
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all objects with pagination."""
        self.cursor.execute(f"SELECT * FROM {self.table_name} LIMIT ? OFFSET ?", (limit, skip))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def update(self, obj_id: int, obj_data: dict) -> Optional[Dict[str, Any]]:
        """Update an object."""
        # Remove None values and id
        obj_data = {k: v for k, v in obj_data.items() if v is not None and k != 'id'}
        
        if not obj_data:
            return self.get_by_id(obj_id)
        
        # Convert metadata to JSON string if present
        if 'metadata' in obj_data:
            obj_data['metadata'] = json.dumps(obj_data['metadata'])
        
        # Add updated_at timestamp
        obj_data['updated_at'] = datetime.now().isoformat()
        
        # Build UPDATE query
        set_clauses = [f"{key} = ?" for key in obj_data.keys()]
        query = f"UPDATE {self.table_name} SET {', '.join(set_clauses)} WHERE id = ?"
        
        self.cursor.execute(query, list(obj_data.values()) + [obj_id])
        return self.get_by_id(obj_id)
    
    def delete(self, obj_id: int) -> bool:
        """Delete an object."""
        self.cursor.execute(f"DELETE FROM {self.table_name} WHERE id = ?", (obj_id,))
        return self.cursor.rowcount > 0
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary."""
        if not row:
            return None
        
        result = dict(row)
        
        # Parse JSON metadata if present
        if 'metadata' in result and result['metadata']:
            try:
                result['metadata'] = json.loads(result['metadata'])
            except (json.JSONDecodeError, TypeError):
                result['metadata'] = {}
        
        # Parse JSON tags if present
        if 'tags' in result and result['tags']:
            try:
                result['tags'] = json.loads(result['tags'])
            except (json.JSONDecodeError, TypeError):
                result['tags'] = []
        
        return result


class PersonDAO(BaseDAO):
    """Data Access Object for Person operations."""
    
    def __init__(self, conn: sqlite3.Connection):
        super().__init__(conn, 'persons')
    
    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get person by email."""
        self.cursor.execute("SELECT * FROM persons WHERE email = ?", (email,))
        row = self.cursor.fetchone()
        return self._row_to_dict(row) if row else None
    
    def get_by_department(self, department: str) -> List[Dict[str, Any]]:
        """Get people by department."""
        self.cursor.execute("SELECT * FROM persons WHERE department = ?", (department,))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Get people by role."""
        self.cursor.execute("SELECT * FROM persons WHERE role = ?", (role,))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]


class PlaceDAO(BaseDAO):
    """Data Access Object for Place operations."""
    
    def __init__(self, conn: sqlite3.Connection):
        super().__init__(conn, 'places')
    
    def get_by_type(self, place_type: str) -> List[Dict[str, Any]]:
        """Get places by type."""
        self.cursor.execute("SELECT * FROM places WHERE place_type = ?", (place_type,))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]


class ThingDAO(BaseDAO):
    """Data Access Object for Thing operations."""
    
    def __init__(self, conn: sqlite3.Connection):
        super().__init__(conn, 'things')
    
    def get_by_type(self, thing_type: str) -> List[Dict[str, Any]]:
        """Get things by type."""
        self.cursor.execute("SELECT * FROM things WHERE thing_type = ?", (thing_type,))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get things by status."""
        self.cursor.execute("SELECT * FROM things WHERE status = ?", (status,))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_available(self) -> List[Dict[str, Any]]:
        """Get available things."""
        return self.get_by_status("available")
    
    def get_by_location(self, location_id: int) -> List[Dict[str, Any]]:
        """Get things at a specific location."""
        self.cursor.execute("SELECT * FROM things WHERE location_id = ?", (location_id,))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]


class ContentDAO(BaseDAO):
    """Data Access Object for Content operations."""
    
    def __init__(self, conn: sqlite3.Connection):
        super().__init__(conn, 'content')
    
    def get_by_type(self, content_type: str) -> List[Dict[str, Any]]:
        """Get content by type."""
        self.cursor.execute("SELECT * FROM content WHERE content_type = ?", (content_type,))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_by_creator(self, creator_id: int) -> List[Dict[str, Any]]:
        """Get content by creator."""
        self.cursor.execute("SELECT * FROM content WHERE creator_id = ?", (creator_id,))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def search_by_title(self, title_part: str) -> List[Dict[str, Any]]:
        """Search content by title."""
        self.cursor.execute("SELECT * FROM content WHERE title LIKE ?", (f"%{title_part}%",))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_recent(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent content."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
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
    
    def get_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get events by type."""
        self.cursor.execute("SELECT * FROM events WHERE event_type = ?", (event_type,))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_by_person(self, person_id: int) -> List[Dict[str, Any]]:
        """Get events for a person."""
        self.cursor.execute("SELECT * FROM events WHERE person_id = ?", (person_id,))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_by_place(self, place_id: int) -> List[Dict[str, Any]]:
        """Get events at a place."""
        self.cursor.execute("SELECT * FROM events WHERE place_id = ?", (place_id,))
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_upcoming(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get upcoming events."""
        now = datetime.now().isoformat()
        until = (datetime.now() + timedelta(days=days)).isoformat()
        self.cursor.execute(
            "SELECT * FROM events WHERE start_time >= ? AND start_time <= ? ORDER BY start_time ASC",
            (now, until)
        )
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_past(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get past events."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        now = datetime.now().isoformat()
        self.cursor.execute(
            "SELECT * FROM events WHERE end_time >= ? AND end_time <= ? ORDER BY end_time DESC",
            (since, now)
        )
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_current(self) -> List[Dict[str, Any]]:
        """Get currently active events."""
        now = datetime.now().isoformat()
        self.cursor.execute(
            "SELECT * FROM events WHERE start_time <= ? AND end_time >= ?",
            (now, now)
        )
        rows = self.cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]


class DataAnalyzer:
    """Utility class for data analysis and preparation for Active Inference."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.person_dao = PersonDAO(conn)
        self.place_dao = PlaceDAO(conn)
        self.thing_dao = ThingDAO(conn)
        self.content_dao = ContentDAO(conn)
        self.event_dao = EventDAO(conn)
    
    def get_user_activity_patterns(self, person_id: int, days: int = 30) -> Dict[str, Any]:
        """Analyze user activity patterns for Active Inference."""
        events = self.event_dao.get_by_person(person_id)
        
        # Filter recent events
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent_events = [e for e in events if e.get('created_at', '') >= cutoff]
        
        # Analyze patterns
        event_types = {}
        hourly_activity = [0] * 24
        daily_activity = [0] * 7
        
        for event in recent_events:
            # Event type frequency
            event_type = event.get('event_type', 'unknown')
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
    
    def get_place_utilization(self, place_id: int, days: int = 30) -> Dict[str, Any]:
        """Analyze place utilization patterns."""
        events = self.event_dao.get_by_place(place_id)
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent_events = [e for e in events if e.get('created_at', '') >= cutoff]
        
        utilization = {
            "total_bookings": len(recent_events),
            "unique_users": len(set(e.get('person_id') for e in recent_events if e.get('person_id'))),
            "avg_duration": 2.0  # Simplified - would calculate from start/end times
        }
        
        return utilization
    
    def get_system_state_vector(self) -> Dict[str, Any]:
        """Get current system state for Active Inference agent."""
        now = datetime.now()
        
        # Current state indicators
        active_events = len(self.event_dao.get_current())
        available_things = len(self.thing_dao.get_available())
        recent_content = len(self.content_dao.get_recent(days=1))
        
        # Get some distribution statistics
        total_people = len(self.person_dao.get_all())
        total_places = len(self.place_dao.get_all())
        total_things = len(self.thing_dao.get_all())
        
        return {
            "timestamp": now.isoformat(),
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