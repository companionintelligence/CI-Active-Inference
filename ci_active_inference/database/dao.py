"""Data Access Object (DAO) classes for database operations."""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from datetime import datetime, timedelta
import logging

from ci_active_inference.models import (
    Person, Place, Thing, Content, Event,
    PersonSchema, PlaceSchema, ThingSchema, ContentSchema, EventSchema
)

logger = logging.getLogger(__name__)


class BaseDAO:
    """Base Data Access Object with common operations."""
    
    def __init__(self, db: Session, model_class, schema_class):
        self.db = db
        self.model_class = model_class
        self.schema_class = schema_class
    
    def create(self, obj_data: dict) -> Any:
        """Create a new object."""
        db_obj = self.model_class(**obj_data)
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    def get_by_id(self, obj_id: int) -> Optional[Any]:
        """Get object by ID."""
        return self.db.query(self.model_class).filter(self.model_class.id == obj_id).first()
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[Any]:
        """Get all objects with pagination."""
        return self.db.query(self.model_class).offset(skip).limit(limit).all()
    
    def update(self, obj_id: int, obj_data: dict) -> Optional[Any]:
        """Update an object."""
        db_obj = self.get_by_id(obj_id)
        if db_obj:
            for key, value in obj_data.items():
                setattr(db_obj, key, value)
            db_obj.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(db_obj)
        return db_obj
    
    def delete(self, obj_id: int) -> bool:
        """Delete an object."""
        db_obj = self.get_by_id(obj_id)
        if db_obj:
            self.db.delete(db_obj)
            self.db.commit()
            return True
        return False


class PersonDAO(BaseDAO):
    """Data Access Object for Person operations."""
    
    def __init__(self, db: Session):
        super().__init__(db, Person, PersonSchema)
    
    def get_by_email(self, email: str) -> Optional[Person]:
        """Get person by email."""
        return self.db.query(Person).filter(Person.email == email).first()
    
    def get_by_department(self, department: str) -> List[Person]:
        """Get people by department."""
        return self.db.query(Person).filter(Person.department == department).all()
    
    def get_by_role(self, role: str) -> List[Person]:
        """Get people by role."""
        return self.db.query(Person).filter(Person.role == role).all()


class PlaceDAO(BaseDAO):
    """Data Access Object for Place operations."""
    
    def __init__(self, db: Session):
        super().__init__(db, Place, PlaceSchema)
    
    def get_by_type(self, place_type: str) -> List[Place]:
        """Get places by type."""
        return self.db.query(Place).filter(Place.place_type == place_type).all()
    
    def get_nearby(self, latitude: float, longitude: float, radius: float = 1.0) -> List[Place]:
        """Get places within radius (simplified distance calculation)."""
        # This is a simplified implementation - in production, use PostGIS
        lat_range = radius / 111.0  # Approximate degrees per km
        lon_range = radius / (111.0 * abs(latitude))
        
        return self.db.query(Place).filter(
            and_(
                Place.latitude.between(latitude - lat_range, latitude + lat_range),
                Place.longitude.between(longitude - lon_range, longitude + lon_range)
            )
        ).all()


class ThingDAO(BaseDAO):
    """Data Access Object for Thing operations."""
    
    def __init__(self, db: Session):
        super().__init__(db, Thing, ThingSchema)
    
    def get_by_type(self, thing_type: str) -> List[Thing]:
        """Get things by type."""
        return self.db.query(Thing).filter(Thing.thing_type == thing_type).all()
    
    def get_by_status(self, status: str) -> List[Thing]:
        """Get things by status."""
        return self.db.query(Thing).filter(Thing.status == status).all()
    
    def get_available(self) -> List[Thing]:
        """Get available things."""
        return self.get_by_status("available")
    
    def get_by_location(self, location_id: int) -> List[Thing]:
        """Get things at a specific location."""
        return self.db.query(Thing).filter(Thing.location_id == location_id).all()


class ContentDAO(BaseDAO):
    """Data Access Object for Content operations."""
    
    def __init__(self, db: Session):
        super().__init__(db, Content, ContentSchema)
    
    def get_by_type(self, content_type: str) -> List[Content]:
        """Get content by type."""
        return self.db.query(Content).filter(Content.content_type == content_type).all()
    
    def get_by_creator(self, creator_id: int) -> List[Content]:
        """Get content by creator."""
        return self.db.query(Content).filter(Content.creator_id == creator_id).all()
    
    def search_by_title(self, title_part: str) -> List[Content]:
        """Search content by title."""
        return self.db.query(Content).filter(Content.title.ilike(f"%{title_part}%")).all()
    
    def get_recent(self, days: int = 7) -> List[Content]:
        """Get recent content."""
        since = datetime.utcnow() - timedelta(days=days)
        return self.db.query(Content).filter(Content.created_at >= since).order_by(desc(Content.created_at)).all()


class EventDAO(BaseDAO):
    """Data Access Object for Event operations."""
    
    def __init__(self, db: Session):
        super().__init__(db, Event, EventSchema)
    
    def get_by_type(self, event_type: str) -> List[Event]:
        """Get events by type."""
        return self.db.query(Event).filter(Event.event_type == event_type).all()
    
    def get_by_person(self, person_id: int) -> List[Event]:
        """Get events for a person."""
        return self.db.query(Event).filter(Event.person_id == person_id).all()
    
    def get_by_place(self, place_id: int) -> List[Event]:
        """Get events at a place."""
        return self.db.query(Event).filter(Event.place_id == place_id).all()
    
    def get_upcoming(self, days: int = 7) -> List[Event]:
        """Get upcoming events."""
        now = datetime.utcnow()
        until = now + timedelta(days=days)
        return self.db.query(Event).filter(
            and_(Event.start_time >= now, Event.start_time <= until)
        ).order_by(asc(Event.start_time)).all()
    
    def get_past(self, days: int = 30) -> List[Event]:
        """Get past events."""
        since = datetime.utcnow() - timedelta(days=days)
        now = datetime.utcnow()
        return self.db.query(Event).filter(
            and_(Event.end_time >= since, Event.end_time <= now)
        ).order_by(desc(Event.end_time)).all()
    
    def get_current(self) -> List[Event]:
        """Get currently active events."""
        now = datetime.utcnow()
        return self.db.query(Event).filter(
            and_(Event.start_time <= now, Event.end_time >= now)
        ).all()


class DataAnalyzer:
    """Utility class for data analysis and preparation for Active Inference."""
    
    def __init__(self, db: Session):
        self.db = db
        self.person_dao = PersonDAO(db)
        self.place_dao = PlaceDAO(db)
        self.thing_dao = ThingDAO(db)
        self.content_dao = ContentDAO(db)
        self.event_dao = EventDAO(db)
    
    def get_user_activity_patterns(self, person_id: int, days: int = 30) -> Dict[str, Any]:
        """Analyze user activity patterns for Active Inference."""
        events = self.event_dao.get_by_person(person_id)
        recent_events = [e for e in events if e.created_at >= datetime.utcnow() - timedelta(days=days)]
        
        # Analyze patterns
        event_types = {}
        hourly_activity = [0] * 24
        daily_activity = [0] * 7
        
        for event in recent_events:
            # Event type frequency
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            
            # Hourly patterns
            if event.start_time:
                hour = event.start_time.hour
                day_of_week = event.start_time.weekday()
                hourly_activity[hour] += 1
                daily_activity[day_of_week] += 1
        
        return {
            "event_types": event_types,
            "hourly_activity": hourly_activity,
            "daily_activity": daily_activity,
            "total_events": len(recent_events)
        }
    
    def get_place_utilization(self, place_id: int, days: int = 30) -> Dict[str, Any]:
        """Analyze place utilization patterns."""
        events = self.place_dao.get_by_place(place_id)
        recent_events = [e for e in events if e.created_at >= datetime.utcnow() - timedelta(days=days)]
        
        utilization = {
            "total_bookings": len(recent_events),
            "unique_users": len(set(e.person_id for e in recent_events if e.person_id)),
            "avg_duration": 0
        }
        
        if recent_events:
            durations = []
            for event in recent_events:
                if event.start_time and event.end_time:
                    duration = (event.end_time - event.start_time).total_seconds() / 3600  # hours
                    durations.append(duration)
            
            if durations:
                utilization["avg_duration"] = sum(durations) / len(durations)
        
        return utilization
    
    def get_system_state_vector(self) -> Dict[str, Any]:
        """Get current system state for Active Inference agent."""
        now = datetime.utcnow()
        
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