"""Database models for CI Active Inference system."""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel

Base = declarative_base()


class Person(Base):
    """Model for people in the CI system."""
    __tablename__ = "persons"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True)
    role = Column(String(100))
    department = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)  # Additional flexible data
    
    # Relationships
    events = relationship("Event", back_populates="person")
    content_created = relationship("Content", back_populates="creator")


class Place(Base):
    """Model for places/locations in the CI system."""
    __tablename__ = "places"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    address = Column(Text)
    latitude = Column(Float)
    longitude = Column(Float)
    place_type = Column(String(100))  # office, meeting_room, building, etc.
    capacity = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)
    
    # Relationships
    events = relationship("Event", back_populates="place")


class Thing(Base):
    """Model for physical/digital things in the CI system."""
    __tablename__ = "things"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    thing_type = Column(String(100))  # equipment, software, resource, etc.
    status = Column(String(50))  # available, in_use, maintenance, etc.
    location_id = Column(Integer, ForeignKey("places.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)
    
    # Relationships
    location = relationship("Place")
    events = relationship("Event", back_populates="thing")


class Content(Base):
    """Model for content/documents in the CI system."""
    __tablename__ = "content"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    content_text = Column(Text)
    content_type = Column(String(100))  # document, email, report, etc.
    creator_id = Column(Integer, ForeignKey("persons.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    tags = Column(JSON)  # List of tags
    metadata = Column(JSON)
    
    # Relationships
    creator = relationship("Person", back_populates="content_created")
    events = relationship("Event", back_populates="content")


class Event(Base):
    """Model for events/activities in the CI system."""
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    event_type = Column(String(100))  # meeting, task, project, etc.
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    place_id = Column(Integer, ForeignKey("places.id"), nullable=True)
    thing_id = Column(Integer, ForeignKey("things.id"), nullable=True)
    content_id = Column(Integer, ForeignKey("content.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)
    
    # Relationships
    person = relationship("Person", back_populates="events")
    place = relationship("Place", back_populates="events")
    thing = relationship("Thing", back_populates="events")
    content = relationship("Content", back_populates="events")


# Pydantic models for API serialization
class PersonSchema(BaseModel):
    id: Optional[int] = None
    name: str
    email: str
    role: Optional[str] = None
    department: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[dict] = None
    
    class Config:
        from_attributes = True


class PlaceSchema(BaseModel):
    id: Optional[int] = None
    name: str
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place_type: Optional[str] = None
    capacity: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[dict] = None
    
    class Config:
        from_attributes = True


class ThingSchema(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    thing_type: Optional[str] = None
    status: Optional[str] = None
    location_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[dict] = None
    
    class Config:
        from_attributes = True


class ContentSchema(BaseModel):
    id: Optional[int] = None
    title: str
    content_text: Optional[str] = None
    content_type: Optional[str] = None
    creator_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: Optional[List[str]] = None
    metadata: Optional[dict] = None
    
    class Config:
        from_attributes = True


class EventSchema(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    event_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    person_id: Optional[int] = None
    place_id: Optional[int] = None
    thing_id: Optional[int] = None
    content_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[dict] = None
    
    class Config:
        from_attributes = True