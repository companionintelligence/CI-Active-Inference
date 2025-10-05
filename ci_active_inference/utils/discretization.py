"""
Data discretization pipeline for Active Inference.

This module converts continuous and categorical data from the CI database
into discrete state indices suitable for Active Inference matrices.
"""

import sqlite3
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class StateMapping:
    """Mapping between raw values and discrete state indices."""
    name: str
    mapping: Dict[str, int]
    num_states: int
    default_state: int = 0


class DataDiscretizer:
    """
    Converts CI database data into discrete states for Active Inference.
    
    This class handles the conversion of temporal patterns, categorical data,
    and continuous measurements into discrete state spaces.
    """
    
    def __init__(self):
        """Initialize discretizer with predefined state mappings."""
        
        # Energy level states (0: low, 1: medium, 2: high)
        self.energy_mapping = StateMapping(
            name="energy",
            mapping={"low": 0, "medium": 1, "high": 2},
            num_states=3,
            default_state=1
        )
        
        # Mood states (0: negative, 1: neutral, 2: positive)
        self.mood_mapping = StateMapping(
            name="mood", 
            mapping={"negative": 0, "neutral": 1, "positive": 2},
            num_states=3,
            default_state=1
        )
        
        # Context states (0: home, 1: work, 2: social, 3: health, 4: other)
        self.context_mapping = StateMapping(
            name="context",
            mapping={"home": 0, "work": 1, "social": 2, "health": 3, "other": 4},
            num_states=5,
            default_state=4
        )
        
        # Digital behavior observations (0: work-focus, 1: communication, 2: entertainment, 3: other)
        self.digital_mapping = StateMapping(
            name="digital_behavior",
            mapping={"work-focus": 0, "communication": 1, "entertainment": 2, "other": 3},
            num_states=4,
            default_state=3
        )
        
        # Location context (0: home, 1: office, 2: outdoor, 3: social-venue, 4: other)
        self.location_mapping = StateMapping(
            name="location_context",
            mapping={"home": 0, "office": 1, "outdoor": 2, "social-venue": 3, "other": 4},
            num_states=5,
            default_state=4
        )
        
        # Social context (0: alone, 1: small-group, 2: large-group)
        self.social_mapping = StateMapping(
            name="social_context",
            mapping={"alone": 0, "small-group": 1, "large-group": 2},
            num_states=3,
            default_state=0
        )
        
        # Content type (0: work, 1: entertainment, 2: communication, 3: other)
        self.content_mapping = StateMapping(
            name="content_type",
            mapping={"work": 0, "entertainment": 1, "communication": 2, "other": 3},
            num_states=4,
            default_state=3
        )
    
    def discretize_hour_of_day(self, hour: int) -> int:
        """
        Discretize hour of day into time periods.
        
        Args:
            hour: Hour (0-23)
        
        Returns:
            Time period index (0: night, 1: morning, 2: afternoon, 3: evening)
        """
        if 22 <= hour or hour < 6:
            return 0  # Night
        elif 6 <= hour < 12:
            return 1  # Morning
        elif 12 <= hour < 18:
            return 2  # Afternoon
        else:
            return 3  # Evening
    
    def discretize_event_count(self, count: int) -> int:
        """
        Discretize event count into activity levels.
        
        Args:
            count: Number of events
        
        Returns:
            Activity level (0: low, 1: medium, 2: high)
        """
        if count <= 1:
            return 0  # Low activity
        elif count <= 3:
            return 1  # Medium activity
        else:
            return 2  # High activity
    
    def discretize_event_duration(self, duration_minutes: int) -> int:
        """
        Discretize event duration.
        
        Args:
            duration_minutes: Duration in minutes
        
        Returns:
            Duration category (0: short, 1: medium, 2: long)
        """
        if duration_minutes <= 30:
            return 0  # Short
        elif duration_minutes <= 120:
            return 1  # Medium
        else:
            return 2  # Long
    
    def classify_event_context(self, event_name: str, event_type: str = None) -> str:
        """
        Classify event into context category based on name and type.
        
        Args:
            event_name: Name of the event
            event_type: Type of the event
        
        Returns:
            Context category string
        """
        if not event_name:
            return "other"
        
        name_lower = event_name.lower()
        type_lower = (event_type or "").lower()
        
        # Health-related keywords (check first to avoid conflicts)
        health_keywords = ["exercise", "gym", "workout", "run", "walk", "doctor", "health", "fitness"]
        if any(kw in name_lower or kw in type_lower for kw in health_keywords):
            return "health"
        
        # Work-related keywords
        work_keywords = ["meeting", "work", "office", "call", "conference", "project", "task"]
        if any(kw in name_lower or kw in type_lower for kw in work_keywords):
            return "work"
        
        # Social keywords
        social_keywords = ["dinner", "lunch", "coffee", "party", "friend", "family", "social"]
        if any(kw in name_lower or kw in type_lower for kw in social_keywords):
            return "social"
        
        # Home keywords
        home_keywords = ["home", "house", "cooking", "cleaning", "rest", "sleep"]
        if any(kw in name_lower or kw in type_lower for kw in home_keywords):
            return "home"
        
        return "other"
    
    def classify_digital_behavior(self, content_type: str = None, title: str = None) -> str:
        """
        Classify digital behavior based on content.
        
        Args:
            content_type: Type of content
            title: Title or text content
        
        Returns:
            Digital behavior category
        """
        if not content_type and not title:
            return "other"
        
        type_lower = (content_type or "").lower()
        title_lower = (title or "").lower()
        
        # Work-focused behavior
        work_keywords = ["document", "spreadsheet", "presentation", "code", "work", "task"]
        if any(kw in type_lower or kw in title_lower for kw in work_keywords):
            return "work-focus"
        
        # Communication behavior
        comm_keywords = ["email", "message", "chat", "call", "video", "communication"]
        if any(kw in type_lower or kw in title_lower for kw in comm_keywords):
            return "communication"
        
        # Entertainment behavior
        ent_keywords = ["video", "game", "music", "entertainment", "movie", "youtube"]
        if any(kw in type_lower or kw in title_lower for kw in ent_keywords):
            return "entertainment"
        
        return "other"
    
    def classify_location_context(self, place_name: str = None, place_type: str = None) -> str:
        """
        Classify location into context category.
        
        Args:
            place_name: Name of the place
            place_type: Type of the place
        
        Returns:
            Location context category
        """
        if not place_name and not place_type:
            return "other"
        
        name_lower = (place_name or "").lower()
        type_lower = (place_type or "").lower()
        
        # Home locations
        home_keywords = ["home", "house", "apartment", "residence"]
        if any(kw in name_lower or kw in type_lower for kw in home_keywords):
            return "home"
        
        # Office/work locations
        office_keywords = ["office", "work", "workplace", "building", "company"]
        if any(kw in name_lower or kw in type_lower for kw in office_keywords):
            return "office"
        
        # Outdoor locations
        outdoor_keywords = ["park", "outdoor", "trail", "beach", "nature", "street"]
        if any(kw in name_lower or kw in type_lower for kw in outdoor_keywords):
            return "outdoor"
        
        # Social venues
        social_keywords = ["restaurant", "cafe", "bar", "pub", "club", "venue", "theater"]
        if any(kw in name_lower or kw in type_lower for kw in social_keywords):
            return "social-venue"
        
        return "other"
    
    def classify_social_context(self, person_count: int) -> str:
        """
        Classify social context based on number of people.
        
        Args:
            person_count: Number of people involved
        
        Returns:
            Social context category
        """
        if person_count == 0:
            return "alone"
        elif person_count <= 2:
            return "small-group"
        else:
            return "large-group"
    
    def map_value_to_state(self, value: str, mapping: StateMapping) -> int:
        """
        Map a categorical value to a discrete state index.
        
        Args:
            value: Value to map
            mapping: State mapping configuration
        
        Returns:
            State index
        """
        return mapping.mapping.get(value, mapping.default_state)
    
    def extract_temporal_features(self, conn: sqlite3.Connection, user_id: str, 
                                 start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Extract temporal features from database for a user and date range.
        
        Args:
            conn: Database connection
            user_id: User ID
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
        
        Returns:
            List of temporal feature records
        """
        query = """
        SELECT 
          strftime('%Y-%m-%d %H:00:00', e.start_time) as time_bucket,
          strftime('%H', e.start_time) as hour_of_day,
          
          -- Event context classification
          CASE 
            WHEN LOWER(e.name) LIKE '%gym%' OR LOWER(e.name) LIKE '%exercise%' OR LOWER(e.name) LIKE '%workout%' THEN 'health'
            WHEN LOWER(e.name) LIKE '%work%' OR LOWER(e.name) LIKE '%meeting%' THEN 'work'
            WHEN LOWER(e.name) LIKE '%social%' OR LOWER(e.name) LIKE '%dinner%' THEN 'social'
            WHEN LOWER(e.name) LIKE '%home%' OR LOWER(e.name) LIKE '%rest%' THEN 'home'
            ELSE 'other'
          END as context_state,
          
          -- Digital behavior from content
          CASE 
            WHEN c.type LIKE '%communication%' THEN 'communication'
            WHEN c.type LIKE '%entertainment%' THEN 'entertainment' 
            WHEN c.type LIKE '%work%' OR c.type LIKE '%document%' THEN 'work-focus'
            ELSE 'other'
          END as digital_behavior,
          
          -- Social context - simplified
          CASE 
            WHEN ep.person_id IS NOT NULL THEN 'small-group'
            ELSE 'alone'
          END as social_context,
          
          -- Location context
          CASE 
            WHEN LOWER(COALESCE(p.name, '')) LIKE '%home%' THEN 'home'
            WHEN LOWER(COALESCE(p.name, '')) LIKE '%office%' OR LOWER(COALESCE(p.name, '')) LIKE '%work%' THEN 'office'
            WHEN LOWER(COALESCE(p.name, '')) LIKE '%park%' OR LOWER(COALESCE(p.name, '')) LIKE '%outdoor%' THEN 'outdoor'
            WHEN LOWER(COALESCE(p.name, '')) LIKE '%restaurant%' OR LOWER(COALESCE(p.name, '')) LIKE '%cafe%' THEN 'social-venue'
            ELSE 'other'
          END as location_context,
          
          -- Content type
          COALESCE(c.type, 'other') as content_type,
          
          -- Event metrics
          1 as event_count,
          CASE 
            WHEN e.end_time IS NOT NULL 
            THEN (strftime('%s', e.end_time) - strftime('%s', e.start_time)) / 60.0 
            ELSE 60 
          END as avg_duration_minutes
          
        FROM events e
        LEFT JOIN locations l ON e.user_id = l.user_id 
          AND datetime(l.datetime) BETWEEN datetime(e.start_time) AND COALESCE(datetime(e.end_time), datetime(e.start_time, '+1 hour'))
        LEFT JOIN places p ON l.place_id = p.id
        LEFT JOIN content c ON e.id = c.event_id
        LEFT JOIN event_person ep ON e.id = ep.event_id
        
        WHERE e.user_id = ? 
          AND e.start_time BETWEEN ? AND ?
        ORDER BY e.start_time
        """
        
        cursor = conn.cursor()
        cursor.execute(query, (user_id, start_date, end_date))
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            results.append(record)
        
        return results
    
    def discretize_features(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert extracted features to discrete state indices.
        
        Args:
            features: List of feature records from extract_temporal_features
        
        Returns:
            List of discretized feature records
        """
        discretized = []
        
        for record in features:
            discretized_record = record.copy()
            
            # Discretize temporal features
            hour = int(record.get('hour_of_day', 12))
            discretized_record['time_period'] = self.discretize_hour_of_day(hour)
            
            # Discretize event metrics
            event_count = int(record.get('event_count', 0))
            discretized_record['activity_level'] = self.discretize_event_count(event_count)
            
            avg_duration = float(record.get('avg_duration_minutes', 60))
            discretized_record['duration_category'] = self.discretize_event_duration(int(avg_duration))
            
            # Map categorical values to state indices
            discretized_record['context_idx'] = self.map_value_to_state(
                record.get('context_state', 'other'), self.context_mapping
            )
            
            discretized_record['digital_idx'] = self.map_value_to_state(
                record.get('digital_behavior', 'other'), self.digital_mapping
            )
            
            discretized_record['social_idx'] = self.map_value_to_state(
                record.get('social_context', 'alone'), self.social_mapping
            )
            
            discretized_record['location_idx'] = self.map_value_to_state(
                record.get('location_context', 'other'), self.location_mapping
            )
            
            discretized_record['content_idx'] = self.map_value_to_state(
                record.get('content_type', 'other'), self.content_mapping
            )
            
            discretized.append(discretized_record)
        
        return discretized
    
    def get_state_dimensions(self) -> Dict[str, int]:
        """
        Get the dimensions of all state spaces.
        
        Returns:
            Dictionary mapping state names to number of states
        """
        return {
            'energy': self.energy_mapping.num_states,
            'mood': self.mood_mapping.num_states,
            'context': self.context_mapping.num_states,
            'digital_behavior': self.digital_mapping.num_states,
            'location_context': self.location_mapping.num_states,
            'social_context': self.social_mapping.num_states,
            'content_type': self.content_mapping.num_states,
            'time_period': 4,  # night, morning, afternoon, evening
            'activity_level': 3,  # low, medium, high
            'duration_category': 3  # short, medium, long
        }
    
    def get_observation_dimensions(self) -> List[int]:
        """
        Get observation dimensions for Active Inference matrices.
        
        Returns:
            List of observation dimensions for each modality
        """
        return [
            self.digital_mapping.num_states,   # Digital behavior observations
            self.location_mapping.num_states,  # Location observations  
            self.social_mapping.num_states,    # Social observations
            self.content_mapping.num_states    # Content observations
        ]
    
    def get_state_factor_dimensions(self) -> List[int]:
        """
        Get state factor dimensions for Active Inference matrices.
        
        Returns:
            List of state dimensions for each factor [Energy, Mood, Context]
        """
        return [
            self.energy_mapping.num_states,    # Energy states
            self.mood_mapping.num_states,      # Mood states
            self.context_mapping.num_states    # Context states
        ]