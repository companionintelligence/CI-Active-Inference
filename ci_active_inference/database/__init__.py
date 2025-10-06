"""Database connection and session management using PostgreSQL implementing CI-Server schema."""

import psycopg2
import psycopg2.extras
import os
import uuid
from typing import Generator, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Database configuration - using PostgreSQL with CI-Server schema
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost:5432/ci_database")


def get_db_connection():
    """Get PostgreSQL database connection."""
    conn = psycopg2.connect(DATABASE_URL)
    return conn


@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    conn = get_db_connection()
    # Use RealDictCursor for dict-like access
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def create_tables():
    """Create all database tables according to CI-Server schema."""
    with get_db_session() as conn:
        cursor = conn.cursor()
        
        # Core user management
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                onboarded BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tags for content organization
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id TEXT PRIMARY KEY,
                label TEXT,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Memory domain - People (contacts and family members)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS people (
                id TEXT PRIMARY KEY,
                type TEXT,
                first_name TEXT,
                last_name TEXT,
                gender TEXT,
                address TEXT,
                street_address TEXT,
                street_address_2 TEXT,
                city TEXT,
                region TEXT,
                country TEXT,
                postal_code TEXT,
                avatar_url TEXT,
                external_id TEXT,
                source_url TEXT,
                source_data JSONB,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Memory domain - Places
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS places (
                id TEXT PRIMARY KEY,
                type TEXT,
                name TEXT,
                geo_format TEXT,
                geolocation JSONB,
                location_boundary JSONB,
                estimated BOOLEAN DEFAULT FALSE,
                tracked BOOLEAN DEFAULT FALSE,
                uploaded BOOLEAN DEFAULT FALSE,
                parent_id TEXT,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES places (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Memory domain - Locations (specific instances of places)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS locations (
                id TEXT PRIMARY KEY,
                identifier TEXT,
                datetime TIMESTAMP,
                geo_format TEXT,
                geolocation JSONB,
                estimated BOOLEAN DEFAULT FALSE,
                tracked BOOLEAN DEFAULT FALSE,
                uploaded BOOLEAN DEFAULT FALSE,
                place_id TEXT,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (place_id) REFERENCES places (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Memory domain - Events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                type TEXT,
                name TEXT,
                description TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                external_id TEXT,
                source_url TEXT,
                source_data JSONB,
                user_id TEXT NOT NULL,
                connection_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Memory domain - Content
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                title TEXT,
                text TEXT,
                url TEXT,
                mimetype TEXT,
                external_id TEXT,
                source_url TEXT,
                source_data JSONB,
                user_id TEXT NOT NULL,
                event_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (event_id) REFERENCES events (id)
            )
        ''')
        
        # Memory domain - Things
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS things (
                id TEXT PRIMARY KEY,
                type TEXT,
                name TEXT,
                description TEXT,
                external_id TEXT,
                source_url TEXT,
                source_data JSONB,
                user_id TEXT NOT NULL,
                event_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (event_id) REFERENCES events (id)
            )
        ''')
        
        # Contact information for people
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contacts (
                id TEXT PRIMARY KEY,
                type TEXT,
                avatar_url TEXT,
                handle TEXT,
                identifier TEXT,
                external_id TEXT,
                source_url TEXT,
                source_data JSONB,
                user_id TEXT NOT NULL,
                person_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (person_id) REFERENCES people (id)
            )
        ''')
        
        # Junction tables for many-to-many relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS person_tag (
                id SERIAL PRIMARY KEY,
                person_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES people (id),
                FOREIGN KEY (tag_id) REFERENCES tags (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_tag (
                id SERIAL PRIMARY KEY,
                event_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events (id),
                FOREIGN KEY (tag_id) REFERENCES tags (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_tag (
                id SERIAL PRIMARY KEY,
                content_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_id) REFERENCES content (id),
                FOREIGN KEY (tag_id) REFERENCES tags (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS thing_tag (
                id SERIAL PRIMARY KEY,
                thing_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thing_id) REFERENCES things (id),
                FOREIGN KEY (tag_id) REFERENCES tags (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS place_tag (
                id SERIAL PRIMARY KEY,
                place_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (place_id) REFERENCES places (id),
                FOREIGN KEY (tag_id) REFERENCES tags (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_person (
                id SERIAL PRIMARY KEY,
                event_id TEXT NOT NULL,
                person_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events (id),
                FOREIGN KEY (person_id) REFERENCES people (id)
            )
        ''')
        
        # Create temporal views for Active Inference feature extraction
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS daily_state_observations AS
            SELECT 
                e.user_id,
                DATE(e.start_time) as day,
                strftime('%H', e.start_time) as hour_bucket,
                
                -- Context classification
                CASE 
                  WHEN LOWER(e.name) LIKE '%work%' OR LOWER(e.name) LIKE '%meeting%' THEN 'work'
                  WHEN LOWER(e.name) LIKE '%exercise%' OR LOWER(e.name) LIKE '%gym%' THEN 'health'
                  WHEN LOWER(e.name) LIKE '%social%' OR LOWER(e.name) LIKE '%dinner%' THEN 'social'
                  WHEN LOWER(e.name) LIKE '%home%' OR LOWER(e.name) LIKE '%rest%' THEN 'home'
                  ELSE 'other'
                END as context_bucket,
                
                -- Energy level inference
                CASE 
                  WHEN COUNT(e.id) > 3 THEN 'high'
                  WHEN COUNT(e.id) > 1 THEN 'medium'
                  ELSE 'low'
                END as energy_bucket,
                
                -- Mood inference from context
                CASE 
                  WHEN LOWER(e.name) LIKE '%social%' OR LOWER(e.name) LIKE '%exercise%' THEN 'positive'
                  WHEN LOWER(e.name) LIKE '%work%' AND COUNT(e.id) > 2 THEN 'neutral'
                  WHEN LOWER(e.name) LIKE '%work%' AND COUNT(e.id) > 4 THEN 'negative'
                  ELSE 'neutral'
                END as mood_bucket,
                
                -- Digital behavior from content
                CASE 
                  WHEN c.type LIKE '%communication%' THEN 'communication'
                  WHEN c.type LIKE '%entertainment%' THEN 'entertainment' 
                  WHEN c.type LIKE '%work%' OR c.type LIKE '%document%' THEN 'work-focus'
                  ELSE 'other'
                END as digital_behavior_type,
                
                -- Location context
                CASE 
                  WHEN LOWER(p.name) LIKE '%home%' THEN 'home'
                  WHEN LOWER(p.name) LIKE '%office%' OR LOWER(p.name) LIKE '%work%' THEN 'office'
                  WHEN LOWER(p.name) LIKE '%park%' OR LOWER(p.name) LIKE '%outdoor%' THEN 'outdoor'
                  WHEN LOWER(p.name) LIKE '%restaurant%' OR LOWER(p.name) LIKE '%cafe%' THEN 'social-venue'
                  ELSE 'other'
                END as location_category,
                
                -- Social context
                CASE 
                  WHEN COUNT(ep.person_id) = 0 THEN 'alone'
                  WHEN COUNT(ep.person_id) <= 2 THEN 'small-group'
                  ELSE 'large-group'
                END as social_category,
                
                COUNT(e.id) as event_count,
                AVG(CASE 
                  WHEN e.end_time IS NOT NULL 
                  THEN (strftime('%s', e.end_time) - strftime('%s', e.start_time)) / 60.0 
                  ELSE 60 
                END) as avg_duration_minutes
                
            FROM events e
            LEFT JOIN locations l ON e.user_id = l.user_id 
              AND datetime(l.datetime) BETWEEN datetime(e.start_time) AND COALESCE(datetime(e.end_time), datetime(e.start_time, '+1 hour'))
            LEFT JOIN places p ON l.place_id = p.id
            LEFT JOIN content c ON e.id = c.event_id
            LEFT JOIN event_person ep ON e.id = ep.event_id
            
            GROUP BY e.user_id, day, hour_bucket
        ''')
        
        # Create hourly aggregation view for real-time features
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS hourly_feature_aggregates AS
            SELECT 
                e.user_id,
                strftime('%Y-%m-%d %H:00:00', e.start_time) as hour_bucket,
                
                COUNT(e.id) as total_events,
                COUNT(DISTINCT c.id) as content_items,
                COUNT(DISTINCT ep.person_id) as people_involved,
                COUNT(DISTINCT l.place_id) as locations_visited,
                
                -- Digital activity aggregation
                COUNT(CASE WHEN c.type LIKE '%work%' THEN 1 END) as work_content,
                COUNT(CASE WHEN c.type LIKE '%communication%' THEN 1 END) as communication_content,
                COUNT(CASE WHEN c.type LIKE '%entertainment%' THEN 1 END) as entertainment_content,
                
                -- Context aggregation
                COUNT(CASE WHEN LOWER(e.name) LIKE '%work%' THEN 1 END) as work_events,
                COUNT(CASE WHEN LOWER(e.name) LIKE '%social%' THEN 1 END) as social_events,
                COUNT(CASE WHEN LOWER(e.name) LIKE '%exercise%' THEN 1 END) as health_events,
                
                AVG(CASE 
                  WHEN e.end_time IS NOT NULL 
                  THEN (strftime('%s', e.end_time) - strftime('%s', e.start_time)) / 60.0 
                  ELSE 60 
                END) as avg_event_duration
                
            FROM events e
            LEFT JOIN content c ON e.id = c.event_id
            LEFT JOIN event_person ep ON e.id = ep.event_id
            LEFT JOIN locations l ON e.user_id = l.user_id 
              AND datetime(l.datetime) BETWEEN datetime(e.start_time) AND COALESCE(datetime(e.end_time), datetime(e.start_time, '+1 hour'))
            
            GROUP BY e.user_id, hour_bucket
        ''')
        
        # Create transition sequences view for B matrix construction
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS state_transition_sequences AS
            WITH ordered_states AS (
              SELECT 
                user_id, day, hour_bucket,
                context_bucket, energy_bucket, mood_bucket,
                digital_behavior_type, location_category, social_category,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY day, hour_bucket) as seq_num
              FROM daily_state_observations
            )
            SELECT 
              curr.user_id,
              curr.day as curr_day,
              curr.hour_bucket as curr_hour,
              curr.context_bucket as curr_context,
              curr.energy_bucket as curr_energy,
              curr.mood_bucket as curr_mood,
              next.context_bucket as next_context,
              next.energy_bucket as next_energy,
              next.mood_bucket as next_mood,
              
              -- Infer action from transition
              CASE 
                WHEN next.context_bucket = 'work' THEN 'work_focus'
                WHEN next.context_bucket = 'social' THEN 'social_engage'
                WHEN next.context_bucket = 'health' THEN 'physical_exercise'
                WHEN next.context_bucket = 'home' THEN 'rest_recover'
                ELSE 'no_action'
              END as inferred_action
              
            FROM ordered_states curr
            JOIN ordered_states next ON curr.user_id = next.user_id 
              AND next.seq_num = curr.seq_num + 1
        ''')

        conn.commit()
        logger.info("Database tables and temporal views created successfully according to CI-Server schema")


def drop_tables():
    """Drop all database tables."""
    with get_db_session() as conn:
        cursor = conn.cursor()
        
        # Drop junction tables first
        junction_tables = [
            'event_person', 'place_tag', 'thing_tag', 'content_tag', 
            'event_tag', 'person_tag'
        ]
        
        # Drop main tables
        main_tables = [
            'contacts', 'things', 'content', 'events', 'locations', 
            'places', 'people', 'tags', 'users'
        ]
        
        for table in junction_tables + main_tables:
            cursor.execute(f'DROP TABLE IF EXISTS {table}')
        
        conn.commit()
        logger.info("Database tables dropped successfully")


def generate_uuid():
    """Generate a UUID string for primary keys."""
    return str(uuid.uuid4())