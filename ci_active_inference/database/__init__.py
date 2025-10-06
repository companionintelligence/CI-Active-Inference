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
        
        conn.commit()
        logger.info("Database tables created successfully according to CI-Server schema")


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