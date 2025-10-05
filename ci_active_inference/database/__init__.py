"""Database connection and session management using SQLite for demo."""

import sqlite3
import os
from typing import Generator, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Database configuration - using SQLite for simplicity
DATABASE_PATH = os.getenv("DATABASE_PATH", "ci_database.db")


def get_db_connection():
    """Get SQLite database connection."""
    return sqlite3.connect(DATABASE_PATH)


@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row  # Enable dict-like access
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
    """Create all database tables."""
    with get_db_session() as conn:
        cursor = conn.cursor()
        
        # Create persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                role TEXT,
                department TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Create places table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS places (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                address TEXT,
                latitude REAL,
                longitude REAL,
                place_type TEXT,
                capacity INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Create things table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS things (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                thing_type TEXT,
                status TEXT,
                location_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (location_id) REFERENCES places (id)
            )
        ''')
        
        # Create content table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content_text TEXT,
                content_type TEXT,
                creator_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT,
                metadata TEXT,
                FOREIGN KEY (creator_id) REFERENCES persons (id)
            )
        ''')
        
        # Create events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                event_type TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                person_id INTEGER,
                place_id INTEGER,
                thing_id INTEGER,
                content_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (person_id) REFERENCES persons (id),
                FOREIGN KEY (place_id) REFERENCES places (id),
                FOREIGN KEY (thing_id) REFERENCES things (id),
                FOREIGN KEY (content_id) REFERENCES content (id)
            )
        ''')
        
        conn.commit()
        logger.info("Database tables created successfully")


def drop_tables():
    """Drop all database tables."""
    with get_db_session() as conn:
        cursor = conn.cursor()
        tables = ['events', 'content', 'things', 'places', 'persons']
        for table in tables:
            cursor.execute(f'DROP TABLE IF EXISTS {table}')
        conn.commit()
        logger.info("Database tables dropped successfully")