"""
Test configuration and fixtures for CI Active Inference Slopbox
"""

import pytest
import tempfile
import os
import shutil
from contextlib import contextmanager

from ci_active_inference.database import get_db_session, create_tables
from ci_active_inference.database.dao import UserDAO


@pytest.fixture(scope="session")
def test_db():
    """Create a temporary test database."""
    # Create temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    # Set environment variable for test database
    original_db_path = os.environ.get('DATABASE_PATH')
    os.environ['DATABASE_PATH'] = db_path
    
    # Create tables
    create_tables()
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)
    
    # Restore original database path
    if original_db_path:
        os.environ['DATABASE_PATH'] = original_db_path
    elif 'DATABASE_PATH' in os.environ:
        del os.environ['DATABASE_PATH']


@pytest.fixture
def test_user(test_db):
    """Create a test user."""
    with get_db_session() as conn:
        user_dao = UserDAO(conn)
        user = user_dao.create_default_user()
        yield user


@contextmanager
def temporary_env_var(key, value):
    """Temporarily set an environment variable."""
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value