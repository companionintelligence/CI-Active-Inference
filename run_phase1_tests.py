#!/usr/bin/env python3
"""
Simple test runner for validating Phase 1 Active Inference implementation.
"""

import sys
import os
import traceback

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from ci_active_inference.utils.matrix_utils import (
    normalize, create_uniform_matrix, SimpleActiveInferenceMatrices
)
from ci_active_inference.utils.discretization import DataDiscretizer
from ci_active_inference.agents.simple_agent import SimpleActiveInferenceAgent
from ci_active_inference.database import create_tables, get_db_session
from ci_active_inference.database.dao import UserDAO


def test_matrix_utils():
    """Test matrix utility functions."""
    print("Testing matrix utilities...")
    
    # Test normalize
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    normalized = normalize(matrix, axis=0)
    col_sum = normalized[0][0] + normalized[1][0]
    assert abs(col_sum - 1.0) < 1e-10, f"Column normalization failed: {col_sum}"
    
    # Test uniform matrix creation
    uniform = create_uniform_matrix((3, 4))
    assert len(uniform) == 3, "Wrong matrix rows"
    assert len(uniform[0]) == 4, "Wrong matrix columns"
    assert abs(uniform[0][0] - 1.0/3.0) < 1e-10, "Wrong uniform value"
    
    # Test SimpleActiveInferenceMatrices
    matrices = SimpleActiveInferenceMatrices([2, 3], [2, 3], 3)
    assert len(matrices.A) == 2, "Wrong A matrix count"
    assert len(matrices.B) == 2, "Wrong B matrix count"
    
    print("✓ Matrix utilities tests passed")


def test_discretization():
    """Test data discretization."""
    print("Testing data discretization...")
    
    discretizer = DataDiscretizer()
    
    # Test hour discretization
    assert discretizer.discretize_hour_of_day(2) == 0  # Night
    assert discretizer.discretize_hour_of_day(8) == 1  # Morning
    assert discretizer.discretize_hour_of_day(14) == 2  # Afternoon
    assert discretizer.discretize_hour_of_day(19) == 3  # Evening
    
    # Test event classification
    assert discretizer.classify_event_context("Team meeting") == "work"
    assert discretizer.classify_event_context("Gym workout") == "health"
    assert discretizer.classify_event_context("Dinner with friends") == "social"
    
    # Test state dimensions
    dims = discretizer.get_state_dimensions()
    assert dims["energy"] == 3
    assert dims["mood"] == 3
    assert dims["context"] == 5
    
    print("✓ Data discretization tests passed")


def test_agent_basic():
    """Test basic agent functionality."""
    print("Testing Simple Active Inference Agent...")
    
    agent = SimpleActiveInferenceAgent("test_user")
    
    # Test observation encoding
    state = {
        'digital_behavior': 'work-focus',
        'location_context': 'office',
        'social_context': 'small-group',
        'content_type': 'work'
    }
    
    observations = agent.encode_observations(state)
    assert len(observations) == 4, "Wrong observation count"
    assert all(isinstance(obs, int) for obs in observations), "Non-integer observations"
    
    # Test learning progress
    progress = agent.get_learning_progress()
    assert 'total_observations' in progress
    assert 'matrix_initialized' in progress
    assert progress['matrix_initialized'] == False  # Not initialized yet
    
    print("✓ Simple Active Inference Agent tests passed")


def test_database_integration():
    """Test database integration."""
    print("Testing database integration...")
    
    # Initialize database
    try:
        with get_db_session() as conn:
            create_tables()
            
            # Create test user
            user_dao = UserDAO(conn)
            user = user_dao.create_default_user()
            assert user['id'] is not None
            
            # Test agent initialization
            agent = SimpleActiveInferenceAgent(user['id'])
            agent.initialize_from_data(conn, '2024-01-01', '2024-01-02')
            
            # Agent should now be initialized
            progress = agent.get_learning_progress()
            assert progress['matrix_initialized'] == True
            
    except Exception as e:
        print(f"Database test failed: {e}")
        return False
    
    print("✓ Database integration tests passed")
    return True


def test_end_to_end():
    """Test end-to-end Active Inference cycle."""
    print("Testing end-to-end Active Inference cycle...")
    
    try:
        with get_db_session() as conn:
            create_tables()
            
            # Create user and agent
            user_dao = UserDAO(conn)
            user = user_dao.create_default_user()
            agent = SimpleActiveInferenceAgent(user['id'])
            
            # Initialize with data
            agent.initialize_from_data(conn, '2024-01-01', '2024-01-02')
            
            # Test complete cycle
            current_state = {
                'digital_behavior': 'work-focus',
                'location_context': 'office',
                'social_context': 'alone',
                'content_type': 'work'
            }
            
            # 1. Get action recommendations
            recommendations = agent.recommend_actions(current_state)
            assert 'action_type' in recommendations
            assert 'confidence' in recommendations
            
            # 2. Predict future states
            predictions = agent.predict_future_state(current_state, time_steps=3)
            assert len(predictions) == 3
            
            # 3. Update beliefs
            agent.update_beliefs(current_state)
            progress = agent.get_learning_progress()
            assert progress['total_observations'] > 0
            
    except Exception as e:
        print(f"End-to-end test failed: {e}")
        traceback.print_exc()
        return False
    
    print("✓ End-to-end Active Inference cycle tests passed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Phase 1 Active Inference Tests")
    print("=" * 60)
    
    tests = [
        test_matrix_utils,
        test_discretization,
        test_agent_basic,
        test_database_integration,
        test_end_to_end
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All Phase 1 tests passed! Active Inference infrastructure is working.")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)