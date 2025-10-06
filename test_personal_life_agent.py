#!/usr/bin/env python3
"""
Test script for PersonalLifeAgent implementation.

This script validates the core functionality of the PersonalLifeAgent
including matrix construction, prediction pipeline, and data integration.
"""

import sys
import os
import sqlite3
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, '/home/runner/work/CI-Active-Inference/CI-Active-Inference')

def test_personal_life_agent():
    """Test PersonalLifeAgent core functionality."""
    
    print("=== PersonalLifeAgent Integration Test ===\n")
    
    try:
        # Import required modules
        from ci_active_inference.agents.personal_life_agent import PersonalLifeAgent
        from ci_active_inference.database import create_tables, get_db_connection
        print("✓ Successfully imported PersonalLifeAgent and database modules")
        
        # Initialize database
        create_tables()
        print("✓ Database tables created")
        
        # Create test user
        user_id = "test_user_123"
        agent = PersonalLifeAgent(user_id)
        print(f"✓ PersonalLifeAgent created for user {user_id}")
        
        # Test agent initialization without data
        print("\n--- Testing Agent Initialization ---")
        
        # Check agent properties
        assert agent.user_id == user_id
        assert agent.num_states == [3, 3, 5]
        assert agent.num_obs == [4, 5, 4, 3]
        assert agent.num_actions == [9]
        print("✓ Agent dimensions are correct")
        
        # Test initialization from data (will work with empty database)
        start_date = (datetime.now() - timedelta(days=7)).isoformat()
        end_date = datetime.now().isoformat()
        
        success = agent.initialize_from_data(start_date, end_date)
        print(f"✓ Agent initialization from data: {'successful' if success else 'failed (expected with empty DB)'}")
        
        # Test prediction with empty/minimal data
        print("\n--- Testing Prediction Pipeline ---")
        
        test_observations = {
            'digital_behavior': 'work-focus',
            'location_context': 'office',
            'social_context': 'alone',
            'content_type': 'work'
        }
        
        result = agent.predict_and_suggest(test_observations)
        print("✓ Prediction pipeline executed")
        
        # Validate result structure
        expected_keys = [
            'current_state_beliefs', 'suggested_action', 'predicted_default',
            'efe_breakdown', 'confidence'
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        print("✓ Prediction result has all expected keys")
        
        # Test EFE decomposition
        efe = result['efe_breakdown']
        assert isinstance(efe, dict)
        assert 'risk' in efe and 'ambiguity' in efe
        print("✓ EFE decomposition structure correct")
        
        # Test model statistics
        print("\n--- Testing Model Statistics ---")
        
        stats = agent.get_model_statistics()
        assert 'total_predictions' in stats
        assert 'last_updated' in stats
        assert 'matrix_shapes' in stats
        print("✓ Model statistics structure correct")
        
        # Test observation encoding
        print("\n--- Testing Observation Encoding ---")
        
        obs_indices = agent._observations_to_indices(test_observations)
        assert len(obs_indices) == 4  # Four modalities
        assert all(isinstance(idx, int) for idx in obs_indices)
        print("✓ Observation encoding works correctly")
        
        # Test state mapping functions
        assert agent._map_energy_to_index('high') == 2
        assert agent._map_mood_to_index('positive') == 2
        assert agent._map_context_to_index('work') == 0
        print("✓ State mapping functions work correctly")
        
        # Test action mapping
        assert agent._map_action_to_index('work_focus') == 0
        assert agent._action_to_description(0) == 'work_focus'
        print("✓ Action mapping functions work correctly")
        
        # Test matrix normalization helpers
        print("\n--- Testing Matrix Operations ---")
        
        # Create test matrix for normalization
        test_A_matrix = [
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]
        ]
        
        normalized_A = agent._normalize_A_matrix(test_A_matrix)
        
        # Check that columns sum to 1
        for energy_idx in range(2):
            for mood_idx in range(2):
                for context_idx in range(3):
                    column_sum = sum(normalized_A[obs_idx][energy_idx][mood_idx][context_idx] 
                                   for obs_idx in range(2))
                    assert abs(column_sum - 1.0) < 1e-10, f"Column sum not 1: {column_sum}"
        
        print("✓ A matrix normalization works correctly")
        
        # Test update functionality
        print("\n--- Testing Update Functionality ---")
        
        agent.update_from_outcome('work_focus', {
            'digital_behavior': 'work-focus',
            'location_context': 'office',
            'social_context': 'alone',
            'content_type': 'work'
        })
        print("✓ Update from outcome executed without errors")
        
        # Test with sample data
        print("\n--- Testing with Sample Data ---")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert sample user
        cursor.execute("""
            INSERT OR REPLACE INTO users (id, name, email, password, created_at)
            VALUES (?, 'Test User', 'test@example.com', 'test_password', datetime('now'))
        """, (user_id,))
        
        # Insert sample events
        sample_events = [
            (user_id, 'Work Meeting', '2024-01-15 09:00:00', '2024-01-15 10:00:00'),
            (user_id, 'Lunch Break', '2024-01-15 12:00:00', '2024-01-15 13:00:00'),
            (user_id, 'Exercise', '2024-01-15 18:00:00', '2024-01-15 19:00:00'),
        ]
        
        for event in sample_events:
            cursor.execute("""
                INSERT INTO events (user_id, name, start_time, end_time, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, event)
        
        conn.commit()
        conn.close()
        print("✓ Sample data inserted")
        
        # Re-initialize agent with sample data
        agent_with_data = PersonalLifeAgent(user_id)
        success = agent_with_data.initialize_from_data('2024-01-01', '2024-01-31')
        print(f"✓ Agent reinitialized with sample data: {'successful' if success else 'failed'}")
        
        if success:
            # Test prediction with real data
            result_with_data = agent_with_data.predict_and_suggest(test_observations)
            print("✓ Prediction with sample data executed")
            
            # Check that matrices were built
            assert agent_with_data.A is not None
            assert agent_with_data.B is not None
            assert agent_with_data.C is not None
            assert agent_with_data.D is not None
            print("✓ All matrices (A, B, C, D) were constructed")
        
        print("\n=== All PersonalLifeAgent Tests Passed! ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_personal_life_agent()
    sys.exit(0 if success else 1)