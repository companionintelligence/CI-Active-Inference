#!/usr/bin/env python3
"""
Manual test runner for PersonalLifeAgent without pytest dependency.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/home/runner/work/CI-Active-Inference/CI-Active-Inference')

from ci_active_inference.agents.personal_life_agent import PersonalLifeAgent
from ci_active_inference.database import create_tables, get_db_connection

def run_comprehensive_tests():
    """Run comprehensive tests for PersonalLifeAgent."""
    
    print("=== PersonalLifeAgent Comprehensive Test Suite ===\n")
    
    try:
        # Initialize database
        create_tables()
        print("✓ Database initialized")
        
        # Test 1: Basic agent creation and properties
        print("\n1. Testing agent creation and properties...")
        user_id = "comprehensive_test_user"
        agent = PersonalLifeAgent(user_id)
        
        assert agent.user_id == user_id
        assert agent.num_states == [3, 3, 5]
        assert agent.num_obs == [4, 5, 4, 3]
        assert agent.num_actions == [9]
        assert agent.prediction_count == 0
        print("   ✓ Agent properties correct")
        
        # Test 2: State mapping functions
        print("\n2. Testing state mapping functions...")
        
        # Energy mapping
        assert agent._map_energy_to_index('low') == 0
        assert agent._map_energy_to_index('medium') == 1
        assert agent._map_energy_to_index('high') == 2
        assert agent._map_energy_to_index('unknown') == 1  # Default
        print("   ✓ Energy mapping correct")
        
        # Mood mapping
        assert agent._map_mood_to_index('negative') == 0
        assert agent._map_mood_to_index('neutral') == 1
        assert agent._map_mood_to_index('positive') == 2
        print("   ✓ Mood mapping correct")
        
        # Context mapping
        assert agent._map_context_to_index('work') == 0
        assert agent._map_context_to_index('social') == 1
        assert agent._map_context_to_index('health') == 2
        assert agent._map_context_to_index('home') == 3
        assert agent._map_context_to_index('other') == 4
        print("   ✓ Context mapping correct")
        
        # Test 3: Action mapping
        print("\n3. Testing action mapping...")
        
        assert agent._map_action_to_index('work_focus') == 0
        assert agent._map_action_to_index('social_engage') == 1
        assert agent._map_action_to_index('physical_exercise') == 2
        assert agent._map_action_to_index('no_action') == 8
        
        assert agent._action_to_description(0) == 'work_focus'
        assert agent._action_to_description(1) == 'social_engage'
        assert agent._action_to_description(8) == 'no_action'
        print("   ✓ Action mapping correct")
        
        # Test 4: Observation encoding
        print("\n4. Testing observation encoding...")
        
        sample_obs = {
            'digital_behavior': 'work-focus',
            'location_context': 'office',
            'social_context': 'alone',
            'content_type': 'work'
        }
        
        obs_indices = agent._observations_to_indices(sample_obs)
        assert len(obs_indices) == 4
        assert all(isinstance(idx, int) for idx in obs_indices)
        assert all(0 <= idx < agent.num_obs[i] for i, idx in enumerate(obs_indices))
        print("   ✓ Observation encoding correct")
        
        # Test 5: Agent initialization from data
        print("\n5. Testing agent initialization from data...")
        
        success = agent.initialize_from_data("2024-01-01", "2024-01-31")
        assert success is True
        assert agent.A is not None
        assert agent.B is not None
        assert agent.C is not None
        assert agent.D is not None
        assert agent.agent is not None
        print("   ✓ Agent initialization successful")
        
        # Test 6: Matrix structure validation
        print("\n6. Testing matrix structures...")
        
        # A matrices
        assert len(agent.A) == 4  # Four observation modalities
        for modality in range(4):
            assert len(agent.A[modality]) == agent.num_obs[modality]
        print("   ✓ A matrices structure correct")
        
        # B matrices
        assert len(agent.B) == 3  # Three state factors
        for factor in range(3):
            assert len(agent.B[factor]) == agent.num_states[factor]
        print("   ✓ B matrices structure correct")
        
        # C vectors
        assert len(agent.C) == 4  # Four observation modalities
        print("   ✓ C vectors structure correct")
        
        # D vectors
        assert len(agent.D) == 3  # Three state factors
        for factor in range(3):
            assert len(agent.D[factor]) == agent.num_states[factor]
            # Check normalization
            total = sum(agent.D[factor])
            assert abs(total - 1.0) < 1e-6, f"D vector {factor} not normalized: {total}"
        print("   ✓ D vectors structure and normalization correct")
        
        # Test 7: Prediction pipeline
        print("\n7. Testing prediction pipeline...")
        
        result = agent.predict_and_suggest(sample_obs)
        
        # Check result structure
        expected_keys = [
            'current_state_beliefs', 'suggested_action', 'predicted_default',
            'efe_breakdown', 'confidence'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        print("   ✓ Prediction result structure correct")
        
        # Check state beliefs structure
        beliefs = result['current_state_beliefs']
        assert len(beliefs) == 3  # Three state factors
        for i, factor_beliefs in enumerate(beliefs):
            assert len(factor_beliefs) == agent.num_states[i]
            # Check normalization
            total = sum(factor_beliefs)
            assert abs(total - 1.0) < 1e-6, f"State beliefs {i} not normalized: {total}"
        print("   ✓ State beliefs structure and normalization correct")
        
        # Check action suggestion
        assert isinstance(result['suggested_action'], str)
        valid_actions = [
            'work_focus', 'social_engage', 'physical_exercise', 'rest_recover',
            'learning', 'creative', 'communication', 'entertainment', 'no_action'
        ]
        assert result['suggested_action'] in valid_actions
        print("   ✓ Action suggestion valid")
        
        # Check confidence and EFE
        assert 0.0 <= result['confidence'] <= 1.0
        efe = result['efe_breakdown']
        assert 'risk' in efe and 'ambiguity' in efe
        assert 0.0 <= efe['risk'] <= 1.0
        assert 0.0 <= efe['ambiguity'] <= 1.0
        print("   ✓ Confidence and EFE values valid")
        
        # Test 8: Model statistics
        print("\n8. Testing model statistics...")
        
        stats = agent.get_model_statistics()
        required_stats = ['total_predictions', 'accuracy_last_week', 'uncertainty_level', 'last_updated', 'matrix_shapes']
        for stat in required_stats:
            assert stat in stats, f"Missing statistic: {stat}"
        
        assert stats['total_predictions'] >= 1  # At least one prediction made
        print("   ✓ Model statistics correct")
        
        # Test 9: Update functionality
        print("\n9. Testing update functionality...")
        
        agent.update_from_outcome('work_focus', {
            'digital_behavior': 'work-focus',
            'location_context': 'office',
            'social_context': 'alone',
            'content_type': 'work'
        })
        print("   ✓ Update from outcome successful")
        
        # Test 10: Load or create functionality
        print("\n10. Testing load_or_create functionality...")
        
        agent2 = PersonalLifeAgent.load_or_create(user_id + "_new")
        assert agent2 is not None
        assert agent2.user_id == user_id + "_new"
        print("   ✓ Load or create successful")
        
        # Test 11: Error handling with uninitialized agent
        print("\n11. Testing error handling...")
        
        uninitialized_agent = PersonalLifeAgent("uninit_user")
        result = uninitialized_agent.predict_and_suggest(sample_obs)
        assert 'error' in result
        assert result['suggested_action'] == 'no_action'
        assert result['confidence'] == 0.0
        print("   ✓ Error handling correct")
        
        # Test 12: Matrix normalization
        print("\n12. Testing matrix normalization...")
        
        # Test A matrix normalization
        test_matrix = [
            [[[2, 4, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]],
            [[[1, 3, 5], [7, 9, 11]], [[13, 15, 17], [19, 21, 23]]]
        ]
        
        normalized = agent._normalize_A_matrix(test_matrix)
        
        # Check normalization
        for energy in range(2):
            for mood in range(2):
                for context in range(3):
                    column_sum = sum(normalized[obs][energy][mood][context] for obs in range(2))
                    assert abs(column_sum - 1.0) < 1e-10, f"Column not normalized: {column_sum}"
        print("   ✓ A matrix normalization correct")
        
        # Test B matrix normalization
        test_B = [
            [[1, 2], [3, 4]],  # next_state 0
            [[5, 6], [7, 8]]   # next_state 1
        ]
        
        normalized_B = agent._normalize_B_matrix(test_B)
        
        for curr in range(2):
            for action in range(2):
                column_sum = sum(normalized_B[next_state][curr][action] for next_state in range(2))
                assert abs(column_sum - 1.0) < 1e-10, f"B matrix column not normalized: {column_sum}"
        print("   ✓ B matrix normalization correct")
        
        print("\n=== All Comprehensive Tests Passed! ===")
        
        # Test with sample data
        print("\n=== Testing with Sample Data ===")
        
        conn = get_db_connection()
        # Disable foreign key constraints for testing
        conn.execute("PRAGMA foreign_keys = OFF")
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
            (user_id, 'Work Focus', '2024-01-16 09:00:00', '2024-01-16 11:00:00'),
            (user_id, 'Social Dinner', '2024-01-16 19:00:00', '2024-01-16 21:00:00'),
        ]
        
        for event in sample_events:
            cursor.execute("""
                INSERT INTO events (user_id, name, start_time, end_time, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, event)
        
        conn.commit()
        # Re-enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        conn.close()
        print("✓ Sample data inserted")
        
        # Test agent with real data
        data_agent = PersonalLifeAgent(user_id)
        success = data_agent.initialize_from_data('2024-01-01', '2024-01-31')
        assert success is True
        print("✓ Agent initialized with sample data")
        
        # Test prediction with real data
        result_with_data = data_agent.predict_and_suggest(sample_obs)
        assert 'current_state_beliefs' in result_with_data
        assert 'suggested_action' in result_with_data
        print("✓ Prediction with sample data successful")
        
        print("\n=== PersonalLifeAgent Implementation Complete and Validated! ===")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)