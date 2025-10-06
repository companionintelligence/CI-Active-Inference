"""
Test PersonalLifeAgent integration with Phase 2 requirements.

This module tests the complete PersonalLifeAgent implementation including:
- Database integration for matrix construction
- Real data-driven A/B/C/D matrix building
- Prediction and suggestion pipeline
- Integration with existing CI infrastructure
"""

import pytest
import sqlite3
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any

from ci_active_inference.agents.personal_life_agent import PersonalLifeAgent
from ci_active_inference.database import create_tables, get_db_connection


class TestPersonalLifeAgent:
    """Test PersonalLifeAgent core functionality."""
    
    @pytest.fixture
    def test_user_id(self):
        """Test user ID."""
        return "test_user_phase2"
    
    @pytest.fixture
    def sample_observations(self):
        """Sample observations for testing."""
        return {
            'digital_behavior': 'work-focus',
            'location_context': 'office',
            'social_context': 'alone',
            'content_type': 'work'
        }
    
    @pytest.fixture
    def sample_data(self, test_user_id):
        """Create sample data in database for testing."""
        # Initialize database
        create_tables()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert test user
        cursor.execute("""
            INSERT OR REPLACE INTO users (id, name, email, password, created_at)
            VALUES (?, 'Test User', 'test@example.com', 'test_password', datetime('now'))
        """, (test_user_id,))
        
        # Insert sample events
        base_date = datetime(2024, 1, 15)
        sample_events = [
            (test_user_id, 'Work Meeting', (base_date + timedelta(hours=9)).isoformat(), 
             (base_date + timedelta(hours=10)).isoformat()),
            (test_user_id, 'Lunch Break', (base_date + timedelta(hours=12)).isoformat(), 
             (base_date + timedelta(hours=13)).isoformat()),
            (test_user_id, 'Exercise', (base_date + timedelta(hours=18)).isoformat(), 
             (base_date + timedelta(hours=19)).isoformat()),
            (test_user_id, 'Work Focus', (base_date + timedelta(days=1, hours=9)).isoformat(), 
             (base_date + timedelta(days=1, hours=11)).isoformat()),
            (test_user_id, 'Social Dinner', (base_date + timedelta(days=1, hours=19)).isoformat(), 
             (base_date + timedelta(days=1, hours=21)).isoformat()),
        ]
        
        for event in sample_events:
            cursor.execute("""
                INSERT INTO events (user_id, name, start_time, end_time, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, event)
        
        # Insert sample content
        event_ids = [row[0] for row in cursor.execute("SELECT id FROM events WHERE user_id = ?", (test_user_id,)).fetchall()]
        for i, event_id in enumerate(event_ids[:3]):
            content_type = ['work', 'communication', 'entertainment'][i % 3]
            cursor.execute("""
                INSERT INTO content (event_id, type, source, created_at)
                VALUES (?, ?, 'test_source', datetime('now'))
            """, (event_id, content_type))
        
        conn.commit()
        conn.close()
        
        yield test_user_id
        
        # Cleanup after test
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM content WHERE event_id IN (SELECT id FROM events WHERE user_id = ?)", (test_user_id,))
        cursor.execute("DELETE FROM events WHERE user_id = ?", (test_user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (test_user_id,))
        conn.commit()
        conn.close()
    
    def test_agent_initialization(self, test_user_id):
        """Test PersonalLifeAgent initialization."""
        agent = PersonalLifeAgent(test_user_id)
        
        assert agent.user_id == test_user_id
        assert agent.num_states == [3, 3, 5]  # [Energy, Mood, Context]
        assert agent.num_obs == [4, 5, 4, 3]  # [Digital, Location, Social, Content]
        assert agent.num_actions == [9]
        assert agent.prediction_count == 0
        assert agent.A is None  # Not initialized yet
        assert agent.B is None
        assert agent.C is None
        assert agent.D is None
    
    def test_initialization_from_data(self, sample_data):
        """Test agent initialization from database data."""
        agent = PersonalLifeAgent(sample_data)
        
        # Initialize from data
        start_date = "2024-01-01"
        end_date = "2024-01-31"
        success = agent.initialize_from_data(start_date, end_date)
        
        assert success is True
        assert agent.A is not None
        assert agent.B is not None
        assert agent.C is not None
        assert agent.D is not None
        assert agent.agent is not None
        assert agent.last_update is not None
    
    def test_matrix_construction(self, sample_data):
        """Test A/B/C/D matrix construction from real data."""
        agent = PersonalLifeAgent(sample_data)
        agent.initialize_from_data("2024-01-01", "2024-01-31")
        
        # Test A matrices (observation likelihood)
        assert len(agent.A) == 4  # Four observation modalities
        for modality in range(4):
            assert len(agent.A[modality]) == agent.num_obs[modality]
        
        # Test B matrices (state transitions)
        assert len(agent.B) == 3  # Three state factors
        for factor in range(3):
            assert len(agent.B[factor]) == agent.num_states[factor]
        
        # Test C vectors (preferences)
        assert len(agent.C) == 4  # Four observation modalities
        for modality in range(4):
            assert len(agent.C[modality]) == agent.num_obs[modality]
        
        # Test D vectors (priors)
        assert len(agent.D) == 3  # Three state factors
        for factor in range(3):
            assert len(agent.D[factor]) == agent.num_states[factor]
            # Check normalization
            assert abs(sum(agent.D[factor]) - 1.0) < 1e-6
    
    def test_predict_and_suggest_pipeline(self, sample_data, sample_observations):
        """Test the main predict_and_suggest pipeline."""
        agent = PersonalLifeAgent(sample_data)
        agent.initialize_from_data("2024-01-01", "2024-01-31")
        
        result = agent.predict_and_suggest(sample_observations)
        
        # Check result structure
        expected_keys = [
            'current_state_beliefs', 'suggested_action', 'predicted_default',
            'efe_breakdown', 'confidence'
        ]
        for key in expected_keys:
            assert key in result
        
        # Check state beliefs
        beliefs = result['current_state_beliefs']
        assert len(beliefs) == 3  # Three state factors
        for factor_beliefs in beliefs:
            assert len(factor_beliefs) == agent.num_states[beliefs.index(factor_beliefs)]
            # Check normalization
            assert abs(sum(factor_beliefs) - 1.0) < 1e-6
        
        # Check action suggestion
        assert isinstance(result['suggested_action'], str)
        assert result['suggested_action'] in [
            'work_focus', 'social_engage', 'physical_exercise', 'rest_recover',
            'learning', 'creative', 'communication', 'entertainment', 'no_action'
        ]
        
        # Check confidence
        assert 0.0 <= result['confidence'] <= 1.0
        
        # Check EFE breakdown
        efe = result['efe_breakdown']
        assert 'risk' in efe and 'ambiguity' in efe
        assert 0.0 <= efe['risk'] <= 1.0
        assert 0.0 <= efe['ambiguity'] <= 1.0
    
    def test_observation_encoding(self, test_user_id, sample_observations):
        """Test observation encoding to indices."""
        agent = PersonalLifeAgent(test_user_id)
        
        obs_indices = agent._observations_to_indices(sample_observations)
        
        assert len(obs_indices) == 4  # Four modalities
        assert all(isinstance(idx, int) for idx in obs_indices)
        assert all(0 <= idx < agent.num_obs[i] for i, idx in enumerate(obs_indices))
    
    def test_state_mapping_functions(self, test_user_id):
        """Test state mapping functions."""
        agent = PersonalLifeAgent(test_user_id)
        
        # Energy mapping
        assert agent._map_energy_to_index('low') == 0
        assert agent._map_energy_to_index('medium') == 1
        assert agent._map_energy_to_index('high') == 2
        assert agent._map_energy_to_index('unknown') == 1  # Default
        
        # Mood mapping
        assert agent._map_mood_to_index('negative') == 0
        assert agent._map_mood_to_index('neutral') == 1
        assert agent._map_mood_to_index('positive') == 2
        
        # Context mapping
        assert agent._map_context_to_index('work') == 0
        assert agent._map_context_to_index('social') == 1
        assert agent._map_context_to_index('health') == 2
        assert agent._map_context_to_index('home') == 3
        assert agent._map_context_to_index('other') == 4
    
    def test_action_mapping(self, test_user_id):
        """Test action mapping functions."""
        agent = PersonalLifeAgent(test_user_id)
        
        # Action to index
        assert agent._map_action_to_index('work_focus') == 0
        assert agent._map_action_to_index('social_engage') == 1
        assert agent._map_action_to_index('physical_exercise') == 2
        assert agent._map_action_to_index('no_action') == 8
        
        # Index to description
        assert agent._action_to_description(0) == 'work_focus'
        assert agent._action_to_description(1) == 'social_engage'
        assert agent._action_to_description(8) == 'no_action'
        assert agent._action_to_description(99) == 'no_action'  # Out of range
    
    def test_matrix_normalization(self, test_user_id):
        """Test matrix normalization functions."""
        agent = PersonalLifeAgent(test_user_id)
        
        # Test A matrix normalization
        test_matrix = [
            [[[2, 4, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]],
            [[[1, 3, 5], [7, 9, 11]], [[13, 15, 17], [19, 21, 23]]]
        ]
        
        normalized = agent._normalize_A_matrix(test_matrix)
        
        # Check that each column sums to 1
        for energy in range(2):
            for mood in range(2):
                for context in range(3):
                    column_sum = sum(normalized[obs][energy][mood][context] for obs in range(2))
                    assert abs(column_sum - 1.0) < 1e-10
        
        # Test B matrix normalization
        test_B = [
            [[1, 2], [3, 4]],  # next_state 0
            [[5, 6], [7, 8]]   # next_state 1
        ]
        
        normalized_B = agent._normalize_B_matrix(test_B)
        
        # Check that each [current_state, action] column sums to 1
        for curr in range(2):
            for action in range(2):
                column_sum = sum(normalized_B[next_state][curr][action] for next_state in range(2))
                assert abs(column_sum - 1.0) < 1e-10
    
    def test_update_from_outcome(self, sample_data):
        """Test updating agent from actual outcomes."""
        agent = PersonalLifeAgent(sample_data)
        agent.initialize_from_data("2024-01-01", "2024-01-31")
        
        # Test update (should not raise errors)
        agent.update_from_outcome('work_focus', {
            'digital_behavior': 'work-focus',
            'location_context': 'office',
            'social_context': 'alone',
            'content_type': 'work'
        })
        
        # Should complete without errors
        assert True
    
    def test_model_statistics(self, sample_data):
        """Test model statistics reporting."""
        agent = PersonalLifeAgent(sample_data)
        agent.initialize_from_data("2024-01-01", "2024-01-31")
        
        # Make a prediction to increase count
        agent.predict_and_suggest({
            'digital_behavior': 'work-focus',
            'location_context': 'office',
            'social_context': 'alone',
            'content_type': 'work'
        })
        
        stats = agent.get_model_statistics()
        
        assert 'total_predictions' in stats
        assert 'accuracy_last_week' in stats
        assert 'uncertainty_level' in stats
        assert 'last_updated' in stats
        assert 'matrix_shapes' in stats
        
        assert stats['total_predictions'] >= 1
        assert isinstance(stats['accuracy_last_week'], float)
        assert isinstance(stats['uncertainty_level'], float)
        assert stats['last_updated'] is not None
    
    def test_uninitialized_agent_handling(self, test_user_id, sample_observations):
        """Test that uninitialized agent handles requests gracefully."""
        agent = PersonalLifeAgent(test_user_id)
        
        # Should not crash and return error information
        result = agent.predict_and_suggest(sample_observations)
        assert 'error' in result
        assert result['suggested_action'] == 'no_action'
        assert result['confidence'] == 0.0
    
    def test_invalid_observation_handling(self, sample_data):
        """Test handling of invalid observations."""
        agent = PersonalLifeAgent(sample_data)
        agent.initialize_from_data("2024-01-01", "2024-01-31")
        
        # Test with empty observations
        result = agent.predict_and_suggest({})
        assert 'current_state_beliefs' in result
        
        # Test with invalid values
        result = agent.predict_and_suggest({
            'digital_behavior': 'invalid_behavior',
            'location_context': 'invalid_location',
            'social_context': 'invalid_social',
            'content_type': 'invalid_content'
        })
        assert 'current_state_beliefs' in result
    
    def test_database_integration(self, sample_data):
        """Test integration with database views and queries."""
        agent = PersonalLifeAgent(sample_data)
        
        # Test that agent can query database views
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check that temporal views exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='daily_state_observations'")
        assert cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='state_transition_sequences'")
        assert cursor.fetchone() is not None
        
        conn.close()
        
        # Test initialization uses these views
        success = agent.initialize_from_data("2024-01-01", "2024-01-31")
        assert success is True
    
    def test_scaling_with_data_volume(self, test_user_id):
        """Test that matrix construction scales with data volume."""
        # This test ensures the agent can handle varying amounts of data
        agent = PersonalLifeAgent(test_user_id)
        
        # Test with minimal data
        success = agent.initialize_from_data("2024-01-01", "2024-01-31")
        assert success is True  # Should work even with empty database
        
        # Matrices should be initialized with smoothing
        assert agent.A is not None
        assert agent.B is not None
        assert agent.C is not None
        assert agent.D is not None