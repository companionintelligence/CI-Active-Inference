"""
Tests for Active Inference agents
"""

import pytest
from ci_active_inference.agents import CIActiveInferenceAgent, CIBehaviorOptimizer
from ci_active_inference.database import get_db_session
from ci_active_inference.database.dao import DataAnalyzer


class TestCIActiveInferenceAgent:
    """Test CI Active Inference Agent."""
    
    def test_agent_initialization(self):
        """Test that agent initializes correctly."""
        agent = CIActiveInferenceAgent()
        
        assert agent is not None
        assert hasattr(agent, 'predict_future_state')
        assert hasattr(agent, 'recommend_actions')
        assert hasattr(agent, 'update_beliefs')
    
    def test_predict_future_state(self, test_user):
        """Test future state prediction."""
        agent = CIActiveInferenceAgent()
        
        # Create a mock current state
        current_state = {
            'timestamp': '2025-01-01T12:00:00',
            'user_id': test_user['id'],
            'active_events': 1,
            'available_resources': 5,
            'recent_content_items': 3,
            'total_people': 2,
            'total_places': 2,
            'total_things': 4,
            'hour_of_day': 12,
            'day_of_week': 1,
            'week_of_year': 1
        }
        
        # Test prediction
        predictions = agent.predict_future_state(current_state, time_steps=3)
        
        assert len(predictions) == 3
        for i, pred in enumerate(predictions):
            assert 'time_step' in pred
            assert pred['time_step'] == i + 1
            assert 'confidence' in pred
            assert 0 <= pred['confidence'] <= 1
    
    def test_recommend_actions(self, test_user):
        """Test action recommendations."""
        agent = CIActiveInferenceAgent()
        
        # Create a mock current state
        current_state = {
            'timestamp': '2025-01-01T12:00:00',
            'user_id': test_user['id'],
            'active_events': 1,
            'available_resources': 5,
            'hour_of_day': 12
        }
        
        # Test recommendations
        recommendations = agent.recommend_actions(current_state)
        
        assert 'action_type' in recommendations
        assert 'confidence' in recommendations
        assert 'timestamp' in recommendations
        assert 0 <= recommendations['confidence'] <= 1
    
    def test_update_beliefs(self, test_user):
        """Test belief updating."""
        agent = CIActiveInferenceAgent()
        
        # Create mock observations
        observations = {
            'timestamp': '2025-01-01T12:00:00',
            'user_id': test_user['id'],
            'active_events': 2,
            'available_resources': 3
        }
        
        # Test belief update (should not raise an error)
        agent.update_beliefs(observations)
    
    def test_get_learning_progress(self):
        """Test learning progress tracking."""
        agent = CIActiveInferenceAgent()
        
        # Get initial progress
        progress = agent.get_learning_progress()
        
        assert 'total_observations' in progress
        assert 'recent_confidence' in progress
        assert 'confidence_trend' in progress
        assert 'learning_rate' in progress
        assert progress['total_observations'] >= 0


class TestCIBehaviorOptimizer:
    """Test CI Behavior Optimizer."""
    
    def test_optimizer_initialization(self, test_user):
        """Test that optimizer initializes correctly."""
        agent = CIActiveInferenceAgent()
        
        with get_db_session() as conn:
            analyzer = DataAnalyzer(conn, test_user['id'])
            optimizer = CIBehaviorOptimizer(agent, analyzer)
            
            assert optimizer is not None
            assert hasattr(optimizer, 'optimize_user_schedule')
            assert hasattr(optimizer, 'optimize_resource_allocation')
    
    def test_optimize_resource_allocation(self, test_user):
        """Test resource allocation optimization."""
        agent = CIActiveInferenceAgent()
        
        with get_db_session() as conn:
            analyzer = DataAnalyzer(conn, test_user['id'])
            optimizer = CIBehaviorOptimizer(agent, analyzer)
            
            # Test resource optimization
            optimization = optimizer.optimize_resource_allocation()
            
            assert 'total_resources' in optimization
            assert 'allocation_strategy' in optimization
            assert 'expected_efficiency' in optimization
            assert 'timestamp' in optimization
            assert 0 <= optimization['expected_efficiency'] <= 1
    
    def test_optimize_user_schedule(self, test_user):
        """Test user schedule optimization."""
        agent = CIActiveInferenceAgent()
        
        with get_db_session() as conn:
            # Create a test person first
            from ci_active_inference.database.dao import PersonDAO
            person_dao = PersonDAO(conn)
            person_data = {'first_name': 'Test', 'last_name': 'User', 'type': 'self'}
            person = person_dao.create(person_data, test_user['id'])
            
            analyzer = DataAnalyzer(conn, test_user['id'])
            optimizer = CIBehaviorOptimizer(agent, analyzer)
            
            # Test schedule optimization
            optimization = optimizer.optimize_user_schedule(person['id'])
            
            assert 'person_id' in optimization
            assert 'current_patterns' in optimization
            assert 'predicted_outcomes' in optimization
            assert 'recommendations' in optimization
            assert 'optimization_score' in optimization
            assert optimization['person_id'] == person['id']