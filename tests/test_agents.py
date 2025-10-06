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
        agent = CIActiveInferenceAgent(user_id="test_user")
        
        assert agent is not None
        assert hasattr(agent, 'predict_and_suggest')
        assert hasattr(agent, 'update_model')
        assert hasattr(agent, 'get_learning_progress')
        assert agent.user_id == "test_user"
    def test_predict_and_suggest(self, test_user):
        """Test prediction and suggestion functionality."""
        agent = CIActiveInferenceAgent(user_id=test_user['id'])
        
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
        
        # Test prediction and suggestion
        result = agent.predict_and_suggest(current_state)
        
        assert 'current_beliefs' in result
        assert 'suggested_action' in result
        assert 'confidence' in result
        assert 'efe_breakdown' in result
        assert 'user_id' in result
        assert result['user_id'] == test_user['id']
        assert 0 <= result['confidence'] <= 1
    
    def test_model_learning(self, test_user):
        """Test model learning with Dirichlet updates."""
        agent = CIActiveInferenceAgent(user_id=test_user['id'])
        
        # Create mock observations
        observations = [1, 2, 0, 1]  # observation indices for each modality
        actions = [3]  # action index
        
        # Test model update
        learning_result = agent.update_model(observations, actions)
        
        assert 'learning_iterations' in learning_result
        assert 'model_entropy' in learning_result
        assert learning_result['learning_iterations'] >= 1
    
    def test_model_versioning(self, test_user):
        """Test model save and load functionality."""
        agent = CIActiveInferenceAgent(user_id=test_user['id'])
        
        # Save model
        model_data = agent.save_model()
        
        assert 'user_id' in model_data
        assert 'A_matrices' in model_data
        assert 'B_matrices' in model_data
        assert 'saved_at' in model_data
        assert model_data['user_id'] == test_user['id']
        
        # Create new agent and load model
        new_agent = CIActiveInferenceAgent(user_id=test_user['id'])
        new_agent.load_model(model_data)
        
        assert new_agent.user_id == test_user['id']
    
    def test_get_learning_progress(self, test_user):
        """Test learning progress tracking."""
        agent = CIActiveInferenceAgent(user_id=test_user['id'])
        
        # Get initial progress
        progress = agent.get_learning_progress()
        
        assert 'total_observations' in progress
        assert 'recent_confidence' in progress
        assert 'confidence_trend' in progress
        assert 'learning_iterations' in progress
        assert 'user_id' in progress
        assert progress['total_observations'] >= 0
        assert progress['user_id'] == test_user['id']


class TestCIBehaviorOptimizer:
    """Test CI Behavior Optimizer."""
    
    def test_optimizer_initialization(self, test_user):
        """Test that optimizer initializes correctly."""
        agent = CIActiveInferenceAgent(user_id=test_user['id'])
        
        with get_db_session() as conn:
            analyzer = DataAnalyzer(conn, test_user['id'])
            optimizer = CIBehaviorOptimizer(agent, analyzer)
            
            assert optimizer is not None
            assert hasattr(optimizer, 'optimize_user_schedule')
            assert hasattr(optimizer, 'optimize_resource_allocation')
    
    def test_optimize_resource_allocation(self, test_user):
        """Test resource allocation optimization."""
        agent = CIActiveInferenceAgent(user_id=test_user['id'])
        
        with get_db_session() as conn:
            analyzer = DataAnalyzer(conn, test_user['id'])
            optimizer = CIBehaviorOptimizer(agent, analyzer)
            
            # Test resource optimization
            optimization = optimizer.optimize_resource_allocation()
            
            assert 'predictions' in optimization
            assert 'recommendations' in optimization
            assert 'optimization_score' in optimization
            assert 'timestamp' in optimization
            assert 0 <= optimization['optimization_score'] <= 1
    
    def test_optimize_user_schedule(self, test_user):
        """Test user schedule optimization."""
        agent = CIActiveInferenceAgent(user_id=test_user['id'])
        
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
            assert 'recommendations' in optimization
            assert 'optimization_score' in optimization
            assert optimization['person_id'] == person['id']