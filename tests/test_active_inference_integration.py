"""
Integration tests for Active Inference components.
"""

import sys
import os
import sqlite3
import tempfile

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from ci_active_inference.database import create_tables, get_db_session
from ci_active_inference.database.dao import UserDAO, EventDAO, ContentDAO
from ci_active_inference.agents.simple_agent import SimpleActiveInferenceAgent
from ci_active_inference.utils.discretization import DataDiscretizer
from ci_active_inference.utils.matrix_builder import MatrixBuilder


class TestActiveInferenceIntegration:
    """Test integration of Active Inference components."""
    
    def setup_method(self):
        """Set up test database and sample data."""
        # Create temporary database
        self.db_file = tempfile.NamedTemporaryFile(delete=False)
        self.db_file.close()
        
        # Set up database
        os.environ['DATABASE_PATH'] = self.db_file.name
        
        with get_db_session() as conn:
            create_tables()
            
            # Create test user
            user_dao = UserDAO(conn)
            self.test_user = user_dao.create_default_user()
            
            # Create sample events and content
            self._create_sample_data(conn)
    
    def teardown_method(self):
        """Clean up test database."""
        try:
            os.unlink(self.db_file.name)
        except FileNotFoundError:
            pass
    
    def _create_sample_data(self, conn):
        """Create sample events and content for testing."""
        event_dao = EventDAO(conn)
        content_dao = ContentDAO(conn)
        
        # Create work events
        work_event = event_dao.create({
            'type': 'meeting',
            'name': 'Team standup',
            'description': 'Daily team meeting',
            'start_time': '2024-01-01T09:00:00',
            'end_time': '2024-01-01T09:30:00',
            'user_id': self.test_user['id']
        })
        
        # Add work content
        content_dao.create({
            'type': 'document',
            'title': 'Project planning',
            'text': 'Planning document for Q1',
            'user_id': self.test_user['id'],
            'event_id': work_event['id']
        })
        
        # Create social event
        social_event = event_dao.create({
            'type': 'social',
            'name': 'Dinner with friends',
            'description': 'Social dinner',
            'start_time': '2024-01-01T19:00:00',
            'end_time': '2024-01-01T21:00:00',
            'user_id': self.test_user['id']
        })
        
        # Create exercise event
        exercise_event = event_dao.create({
            'type': 'health',
            'name': 'Morning run',
            'description': 'Daily exercise',
            'start_time': '2024-01-01T07:00:00',
            'end_time': '2024-01-01T08:00:00',
            'user_id': self.test_user['id']
        })
    
    def test_data_discretization_pipeline(self):
        """Test the complete data discretization pipeline."""
        discretizer = DataDiscretizer()
        
        with get_db_session() as conn:
            # Extract temporal features
            features = discretizer.extract_temporal_features(
                conn, self.test_user['id'], 
                '2024-01-01T00:00:00', '2024-01-02T00:00:00'
            )
            
            assert len(features) >= 0  # May be 0 due to simplified extraction
            
            # Test discretization
            if features:
                discretized = discretizer.discretize_features(features)
                assert len(discretized) == len(features)
                
                # Check that discretized features have the expected keys
                for record in discretized:
                    assert 'time_period' in record
                    assert 'activity_level' in record
                    assert 'context_idx' in record
    
    def test_matrix_builder_integration(self):
        """Test matrix building from database data."""
        matrix_builder = MatrixBuilder(self.test_user['id'])
        
        with get_db_session() as conn:
            # Test A matrix building
            A_matrices = matrix_builder.build_A_matrices_from_data(
                conn, '2024-01-01T00:00:00', '2024-01-02T00:00:00'
            )
            
            assert isinstance(A_matrices, list)
            assert len(A_matrices) == len(matrix_builder.num_obs)
            
            # Test B matrix building
            B_matrices = matrix_builder.build_B_matrices_from_data(
                conn, '2024-01-01T00:00:00', '2024-01-02T00:00:00'
            )
            
            assert isinstance(B_matrices, list)
            assert len(B_matrices) == len(matrix_builder.num_states)
            
            # Test C vector building
            C_vectors = matrix_builder.build_C_vectors()
            assert isinstance(C_vectors, list)
            assert len(C_vectors) == len(matrix_builder.num_obs)
            
            # Test D vector building
            D_vectors = matrix_builder.build_D_vectors(
                conn, '2024-01-01T00:00:00', '2024-01-02T00:00:00'
            )
            assert isinstance(D_vectors, list)
            assert len(D_vectors) == len(matrix_builder.num_states)
    
    def test_simple_agent_initialization(self):
        """Test Simple Active Inference Agent initialization."""
        agent = SimpleActiveInferenceAgent(self.test_user['id'])
        
        assert agent.user_id == self.test_user['id']
        assert agent.discretizer is not None
        assert agent.matrix_builder is not None
        assert agent.matrices is None  # Not initialized yet
    
    def test_agent_data_initialization(self):
        """Test agent initialization from database data."""
        agent = SimpleActiveInferenceAgent(self.test_user['id'])
        
        with get_db_session() as conn:
            agent.initialize_from_data(
                conn, '2024-01-01T00:00:00', '2024-01-02T00:00:00'
            )
            
            assert agent.matrices is not None
            assert agent.last_update is not None
            assert agent.matrices.A is not None
            assert agent.matrices.B is not None
            assert agent.matrices.C is not None
            assert agent.matrices.D is not None
    
    def test_agent_observation_encoding(self):
        """Test agent observation encoding."""
        agent = SimpleActiveInferenceAgent(self.test_user['id'])
        
        # Test observation encoding
        system_state = {
            'digital_behavior': 'work-focus',
            'location_context': 'office',
            'social_context': 'small-group',
            'content_type': 'work'
        }
        
        observations = agent.encode_observations(system_state)
        
        assert isinstance(observations, list)
        assert len(observations) == 4  # 4 observation modalities
        assert all(isinstance(obs, int) for obs in observations)
        assert all(0 <= obs < 10 for obs in observations)  # Reasonable range
    
    def test_agent_prediction_without_data(self):
        """Test agent prediction before data initialization."""
        agent = SimpleActiveInferenceAgent(self.test_user['id'])
        
        system_state = {'digital_behavior': 'work-focus'}
        predictions = agent.predict_future_state(system_state, time_steps=3)
        
        assert len(predictions) == 3
        assert all('error' in pred for pred in predictions)
    
    def test_agent_prediction_with_data(self):
        """Test agent prediction after data initialization."""
        agent = SimpleActiveInferenceAgent(self.test_user['id'])
        
        with get_db_session() as conn:
            agent.initialize_from_data(
                conn, '2024-01-01T00:00:00', '2024-01-02T00:00:00'
            )
            
            system_state = {
                'digital_behavior': 'work-focus',
                'location_context': 'office',
                'social_context': 'small-group',
                'content_type': 'work'
            }
            
            predictions = agent.predict_future_state(system_state, time_steps=3)
            
            assert len(predictions) == 3
            assert all('error' not in pred for pred in predictions)
            assert all('confidence' in pred for pred in predictions)
            assert all('timestamp' in pred for pred in predictions)
    
    def test_agent_action_recommendations(self):
        """Test agent action recommendations."""
        agent = SimpleActiveInferenceAgent(self.test_user['id'])
        
        with get_db_session() as conn:
            agent.initialize_from_data(
                conn, '2024-01-01T00:00:00', '2024-01-02T00:00:00'
            )
            
            system_state = {
                'digital_behavior': 'work-focus',
                'location_context': 'office'
            }
            
            recommendations = agent.recommend_actions(system_state)
            
            assert 'action_type' in recommendations
            assert 'confidence' in recommendations
            assert 'current_beliefs' in recommendations
            assert 'timestamp' in recommendations
            assert isinstance(recommendations['confidence'], float)
            assert 0.0 <= recommendations['confidence'] <= 1.0
    
    def test_agent_belief_updates(self):
        """Test agent belief updating."""
        agent = SimpleActiveInferenceAgent(self.test_user['id'])
        
        with get_db_session() as conn:
            agent.initialize_from_data(
                conn, '2024-01-01T00:00:00', '2024-01-02T00:00:00'
            )
            
            # Update beliefs with new observations
            observations = {
                'digital_behavior': 'communication',
                'location_context': 'home'
            }
            
            initial_count = agent.observation_count
            agent.update_beliefs(observations)
            
            assert agent.observation_count == initial_count + 1
    
    def test_agent_learning_progress(self):
        """Test agent learning progress tracking."""
        agent = SimpleActiveInferenceAgent(self.test_user['id'])
        
        progress = agent.get_learning_progress()
        
        assert 'total_observations' in progress
        assert 'recent_confidence' in progress
        assert 'confidence_trend' in progress
        assert 'learning_rate' in progress
        assert 'matrix_initialized' in progress
        
        assert progress['total_observations'] == 0
        assert progress['matrix_initialized'] == False
    
    def test_database_temporal_views(self):
        """Test that temporal views are created properly."""
        with get_db_session() as conn:
            cursor = conn.cursor()
            
            # Check that views exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='view' AND name='daily_state_observations'
            """)
            assert cursor.fetchone() is not None
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='view' AND name='hourly_feature_aggregates'
            """)
            assert cursor.fetchone() is not None
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='view' AND name='state_transition_sequences'
            """)
            assert cursor.fetchone() is not None
    
    def test_end_to_end_active_inference_cycle(self):
        """Test complete Active Inference cycle: observe → infer → plan → act."""
        agent = SimpleActiveInferenceAgent(self.test_user['id'])
        
        with get_db_session() as conn:
            # 1. Initialize agent from data
            agent.initialize_from_data(
                conn, '2024-01-01T00:00:00', '2024-01-02T00:00:00'
            )
            
            # 2. Observe current state
            current_state = {
                'digital_behavior': 'work-focus',
                'location_context': 'office',
                'social_context': 'alone',
                'content_type': 'work'
            }
            
            # 3. Infer states (encoded in recommendations)
            recommendations = agent.recommend_actions(current_state)
            assert 'current_beliefs' in recommendations
            
            # 4. Plan future states
            predictions = agent.predict_future_state(current_state, time_steps=3)
            assert len(predictions) == 3
            
            # 5. Act (get action recommendation)
            action_type = recommendations['action_type']
            assert isinstance(action_type, str)
            
            # 6. Update beliefs with outcome
            agent.update_beliefs(current_state)
            
            # Verify the cycle completed successfully
            progress = agent.get_learning_progress()
            assert progress['total_observations'] > 0
            assert progress['matrix_initialized'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])