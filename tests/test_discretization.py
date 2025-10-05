"""
Unit tests for data discretization pipeline.
"""

import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from ci_active_inference.utils.discretization import DataDiscretizer, StateMapping


class TestStateMapping:
    """Test StateMapping dataclass."""
    
    def test_state_mapping_creation(self):
        """Test creation of state mapping."""
        mapping = StateMapping(
            name="test",
            mapping={"a": 0, "b": 1, "c": 2},
            num_states=3,
            default_state=0
        )
        
        assert mapping.name == "test"
        assert mapping.num_states == 3
        assert mapping.mapping["a"] == 0


class TestDataDiscretizer:
    """Test data discretization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.discretizer = DataDiscretizer()
    
    def test_discretize_hour_of_day(self):
        """Test hour discretization."""
        # Night hours
        assert self.discretizer.discretize_hour_of_day(2) == 0
        assert self.discretizer.discretize_hour_of_day(23) == 0
        
        # Morning hours
        assert self.discretizer.discretize_hour_of_day(8) == 1
        assert self.discretizer.discretize_hour_of_day(11) == 1
        
        # Afternoon hours
        assert self.discretizer.discretize_hour_of_day(14) == 2
        assert self.discretizer.discretize_hour_of_day(17) == 2
        
        # Evening hours
        assert self.discretizer.discretize_hour_of_day(19) == 3
        assert self.discretizer.discretize_hour_of_day(21) == 3
    
    def test_discretize_event_count(self):
        """Test event count discretization."""
        assert self.discretizer.discretize_event_count(0) == 0  # Low
        assert self.discretizer.discretize_event_count(1) == 0  # Low
        assert self.discretizer.discretize_event_count(2) == 1  # Medium
        assert self.discretizer.discretize_event_count(3) == 1  # Medium
        assert self.discretizer.discretize_event_count(5) == 2  # High
    
    def test_discretize_event_duration(self):
        """Test event duration discretization."""
        assert self.discretizer.discretize_event_duration(15) == 0   # Short
        assert self.discretizer.discretize_event_duration(30) == 0   # Short
        assert self.discretizer.discretize_event_duration(60) == 1   # Medium
        assert self.discretizer.discretize_event_duration(120) == 1  # Medium
        assert self.discretizer.discretize_event_duration(180) == 2  # Long
    
    def test_classify_event_context(self):
        """Test event context classification."""
        # Work events
        assert self.discretizer.classify_event_context("Team meeting") == "work"
        assert self.discretizer.classify_event_context("Project call") == "work"
        
        # Health events
        assert self.discretizer.classify_event_context("Gym workout") == "health"
        assert self.discretizer.classify_event_context("Morning run") == "health"
        
        # Social events
        assert self.discretizer.classify_event_context("Dinner with friends") == "social"
        assert self.discretizer.classify_event_context("Coffee meeting") == "social"
        
        # Home events
        assert self.discretizer.classify_event_context("Cooking at home") == "home"
        assert self.discretizer.classify_event_context("Rest time") == "home"
        
        # Other events
        assert self.discretizer.classify_event_context("Random stuff") == "other"
        assert self.discretizer.classify_event_context("") == "other"
    
    def test_classify_digital_behavior(self):
        """Test digital behavior classification."""
        # Work-focused behavior
        assert self.discretizer.classify_digital_behavior("document", "Work Report") == "work-focus"
        assert self.discretizer.classify_digital_behavior("spreadsheet") == "work-focus"
        
        # Communication behavior
        assert self.discretizer.classify_digital_behavior("email") == "communication"
        assert self.discretizer.classify_digital_behavior("chat", "Team discussion") == "communication"
        
        # Entertainment behavior
        assert self.discretizer.classify_digital_behavior("video", "YouTube") == "entertainment"
        assert self.discretizer.classify_digital_behavior("game") == "entertainment"
        
        # Other behavior
        assert self.discretizer.classify_digital_behavior("unknown") == "other"
        assert self.discretizer.classify_digital_behavior(None, None) == "other"
    
    def test_classify_location_context(self):
        """Test location context classification."""
        # Home locations
        assert self.discretizer.classify_location_context("Home") == "home"
        assert self.discretizer.classify_location_context("My apartment") == "home"
        
        # Office locations
        assert self.discretizer.classify_location_context("Office building") == "office"
        assert self.discretizer.classify_location_context("Workplace") == "office"
        
        # Outdoor locations
        assert self.discretizer.classify_location_context("Central Park") == "outdoor"
        assert self.discretizer.classify_location_context("Beach walk") == "outdoor"
        
        # Social venues
        assert self.discretizer.classify_location_context("Restaurant") == "social-venue"
        assert self.discretizer.classify_location_context("Coffee shop") == "social-venue"
        
        # Other locations
        assert self.discretizer.classify_location_context("Random place") == "other"
    
    def test_classify_social_context(self):
        """Test social context classification."""
        assert self.discretizer.classify_social_context(0) == "alone"
        assert self.discretizer.classify_social_context(1) == "small-group"
        assert self.discretizer.classify_social_context(2) == "small-group"
        assert self.discretizer.classify_social_context(5) == "large-group"
    
    def test_map_value_to_state(self):
        """Test mapping values to state indices."""
        # Test with existing value
        context_idx = self.discretizer.map_value_to_state("work", self.discretizer.context_mapping)
        assert context_idx == 1
        
        # Test with non-existing value (should return default)
        unknown_idx = self.discretizer.map_value_to_state("unknown", self.discretizer.context_mapping)
        assert unknown_idx == self.discretizer.context_mapping.default_state
    
    def test_get_state_dimensions(self):
        """Test getting state dimensions."""
        dimensions = self.discretizer.get_state_dimensions()
        
        assert isinstance(dimensions, dict)
        assert "energy" in dimensions
        assert "mood" in dimensions
        assert "context" in dimensions
        assert dimensions["energy"] == 3
        assert dimensions["mood"] == 3
        assert dimensions["context"] == 5
    
    def test_get_observation_dimensions(self):
        """Test getting observation dimensions."""
        obs_dims = self.discretizer.get_observation_dimensions()
        
        assert isinstance(obs_dims, list)
        assert len(obs_dims) == 4  # 4 observation modalities
        assert obs_dims[0] == 4    # Digital behavior
        assert obs_dims[1] == 5    # Location context
        assert obs_dims[2] == 3    # Social context
        assert obs_dims[3] == 4    # Content type
    
    def test_get_state_factor_dimensions(self):
        """Test getting state factor dimensions."""
        state_dims = self.discretizer.get_state_factor_dimensions()
        
        assert isinstance(state_dims, list)
        assert len(state_dims) == 3  # 3 state factors
        assert state_dims[0] == 3    # Energy
        assert state_dims[1] == 3    # Mood
        assert state_dims[2] == 5    # Context
    
    def test_discretize_features(self):
        """Test feature discretization."""
        # Mock feature record
        features = [{
            'hour_of_day': '14',
            'event_count': 3,
            'avg_duration_minutes': 45.0,
            'context_state': 'work',
            'digital_behavior': 'work-focus',
            'social_context': 'small-group',
            'location_context': 'office',
            'content_type': 'document'
        }]
        
        discretized = self.discretizer.discretize_features(features)
        
        assert len(discretized) == 1
        record = discretized[0]
        
        assert record['time_period'] == 2  # Afternoon
        assert record['activity_level'] == 1  # Medium
        assert record['duration_category'] == 1  # Medium
        assert record['context_idx'] == 1  # Work
        assert record['digital_idx'] == 0  # Work-focus
        assert record['social_idx'] == 1  # Small-group
        assert record['location_idx'] == 1  # Office
        assert record['content_idx'] == 0  # Work content


if __name__ == "__main__":
    pytest.main([__file__])