"""Tests for Phase 5 advanced features."""

import pytest
from datetime import datetime, timedelta
import json

from ci_active_inference.planning.multi_step_planner import MultiStepPlanner, PolicyTree
from ci_active_inference.explanation.explanation_generator import ExplanationGenerator
from ci_active_inference.ab_testing.experiment_manager import ExperimentManager, ExperimentConfig
from ci_active_inference.monitoring.data_quality_monitor import DataQualityMonitor
from ci_active_inference.monitoring.anomaly_detector import AnomalyDetector
from ci_active_inference.utils.matrix_utils import SimpleActiveInferenceMatrices


class TestMultiStepPlanning:
    """Test multi-step planning functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create simple matrices for testing
        self.matrices = SimpleActiveInferenceMatrices(
            num_obs=[4, 4, 3, 3],  # digital, location, social, content
            num_states=[3, 3, 3],  # energy, mood, context
            num_actions=9
        )
        
        # Initialize with some basic matrices
        self.matrices.A = [
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.0, 0.0, 0.0]],  # digital
            [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.0, 0.0, 0.0]],  # location
            [[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.1, 0.3, 0.6]],  # social
            [[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]   # content
        ]
        
        self.matrices.B = [
            # Energy factor transitions
            [[[0.8, 0.1, 0.1] for _ in range(9)] for _ in range(3)],
            # Mood factor transitions  
            [[[0.7, 0.2, 0.1] for _ in range(9)] for _ in range(3)],
            # Context factor transitions
            [[[0.6, 0.3, 0.1] for _ in range(9)] for _ in range(3)]
        ]
        
        self.matrices.C = [
            [0.9, 0.5, 0.1, 0.0],  # digital preferences
            [0.8, 0.6, 0.2, 0.0],  # location preferences
            [0.7, 0.5, 0.3],       # social preferences
            [0.6, 0.4, 0.2]        # content preferences
        ]
        
    def test_policy_tree_creation(self):
        """Test PolicyTree creation and structure."""
        root = PolicyTree()
        assert root.action is None
        assert root.depth == 0
        assert len(root.children) == 0
        
        child = root.add_child(1)
        assert child.action == 1
        assert child.depth == 1
        assert len(root.children) == 1
        
    def test_multi_step_planner_initialization(self):
        """Test MultiStepPlanner initialization."""
        planner = MultiStepPlanner(self.matrices, max_horizon=5)
        
        assert planner.matrices == self.matrices
        assert planner.max_horizon == 5
        assert planner.branching_factor == 9  # num_actions
        
    def test_multi_step_planning(self):
        """Test multi-step planning execution."""
        planner = MultiStepPlanner(self.matrices, max_horizon=3)
        
        # Create initial beliefs
        initial_beliefs = [
            [0.5, 0.3, 0.2],  # energy
            [0.4, 0.4, 0.2],  # mood  
            [0.3, 0.5, 0.2]   # context
        ]
        
        # Plan multiple steps
        action_sequence, expected_utility = planner.plan_multi_step(initial_beliefs, horizon=3)
        
        assert isinstance(action_sequence, list)
        assert len(action_sequence) >= 1
        assert isinstance(expected_utility, float)
        assert all(isinstance(action, int) for action in action_sequence)
        
    def test_planning_statistics(self):
        """Test planning statistics generation."""
        planner = MultiStepPlanner(self.matrices)
        stats = planner.get_planning_statistics()
        
        assert "max_horizon" in stats
        assert "branching_factor" in stats
        assert "num_states" in stats
        assert "num_actions" in stats
        assert stats["num_actions"] == 9


class TestExplanationGeneration:
    """Test explanation generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.matrices = SimpleActiveInferenceMatrices(
            num_obs=[4, 4, 3, 3],
            num_states=[3, 3, 3], 
            num_actions=9
        )
        
        # Initialize matrices (simplified)
        self.matrices.A = [
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.0, 0.0, 0.0]],
            [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.0, 0.0, 0.0]],
            [[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.1, 0.3, 0.6]],
            [[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
        ]
        
        self.matrices.C = [
            [0.9, 0.5, 0.1, 0.0],
            [0.8, 0.6, 0.2, 0.0], 
            [0.7, 0.5, 0.3],
            [0.6, 0.4, 0.2]
        ]
        
    def test_explanation_generator_initialization(self):
        """Test ExplanationGenerator initialization."""
        generator = ExplanationGenerator(self.matrices)
        
        assert generator.matrices == self.matrices
        assert generator.efe_decomposer is not None
        assert generator.causality_analyzer is not None
        assert generator.templates is not None
        
    def test_detailed_explanation_generation(self):
        """Test detailed explanation generation."""
        generator = ExplanationGenerator(self.matrices)
        
        current_state = {
            "digital_behavior": "work",
            "location_context": "office", 
            "social_context": "alone",
            "timestamp": datetime.now().isoformat()
        }
        
        beliefs = [
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.3, 0.5, 0.2]
        ]
        
        explanation = generator.generate_detailed_explanation(
            current_state=current_state,
            suggested_action="exercise",
            predicted_default="work",
            confidence=0.8,
            efe_breakdown={"risk": 0.3, "ambiguity": 0.4},
            beliefs=beliefs
        )
        
        assert isinstance(explanation, dict)
        assert "summary" in explanation
        assert "reasoning" in explanation
        assert "causal_factors" in explanation
        assert "natural_language" in explanation
        
    def test_efe_decomposition(self):
        """Test EFE decomposition functionality."""
        from ci_active_inference.explanation.efe_decomposer import EFEDecomposer
        
        decomposer = EFEDecomposer(self.matrices)
        
        efe_breakdown = {"risk": 0.6, "ambiguity": 0.4}
        analysis = decomposer.decompose_efe(efe_breakdown)
        
        assert isinstance(analysis, dict)
        assert "risk_component" in analysis
        assert "ambiguity_component" in analysis
        assert "overall_efe" in analysis
        assert "dominant_component" in analysis


class TestABTesting:
    """Test A/B testing functionality."""
    
    def test_experiment_manager_initialization(self):
        """Test ExperimentManager initialization."""
        manager = ExperimentManager()
        
        assert manager.active_experiments == {}
        assert manager.completed_experiments == {}
        assert manager.user_allocations == {}
        
    def test_experiment_creation(self):
        """Test experiment creation."""
        manager = ExperimentManager()
        
        config = ExperimentConfig(
            name="Test Experiment",
            description="Testing preference variants",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=7),
            variants=[
                {"name": "control", "config": {"type": "baseline"}},
                {"name": "variant_a", "config": {"type": "health_focused"}}
            ],
            success_metrics=["user_satisfaction"],
            allocation={"control": 0.5, "variant_a": 0.5}
        )
        
        experiment_id = manager.create_experiment(config)
        
        assert experiment_id in manager.active_experiments
        assert manager.active_experiments[experiment_id]["config"] == config
        
    def test_user_allocation(self):
        """Test user allocation to variants."""
        manager = ExperimentManager()
        
        config = ExperimentConfig(
            name="Test Experiment",
            description="Testing allocation",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=7),
            variants=[
                {"name": "control", "config": {}},
                {"name": "variant_a", "config": {}}
            ],
            success_metrics=["metric1"],
            allocation={"control": 0.5, "variant_a": 0.5}
        )
        
        experiment_id = manager.create_experiment(config)
        
        # Allocate user
        allocations = manager.allocate_user_to_variant("user123", experiment_id)
        
        assert experiment_id in allocations
        assert allocations[experiment_id] in ["control", "variant_a"]
        
        # Check consistency
        allocations2 = manager.allocate_user_to_variant("user123", experiment_id)
        assert allocations == allocations2
        
    def test_data_point_recording(self):
        """Test recording data points for experiments."""
        manager = ExperimentManager()
        
        config = ExperimentConfig(
            name="Test Experiment",
            description="Testing data recording",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=7),
            variants=[{"name": "control", "config": {}}],
            success_metrics=["satisfaction"],
            allocation={"control": 1.0}
        )
        
        experiment_id = manager.create_experiment(config)
        manager.allocate_user_to_variant("user123", experiment_id)
        
        # Record data point
        success = manager.record_data_point(
            "user123", experiment_id, "satisfaction", 4.5
        )
        
        assert success
        assert len(manager.active_experiments[experiment_id]["data_points"]) == 1
        
    def test_preference_variants_generation(self):
        """Test preference variants generation."""
        from ci_active_inference.ab_testing.preference_variants import PreferenceVariants
        
        generator = PreferenceVariants()
        
        variants = generator.generate_variants(["health_focused", "productivity_focused"])
        
        assert "control" in variants
        assert "health_focused" in variants
        assert "productivity_focused" in variants
        
        # Check that variants have different configurations
        health_prefs = variants["health_focused"]["activity_preferences"]["exercise"]
        prod_prefs = variants["productivity_focused"]["activity_preferences"]["work"]
        
        assert health_prefs > variants["control"]["activity_preferences"]["exercise"]
        assert prod_prefs > variants["control"]["activity_preferences"]["work"]


class TestDataQualityMonitoring:
    """Test data quality monitoring functionality."""
    
    def test_data_quality_monitor_initialization(self):
        """Test DataQualityMonitor initialization."""
        monitor = DataQualityMonitor()
        
        assert monitor.alert_thresholds is not None
        assert monitor.alerts == []
        assert monitor.quality_history == []
        
    def test_data_quality_assessment(self):
        """Test data quality assessment."""
        monitor = DataQualityMonitor()
        
        # Create sample data batch
        data_batch = [
            {
                "timestamp": datetime.now().isoformat(),
                "user_id": "user123",
                "digital_behavior": "work",
                "location_context": "office",
                "social_context": "alone"
            },
            {
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
                "user_id": "user456", 
                "digital_behavior": "entertainment",
                "location_context": "home",
                "social_context": "family-time"
            }
        ]
        
        assessment = monitor.assess_data_quality(data_batch)
        
        assert isinstance(assessment, dict)
        assert "timestamp" in assessment
        assert "batch_size" in assessment
        assert "metrics" in assessment
        assert "overall_score" in assessment
        assert "quality_grade" in assessment
        
        assert assessment["batch_size"] == 2
        assert 0 <= assessment["overall_score"] <= 1
        
    def test_completeness_assessment(self):
        """Test completeness assessment specifically."""
        monitor = DataQualityMonitor()
        
        # Complete data
        complete_data = [
            {
                "timestamp": datetime.now().isoformat(),
                "user_id": "user123",
                "digital_behavior": "work",
                "location_context": "office", 
                "social_context": "alone"
            }
        ]
        
        assessment = monitor.assess_data_quality(complete_data)
        completeness = assessment["metrics"]["completeness"]
        
        assert completeness["score"] > 0.8  # Should be high for complete data
        
        # Incomplete data
        incomplete_data = [
            {
                "timestamp": datetime.now().isoformat(),
                "user_id": "user123"
                # Missing other required fields
            }
        ]
        
        assessment2 = monitor.assess_data_quality(incomplete_data)
        completeness2 = assessment2["metrics"]["completeness"]
        
        assert completeness2["score"] < completeness["score"]


class TestAnomalyDetection:
    """Test anomaly detection functionality."""
    
    def test_anomaly_detector_initialization(self):
        """Test AnomalyDetector initialization."""
        detector = AnomalyDetector()
        
        assert detector.sensitivity == 0.8
        assert detector.detected_anomalies == []
        assert detector.baseline_stats == {}
        
    def test_statistical_anomaly_detection(self):
        """Test statistical anomaly detection."""
        detector = AnomalyDetector()
        
        # Create data with outlier
        data_batch = []
        for i in range(20):
            data_batch.append({
                "user_id": f"user{i}",
                "hour_of_day": 12,  # Normal hour
                "confidence": 0.8   # Normal confidence
            })
        
        # Add outlier
        data_batch.append({
            "user_id": "outlier_user",
            "hour_of_day": 25,  # Invalid hour (outlier)
            "confidence": 0.8
        })
        
        anomalies = detector.detect_anomalies(data_batch)
        
        # Should detect the hour outlier
        assert len(anomalies) > 0
        hour_anomalies = [a for a in anomalies if "hour_of_day" in str(a.data)]
        assert len(hour_anomalies) > 0
        
    def test_volume_anomaly_detection(self):
        """Test volume anomaly detection."""
        detector = AnomalyDetector()
        
        # Establish baseline with normal batches
        for _ in range(5):
            normal_batch = [{"user_id": f"user{i}"} for i in range(10)]
            detector.detect_anomalies(normal_batch)
        
        # Test with unusual batch size
        large_batch = [{"user_id": f"user{i}"} for i in range(100)]
        anomalies = detector.detect_anomalies(large_batch)
        
        volume_anomalies = [a for a in anomalies if a.anomaly_type == "volume_anomaly"]
        assert len(volume_anomalies) > 0
        
    def test_empty_batch_detection(self):
        """Test empty batch detection."""
        detector = AnomalyDetector()
        
        anomalies = detector.detect_anomalies([])
        
        assert len(anomalies) > 0
        empty_anomalies = [a for a in anomalies if a.anomaly_type == "empty_batch"]
        assert len(empty_anomalies) > 0
        
    def test_anomaly_summary(self):
        """Test anomaly summary generation."""
        detector = AnomalyDetector()
        
        # Add some fake anomalies
        from ci_active_inference.monitoring.anomaly_detector import Anomaly
        
        detector.detected_anomalies.extend([
            Anomaly(
                timestamp=datetime.now(),
                anomaly_type="test_anomaly",
                severity="medium",
                description="Test anomaly",
                confidence=0.8,
                affected_entity="test_user",
                data={}
            )
        ])
        
        summary = detector.get_anomaly_summary(days=7)
        
        assert isinstance(summary, dict)
        assert "total_anomalies" in summary
        assert "type_counts" in summary
        assert "severity_counts" in summary
        assert summary["total_anomalies"] >= 1


if __name__ == "__main__":
    pytest.main([__file__])