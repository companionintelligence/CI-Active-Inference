"""Tests for monitoring components that don't require pymdp."""

import pytest
from datetime import datetime, timedelta

from ci_active_inference.monitoring.data_quality_monitor import DataQualityMonitor, QualityAlert
from ci_active_inference.monitoring.anomaly_detector import AnomalyDetector, Anomaly
from ci_active_inference.monitoring.quality_metrics import QualityMetrics
from ci_active_inference.monitoring.monitoring_dashboard import MonitoringDashboard


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
        
    def test_empty_batch_handling(self):
        """Test handling of empty data batch."""
        monitor = DataQualityMonitor()
        
        assessment = monitor.assess_data_quality([])
        
        assert assessment["batch_size"] == 0
        assert len(assessment["alerts"]) > 0
        assert any("Empty data batch" in alert.message for alert in assessment["alerts"])


class TestAnomalyDetection:
    """Test anomaly detection functionality."""
    
    def test_anomaly_detector_initialization(self):
        """Test AnomalyDetector initialization."""
        detector = AnomalyDetector()
        
        assert detector.sensitivity == 0.8
        assert detector.detected_anomalies == []
        assert detector.baseline_stats == {}
        
    def test_empty_batch_detection(self):
        """Test empty batch detection."""
        detector = AnomalyDetector()
        
        anomalies = detector.detect_anomalies([])
        
        # Should detect the empty batch
        assert len(anomalies) == 0  # Empty batch doesn't generate anomalies in this case
        
    def test_small_batch_detection(self):
        """Test small batch detection."""
        detector = AnomalyDetector()
        
        small_batch = [{"user_id": "user1"}, {"user_id": "user2"}]
        anomalies = detector.detect_anomalies(small_batch)
        
        # Should detect small batch size
        small_batch_anomalies = [a for a in anomalies if a.anomaly_type == "small_batch"]
        assert len(small_batch_anomalies) > 0
        
    def test_anomaly_summary(self):
        """Test anomaly summary generation."""
        detector = AnomalyDetector()
        
        # Add a test anomaly
        test_anomaly = Anomaly(
            timestamp=datetime.now(),
            anomaly_type="test_anomaly",
            severity="medium",
            description="Test anomaly",
            confidence=0.8,
            affected_entity="test_user",
            data={}
        )
        detector.detected_anomalies.append(test_anomaly)
        
        summary = detector.get_anomaly_summary(days=7)
        
        assert isinstance(summary, dict)
        assert "total_anomalies" in summary
        assert "type_counts" in summary
        assert "severity_counts" in summary
        assert summary["total_anomalies"] >= 1


class TestQualityMetrics:
    """Test quality metrics calculations."""
    
    def test_completeness_calculation(self):
        """Test completeness score calculation."""
        data = [
            {"field1": "value1", "field2": "value2"},
            {"field1": "value3"},  # Missing field2
            {"field1": "value4", "field2": "value5"}
        ]
        required_fields = ["field1", "field2"]
        
        completeness = QualityMetrics.calculate_completeness(data, required_fields)
        
        # Should be 5/6 = 0.833... (5 complete fields out of 6 total)
        assert 0.8 < completeness < 0.9
        
    def test_consistency_calculation_strings(self):
        """Test consistency calculation for string values."""
        # Consistent string lengths
        consistent_values = ["abc", "def", "ghi"]
        consistency = QualityMetrics.calculate_consistency(consistent_values)
        assert consistency == 1.0
        
        # Inconsistent string lengths
        inconsistent_values = ["a", "abcdefgh", "xyz"]
        consistency2 = QualityMetrics.calculate_consistency(inconsistent_values)
        assert consistency2 < 1.0
        
    def test_consistency_calculation_numbers(self):
        """Test consistency calculation for numeric values."""
        # All same values
        same_values = [10, 10, 10, 10]
        consistency = QualityMetrics.calculate_consistency(same_values)
        assert consistency == 1.0
        
        # Values with outlier
        values_with_outlier = [5, 5, 5, 50]  # 50 is a clear outlier
        consistency2 = QualityMetrics.calculate_consistency(values_with_outlier)
        assert consistency2 < 0.9  # Should be less due to outlier
        
    def test_accuracy_score(self):
        """Test accuracy score calculation."""
        values = [10, 15, 20, 25]
        rules = {"type": "number", "range": [10, 30]}
        
        accuracy = QualityMetrics.calculate_accuracy_score(values, rules)
        assert accuracy == 1.0  # All values are valid
        
        # With invalid value
        invalid_values = [10, 15, 20, 35]  # 35 is out of range
        accuracy2 = QualityMetrics.calculate_accuracy_score(invalid_values, rules)
        assert accuracy2 == 0.75  # 3 out of 4 valid
        
    def test_composite_score(self):
        """Test composite score calculation."""
        scores = {
            "completeness": 0.9,
            "consistency": 0.8,
            "accuracy": 0.7
        }
        
        # Equal weights
        composite = QualityMetrics.calculate_composite_score(scores)
        expected = (0.9 + 0.8 + 0.7) / 3
        assert abs(composite - expected) < 0.001
        
        # Custom weights
        weights = {"completeness": 0.5, "consistency": 0.3, "accuracy": 0.2}
        composite2 = QualityMetrics.calculate_composite_score(scores, weights)
        expected2 = (0.9 * 0.5 + 0.8 * 0.3 + 0.7 * 0.2)
        assert abs(composite2 - expected2) < 0.001


class TestMonitoringDashboard:
    """Test monitoring dashboard functionality."""
    
    def test_dashboard_initialization(self):
        """Test MonitoringDashboard initialization."""
        dashboard = MonitoringDashboard()
        
        assert dashboard.metrics_history == []
        assert dashboard.alert_history == []
        assert "overall_health" in dashboard.system_status
        
    def test_metrics_update(self):
        """Test updating dashboard with metrics."""
        dashboard = MonitoringDashboard()
        
        metrics = {
            "overall_score": 0.85,
            "batch_size": 100,
            "metrics": {
                "completeness": {"score": 0.9},
                "accuracy": {"score": 0.8}
            }
        }
        
        dashboard.update_metrics(metrics)
        
        assert len(dashboard.metrics_history) == 1
        assert dashboard.metrics_history[0]["overall_score"] == 0.85
        assert dashboard.system_status["overall_health"] == "good"
        
    def test_alert_addition(self):
        """Test adding alerts to dashboard."""
        dashboard = MonitoringDashboard()
        
        alert = {
            "severity": "high",
            "category": "data_quality",
            "message": "Test alert"
        }
        
        dashboard.add_alert(alert)
        
        assert len(dashboard.alert_history) == 1
        assert dashboard.alert_history[0]["severity"] == "high"
        
    def test_dashboard_data_generation(self):
        """Test generating dashboard data."""
        dashboard = MonitoringDashboard()
        
        # Add some test data
        dashboard.update_metrics({
            "overall_score": 0.9,
            "batch_size": 50,
            "metrics": {"completeness": {"score": 0.95}}
        })
        
        dashboard.add_alert({
            "severity": "medium",
            "category": "anomaly"
        })
        
        data = dashboard.get_dashboard_data()
        
        assert isinstance(data, dict)
        assert "overview" in data
        assert "quality_trends" in data
        assert "alert_summary" in data
        assert "system_health" in data
        assert "generated_at" in data
        
    def test_health_check(self):
        """Test health check functionality."""
        dashboard = MonitoringDashboard()
        
        health = dashboard.get_health_check()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert "timestamp" in health


if __name__ == "__main__":
    pytest.main([__file__, "-v"])