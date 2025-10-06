"""Monitoring dashboard for data quality and system health."""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json


class MonitoringDashboard:
    """
    Provides dashboard functionality for monitoring data quality and system health.
    
    Aggregates metrics from various monitoring components and provides
    visualization-ready data structures.
    """
    
    def __init__(self):
        """Initialize monitoring dashboard."""
        self.metrics_history = []
        self.alert_history = []
        self.system_status = {
            "overall_health": "unknown",
            "last_updated": None,
            "components": {}
        }
        
    def get_dashboard_data(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for specified time range.
        
        Args:
            time_range_hours: Time range in hours to include in dashboard
            
        Returns:
            Dashboard data structure
        """
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        
        dashboard = {
            "overview": self._get_overview_metrics(cutoff_time),
            "quality_trends": self._get_quality_trends(cutoff_time),
            "alert_summary": self._get_alert_summary(cutoff_time),
            "system_health": self._get_system_health(),
            "anomaly_insights": self._get_anomaly_insights(cutoff_time),
            "performance_metrics": self._get_performance_metrics(cutoff_time),
            "generated_at": datetime.now().isoformat()
        }
        
        return dashboard
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update dashboard with new metrics.
        
        Args:
            metrics: New metrics to add to dashboard
        """
        metrics["timestamp"] = datetime.now()
        self.metrics_history.append(metrics)
        
        # Keep only recent history (last 1000 entries)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        # Update system status
        self._update_system_status(metrics)
    
    def add_alert(self, alert_data: Dict[str, Any]):
        """
        Add alert to dashboard history.
        
        Args:
            alert_data: Alert information
        """
        alert_data["timestamp"] = datetime.now()
        self.alert_history.append(alert_data)
        
        # Keep only recent alerts (last 500)
        if len(self.alert_history) > 500:
            self.alert_history = self.alert_history[-500:]
    
    def _get_overview_metrics(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get high-level overview metrics."""
        recent_metrics = [m for m in self.metrics_history if m.get("timestamp", datetime.min) > cutoff_time]
        
        if not recent_metrics:
            return {
                "total_data_points": 0,
                "average_quality_score": 0.0,
                "quality_grade": "N/A",
                "trending": "unknown"
            }
        
        total_points = sum(m.get("batch_size", 0) for m in recent_metrics)
        quality_scores = [m.get("overall_score", 0.0) for m in recent_metrics if "overall_score" in m]
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Calculate trend
        if len(quality_scores) >= 2:
            first_half = quality_scores[:len(quality_scores)//2]
            second_half = quality_scores[len(quality_scores)//2:]
            
            first_avg = sum(first_half) / len(first_half) if first_half else 0
            second_avg = sum(second_half) / len(second_half) if second_half else 0
            
            if second_avg > first_avg + 0.05:
                trending = "improving"
            elif second_avg < first_avg - 0.05:
                trending = "declining"
            else:
                trending = "stable"
        else:
            trending = "insufficient_data"
        
        return {
            "total_data_points": total_points,
            "average_quality_score": avg_quality,
            "quality_grade": self._score_to_grade(avg_quality),
            "trending": trending,
            "assessments_count": len(recent_metrics)
        }
    
    def _get_quality_trends(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get quality trend data for charts."""
        recent_metrics = [m for m in self.metrics_history if m.get("timestamp", datetime.min) > cutoff_time]
        
        trends = {
            "timestamps": [],
            "overall_scores": [],
            "completeness_scores": [],
            "consistency_scores": [],
            "accuracy_scores": [],
            "timeliness_scores": [],
            "validity_scores": []
        }
        
        for metric in recent_metrics:
            trends["timestamps"].append(metric.get("timestamp", datetime.now()).isoformat())
            trends["overall_scores"].append(metric.get("overall_score", 0.0))
            
            metrics_detail = metric.get("metrics", {})
            trends["completeness_scores"].append(metrics_detail.get("completeness", {}).get("score", 0.0))
            trends["consistency_scores"].append(metrics_detail.get("consistency", {}).get("score", 0.0))
            trends["accuracy_scores"].append(metrics_detail.get("accuracy", {}).get("score", 0.0))
            trends["timeliness_scores"].append(metrics_detail.get("timeliness", {}).get("score", 0.0))
            trends["validity_scores"].append(metrics_detail.get("validity", {}).get("score", 0.0))
        
        return trends
    
    def _get_alert_summary(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get alert summary for dashboard."""
        recent_alerts = [a for a in self.alert_history if a.get("timestamp", datetime.min) > cutoff_time]
        
        # Count by severity
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        category_counts = {}
        
        for alert in recent_alerts:
            severity = alert.get("severity", "unknown")
            if severity in severity_counts:
                severity_counts[severity] += 1
            
            category = alert.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Recent critical alerts
        critical_alerts = [a for a in recent_alerts if a.get("severity") == "critical"]
        critical_alerts.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
        
        return {
            "total_alerts": len(recent_alerts),
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "critical_alerts": critical_alerts[:5],  # Top 5 recent critical
            "alert_rate": len(recent_alerts) / 24 if recent_alerts else 0  # per hour
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        return self.system_status.copy()
    
    def _get_anomaly_insights(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get anomaly detection insights."""
        # This would integrate with anomaly detector
        # For now, return placeholder structure
        return {
            "total_anomalies": 0,
            "anomaly_types": {},
            "most_affected_entities": [],
            "anomaly_timeline": [],
            "severity_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0}
        }
    
    def _get_performance_metrics(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get system performance metrics."""
        return {
            "processing_times": {
                "average_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0
            },
            "throughput": {
                "records_per_second": 0,
                "batches_per_hour": 0
            },
            "resource_usage": {
                "memory_mb": 0,
                "cpu_percent": 0
            }
        }
    
    def _update_system_status(self, metrics: Dict[str, Any]):
        """Update overall system status based on metrics."""
        overall_score = metrics.get("overall_score", 0.0)
        
        # Determine health based on quality score
        if overall_score >= 0.9:
            health = "excellent"
        elif overall_score >= 0.8:
            health = "good"
        elif overall_score >= 0.7:
            health = "fair"
        elif overall_score >= 0.5:
            health = "poor"
        else:
            health = "critical"
        
        self.system_status.update({
            "overall_health": health,
            "last_updated": datetime.now(),
            "latest_quality_score": overall_score
        })
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def export_dashboard_data(self, format_type: str = "json") -> str:
        """
        Export dashboard data in specified format.
        
        Args:
            format_type: Export format ("json", "csv")
            
        Returns:
            Formatted data string
        """
        dashboard_data = self.get_dashboard_data()
        
        if format_type == "json":
            return json.dumps(dashboard_data, indent=2, default=str)
        elif format_type == "csv":
            # Simple CSV export of overview metrics
            csv_lines = [
                "metric,value",
                f"total_data_points,{dashboard_data['overview']['total_data_points']}",
                f"average_quality_score,{dashboard_data['overview']['average_quality_score']}",
                f"quality_grade,{dashboard_data['overview']['quality_grade']}",
                f"trending,{dashboard_data['overview']['trending']}",
                f"total_alerts,{dashboard_data['alert_summary']['total_alerts']}"
            ]
            return "\n".join(csv_lines)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get simple health check response."""
        return {
            "status": self.system_status["overall_health"],
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": 24,  # Placeholder
            "last_quality_score": self.system_status.get("latest_quality_score", 0.0)
        }