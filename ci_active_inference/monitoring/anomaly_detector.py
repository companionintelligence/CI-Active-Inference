"""Anomaly detection for Active Inference system data and behavior."""

import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    timestamp: datetime
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    confidence: float
    affected_entity: str  # user_id, system, etc.
    data: Dict[str, Any]


class AnomalyDetector:
    """
    Detects anomalies in Active Inference system data and behavior.
    
    Uses statistical methods and pattern recognition to identify
    unusual patterns that might indicate data issues or system problems.
    """
    
    def __init__(self, sensitivity: float = 0.8):
        """
        Initialize anomaly detector.
        
        Args:
            sensitivity: Detection sensitivity (0.0 to 1.0)
        """
        self.sensitivity = sensitivity
        self.detected_anomalies = []
        self.baseline_stats = {}
        self.pattern_history = {}
        
    def detect_anomalies(self, data_batch: List[Dict[str, Any]], 
                        context: Dict[str, Any] = None) -> List[Anomaly]:
        """
        Detect anomalies in a data batch.
        
        Args:
            data_batch: Batch of data to analyze
            context: Additional context information
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if not data_batch:
            return anomalies
        
        # Statistical anomalies
        stat_anomalies = self._detect_statistical_anomalies(data_batch)
        anomalies.extend(stat_anomalies)
        
        # Temporal pattern anomalies
        temporal_anomalies = self._detect_temporal_anomalies(data_batch)
        anomalies.extend(temporal_anomalies)
        
        # Behavioral anomalies
        behavioral_anomalies = self._detect_behavioral_anomalies(data_batch)
        anomalies.extend(behavioral_anomalies)
        
        # Volume anomalies
        volume_anomalies = self._detect_volume_anomalies(data_batch)
        anomalies.extend(volume_anomalies)
        
        # Store detected anomalies
        self.detected_anomalies.extend(anomalies)
        
        # Update baseline statistics
        self._update_baseline_stats(data_batch)
        
        return anomalies
    
    def _detect_statistical_anomalies(self, data_batch: List[Dict[str, Any]]) -> List[Anomaly]:
        """Detect statistical anomalies using Z-score and IQR methods."""
        anomalies = []
        
        # Analyze numeric fields
        numeric_fields = ["hour_of_day", "confidence", "engagement_duration"]
        
        for field in numeric_fields:
            values = []
            for record in data_batch:
                if field in record and record[field] is not None:
                    try:
                        values.append(float(record[field]))
                    except (ValueError, TypeError):
                        continue
            
            if len(values) < 3:  # Need minimum samples
                continue
            
            # Calculate statistics
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            median_val = statistics.median(values)
            
            # Skip if no variation
            if std_val == 0:
                continue
            
            # Z-score anomaly detection
            z_threshold = 2.5 * self.sensitivity
            for i, value in enumerate(values):
                z_score = abs(value - mean_val) / std_val
                
                if z_score > z_threshold:
                    anomalies.append(Anomaly(
                        timestamp=datetime.now(),
                        anomaly_type="statistical_outlier",
                        severity="medium" if z_score > 3 else "low",
                        description=f"Statistical outlier in {field}: {value} (z-score: {z_score:.2f})",
                        confidence=min(1.0, z_score / 4.0),
                        affected_entity=data_batch[i].get("user_id", "unknown"),
                        data={"field": field, "value": value, "z_score": z_score, "mean": mean_val, "std": std_val}
                    ))
            
            # IQR-based detection
            if len(values) >= 4:
                q1 = statistics.quantiles(values, n=4)[0]
                q3 = statistics.quantiles(values, n=4)[2]
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                for i, value in enumerate(values):
                    if value < lower_bound or value > upper_bound:
                        anomalies.append(Anomaly(
                            timestamp=datetime.now(),
                            anomaly_type="iqr_outlier",
                            severity="low",
                            description=f"IQR outlier in {field}: {value}",
                            confidence=0.7,
                            affected_entity=data_batch[i].get("user_id", "unknown"),
                            data={"field": field, "value": value, "q1": q1, "q3": q3, "iqr": iqr}
                        ))
        
        return anomalies
    
    def _detect_temporal_anomalies(self, data_batch: List[Dict[str, Any]]) -> List[Anomaly]:
        """Detect temporal pattern anomalies."""
        anomalies = []
        
        # Extract timestamps
        timestamps = []
        for record in data_batch:
            if "timestamp" in record:
                try:
                    ts = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
                    timestamps.append((ts, record))
                except:
                    continue
        
        if len(timestamps) < 2:
            return anomalies
        
        # Sort by timestamp
        timestamps.sort(key=lambda x: x[0])
        
        # Check for unusual time gaps
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = (timestamps[i][0] - timestamps[i-1][0]).total_seconds()
            time_diffs.append(diff)
        
        if time_diffs:
            mean_diff = statistics.mean(time_diffs)
            std_diff = statistics.stdev(time_diffs) if len(time_diffs) > 1 else 0
            
            for i, diff in enumerate(time_diffs):
                if std_diff > 0:
                    z_score = abs(diff - mean_diff) / std_diff
                    
                    if z_score > 2.0 * self.sensitivity:
                        severity = "high" if z_score > 4 else "medium"
                        anomalies.append(Anomaly(
                            timestamp=timestamps[i+1][0],
                            anomaly_type="temporal_gap",
                            severity=severity,
                            description=f"Unusual time gap: {diff:.0f} seconds",
                            confidence=min(1.0, z_score / 5.0),
                            affected_entity=timestamps[i+1][1].get("user_id", "unknown"),
                            data={"time_gap": diff, "z_score": z_score, "mean_gap": mean_diff}
                        ))
        
        # Check for out-of-order timestamps
        for i in range(1, len(timestamps)):
            if timestamps[i][0] < timestamps[i-1][0]:
                anomalies.append(Anomaly(
                    timestamp=timestamps[i][0],
                    anomaly_type="temporal_disorder",
                    severity="medium",
                    description="Out-of-order timestamp detected",
                    confidence=1.0,
                    affected_entity=timestamps[i][1].get("user_id", "unknown"),
                    data={"current_ts": timestamps[i][0], "previous_ts": timestamps[i-1][0]}
                ))
        
        return anomalies
    
    def _detect_behavioral_anomalies(self, data_batch: List[Dict[str, Any]]) -> List[Anomaly]:
        """Detect behavioral pattern anomalies."""
        anomalies = []
        
        # Group by user
        user_data = {}
        for record in data_batch:
            user_id = record.get("user_id", "unknown")
            if user_id not in user_data:
                user_data[user_id] = []
            user_data[user_id].append(record)
        
        # Analyze each user's behavior
        for user_id, records in user_data.items():
            if len(records) < 3:  # Need minimum records
                continue
            
            # Check for unusual activity patterns
            digital_behaviors = [r.get("digital_behavior") for r in records if r.get("digital_behavior")]
            location_contexts = [r.get("location_context") for r in records if r.get("location_context")]
            
            # Detect repetitive patterns (potential bot behavior)
            if digital_behaviors:
                behavior_counts = {}
                for behavior in digital_behaviors:
                    behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                
                max_count = max(behavior_counts.values())
                if max_count > len(digital_behaviors) * 0.9:  # 90% same behavior
                    anomalies.append(Anomaly(
                        timestamp=datetime.now(),
                        anomaly_type="repetitive_behavior",
                        severity="medium",
                        description=f"Highly repetitive digital behavior for user {user_id}",
                        confidence=0.8,
                        affected_entity=user_id,
                        data={"behavior_counts": behavior_counts, "repetition_rate": max_count / len(digital_behaviors)}
                    ))
            
            # Detect sudden behavior changes
            if len(records) >= 6:  # Need enough data to detect changes
                first_half = records[:len(records)//2]
                second_half = records[len(records)//2:]
                
                first_behaviors = set(r.get("digital_behavior") for r in first_half if r.get("digital_behavior"))
                second_behaviors = set(r.get("digital_behavior") for r in second_half if r.get("digital_behavior"))
                
                overlap = len(first_behaviors.intersection(second_behaviors))
                total_unique = len(first_behaviors.union(second_behaviors))
                
                if total_unique > 0:
                    similarity = overlap / total_unique
                    
                    if similarity < 0.3:  # Very different behavior patterns
                        anomalies.append(Anomaly(
                            timestamp=datetime.now(),
                            anomaly_type="behavior_change",
                            severity="low",
                            description=f"Significant behavior change detected for user {user_id}",
                            confidence=1.0 - similarity,
                            affected_entity=user_id,
                            data={"first_behaviors": list(first_behaviors), "second_behaviors": list(second_behaviors), "similarity": similarity}
                        ))
        
        return anomalies
    
    def _detect_volume_anomalies(self, data_batch: List[Dict[str, Any]]) -> List[Anomaly]:
        """Detect volume-based anomalies."""
        anomalies = []
        
        batch_size = len(data_batch)
        
        # Check against historical volume if available
        if "batch_size" in self.baseline_stats:
            historical_sizes = self.baseline_stats["batch_size"]
            
            if len(historical_sizes) >= 3:
                mean_size = statistics.mean(historical_sizes)
                std_size = statistics.stdev(historical_sizes) if len(historical_sizes) > 1 else 0
                
                if std_size > 0:
                    z_score = abs(batch_size - mean_size) / std_size
                    
                    if z_score > 2.0 * self.sensitivity:
                        severity = "high" if z_score > 4 else "medium"
                        anomalies.append(Anomaly(
                            timestamp=datetime.now(),
                            anomaly_type="volume_anomaly",
                            severity=severity,
                            description=f"Unusual batch size: {batch_size} (expected ~{mean_size:.0f})",
                            confidence=min(1.0, z_score / 4.0),
                            affected_entity="system",
                            data={"batch_size": batch_size, "mean_size": mean_size, "z_score": z_score}
                        ))
        
        # Check for empty or very small batches
        if batch_size == 0:
            anomalies.append(Anomaly(
                timestamp=datetime.now(),
                anomaly_type="empty_batch",
                severity="high",
                description="Empty data batch received",
                confidence=1.0,
                affected_entity="system",
                data={"batch_size": 0}
            ))
        elif batch_size < 5:  # Unusually small batch
            anomalies.append(Anomaly(
                timestamp=datetime.now(),
                anomaly_type="small_batch",
                severity="medium",
                description=f"Unusually small batch size: {batch_size}",
                confidence=0.7,
                affected_entity="system",
                data={"batch_size": batch_size}
            ))
        
        return anomalies
    
    def _update_baseline_stats(self, data_batch: List[Dict[str, Any]]):
        """Update baseline statistics for future anomaly detection."""
        batch_size = len(data_batch)
        
        # Update batch size history
        if "batch_size" not in self.baseline_stats:
            self.baseline_stats["batch_size"] = []
        
        self.baseline_stats["batch_size"].append(batch_size)
        
        # Keep only recent history (last 50 batches)
        if len(self.baseline_stats["batch_size"]) > 50:
            self.baseline_stats["batch_size"] = self.baseline_stats["batch_size"][-50:]
        
        # Update user behavior patterns
        user_behaviors = {}
        for record in data_batch:
            user_id = record.get("user_id", "unknown")
            behavior = record.get("digital_behavior")
            
            if behavior:
                if user_id not in user_behaviors:
                    user_behaviors[user_id] = []
                user_behaviors[user_id].append(behavior)
        
        # Store recent behavior patterns
        for user_id, behaviors in user_behaviors.items():
            if user_id not in self.pattern_history:
                self.pattern_history[user_id] = []
            
            self.pattern_history[user_id].extend(behaviors)
            
            # Keep only recent history
            if len(self.pattern_history[user_id]) > 100:
                self.pattern_history[user_id] = self.pattern_history[user_id][-100:]
    
    def get_anomaly_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of anomalies detected in the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_anomalies = [a for a in self.detected_anomalies if a.timestamp > cutoff]
        
        # Count by type and severity
        type_counts = {}
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for anomaly in recent_anomalies:
            # Count by type
            if anomaly.anomaly_type not in type_counts:
                type_counts[anomaly.anomaly_type] = 0
            type_counts[anomaly.anomaly_type] += 1
            
            # Count by severity
            severity_counts[anomaly.severity] += 1
        
        # Calculate average confidence
        confidences = [a.confidence for a in recent_anomalies]
        avg_confidence = statistics.mean(confidences) if confidences else 0.0
        
        return {
            "period_days": days,
            "total_anomalies": len(recent_anomalies),
            "type_counts": type_counts,
            "severity_counts": severity_counts,
            "average_confidence": avg_confidence,
            "most_common_type": max(type_counts.keys(), key=type_counts.get) if type_counts else None,
            "highest_severity": max(severity_counts.keys(), key=severity_counts.get) if any(severity_counts.values()) else None
        }
    
    def adjust_sensitivity(self, new_sensitivity: float):
        """Adjust detection sensitivity."""
        self.sensitivity = max(0.0, min(1.0, new_sensitivity))
    
    def clear_old_anomalies(self, days: int = 30):
        """Clear anomalies older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        self.detected_anomalies = [a for a in self.detected_anomalies if a.timestamp > cutoff]