"""Quality metrics calculations for data monitoring."""

from typing import Dict, List, Any
import statistics
import math


class QualityMetrics:
    """
    Provides standardized quality metric calculations.
    
    This class contains methods for calculating various data quality
    metrics used throughout the monitoring system.
    """
    
    @staticmethod
    def calculate_completeness(data: List[Dict[str, Any]], 
                             required_fields: List[str]) -> float:
        """
        Calculate completeness score for a dataset.
        
        Args:
            data: List of data records
            required_fields: List of required field names
            
        Returns:
            Completeness score (0.0 to 1.0)
        """
        if not data or not required_fields:
            return 0.0
        
        total_fields = len(data) * len(required_fields)
        complete_fields = 0
        
        for record in data:
            for field in required_fields:
                if field in record and record[field] is not None:
                    complete_fields += 1
        
        return complete_fields / total_fields if total_fields > 0 else 0.0
    
    @staticmethod
    def calculate_consistency(values: List[Any]) -> float:
        """
        Calculate consistency score for a list of values.
        
        Args:
            values: List of values to check for consistency
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        if not values:
            return 0.0
        
        # For string values, check format consistency
        if all(isinstance(v, str) for v in values):
            # Simple consistency check: similar length values are more consistent
            lengths = [len(v) for v in values]
            if len(set(lengths)) == 1:
                return 1.0
            else:
                length_std = statistics.stdev(lengths) if len(lengths) > 1 else 0
                avg_length = statistics.mean(lengths)
                if avg_length > 0:
                    return max(0.0, 1.0 - (length_std / avg_length))
                return 0.0
        
        # For numeric values, check for outliers
        elif all(isinstance(v, (int, float)) for v in values):
            if len(values) < 2:
                return 1.0
            
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            
            if std_val == 0:
                return 1.0  # All values are the same
            
            # Count outliers (more than 1 std deviation from mean)
            outliers = sum(1 for v in values if abs(v - mean_val) > 1.0 * std_val)
            consistency = 1.0 - (outliers / len(values))
            return max(0.0, consistency)
        
        # For mixed types, consistency is low
        types = set(type(v) for v in values)
        return 1.0 - (len(types) - 1) / len(values)
    
    @staticmethod
    def calculate_accuracy_score(values: List[Any], 
                               validation_rules: Dict[str, Any]) -> float:
        """
        Calculate accuracy score based on validation rules.
        
        Args:
            values: List of values to validate
            validation_rules: Dictionary of validation rules
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not values:
            return 0.0
        
        valid_count = 0
        total_count = len(values)
        
        for value in values:
            is_valid = QualityMetrics._validate_value(value, validation_rules)
            if is_valid:
                valid_count += 1
        
        return valid_count / total_count if total_count > 0 else 0.0
    
    @staticmethod
    def calculate_timeliness_score(timestamps: List[str], 
                                 max_delay_hours: float = 24.0) -> float:
        """
        Calculate timeliness score based on timestamp delays.
        
        Args:
            timestamps: List of timestamp strings
            max_delay_hours: Maximum acceptable delay in hours
            
        Returns:
            Timeliness score (0.0 to 1.0)
        """
        if not timestamps:
            return 0.0
        
        from datetime import datetime
        
        now = datetime.now()
        delays = []
        
        for ts_str in timestamps:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                delay_hours = (now - ts).total_seconds() / 3600
                delays.append(delay_hours)
            except:
                delays.append(max_delay_hours)  # Treat invalid timestamps as max delay
        
        if not delays:
            return 0.0
        
        avg_delay = statistics.mean(delays)
        timeliness = max(0.0, 1.0 - (avg_delay / max_delay_hours))
        return timeliness
    
    @staticmethod
    def calculate_validity_score(data: List[Dict[str, Any]], 
                               schema: Dict[str, Any]) -> float:
        """
        Calculate validity score against a schema.
        
        Args:
            data: List of data records
            schema: Schema definition
            
        Returns:
            Validity score (0.0 to 1.0)
        """
        if not data:
            return 0.0
        
        valid_records = 0
        
        for record in data:
            if QualityMetrics._validate_record_against_schema(record, schema):
                valid_records += 1
        
        return valid_records / len(data)
    
    @staticmethod
    def _validate_value(value: Any, rules: Dict[str, Any]) -> bool:
        """Validate a single value against rules."""
        if "type" in rules:
            expected_type = rules["type"]
            if expected_type == "string" and not isinstance(value, str):
                return False
            elif expected_type == "number" and not isinstance(value, (int, float)):
                return False
            elif expected_type == "boolean" and not isinstance(value, bool):
                return False
        
        if "range" in rules and isinstance(value, (int, float)):
            min_val, max_val = rules["range"]
            if not (min_val <= value <= max_val):
                return False
        
        if "enum" in rules:
            if value not in rules["enum"]:
                return False
        
        if "pattern" in rules and isinstance(value, str):
            import re
            pattern = rules["pattern"]
            if not re.match(pattern, value):
                return False
        
        return True
    
    @staticmethod
    def _validate_record_against_schema(record: Dict[str, Any], 
                                      schema: Dict[str, Any]) -> bool:
        """Validate a record against schema."""
        required_fields = schema.get("required", [])
        
        # Check required fields
        for field in required_fields:
            if field not in record:
                return False
        
        # Check field types and constraints
        properties = schema.get("properties", {})
        for field, value in record.items():
            if field in properties:
                field_rules = properties[field]
                if not QualityMetrics._validate_value(value, field_rules):
                    return False
        
        return True
    
    @staticmethod
    def calculate_composite_score(scores: Dict[str, float], 
                                weights: Dict[str, float] = None) -> float:
        """
        Calculate composite quality score from individual metric scores.
        
        Args:
            scores: Dictionary of metric_name -> score
            weights: Dictionary of metric_name -> weight
            
        Returns:
            Weighted composite score (0.0 to 1.0)
        """
        if not scores:
            return 0.0
        
        if weights is None:
            # Equal weights
            weights = {metric: 1.0 for metric in scores.keys()}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            weight = weights.get(metric, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0