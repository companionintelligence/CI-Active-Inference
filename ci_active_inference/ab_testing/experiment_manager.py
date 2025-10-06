"""A/B testing experiment manager for agent configurations."""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    variants: List[Dict[str, Any]]
    success_metrics: List[str]
    allocation: Dict[str, float]  # variant_name -> allocation percentage
    min_sample_size: int = 100
    significance_threshold: float = 0.05
    

class ExperimentManager:
    """
    Manages A/B testing experiments for Active Inference agent configurations.
    
    Handles experiment creation, user allocation, data collection,
    and statistical analysis of results.
    """
    
    def __init__(self, storage_path: str = None):
        """
        Initialize experiment manager.
        
        Args:
            storage_path: Path to store experiment data
        """
        self.storage_path = storage_path or "/tmp/ab_experiments"
        self.active_experiments = {}
        self.completed_experiments = {}
        self.user_allocations = {}  # user_id -> {experiment_id: variant}
        
    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create a new A/B testing experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        experiment = {
            "id": experiment_id,
            "config": config,
            "created_at": datetime.now(),
            "status": "active",
            "participants": {},
            "results": {},
            "data_points": []
        }
        
        self.active_experiments[experiment_id] = experiment
        
        # Validate allocation percentages
        total_allocation = sum(config.allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Allocation percentages must sum to 1.0, got {total_allocation}")
        
        return experiment_id
    
    def allocate_user_to_variant(self, user_id: str, 
                                experiment_id: str = None) -> Dict[str, str]:
        """
        Allocate user to variant(s) for active experiments.
        
        Args:
            user_id: User identifier
            experiment_id: Specific experiment (None for all active)
            
        Returns:
            Dictionary of experiment_id -> variant_name
        """
        if user_id not in self.user_allocations:
            self.user_allocations[user_id] = {}
        
        allocations = {}
        
        experiments_to_process = (
            [experiment_id] if experiment_id else list(self.active_experiments.keys())
        )
        
        for exp_id in experiments_to_process:
            if exp_id not in self.active_experiments:
                continue
                
            experiment = self.active_experiments[exp_id]
            config = experiment["config"]
            
            # Check if user already allocated
            if exp_id in self.user_allocations[user_id]:
                allocations[exp_id] = self.user_allocations[user_id][exp_id]
                continue
            
            # Check if experiment is active
            now = datetime.now()
            if not (config.start_date <= now <= config.end_date):
                continue
            
            # Allocate user to variant based on allocation percentages
            variant = self._allocate_to_variant(user_id, config.allocation)
            
            self.user_allocations[user_id][exp_id] = variant
            allocations[exp_id] = variant
            
            # Add user to experiment participants
            if user_id not in experiment["participants"]:
                experiment["participants"][user_id] = {
                    "variant": variant,
                    "allocated_at": now,
                    "data_points": []
                }
        
        return allocations
    
    def record_data_point(self, user_id: str, experiment_id: str, 
                         metric_name: str, value: float, 
                         timestamp: datetime = None) -> bool:
        """
        Record a data point for experiment analysis.
        
        Args:
            user_id: User identifier
            experiment_id: Experiment identifier
            metric_name: Name of the metric
            value: Metric value
            timestamp: When metric was recorded
            
        Returns:
            Success status
        """
        if experiment_id not in self.active_experiments:
            return False
        
        if user_id not in self.user_allocations.get(user_id, {}):
            return False
        
        experiment = self.active_experiments[experiment_id]
        timestamp = timestamp or datetime.now()
        
        data_point = {
            "user_id": user_id,
            "variant": self.user_allocations[user_id][experiment_id],
            "metric_name": metric_name,
            "value": value,
            "timestamp": timestamp
        }
        
        # Add to experiment data
        experiment["data_points"].append(data_point)
        
        # Add to user's data in experiment
        if user_id in experiment["participants"]:
            experiment["participants"][user_id]["data_points"].append(data_point)
        
        return True
    
    def get_user_variant(self, user_id: str, experiment_id: str) -> Optional[str]:
        """
        Get user's variant for a specific experiment.
        
        Args:
            user_id: User identifier
            experiment_id: Experiment identifier
            
        Returns:
            Variant name or None
        """
        return self.user_allocations.get(user_id, {}).get(experiment_id)
    
    def end_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        End an experiment and analyze results.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment results
        """
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        experiment["status"] = "completed"
        experiment["completed_at"] = datetime.now()
        
        # Analyze results
        from .statistical_analyzer import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        results = analyzer.analyze_experiment(experiment)
        
        experiment["results"] = results
        
        # Move to completed experiments
        self.completed_experiments[experiment_id] = experiment
        del self.active_experiments[experiment_id]
        
        return results
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get current status of an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment status information
        """
        experiment = (self.active_experiments.get(experiment_id) or 
                     self.completed_experiments.get(experiment_id))
        
        if not experiment:
            return {"error": "Experiment not found"}
        
        config = experiment["config"]
        participants = experiment["participants"]
        
        status = {
            "id": experiment_id,
            "name": config.name,
            "status": experiment["status"],
            "start_date": config.start_date,
            "end_date": config.end_date,
            "total_participants": len(participants),
            "data_points": len(experiment["data_points"]),
            "variants": {}
        }
        
        # Count participants per variant
        for user_data in participants.values():
            variant = user_data["variant"]
            if variant not in status["variants"]:
                status["variants"][variant] = {"participants": 0, "data_points": 0}
            status["variants"][variant]["participants"] += 1
            status["variants"][variant]["data_points"] += len(user_data["data_points"])
        
        return status
    
    def list_experiments(self, status: str = "all") -> List[Dict[str, Any]]:
        """
        List experiments by status.
        
        Args:
            status: "active", "completed", or "all"
            
        Returns:
            List of experiment summaries
        """
        experiments = []
        
        if status in ["active", "all"]:
            for exp_id, exp in self.active_experiments.items():
                experiments.append(self._experiment_summary(exp_id, exp))
        
        if status in ["completed", "all"]:
            for exp_id, exp in self.completed_experiments.items():
                experiments.append(self._experiment_summary(exp_id, exp))
        
        return experiments
    
    def _allocate_to_variant(self, user_id: str, 
                           allocation: Dict[str, float]) -> str:
        """
        Allocate user to variant based on allocation percentages.
        
        Args:
            user_id: User identifier for consistent allocation
            allocation: Variant allocation percentages
            
        Returns:
            Selected variant name
        """
        # Use hash of user_id for consistent allocation
        random.seed(hash(user_id) % (2**32))
        
        cumulative = 0.0
        rand_val = random.random()
        
        for variant, percentage in allocation.items():
            cumulative += percentage
            if rand_val <= cumulative:
                return variant
        
        # Fallback to first variant
        return list(allocation.keys())[0]
    
    def _experiment_summary(self, exp_id: str, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Create experiment summary."""
        config = experiment["config"]
        return {
            "id": exp_id,
            "name": config.name,
            "description": config.description,
            "status": experiment["status"],
            "start_date": config.start_date,
            "end_date": config.end_date,
            "participants": len(experiment["participants"]),
            "variants": list(config.allocation.keys())
        }