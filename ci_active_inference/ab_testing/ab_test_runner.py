"""A/B test runner for orchestrating experiments with Active Inference agents."""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from .experiment_manager import ExperimentManager, ExperimentConfig
from .preference_variants import PreferenceVariants
from ..agents.simple_agent import SimpleActiveInferenceAgent


class ABTestRunner:
    """
    Orchestrates A/B testing experiments with Active Inference agents.
    
    Manages the lifecycle of experiments including agent configuration,
    user allocation, data collection, and result analysis.
    """
    
    def __init__(self, experiment_manager: ExperimentManager = None):
        """
        Initialize A/B test runner.
        
        Args:
            experiment_manager: Experiment manager instance
        """
        self.experiment_manager = experiment_manager or ExperimentManager()
        self.preference_generator = PreferenceVariants()
        self.active_agent_configs = {}  # user_id -> agent_config
        
    def create_preference_experiment(self, 
                                   experiment_name: str,
                                   duration_days: int = 7,
                                   variant_types: List[str] = None,
                                   success_metrics: List[str] = None) -> str:
        """
        Create an A/B test experiment for different preference settings.
        
        Args:
            experiment_name: Name of the experiment
            duration_days: Duration in days
            variant_types: Types of preference variants to test
            success_metrics: Metrics to track for success
            
        Returns:
            Experiment ID
        """
        variant_types = variant_types or ["health_focused", "productivity_focused"]
        success_metrics = success_metrics or ["user_satisfaction", "recommendation_adoption"]
        
        # Generate preference variants
        variants = self.preference_generator.generate_variants(variant_types)
        
        # Create experiment config
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)
        
        # Equal allocation across variants
        allocation = {variant: 1.0 / len(variants) for variant in variants.keys()}
        
        config = ExperimentConfig(
            name=experiment_name,
            description=f"Testing {len(variants)} preference variants over {duration_days} days",
            start_date=start_date,
            end_date=end_date,
            variants=[{"name": k, "config": v} for k, v in variants.items()],
            success_metrics=success_metrics,
            allocation=allocation
        )
        
        experiment_id = self.experiment_manager.create_experiment(config)
        
        return experiment_id
    
    def configure_agent_for_user(self, user_id: str, 
                                experiment_id: str = None) -> Dict[str, Any]:
        """
        Configure Active Inference agent for user based on A/B test allocation.
        
        Args:
            user_id: User identifier
            experiment_id: Specific experiment (None for all active)
            
        Returns:
            Agent configuration details
        """
        # Get user's variant allocations
        allocations = self.experiment_manager.allocate_user_to_variant(user_id, experiment_id)
        
        if not allocations:
            # Use default configuration
            return self._get_default_agent_config(user_id)
        
        # Use the first active experiment's variant
        experiment_id = list(allocations.keys())[0]
        variant = allocations[experiment_id]
        
        # Get experiment details
        experiment = self.experiment_manager.active_experiments.get(experiment_id)
        if not experiment:
            return self._get_default_agent_config(user_id)
        
        # Find variant configuration
        variant_config = None
        for v in experiment["config"].variants:
            if v["name"] == variant:
                variant_config = v["config"]
                break
        
        if not variant_config:
            return self._get_default_agent_config(user_id)
        
        # Configure agent with variant preferences
        agent_config = {
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant": variant,
            "preferences": variant_config,
            "configured_at": datetime.now()
        }
        
        self.active_agent_configs[user_id] = agent_config
        
        return agent_config
    
    def record_agent_interaction(self, user_id: str, 
                               interaction_data: Dict[str, Any]) -> bool:
        """
        Record agent interaction data for A/B testing analysis.
        
        Args:
            user_id: User identifier
            interaction_data: Data about the agent interaction
            
        Returns:
            Success status
        """
        if user_id not in self.active_agent_configs:
            return False
        
        agent_config = self.active_agent_configs[user_id]
        experiment_id = agent_config["experiment_id"]
        
        # Extract metrics from interaction data
        metrics_recorded = 0
        
        # User satisfaction (if user provided feedback)
        if "user_feedback" in interaction_data:
            feedback = interaction_data["user_feedback"]
            if "satisfaction" in feedback:
                self.experiment_manager.record_data_point(
                    user_id, experiment_id, "user_satisfaction", 
                    feedback["satisfaction"]
                )
                metrics_recorded += 1
        
        # Recommendation adoption
        if "recommendation_followed" in interaction_data:
            adopted = 1.0 if interaction_data["recommendation_followed"] else 0.0
            self.experiment_manager.record_data_point(
                user_id, experiment_id, "recommendation_adoption", adopted
            )
            metrics_recorded += 1
        
        # Engagement metrics
        if "engagement_duration" in interaction_data:
            self.experiment_manager.record_data_point(
                user_id, experiment_id, "engagement_duration",
                interaction_data["engagement_duration"]
            )
            metrics_recorded += 1
        
        # Confidence in recommendation
        if "confidence" in interaction_data:
            self.experiment_manager.record_data_point(
                user_id, experiment_id, "agent_confidence",
                interaction_data["confidence"]
            )
            metrics_recorded += 1
        
        return metrics_recorded > 0
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of an A/B test experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment summary with current status and preliminary results
        """
        # Get basic experiment status
        status = self.experiment_manager.get_experiment_status(experiment_id)
        
        if "error" in status:
            return status
        
        # Add preliminary analysis if enough data
        if status["data_points"] >= 10:  # Minimum data threshold
            experiment = (self.experiment_manager.active_experiments.get(experiment_id) or
                         self.experiment_manager.completed_experiments.get(experiment_id))
            
            if experiment:
                from .statistical_analyzer import StatisticalAnalyzer
                analyzer = StatisticalAnalyzer()
                preliminary_analysis = analyzer.analyze_experiment(experiment)
                status["preliminary_analysis"] = preliminary_analysis
        
        return status
    
    def end_experiment_and_analyze(self, experiment_id: str) -> Dict[str, Any]:
        """
        End an A/B test experiment and provide full analysis.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Complete experiment analysis and recommendations
        """
        results = self.experiment_manager.end_experiment(experiment_id)
        
        # Clean up agent configurations for this experiment
        users_to_remove = []
        for user_id, config in self.active_agent_configs.items():
            if config.get("experiment_id") == experiment_id:
                users_to_remove.append(user_id)
        
        for user_id in users_to_remove:
            del self.active_agent_configs[user_id]
        
        return results
    
    def create_agent_with_variant(self, user_id: str, 
                                 variant_preferences: Dict[str, Any]) -> SimpleActiveInferenceAgent:
        """
        Create an Active Inference agent configured with specific variant preferences.
        
        Args:
            user_id: User identifier
            variant_preferences: Preference configuration for the variant
            
        Returns:
            Configured SimpleActiveInferenceAgent
        """
        agent = SimpleActiveInferenceAgent(user_id)
        
        # Apply preference modifications
        if hasattr(agent, 'preference_config'):
            agent.preference_config.update(variant_preferences)
        else:
            agent.preference_config = variant_preferences
        
        return agent
    
    def _get_default_agent_config(self, user_id: str) -> Dict[str, Any]:
        """Get default agent configuration when no experiments are active."""
        return {
            "user_id": user_id,
            "experiment_id": None,
            "variant": "default",
            "preferences": self.preference_generator.base_preferences,
            "configured_at": datetime.now()
        }