"""Statistical analyzer for A/B testing results."""

import math
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import numpy as np


class StatisticalAnalyzer:
    """
    Statistical analyzer for A/B testing experiments.
    
    Provides statistical significance testing, effect size calculation,
    and comprehensive analysis of experiment results.
    """
    
    def __init__(self, significance_threshold: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            significance_threshold: P-value threshold for significance
        """
        self.significance_threshold = significance_threshold
        
    def analyze_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze A/B testing experiment results.
        
        Args:
            experiment: Experiment data structure
            
        Returns:
            Comprehensive statistical analysis
        """
        config = experiment["config"]
        data_points = experiment["data_points"]
        
        analysis = {
            "experiment_id": experiment["id"],
            "experiment_name": config.name,
            "total_participants": len(experiment["participants"]),
            "total_data_points": len(data_points),
            "analysis_date": "now",  # Would be datetime.now() in real implementation
            "metrics": {},
            "overall_results": {},
            "recommendations": []
        }
        
        # Analyze each success metric
        for metric in config.success_metrics:
            metric_analysis = self._analyze_metric(data_points, metric, config.variants)
            analysis["metrics"][metric] = metric_analysis
        
        # Overall experiment conclusion
        analysis["overall_results"] = self._conclude_experiment(analysis["metrics"])
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_metric(self, data_points: List[Dict[str, Any]], 
                       metric_name: str, 
                       variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a specific metric across variants.
        
        Args:
            data_points: All experiment data points
            metric_name: Name of metric to analyze
            variants: Experiment variants
            
        Returns:
            Metric analysis results
        """
        # Group data by variant
        variant_data = {}
        for point in data_points:
            if point["metric_name"] == metric_name:
                variant = point["variant"]
                if variant not in variant_data:
                    variant_data[variant] = []
                variant_data[variant].append(point["value"])
        
        if len(variant_data) < 2:
            return {"error": "Insufficient data for comparison"}
        
        analysis = {
            "metric_name": metric_name,
            "variant_stats": {},
            "comparisons": {},
            "significant_differences": [],
            "winner": None,
            "confidence_level": 0.0
        }
        
        # Calculate statistics for each variant
        for variant, values in variant_data.items():
            if values:  # Check if we have data
                analysis["variant_stats"][variant] = self._calculate_descriptive_stats(values)
        
        # Pairwise comparisons
        variant_names = list(variant_data.keys())
        for i, variant_a in enumerate(variant_names):
            for variant_b in variant_names[i+1:]:
                if variant_a in variant_data and variant_b in variant_data:
                    comparison = self._compare_variants(
                        variant_data[variant_a], 
                        variant_data[variant_b],
                        variant_a, 
                        variant_b
                    )
                    comparison_key = f"{variant_a}_vs_{variant_b}"
                    analysis["comparisons"][comparison_key] = comparison
                    
                    if comparison["significant"]:
                        analysis["significant_differences"].append(comparison_key)
        
        # Determine winner
        analysis["winner"] = self._determine_winner(analysis["variant_stats"])
        analysis["confidence_level"] = self._calculate_confidence_level(analysis["comparisons"])
        
        return analysis
    
    def _calculate_descriptive_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics for a set of values."""
        if not values:
            return {"error": "No values provided"}
        
        values_array = np.array(values)
        
        return {
            "count": len(values),
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array, ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "q25": float(np.percentile(values_array, 25)),
            "q75": float(np.percentile(values_array, 75))
        }
    
    def _compare_variants(self, values_a: List[float], values_b: List[float],
                         variant_a: str, variant_b: str) -> Dict[str, Any]:
        """
        Compare two variants statistically.
        
        Args:
            values_a: Values for variant A
            values_b: Values for variant B
            variant_a: Name of variant A
            variant_b: Name of variant B
            
        Returns:
            Statistical comparison results
        """
        if len(values_a) < 2 or len(values_b) < 2:
            return {"error": "Insufficient sample size for comparison"}
        
        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            
            # Calculate effect size (Cohen's d)
            pooled_std = math.sqrt(((len(values_a) - 1) * np.var(values_a, ddof=1) + 
                                   (len(values_b) - 1) * np.var(values_b, ddof=1)) / 
                                  (len(values_a) + len(values_b) - 2))
            
            cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std if pooled_std > 0 else 0
            
            # Determine significance and practical significance
            significant = p_value < self.significance_threshold
            practical_significance = abs(cohens_d) > 0.2  # Small effect size threshold
            
            comparison = {
                "variant_a": variant_a,
                "variant_b": variant_b,
                "mean_a": float(np.mean(values_a)),
                "mean_b": float(np.mean(values_b)),
                "mean_difference": float(np.mean(values_a) - np.mean(values_b)),
                "relative_difference": float((np.mean(values_a) - np.mean(values_b)) / np.mean(values_b)) if np.mean(values_b) != 0 else 0,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": significant,
                "cohens_d": float(cohens_d),
                "effect_size": self._interpret_effect_size(cohens_d),
                "practical_significance": practical_significance,
                "sample_size_a": len(values_a),
                "sample_size_b": len(values_b)
            }
            
            # Power analysis
            comparison["statistical_power"] = self._calculate_power(values_a, values_b, cohens_d)
            
            return comparison
            
        except Exception as e:
            return {"error": f"Statistical test failed: {str(e)}"}
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_power(self, values_a: List[float], values_b: List[float], 
                        effect_size: float) -> float:
        """Calculate statistical power of the test."""
        # Simplified power calculation
        n1, n2 = len(values_a), len(values_b)
        
        # For equal sample sizes, approximate power calculation
        if n1 == n2:
            n = n1
            # Simplified power calculation for t-test
            ncp = abs(effect_size) * math.sqrt(n / 2)  # Non-centrality parameter
            
            # Approximate power using normal distribution
            critical_t = 1.96  # For alpha = 0.05, two-tailed
            power = 1 - stats.norm.cdf(critical_t - ncp) + stats.norm.cdf(-critical_t - ncp)
            
            return min(1.0, max(0.0, power))
        
        return 0.5  # Default moderate power assumption
    
    def _determine_winner(self, variant_stats: Dict[str, Dict[str, float]]) -> Optional[str]:
        """Determine the winning variant based on mean performance."""
        if not variant_stats:
            return None
        
        best_variant = None
        best_mean = float('-inf')
        
        for variant, stats_dict in variant_stats.items():
            if "mean" in stats_dict and stats_dict["mean"] > best_mean:
                best_mean = stats_dict["mean"]
                best_variant = variant
        
        return best_variant
    
    def _calculate_confidence_level(self, comparisons: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall confidence level in results."""
        if not comparisons:
            return 0.0
        
        p_values = []
        for comparison in comparisons.values():
            if "p_value" in comparison:
                p_values.append(comparison["p_value"])
        
        if not p_values:
            return 0.0
        
        # Use minimum p-value as conservative estimate
        min_p_value = min(p_values)
        confidence = 1.0 - min_p_value
        
        return confidence
    
    def _conclude_experiment(self, metric_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Draw overall conclusions from metric analyses."""
        conclusion = {
            "clear_winner": None,
            "significant_metrics": [],
            "inconclusive_metrics": [],
            "overall_confidence": 0.0,
            "recommendation": "continue_experiment"
        }
        
        winners = {}
        significant_count = 0
        total_metrics = len(metric_analyses)
        
        for metric_name, analysis in metric_analyses.items():
            if "winner" in analysis and analysis["winner"]:
                winner = analysis["winner"]
                if winner not in winners:
                    winners[winner] = 0
                winners[winner] += 1
                
                if analysis.get("significant_differences"):
                    conclusion["significant_metrics"].append(metric_name)
                    significant_count += 1
                else:
                    conclusion["inconclusive_metrics"].append(metric_name)
        
        # Determine overall winner
        if winners:
            conclusion["clear_winner"] = max(winners.keys(), key=lambda k: winners[k])
        
        # Calculate overall confidence
        if total_metrics > 0:
            conclusion["overall_confidence"] = significant_count / total_metrics
        
        # Make recommendation
        if conclusion["overall_confidence"] > 0.7:
            conclusion["recommendation"] = "implement_winner"
        elif conclusion["overall_confidence"] > 0.3:
            conclusion["recommendation"] = "extend_experiment"
        else:
            conclusion["recommendation"] = "redesign_experiment"
        
        return conclusion
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        overall_results = analysis["overall_results"]
        
        if overall_results["recommendation"] == "implement_winner":
            winner = overall_results["clear_winner"]
            recommendations.append(f"Implement variant '{winner}' - statistically significant improvement")
            
        elif overall_results["recommendation"] == "extend_experiment":
            recommendations.append("Extend experiment duration to gather more data")
            recommendations.append("Consider increasing sample size for better power")
            
        elif overall_results["recommendation"] == "redesign_experiment":
            recommendations.append("Redesign experiment with larger effect sizes")
            recommendations.append("Consider different metrics or variants")
        
        # Sample size recommendations
        for metric_name, metric_analysis in analysis["metrics"].items():
            for comparison in metric_analysis.get("comparisons", {}).values():
                if "statistical_power" in comparison:
                    power = comparison["statistical_power"]
                    if power < 0.8:
                        recommendations.append(
                            f"Increase sample size for {metric_name} (current power: {power:.2f})"
                        )
        
        return recommendations