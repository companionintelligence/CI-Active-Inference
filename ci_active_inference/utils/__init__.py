"""Utility functions for CI Active Inference system."""

import os
import logging
from typing import Dict, Any, List
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Configuration management."""
    
    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost:5432/ci_database")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "ci_database")
    DB_USER = os.getenv("DB_USER", "username")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    
    # Active Inference settings
    INFERENCE_UPDATE_INTERVAL = int(os.getenv("INFERENCE_UPDATE_INTERVAL", "60"))
    PREDICTION_HORIZON = int(os.getenv("PREDICTION_HORIZON", "10"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.01"))
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def setup_logging():
    """Configure logging for the application."""
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ci_active_inference.log') if not Config.DEBUG else logging.NullHandler()
        ]
    )


def normalize_data(data: List[float], min_val: float = 0.0, max_val: float = 1.0) -> List[float]:
    """
    Normalize data to a specified range.
    
    Args:
        data: Input data to normalize
        min_val: Minimum value for normalized range
        max_val: Maximum value for normalized range
        
    Returns:
        Normalized data
    """
    if not data:
        return data
    
    data_array = np.array(data)
    data_min = np.min(data_array)
    data_max = np.max(data_array)
    
    if data_max == data_min:
        return [min_val] * len(data)
    
    normalized = (data_array - data_min) / (data_max - data_min)
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized.tolist()


def calculate_entropy(probabilities: List[float]) -> float:
    """
    Calculate Shannon entropy of a probability distribution.
    
    Args:
        probabilities: List of probabilities
        
    Returns:
        Entropy value
    """
    prob_array = np.array(probabilities)
    prob_array = prob_array[prob_array > 0]  # Remove zeros to avoid log(0)
    
    if len(prob_array) == 0:
        return 0.0
    
    return -np.sum(prob_array * np.log2(prob_array))


def calculate_kl_divergence(p: List[float], q: List[float]) -> float:
    """
    Calculate Kullback-Leibler divergence between two distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        KL divergence value
    """
    p_array = np.array(p)
    q_array = np.array(q)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    q_array = q_array + epsilon
    
    # Only consider non-zero elements of p
    mask = p_array > 0
    
    if not np.any(mask):
        return 0.0
    
    return np.sum(p_array[mask] * np.log(p_array[mask] / q_array[mask]))


def softmax(x: List[float], temperature: float = 1.0) -> List[float]:
    """
    Apply softmax function with temperature scaling.
    
    Args:
        x: Input values
        temperature: Temperature parameter for scaling
        
    Returns:
        Softmax probabilities
    """
    x_array = np.array(x) / temperature
    exp_x = np.exp(x_array - np.max(x_array))  # Subtract max for numerical stability
    return (exp_x / np.sum(exp_x)).tolist()


def moving_average(data: List[float], window: int) -> List[float]:
    """
    Calculate moving average of data.
    
    Args:
        data: Input data
        window: Window size for averaging
        
    Returns:
        Moving averages
    """
    if len(data) < window:
        return data
    
    result = []
    for i in range(len(data)):
        if i < window - 1:
            result.append(np.mean(data[:i+1]))
        else:
            result.append(np.mean(data[i-window+1:i+1]))
    
    return result


def time_to_discrete(time: datetime, bins: int = 24) -> int:
    """
    Convert time to discrete bin (e.g., hour of day).
    
    Args:
        time: Datetime object
        bins: Number of discrete bins
        
    Returns:
        Discrete bin index
    """
    if bins == 24:  # Hour of day
        return time.hour
    elif bins == 7:  # Day of week
        return time.weekday()
    else:
        # General binning based on hour
        return min(int(time.hour * bins / 24), bins - 1)


def create_time_series_features(timestamps: List[datetime]) -> Dict[str, List[float]]:
    """
    Create time series features from timestamps.
    
    Args:
        timestamps: List of datetime objects
        
    Returns:
        Dictionary of time features
    """
    features = {
        "hour_of_day": [],
        "day_of_week": [],
        "week_of_year": [],
        "month": [],
        "is_weekend": [],
        "is_business_hours": []
    }
    
    for ts in timestamps:
        features["hour_of_day"].append(ts.hour / 24.0)  # Normalize to [0, 1]
        features["day_of_week"].append(ts.weekday() / 7.0)
        features["week_of_year"].append(ts.isocalendar()[1] / 52.0)
        features["month"].append(ts.month / 12.0)
        features["is_weekend"].append(1.0 if ts.weekday() >= 5 else 0.0)
        features["is_business_hours"].append(1.0 if 9 <= ts.hour <= 17 else 0.0)
    
    return features


def validate_probability_distribution(probs: List[float], tolerance: float = 1e-6) -> bool:
    """
    Validate that a list represents a valid probability distribution.
    
    Args:
        probs: List of probabilities
        tolerance: Tolerance for sum validation
        
    Returns:
        True if valid probability distribution
    """
    if not probs:
        return False
    
    # Check non-negative
    if any(p < 0 for p in probs):
        return False
    
    # Check sums to 1
    prob_sum = sum(probs)
    if abs(prob_sum - 1.0) > tolerance:
        return False
    
    return True


def exponential_smoothing(data: List[float], alpha: float = 0.3) -> List[float]:
    """
    Apply exponential smoothing to data.
    
    Args:
        data: Input data
        alpha: Smoothing parameter (0 < alpha < 1)
        
    Returns:
        Smoothed data
    """
    if not data:
        return data
    
    smoothed = [data[0]]
    
    for i in range(1, len(data)):
        smoothed_value = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        smoothed.append(smoothed_value)
    
    return smoothed


class MetricsCalculator:
    """Utility class for calculating various metrics."""
    
    @staticmethod
    def accuracy(predictions: List[int], targets: List[int]) -> float:
        """Calculate prediction accuracy."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")
        
        if not predictions:
            return 0.0
        
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        return correct / len(predictions)
    
    @staticmethod
    def mean_absolute_error(predictions: List[float], targets: List[float]) -> float:
        """Calculate mean absolute error."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")
        
        if not predictions:
            return 0.0
        
        return np.mean([abs(p - t) for p, t in zip(predictions, targets)])
    
    @staticmethod
    def root_mean_square_error(predictions: List[float], targets: List[float]) -> float:
        """Calculate root mean square error."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")
        
        if not predictions:
            return 0.0
        
        mse = np.mean([(p - t) ** 2 for p, t in zip(predictions, targets)])
        return np.sqrt(mse)
    
    @staticmethod
    def confidence_interval(data: List[float], confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for data."""
        if not data:
            return (0.0, 0.0)
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        n = len(data)
        
        # Using t-distribution for small samples
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_value * (std / np.sqrt(n))
        
        return (mean - margin_error, mean + margin_error)


def format_timestamp(dt: datetime = None) -> str:
    """Format datetime for consistent string representation."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime object."""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        # Try alternative formats
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%H:%M:%S"]:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Could not parse timestamp: {timestamp_str}")


def generate_sample_data() -> Dict[str, Any]:
    """Generate sample data for testing and demonstration."""
    now = datetime.now()
    
    sample_data = {
        "persons": [
            {
                "name": "Alice Johnson",
                "email": "alice@company.com",
                "role": "Data Scientist",
                "department": "AI Research"
            },
            {
                "name": "Bob Smith", 
                "email": "bob@company.com",
                "role": "Software Engineer",
                "department": "Engineering"
            },
            {
                "name": "Carol Wilson",
                "email": "carol@company.com", 
                "role": "Project Manager",
                "department": "Operations"
            }
        ],
        "places": [
            {
                "name": "Conference Room A",
                "place_type": "meeting_room",
                "capacity": 10
            },
            {
                "name": "Main Office",
                "place_type": "office",
                "capacity": 50
            },
            {
                "name": "Lab Space",
                "place_type": "laboratory", 
                "capacity": 8
            }
        ],
        "things": [
            {
                "name": "Laptop Dell XPS",
                "thing_type": "equipment",
                "status": "available"
            },
            {
                "name": "Projector Sony",
                "thing_type": "equipment", 
                "status": "in_use"
            },
            {
                "name": "AI Development Server",
                "thing_type": "software",
                "status": "available"
            }
        ],
        "content": [
            {
                "title": "Q4 Progress Report",
                "content_type": "document",
                "creator_id": 1
            },
            {
                "title": "Team Meeting Notes", 
                "content_type": "document",
                "creator_id": 2
            }
        ],
        "events": [
            {
                "name": "Daily Standup",
                "event_type": "meeting",
                "start_time": now + timedelta(hours=1),
                "end_time": now + timedelta(hours=1.5),
                "person_id": 1,
                "place_id": 1
            },
            {
                "name": "Code Review",
                "event_type": "task", 
                "start_time": now + timedelta(hours=2),
                "end_time": now + timedelta(hours=3),
                "person_id": 2
            }
        ]
    }
    
    return sample_data