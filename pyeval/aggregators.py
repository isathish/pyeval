"""
PyEval Aggregators - Metric Aggregation Utilities
=================================================

Provides aggregation strategies for:
- Statistical aggregation
- Weighted combinations
- Ensemble evaluation
- Cross-validation support
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math

T = TypeVar('T')


# =============================================================================
# Base Aggregator
# =============================================================================

class Aggregator(ABC):
    """Abstract base class for aggregators."""
    
    @abstractmethod
    def aggregate(self, values: List[float]) -> float:
        """Aggregate a list of values."""
        pass
    
    def __call__(self, values: List[float]) -> float:
        return self.aggregate(values)


# =============================================================================
# Statistical Aggregators
# =============================================================================

class MeanAggregator(Aggregator):
    """Compute arithmetic mean."""
    
    def aggregate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)


class WeightedMeanAggregator(Aggregator):
    """Compute weighted mean."""
    
    def __init__(self, weights: List[float] = None):
        self.weights = weights
    
    def aggregate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        weights = self.weights or [1.0] * len(values)
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        return weighted_sum / sum(weights)


class MedianAggregator(Aggregator):
    """Compute median."""
    
    def aggregate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2
        return sorted_values[mid]


class MinAggregator(Aggregator):
    """Compute minimum."""
    
    def aggregate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return min(values)


class MaxAggregator(Aggregator):
    """Compute maximum."""
    
    def aggregate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return max(values)


class SumAggregator(Aggregator):
    """Compute sum."""
    
    def aggregate(self, values: List[float]) -> float:
        return sum(values)


class StdAggregator(Aggregator):
    """Compute standard deviation."""
    
    def __init__(self, sample: bool = True):
        self.sample = sample
    
    def aggregate(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values)
        divisor = len(values) - 1 if self.sample else len(values)
        return math.sqrt(variance / divisor)


class VarianceAggregator(Aggregator):
    """Compute variance."""
    
    def __init__(self, sample: bool = True):
        self.sample = sample
    
    def aggregate(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values)
        divisor = len(values) - 1 if self.sample else len(values)
        return variance / divisor


class PercentileAggregator(Aggregator):
    """Compute percentile."""
    
    def __init__(self, percentile: float = 50.0):
        self.percentile = percentile
    
    def aggregate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (self.percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(k)]
        return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


# =============================================================================
# Custom Aggregators
# =============================================================================

class GeometricMeanAggregator(Aggregator):
    """Compute geometric mean."""
    
    def aggregate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        # Filter positive values
        positive_values = [v for v in values if v > 0]
        if not positive_values:
            return 0.0
        product = 1.0
        for v in positive_values:
            product *= v
        return product ** (1.0 / len(positive_values))


class HarmonicMeanAggregator(Aggregator):
    """Compute harmonic mean."""
    
    def aggregate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        # Filter positive values
        positive_values = [v for v in values if v > 0]
        if not positive_values:
            return 0.0
        reciprocal_sum = sum(1.0 / v for v in positive_values)
        return len(positive_values) / reciprocal_sum


class TrimmedMeanAggregator(Aggregator):
    """Compute trimmed mean (removes outliers)."""
    
    def __init__(self, trim_percent: float = 10.0):
        self.trim_percent = trim_percent
    
    def aggregate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        trim_count = int(n * (self.trim_percent / 100.0))
        if trim_count * 2 >= n:
            return sum(values) / n
        trimmed = sorted_values[trim_count:n - trim_count]
        return sum(trimmed) / len(trimmed) if trimmed else 0.0


class WinsorizedMeanAggregator(Aggregator):
    """Compute winsorized mean (replaces outliers with boundary values)."""
    
    def __init__(self, trim_percent: float = 10.0):
        self.trim_percent = trim_percent
    
    def aggregate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        trim_count = int(n * (self.trim_percent / 100.0))
        if trim_count >= n // 2:
            return sum(values) / n
        
        lower_bound = sorted_values[trim_count]
        upper_bound = sorted_values[n - trim_count - 1]
        
        winsorized = [
            lower_bound if v < lower_bound else 
            upper_bound if v > upper_bound else v
            for v in values
        ]
        return sum(winsorized) / len(winsorized)


# =============================================================================
# Metric Aggregation Results
# =============================================================================

@dataclass
class AggregationResult:
    """Result of metric aggregation."""
    metric_name: str
    values: List[float]
    aggregations: Dict[str, float] = field(default_factory=dict)
    
    def add_aggregation(self, name: str, value: float):
        """Add an aggregation result."""
        self.aggregations[name] = value
    
    def summary(self) -> str:
        """Get summary string."""
        lines = [f"Metric: {self.metric_name}"]
        lines.append(f"  Samples: {len(self.values)}")
        for name, value in self.aggregations.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


# =============================================================================
# Multi-Metric Aggregator
# =============================================================================

class MetricAggregator:
    """
    Aggregator for multiple metrics across multiple evaluations.
    
    Example:
        aggregator = MetricAggregator()
        aggregator.add_result('accuracy', 0.95)
        aggregator.add_result('accuracy', 0.92)
        aggregator.add_result('f1', 0.90)
        aggregator.add_result('f1', 0.88)
        
        stats = aggregator.get_statistics()
    """
    
    def __init__(self, aggregators: List[Aggregator] = None):
        self.aggregators = aggregators or [
            ('mean', MeanAggregator()),
            ('std', StdAggregator()),
            ('min', MinAggregator()),
            ('max', MaxAggregator())
        ]
        self._results: Dict[str, List[float]] = {}
    
    def add_result(self, metric_name: str, value: float):
        """Add a metric result."""
        if metric_name not in self._results:
            self._results[metric_name] = []
        self._results[metric_name].append(value)
    
    def add_results(self, results: Dict[str, float]):
        """Add multiple metric results."""
        for name, value in results.items():
            self.add_result(name, value)
    
    def get_values(self, metric_name: str) -> List[float]:
        """Get all values for a metric."""
        return self._results.get(metric_name, [])
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated statistics for all metrics."""
        stats = {}
        for metric_name, values in self._results.items():
            stats[metric_name] = {}
            for agg_name, aggregator in self.aggregators:
                stats[metric_name][agg_name] = aggregator.aggregate(values)
            stats[metric_name]['count'] = len(values)
        return stats
    
    def get_metric_statistics(self, metric_name: str) -> AggregationResult:
        """Get detailed statistics for one metric."""
        values = self._results.get(metric_name, [])
        result = AggregationResult(metric_name=metric_name, values=values)
        for agg_name, aggregator in self.aggregators:
            result.add_aggregation(agg_name, aggregator.aggregate(values))
        return result
    
    def reset(self):
        """Reset all results."""
        self._results.clear()
    
    def summary(self) -> str:
        """Get summary of all metrics."""
        stats = self.get_statistics()
        lines = ["Metric Aggregation Summary", "=" * 40]
        for metric_name, metric_stats in stats.items():
            lines.append(f"\n{metric_name}:")
            for stat_name, value in metric_stats.items():
                if stat_name == 'count':
                    lines.append(f"  {stat_name}: {int(value)}")
                else:
                    lines.append(f"  {stat_name}: {value:.4f}")
        return "\n".join(lines)


# =============================================================================
# Cross-Validation Aggregator
# =============================================================================

@dataclass
class FoldResult:
    """Result from a single fold."""
    fold_index: int
    metrics: Dict[str, float]
    train_size: int = 0
    test_size: int = 0


class CrossValidationAggregator:
    """
    Aggregator for cross-validation results.
    
    Example:
        cv_aggregator = CrossValidationAggregator(n_folds=5)
        
        for fold_idx, (y_true, y_pred) in enumerate(cv_splits):
            cv_aggregator.add_fold_result(fold_idx, {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
            })
        
        summary = cv_aggregator.get_summary()
    """
    
    def __init__(self, n_folds: int):
        self.n_folds = n_folds
        self._fold_results: List[FoldResult] = []
    
    def add_fold_result(self, fold_index: int, metrics: Dict[str, float],
                        train_size: int = 0, test_size: int = 0):
        """Add results from a fold."""
        self._fold_results.append(FoldResult(
            fold_index=fold_index,
            metrics=metrics,
            train_size=train_size,
            test_size=test_size
        ))
    
    def get_fold_results(self) -> List[FoldResult]:
        """Get all fold results."""
        return self._fold_results
    
    def get_metric_values(self, metric_name: str) -> List[float]:
        """Get values for a metric across folds."""
        return [fr.metrics.get(metric_name, 0.0) for fr in self._fold_results]
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics."""
        if not self._fold_results:
            return {}
        
        # Get all metric names
        metric_names = set()
        for fr in self._fold_results:
            metric_names.update(fr.metrics.keys())
        
        summary = {}
        for metric_name in metric_names:
            values = self.get_metric_values(metric_name)
            n = len(values)
            mean = sum(values) / n
            std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0
            
            summary[metric_name] = {
                'mean': mean,
                'std': std,
                'min': min(values),
                'max': max(values),
                'folds': n
            }
        
        return summary
    
    def get_best_fold(self, metric_name: str, maximize: bool = True) -> FoldResult:
        """Get the best performing fold for a metric."""
        if not self._fold_results:
            return None
        
        key_func = lambda fr: fr.metrics.get(metric_name, float('-inf') if maximize else float('inf'))
        return max(self._fold_results, key=key_func) if maximize else min(self._fold_results, key=key_func)
    
    def summary_table(self) -> str:
        """Get summary as formatted table."""
        summary = self.get_summary()
        if not summary:
            return "No results"
        
        lines = [f"Cross-Validation Results ({self.n_folds} folds)", "=" * 60]
        lines.append(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        lines.append("-" * 60)
        
        for metric_name, stats in summary.items():
            lines.append(
                f"{metric_name:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                f"{stats['min']:>10.4f} {stats['max']:>10.4f}"
            )
        
        return "\n".join(lines)


# =============================================================================
# Ensemble Aggregator
# =============================================================================

class EnsembleAggregator:
    """
    Aggregator for ensemble model predictions.
    
    Example:
        ensemble = EnsembleAggregator(strategy='majority_vote')
        
        # Add predictions from each model
        ensemble.add_predictions('model1', [1, 0, 1, 1])
        ensemble.add_predictions('model2', [1, 1, 1, 0])
        ensemble.add_predictions('model3', [0, 0, 1, 1])
        
        final_predictions = ensemble.aggregate()  # [1, 0, 1, 1]
    """
    
    def __init__(self, strategy: str = 'majority_vote', 
                 weights: Dict[str, float] = None):
        self.strategy = strategy
        self.weights = weights or {}
        self._predictions: Dict[str, List] = {}
    
    def add_predictions(self, model_name: str, predictions: List):
        """Add predictions from a model."""
        self._predictions[model_name] = predictions
    
    def set_weight(self, model_name: str, weight: float):
        """Set weight for a model."""
        self.weights[model_name] = weight
    
    def aggregate(self) -> List:
        """Aggregate predictions based on strategy."""
        if not self._predictions:
            return []
        
        if self.strategy == 'majority_vote':
            return self._majority_vote()
        elif self.strategy == 'weighted_vote':
            return self._weighted_vote()
        elif self.strategy == 'average':
            return self._average()
        elif self.strategy == 'weighted_average':
            return self._weighted_average()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _majority_vote(self) -> List:
        """Aggregate by majority voting."""
        n_samples = len(next(iter(self._predictions.values())))
        results = []
        
        for i in range(n_samples):
            votes = {}
            for preds in self._predictions.values():
                vote = preds[i]
                votes[vote] = votes.get(vote, 0) + 1
            results.append(max(votes, key=votes.get))
        
        return results
    
    def _weighted_vote(self) -> List:
        """Aggregate by weighted voting."""
        n_samples = len(next(iter(self._predictions.values())))
        results = []
        
        # Default weights
        default_weight = 1.0 / len(self._predictions)
        
        for i in range(n_samples):
            votes = {}
            for model_name, preds in self._predictions.items():
                vote = preds[i]
                weight = self.weights.get(model_name, default_weight)
                votes[vote] = votes.get(vote, 0) + weight
            results.append(max(votes, key=votes.get))
        
        return results
    
    def _average(self) -> List:
        """Aggregate by averaging (for numeric predictions)."""
        n_samples = len(next(iter(self._predictions.values())))
        n_models = len(self._predictions)
        
        results = []
        for i in range(n_samples):
            total = sum(preds[i] for preds in self._predictions.values())
            results.append(total / n_models)
        
        return results
    
    def _weighted_average(self) -> List:
        """Aggregate by weighted averaging."""
        n_samples = len(next(iter(self._predictions.values())))
        default_weight = 1.0 / len(self._predictions)
        
        results = []
        for i in range(n_samples):
            weighted_sum = 0
            weight_sum = 0
            for model_name, preds in self._predictions.items():
                weight = self.weights.get(model_name, default_weight)
                weighted_sum += preds[i] * weight
                weight_sum += weight
            results.append(weighted_sum / weight_sum)
        
        return results


# =============================================================================
# Factory Functions
# =============================================================================

def create_aggregator(name: str, **kwargs) -> Aggregator:
    """
    Create an aggregator by name.
    
    Example:
        agg = create_aggregator('mean')
        agg = create_aggregator('percentile', percentile=75)
    """
    aggregators = {
        'mean': MeanAggregator,
        'weighted_mean': WeightedMeanAggregator,
        'median': MedianAggregator,
        'min': MinAggregator,
        'max': MaxAggregator,
        'sum': SumAggregator,
        'std': StdAggregator,
        'variance': VarianceAggregator,
        'percentile': PercentileAggregator,
        'geometric_mean': GeometricMeanAggregator,
        'harmonic_mean': HarmonicMeanAggregator,
        'trimmed_mean': TrimmedMeanAggregator,
        'winsorized_mean': WinsorizedMeanAggregator,
    }
    
    if name not in aggregators:
        raise ValueError(f"Unknown aggregator: {name}")
    
    return aggregators[name](**kwargs)


# =============================================================================
# Export all
# =============================================================================

__all__ = [
    # Base
    'Aggregator',
    
    # Statistical Aggregators
    'MeanAggregator',
    'WeightedMeanAggregator',
    'MedianAggregator',
    'MinAggregator',
    'MaxAggregator',
    'SumAggregator',
    'StdAggregator',
    'VarianceAggregator',
    'PercentileAggregator',
    
    # Custom Aggregators
    'GeometricMeanAggregator',
    'HarmonicMeanAggregator',
    'TrimmedMeanAggregator',
    'WinsorizedMeanAggregator',
    
    # Results
    'AggregationResult',
    
    # Multi-Metric
    'MetricAggregator',
    
    # Cross-Validation
    'FoldResult',
    'CrossValidationAggregator',
    
    # Ensemble
    'EnsembleAggregator',
    
    # Factory
    'create_aggregator',
]
