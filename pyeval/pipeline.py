"""
PyEval Pipeline - Evaluation Pipeline Builder
==============================================

Provides fluent pipeline building for:
- Data preprocessing
- Metric computation
- Result aggregation
- Report generation
"""

from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import time

T = TypeVar('T')


# =============================================================================
# Pipeline Steps
# =============================================================================

class StepType(Enum):
    """Types of pipeline steps."""
    PREPROCESSOR = "preprocessor"
    TRANSFORMER = "transformer"
    METRIC = "metric"
    AGGREGATOR = "aggregator"
    REPORTER = "reporter"
    VALIDATOR = "validator"


@dataclass
class PipelineStepConfig:
    """Configuration for a pipeline step."""
    name: str
    step_type: StepType
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    enabled: bool = True
    continue_on_error: bool = False


@dataclass
class StepResult:
    """Result from a pipeline step."""
    name: str
    step_type: StepType
    output: Any
    duration: float
    success: bool
    error: Optional[str] = None


# =============================================================================
# Pipeline Stage Interface
# =============================================================================

class PipelineStage(ABC):
    """Abstract pipeline stage."""
    
    @abstractmethod
    def execute(self, data: Any) -> Any:
        """Execute the stage."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name."""
        pass


# =============================================================================
# Built-in Stages
# =============================================================================

class PreprocessorStage(PipelineStage):
    """
    Preprocessor stage for data transformation.
    
    Example:
        stage = PreprocessorStage('tokenize', tokenize_func)
    """
    
    def __init__(self, name: str, func: Callable, **kwargs):
        self._name = name
        self._func = func
        self._kwargs = kwargs
    
    def execute(self, data: Tuple[List, List]) -> Tuple[List, List]:
        y_true, y_pred = data
        return self._func(y_true, y_pred, **self._kwargs)
    
    @property
    def name(self) -> str:
        return f"preprocess:{self._name}"


class MetricStage(PipelineStage):
    """
    Metric computation stage.
    
    Example:
        stage = MetricStage('accuracy', accuracy_score)
    """
    
    def __init__(self, name: str, func: Callable, **kwargs):
        self._name = name
        self._func = func
        self._kwargs = kwargs
    
    def execute(self, data: Tuple[List, List]) -> Tuple[str, float]:
        y_true, y_pred = data
        value = self._func(y_true, y_pred, **self._kwargs)
        return (self._name, value)
    
    @property
    def name(self) -> str:
        return f"metric:{self._name}"


class AggregatorStage(PipelineStage):
    """
    Aggregator stage for combining results.
    
    Example:
        stage = AggregatorStage('mean', lambda results: sum(results.values()) / len(results))
    """
    
    def __init__(self, name: str, func: Callable):
        self._name = name
        self._func = func
    
    def execute(self, data: Dict[str, float]) -> Any:
        return self._func(data)
    
    @property
    def name(self) -> str:
        return f"aggregate:{self._name}"


class ValidatorStage(PipelineStage):
    """
    Validator stage for input validation.
    
    Example:
        stage = ValidatorStage('non_empty', lambda x: len(x[0]) > 0)
    """
    
    def __init__(self, name: str, func: Callable[[Tuple], bool], 
                 error_message: str = "Validation failed"):
        self._name = name
        self._func = func
        self._error_message = error_message
    
    def execute(self, data: Tuple[List, List]) -> Tuple[List, List]:
        if not self._func(data):
            raise ValueError(f"{self._name}: {self._error_message}")
        return data
    
    @property
    def name(self) -> str:
        return f"validate:{self._name}"


# =============================================================================
# Pipeline Builder
# =============================================================================

class Pipeline:
    """
    Fluent evaluation pipeline.
    
    Example:
        pipeline = (
            Pipeline()
            .validate(lambda x: len(x[0]) == len(x[1]), "Length mismatch")
            .preprocess(lowercase)
            .add_metric('accuracy', accuracy_score)
            .add_metric('f1', f1_score, average='macro')
            .aggregate('mean', lambda r: sum(r.values()) / len(r))
        )
        
        results = pipeline.run(y_true, y_pred)
    """
    
    def __init__(self, name: str = "evaluation_pipeline"):
        self.name = name
        self._validators: List[ValidatorStage] = []
        self._preprocessors: List[PreprocessorStage] = []
        self._metrics: List[MetricStage] = []
        self._aggregators: List[AggregatorStage] = []
        self._hooks: Dict[str, List[Callable]] = {
            'before_run': [],
            'after_run': [],
            'before_metric': [],
            'after_metric': [],
            'on_error': []
        }
        self._results_history: List[Dict] = []
    
    # =========================================================================
    # Builder Methods
    # =========================================================================
    
    def validate(self, func: Callable[[Tuple], bool], 
                 error_message: str = "Validation failed",
                 name: str = None) -> 'Pipeline':
        """Add a validation step."""
        name = name or f"validator_{len(self._validators)}"
        self._validators.append(ValidatorStage(name, func, error_message))
        return self
    
    def preprocess(self, func: Callable, name: str = None, **kwargs) -> 'Pipeline':
        """Add a preprocessing step."""
        name = name or func.__name__
        self._preprocessors.append(PreprocessorStage(name, func, **kwargs))
        return self
    
    def add_metric(self, name: str, func: Callable, **kwargs) -> 'Pipeline':
        """Add a metric computation step."""
        self._metrics.append(MetricStage(name, func, **kwargs))
        return self
    
    def add_metrics(self, metrics: Dict[str, Callable]) -> 'Pipeline':
        """Add multiple metrics at once."""
        for name, func in metrics.items():
            self.add_metric(name, func)
        return self
    
    def aggregate(self, name: str, func: Callable) -> 'Pipeline':
        """Add an aggregation step."""
        self._aggregators.append(AggregatorStage(name, func))
        return self
    
    def add_hook(self, event: str, func: Callable) -> 'Pipeline':
        """Add a hook for an event."""
        if event in self._hooks:
            self._hooks[event].append(func)
        return self
    
    # =========================================================================
    # Execution Methods
    # =========================================================================
    
    def _call_hooks(self, event: str, *args, **kwargs):
        """Call all hooks for an event."""
        for hook in self._hooks.get(event, []):
            hook(*args, **kwargs)
    
    def run(self, y_true: List, y_pred: List, 
            return_details: bool = False) -> Union[Dict[str, float], 'PipelineResult']:
        """Execute the pipeline."""
        start_time = time.perf_counter()
        step_results: List[StepResult] = []
        data = (y_true, y_pred)
        
        self._call_hooks('before_run', data)
        
        try:
            # Validation
            for validator in self._validators:
                step_start = time.perf_counter()
                try:
                    data = validator.execute(data)
                    step_results.append(StepResult(
                        name=validator.name,
                        step_type=StepType.VALIDATOR,
                        output=True,
                        duration=time.perf_counter() - step_start,
                        success=True
                    ))
                except Exception as e:
                    step_results.append(StepResult(
                        name=validator.name,
                        step_type=StepType.VALIDATOR,
                        output=False,
                        duration=time.perf_counter() - step_start,
                        success=False,
                        error=str(e)
                    ))
                    raise
            
            # Preprocessing
            for preprocessor in self._preprocessors:
                step_start = time.perf_counter()
                data = preprocessor.execute(data)
                step_results.append(StepResult(
                    name=preprocessor.name,
                    step_type=StepType.PREPROCESSOR,
                    output=None,
                    duration=time.perf_counter() - step_start,
                    success=True
                ))
            
            # Metrics
            metric_results: Dict[str, float] = {}
            for metric in self._metrics:
                self._call_hooks('before_metric', metric.name)
                step_start = time.perf_counter()
                
                try:
                    name, value = metric.execute(data)
                    metric_results[name] = value
                    step_results.append(StepResult(
                        name=metric.name,
                        step_type=StepType.METRIC,
                        output=value,
                        duration=time.perf_counter() - step_start,
                        success=True
                    ))
                except Exception as e:
                    step_results.append(StepResult(
                        name=metric.name,
                        step_type=StepType.METRIC,
                        output=None,
                        duration=time.perf_counter() - step_start,
                        success=False,
                        error=str(e)
                    ))
                    self._call_hooks('on_error', metric.name, e)
                
                self._call_hooks('after_metric', metric.name, metric_results.get(metric._name))
            
            # Aggregation
            aggregated = {}
            for aggregator in self._aggregators:
                step_start = time.perf_counter()
                result = aggregator.execute(metric_results)
                aggregated[aggregator._name] = result
                step_results.append(StepResult(
                    name=aggregator.name,
                    step_type=StepType.AGGREGATOR,
                    output=result,
                    duration=time.perf_counter() - step_start,
                    success=True
                ))
            
            total_duration = time.perf_counter() - start_time
            
            # Store in history
            self._results_history.append({
                'metrics': metric_results,
                'aggregated': aggregated,
                'duration': total_duration
            })
            
            self._call_hooks('after_run', metric_results)
            
            if return_details:
                return PipelineResult(
                    metrics=metric_results,
                    aggregated=aggregated,
                    step_results=step_results,
                    duration=total_duration
                )
            
            return {**metric_results, **aggregated}
            
        except Exception as e:
            self._call_hooks('on_error', 'pipeline', e)
            raise
    
    def run_batch(self, batches: List[Tuple[List, List]]) -> List[Dict[str, float]]:
        """Execute pipeline on multiple batches."""
        return [self.run(y_true, y_pred) for y_true, y_pred in batches]
    
    def get_history(self) -> List[Dict]:
        """Get execution history."""
        return self._results_history
    
    def clear_history(self):
        """Clear execution history."""
        self._results_history.clear()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def clone(self) -> 'Pipeline':
        """Create a copy of the pipeline."""
        new_pipeline = Pipeline(self.name)
        new_pipeline._validators = list(self._validators)
        new_pipeline._preprocessors = list(self._preprocessors)
        new_pipeline._metrics = list(self._metrics)
        new_pipeline._aggregators = list(self._aggregators)
        return new_pipeline
    
    def extend(self, other: 'Pipeline') -> 'Pipeline':
        """Extend pipeline with another pipeline's stages."""
        self._validators.extend(other._validators)
        self._preprocessors.extend(other._preprocessors)
        self._metrics.extend(other._metrics)
        self._aggregators.extend(other._aggregators)
        return self
    
    def summary(self) -> str:
        """Get pipeline summary."""
        lines = [f"Pipeline: {self.name}", "=" * 40]
        
        if self._validators:
            lines.append(f"Validators ({len(self._validators)}):")
            for v in self._validators:
                lines.append(f"  - {v._name}")
        
        if self._preprocessors:
            lines.append(f"Preprocessors ({len(self._preprocessors)}):")
            for p in self._preprocessors:
                lines.append(f"  - {p._name}")
        
        if self._metrics:
            lines.append(f"Metrics ({len(self._metrics)}):")
            for m in self._metrics:
                lines.append(f"  - {m._name}")
        
        if self._aggregators:
            lines.append(f"Aggregators ({len(self._aggregators)}):")
            for a in self._aggregators:
                lines.append(f"  - {a._name}")
        
        return "\n".join(lines)


@dataclass
class PipelineResult:
    """Detailed result from pipeline execution."""
    metrics: Dict[str, float]
    aggregated: Dict[str, Any]
    step_results: List[StepResult]
    duration: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'metrics': self.metrics,
            'aggregated': self.aggregated,
            'duration': self.duration,
            'steps': [
                {
                    'name': s.name,
                    'type': s.step_type.value,
                    'success': s.success,
                    'duration': s.duration,
                    'error': s.error
                }
                for s in self.step_results
            ]
        }
    
    def summary(self) -> str:
        """Get result summary."""
        lines = ["Pipeline Result", "=" * 40]
        
        lines.append(f"Duration: {self.duration:.4f}s")
        lines.append("")
        
        lines.append("Metrics:")
        for name, value in self.metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        
        if self.aggregated:
            lines.append("")
            lines.append("Aggregated:")
            for name, value in self.aggregated.items():
                lines.append(f"  {name}: {value}")
        
        return "\n".join(lines)


# =============================================================================
# Pipeline Presets
# =============================================================================

def create_classification_pipeline() -> Pipeline:
    """Create a preset classification evaluation pipeline."""
    from pyeval.ml import accuracy_score, precision_score, recall_score, f1_score
    
    return (
        Pipeline("classification")
        .validate(
            lambda x: len(x[0]) == len(x[1]),
            "y_true and y_pred must have same length"
        )
        .validate(
            lambda x: len(x[0]) > 0,
            "Input arrays cannot be empty"
        )
        .add_metric('accuracy', accuracy_score)
        .add_metric('precision', precision_score)
        .add_metric('recall', recall_score)
        .add_metric('f1', f1_score)
        .aggregate('mean_score', lambda r: sum(r.values()) / len(r))
    )


def create_regression_pipeline() -> Pipeline:
    """Create a preset regression evaluation pipeline."""
    from pyeval.ml import mean_squared_error, mean_absolute_error, r2_score
    
    return (
        Pipeline("regression")
        .validate(
            lambda x: len(x[0]) == len(x[1]),
            "y_true and y_pred must have same length"
        )
        .add_metric('mse', mean_squared_error)
        .add_metric('mae', mean_absolute_error)
        .add_metric('r2', r2_score)
    )


def create_nlp_pipeline() -> Pipeline:
    """Create a preset NLP evaluation pipeline."""
    from pyeval.nlp import bleu_score, rouge_scores
    
    def bleu_wrapper(y_true: List, y_pred: List) -> float:
        # Compute average BLEU across all pairs
        scores = [bleu_score(ref, hyp) for ref, hyp in zip(y_true, y_pred)]
        return sum(scores) / len(scores) if scores else 0.0
    
    def rouge_wrapper(y_true: List, y_pred: List) -> float:
        # Compute average ROUGE-L across all pairs
        scores = []
        for ref, hyp in zip(y_true, y_pred):
            rouge = rouge_scores(ref, hyp)
            scores.append(rouge.get('rouge_l', {}).get('f', 0.0))
        return sum(scores) / len(scores) if scores else 0.0
    
    return (
        Pipeline("nlp")
        .add_metric('bleu', bleu_wrapper)
        .add_metric('rouge_l', rouge_wrapper)
    )


# =============================================================================
# Pipeline Registry
# =============================================================================

class PipelineRegistry:
    """
    Registry for reusable pipelines.
    
    Example:
        registry = PipelineRegistry()
        registry.register('classification', create_classification_pipeline())
        
        pipeline = registry.get('classification')
        results = pipeline.run(y_true, y_pred)
    """
    
    def __init__(self):
        self._pipelines: Dict[str, Pipeline] = {}
    
    def register(self, name: str, pipeline: Pipeline):
        """Register a pipeline."""
        self._pipelines[name] = pipeline
    
    def get(self, name: str) -> Pipeline:
        """Get a pipeline (returns a clone)."""
        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not found")
        return self._pipelines[name].clone()
    
    def list_pipelines(self) -> List[str]:
        """List registered pipeline names."""
        return list(self._pipelines.keys())
    
    def unregister(self, name: str):
        """Remove a pipeline."""
        self._pipelines.pop(name, None)


# Global registry
pipeline_registry = PipelineRegistry()


# =============================================================================
# Export all
# =============================================================================

__all__ = [
    # Types
    'StepType',
    'PipelineStepConfig',
    'StepResult',
    
    # Stages
    'PipelineStage',
    'PreprocessorStage',
    'MetricStage',
    'AggregatorStage',
    'ValidatorStage',
    
    # Pipeline
    'Pipeline',
    'PipelineResult',
    
    # Presets
    'create_classification_pipeline',
    'create_regression_pipeline',
    'create_nlp_pipeline',
    
    # Registry
    'PipelineRegistry',
    'pipeline_registry',
]
