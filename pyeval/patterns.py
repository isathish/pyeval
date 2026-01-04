"""
PyEval Design Patterns - Design Patterns for Evaluation
========================================================

Implements common design patterns:
- Strategy Pattern: Interchangeable metric algorithms
- Factory Pattern: Metric creation
- Builder Pattern: Pipeline construction
- Observer Pattern: Event callbacks
- Composite Pattern: Metric composition
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, List, Optional, TypeVar
from dataclasses import dataclass, field
from enum import Enum

T = TypeVar("T")


# =============================================================================
# Strategy Pattern - Interchangeable Metric Algorithms
# =============================================================================


class MetricStrategy(ABC):
    """
    Abstract base class for metric computation strategies.

    Implement this to create interchangeable metric algorithms.
    """

    @abstractmethod
    def compute(self, y_true: List, y_pred: List) -> float:
        """Compute the metric value."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        pass


class AccuracyStrategy(MetricStrategy):
    """Accuracy computation strategy."""

    def compute(self, y_true: List, y_pred: List) -> float:
        if len(y_true) == 0:
            return 0.0
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true)

    @property
    def name(self) -> str:
        return "accuracy"


class PrecisionStrategy(MetricStrategy):
    """Precision computation strategy (binary)."""

    def __init__(self, positive_class: Any = 1):
        self.positive_class = positive_class

    def compute(self, y_true: List, y_pred: List) -> float:
        tp = sum(
            1
            for t, p in zip(y_true, y_pred)
            if p == self.positive_class and t == self.positive_class
        )
        fp = sum(
            1
            for t, p in zip(y_true, y_pred)
            if p == self.positive_class and t != self.positive_class
        )
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def name(self) -> str:
        return "precision"


class RecallStrategy(MetricStrategy):
    """Recall computation strategy (binary)."""

    def __init__(self, positive_class: Any = 1):
        self.positive_class = positive_class

    def compute(self, y_true: List, y_pred: List) -> float:
        tp = sum(
            1
            for t, p in zip(y_true, y_pred)
            if p == self.positive_class and t == self.positive_class
        )
        fn = sum(
            1
            for t, p in zip(y_true, y_pred)
            if p != self.positive_class and t == self.positive_class
        )
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def name(self) -> str:
        return "recall"


class F1Strategy(MetricStrategy):
    """F1 score computation strategy."""

    def __init__(self, positive_class: Any = 1):
        self.positive_class = positive_class
        self.precision_strategy = PrecisionStrategy(positive_class)
        self.recall_strategy = RecallStrategy(positive_class)

    def compute(self, y_true: List, y_pred: List) -> float:
        precision = self.precision_strategy.compute(y_true, y_pred)
        recall = self.recall_strategy.compute(y_true, y_pred)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @property
    def name(self) -> str:
        return "f1"


class MetricCalculator:
    """
    Context class for using metric strategies.

    Example:
        calculator = MetricCalculator(AccuracyStrategy())
        score = calculator.calculate([0, 1, 1], [0, 1, 0])

        calculator.set_strategy(F1Strategy())
        f1 = calculator.calculate([0, 1, 1], [0, 1, 0])
    """

    def __init__(self, strategy: MetricStrategy = None):
        self._strategy = strategy
        self._history: List[Dict] = []

    def set_strategy(self, strategy: MetricStrategy):
        """Change the metric strategy."""
        self._strategy = strategy

    def calculate(self, y_true: List, y_pred: List) -> float:
        """Calculate metric using current strategy."""
        if self._strategy is None:
            raise ValueError("No strategy set")

        result = self._strategy.compute(y_true, y_pred)
        self._history.append({"metric": self._strategy.name, "result": result})
        return result

    def get_history(self) -> List[Dict]:
        """Get calculation history."""
        return self._history


# =============================================================================
# Factory Pattern - Metric Creation
# =============================================================================


class MetricType(Enum):
    """Enum for metric types."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    R2 = "r2"
    BLEU = "bleu"
    ROUGE = "rouge"
    WER = "wer"


class MetricFactory:
    """
    Factory for creating metric computation strategies.

    Example:
        factory = MetricFactory()
        accuracy_metric = factory.create(MetricType.ACCURACY)
        score = accuracy_metric.compute(y_true, y_pred)

        # Register custom metric
        factory.register(MetricType.F1, CustomF1Strategy)
    """

    def __init__(self):
        self._creators: Dict[MetricType, type] = {
            MetricType.ACCURACY: AccuracyStrategy,
            MetricType.PRECISION: PrecisionStrategy,
            MetricType.RECALL: RecallStrategy,
            MetricType.F1: F1Strategy,
        }

    def register(self, metric_type: MetricType, creator: type):
        """Register a metric creator."""
        self._creators[metric_type] = creator

    def create(self, metric_type: MetricType, **kwargs) -> MetricStrategy:
        """Create a metric strategy instance."""
        if metric_type not in self._creators:
            raise ValueError(f"Unknown metric type: {metric_type}")
        return self._creators[metric_type](**kwargs)

    def available_metrics(self) -> List[MetricType]:
        """List available metric types."""
        return list(self._creators.keys())


# Global factory instance
metric_factory = MetricFactory()


# =============================================================================
# Builder Pattern - Pipeline Construction
# =============================================================================


@dataclass
class PipelineStep:
    """A single step in the evaluation pipeline."""

    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)


class EvaluationPipelineBuilder:
    """
    Builder for constructing evaluation pipelines.

    Example:
        pipeline = (
            EvaluationPipelineBuilder()
            .add_preprocessing(tokenize)
            .add_metric('accuracy', accuracy_score)
            .add_metric('f1', f1_score, average='macro')
            .add_postprocessing(round_scores)
            .build()
        )

        results = pipeline.run(y_true, y_pred)
    """

    def __init__(self):
        self._preprocessing: List[PipelineStep] = []
        self._metrics: List[PipelineStep] = []
        self._postprocessing: List[PipelineStep] = []
        self._validators: List[Callable] = []

    def add_preprocessing(self, func: Callable, *args, **kwargs):
        """Add a preprocessing step."""
        self._preprocessing.append(
            PipelineStep(name=func.__name__, func=func, args=args, kwargs=kwargs)
        )
        return self

    def add_metric(self, name: str, func: Callable, *args, **kwargs):
        """Add a metric computation step."""
        self._metrics.append(
            PipelineStep(name=name, func=func, args=args, kwargs=kwargs)
        )
        return self

    def add_postprocessing(self, func: Callable, *args, **kwargs):
        """Add a postprocessing step."""
        self._postprocessing.append(
            PipelineStep(name=func.__name__, func=func, args=args, kwargs=kwargs)
        )
        return self

    def add_validator(self, validator: Callable):
        """Add a validation function."""
        self._validators.append(validator)
        return self

    def build(self) -> "EvaluationPipeline":
        """Build the pipeline."""
        return EvaluationPipeline(
            preprocessing=self._preprocessing,
            metrics=self._metrics,
            postprocessing=self._postprocessing,
            validators=self._validators,
        )


class EvaluationPipeline:
    """
    Evaluation pipeline constructed by builder.
    """

    def __init__(
        self,
        preprocessing: List[PipelineStep],
        metrics: List[PipelineStep],
        postprocessing: List[PipelineStep],
        validators: List[Callable],
    ):
        self.preprocessing = preprocessing
        self.metrics = metrics
        self.postprocessing = postprocessing
        self.validators = validators

    def run(self, y_true: List, y_pred: List) -> Dict[str, Any]:
        """Execute the pipeline."""
        # Validate inputs
        for validator in self.validators:
            if not validator(y_true, y_pred):
                raise ValueError(f"Validation failed: {validator.__name__}")

        # Preprocessing
        for step in self.preprocessing:
            y_true, y_pred = step.func(y_true, y_pred, *step.args, **step.kwargs)

        # Compute metrics
        results = {}
        for step in self.metrics:
            results[step.name] = step.func(y_true, y_pred, *step.args, **step.kwargs)

        # Postprocessing
        for step in self.postprocessing:
            results = step.func(results, *step.args, **step.kwargs)

        return results


# =============================================================================
# Observer Pattern - Event Callbacks
# =============================================================================


class Event(Enum):
    """Evaluation events."""

    METRIC_START = "metric_start"
    METRIC_END = "metric_end"
    BATCH_START = "batch_start"
    BATCH_END = "batch_end"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class EventData:
    """Data associated with an event."""

    event: Event
    metric_name: str = ""
    value: Any = None
    metadata: Dict = field(default_factory=dict)


class EventObserver(ABC):
    """Abstract observer for evaluation events."""

    @abstractmethod
    def update(self, event_data: EventData):
        """Handle event notification."""
        pass


class LoggingObserver(EventObserver):
    """Observer that logs events."""

    def __init__(self):
        self.logs: List[str] = []

    def update(self, event_data: EventData):
        log_msg = (
            f"[{event_data.event.value}] {event_data.metric_name}: {event_data.value}"
        )
        self.logs.append(log_msg)
        print(log_msg)


class ProgressObserver(EventObserver):
    """Observer that tracks progress."""

    def __init__(self, total_metrics: int):
        self.total = total_metrics
        self.completed = 0

    def update(self, event_data: EventData):
        if event_data.event == Event.METRIC_END:
            self.completed += 1
            progress = (self.completed / self.total) * 100
            print(f"Progress: {progress:.1f}% ({self.completed}/{self.total})")


class CallbackObserver(EventObserver):
    """Observer that calls a callback function."""

    def __init__(self, callback: Callable[[EventData], None]):
        self.callback = callback

    def update(self, event_data: EventData):
        self.callback(event_data)


class MetricSubject:
    """
    Subject that notifies observers of metric events.

    Example:
        subject = MetricSubject()
        subject.attach(LoggingObserver())
        subject.attach(ProgressObserver(total_metrics=5))

        subject.notify(EventData(
            event=Event.METRIC_END,
            metric_name='accuracy',
            value=0.95
        ))
    """

    def __init__(self):
        self._observers: List[EventObserver] = []

    def attach(self, observer: EventObserver):
        """Attach an observer."""
        self._observers.append(observer)

    def detach(self, observer: EventObserver):
        """Detach an observer."""
        self._observers.remove(observer)

    def notify(self, event_data: EventData):
        """Notify all observers."""
        for observer in self._observers:
            observer.update(event_data)


# =============================================================================
# Composite Pattern - Metric Composition
# =============================================================================


class MetricComponent(ABC):
    """Abstract component for composite metrics."""

    @abstractmethod
    def compute(self, y_true: List, y_pred: List) -> Dict[str, float]:
        """Compute the metric(s)."""
        pass

    @property
    @abstractmethod
    def names(self) -> List[str]:
        """Return metric names."""
        pass


class SingleMetric(MetricComponent):
    """Leaf component for a single metric."""

    def __init__(self, name: str, func: Callable, **kwargs):
        self._name = name
        self._func = func
        self._kwargs = kwargs

    def compute(self, y_true: List, y_pred: List) -> Dict[str, float]:
        return {self._name: self._func(y_true, y_pred, **self._kwargs)}

    @property
    def names(self) -> List[str]:
        return [self._name]


class CompositeMetric(MetricComponent):
    """
    Composite for multiple metrics.

    Example:
        # Create individual metrics
        accuracy = SingleMetric('accuracy', accuracy_score)
        f1 = SingleMetric('f1', f1_score, average='macro')

        # Compose them
        classification_metrics = CompositeMetric('classification')
        classification_metrics.add(accuracy)
        classification_metrics.add(f1)

        # Compute all at once
        results = classification_metrics.compute(y_true, y_pred)
        # {'accuracy': 0.95, 'f1': 0.92}
    """

    def __init__(self, name: str):
        self._name = name
        self._children: List[MetricComponent] = []

    def add(self, component: MetricComponent):
        """Add a child component."""
        self._children.append(component)

    def remove(self, component: MetricComponent):
        """Remove a child component."""
        self._children.remove(component)

    def compute(self, y_true: List, y_pred: List) -> Dict[str, float]:
        results = {}
        for child in self._children:
            results.update(child.compute(y_true, y_pred))
        return results

    @property
    def names(self) -> List[str]:
        names = []
        for child in self._children:
            names.extend(child.names)
        return names


# =============================================================================
# Chain of Responsibility Pattern - Validation Chain
# =============================================================================


class ValidationHandler(ABC):
    """Abstract handler for validation chain."""

    def __init__(self):
        self._next: Optional["ValidationHandler"] = None

    def set_next(self, handler: "ValidationHandler") -> "ValidationHandler":
        """Set the next handler in the chain."""
        self._next = handler
        return handler

    def handle(self, y_true: List, y_pred: List) -> bool:
        """Handle validation and pass to next handler."""
        if not self._validate(y_true, y_pred):
            return False
        if self._next:
            return self._next.handle(y_true, y_pred)
        return True

    @abstractmethod
    def _validate(self, y_true: List, y_pred: List) -> bool:
        """Perform validation."""
        pass


class NonEmptyHandler(ValidationHandler):
    """Validates that inputs are non-empty."""

    def _validate(self, y_true: List, y_pred: List) -> bool:
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Inputs cannot be empty")
        return True


class SameLengthHandler(ValidationHandler):
    """Validates that inputs have same length."""

    def _validate(self, y_true: List, y_pred: List) -> bool:
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: {len(y_true)} != {len(y_pred)}")
        return True


class TypeCheckHandler(ValidationHandler):
    """Validates input types."""

    def __init__(self, allowed_types: tuple = (list, tuple)):
        super().__init__()
        self.allowed_types = allowed_types

    def _validate(self, y_true: List, y_pred: List) -> bool:
        if not isinstance(y_true, self.allowed_types):
            raise TypeError(f"y_true must be one of {self.allowed_types}")
        if not isinstance(y_pred, self.allowed_types):
            raise TypeError(f"y_pred must be one of {self.allowed_types}")
        return True


def create_validation_chain() -> ValidationHandler:
    """
    Create a standard validation chain.

    Example:
        validator = create_validation_chain()
        try:
            validator.handle(y_true, y_pred)
        except ValueError as e:
            print(f"Validation failed: {e}")
    """
    type_handler = TypeCheckHandler()
    non_empty_handler = NonEmptyHandler()
    same_length_handler = SameLengthHandler()

    type_handler.set_next(non_empty_handler).set_next(same_length_handler)

    return type_handler


# =============================================================================
# Template Method Pattern - Evaluation Workflow
# =============================================================================


class EvaluationTemplate(ABC):
    """
    Template for evaluation workflow.

    Subclasses implement specific steps while the overall
    algorithm structure remains fixed.
    """

    def evaluate(self, y_true: List, y_pred: List) -> Dict[str, Any]:
        """Execute the evaluation workflow."""
        # Template method
        self.validate(y_true, y_pred)
        y_true, y_pred = self.preprocess(y_true, y_pred)
        results = self.compute_metrics(y_true, y_pred)
        results = self.postprocess(results)
        return results

    def validate(self, y_true: List, y_pred: List):
        """Default validation - can be overridden."""
        if len(y_true) != len(y_pred):
            raise ValueError("Length mismatch")

    def preprocess(self, y_true: List, y_pred: List) -> tuple:
        """Default preprocessing - can be overridden."""
        return y_true, y_pred

    @abstractmethod
    def compute_metrics(self, y_true: List, y_pred: List) -> Dict[str, float]:
        """Must be implemented by subclasses."""
        pass

    def postprocess(self, results: Dict) -> Dict:
        """Default postprocessing - can be overridden."""
        return results


class ClassificationEvaluation(EvaluationTemplate):
    """Classification evaluation workflow."""

    def compute_metrics(self, y_true: List, y_pred: List) -> Dict[str, float]:
        calculator = MetricCalculator()
        results = {}

        calculator.set_strategy(AccuracyStrategy())
        results["accuracy"] = calculator.calculate(y_true, y_pred)

        calculator.set_strategy(PrecisionStrategy())
        results["precision"] = calculator.calculate(y_true, y_pred)

        calculator.set_strategy(RecallStrategy())
        results["recall"] = calculator.calculate(y_true, y_pred)

        calculator.set_strategy(F1Strategy())
        results["f1"] = calculator.calculate(y_true, y_pred)

        return results


# =============================================================================
# Export all
# =============================================================================

__all__ = [
    # Strategy Pattern
    "MetricStrategy",
    "AccuracyStrategy",
    "PrecisionStrategy",
    "RecallStrategy",
    "F1Strategy",
    "MetricCalculator",
    # Factory Pattern
    "MetricType",
    "MetricFactory",
    "metric_factory",
    # Builder Pattern
    "PipelineStep",
    "EvaluationPipelineBuilder",
    "EvaluationPipeline",
    # Observer Pattern
    "Event",
    "EventData",
    "EventObserver",
    "LoggingObserver",
    "ProgressObserver",
    "CallbackObserver",
    "MetricSubject",
    # Composite Pattern
    "MetricComponent",
    "SingleMetric",
    "CompositeMetric",
    # Chain of Responsibility
    "ValidationHandler",
    "NonEmptyHandler",
    "SameLengthHandler",
    "TypeCheckHandler",
    "create_validation_chain",
    # Template Method
    "EvaluationTemplate",
    "ClassificationEvaluation",
]
