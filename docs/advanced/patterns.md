# Design Patterns

PyEval provides several design patterns for flexible and extensible metric computation.

---

## Strategy Pattern

The Strategy pattern allows switching between different metric algorithms at runtime.

### Basic Usage

```python
from pyeval import (
    MetricCalculator, 
    AccuracyStrategy, 
    F1Strategy, 
    PrecisionStrategy,
    RecallStrategy
)

# Create calculator with initial strategy
calculator = MetricCalculator(AccuracyStrategy())

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 0]

# Compute accuracy
accuracy = calculator.calculate(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Switch strategy
calculator.set_strategy(F1Strategy())
f1 = calculator.calculate(y_true, y_pred)
print(f"F1: {f1:.4f}")

# Switch again
calculator.set_strategy(PrecisionStrategy())
precision = calculator.calculate(y_true, y_pred)
print(f"Precision: {precision:.4f}")
```

### Creating Custom Strategies

```python
from pyeval import MetricStrategy, MetricCalculator

class BalancedAccuracyStrategy(MetricStrategy):
    """Custom strategy for balanced accuracy."""
    
    def compute(self, y_true, y_pred, **kwargs):
        # Get unique classes
        classes = set(y_true)
        
        # Compute recall for each class
        recalls = []
        for cls in classes:
            true_positives = sum(1 for t, p in zip(y_true, y_pred) 
                               if t == cls and p == cls)
            actual_positives = sum(1 for t in y_true if t == cls)
            
            if actual_positives > 0:
                recalls.append(true_positives / actual_positives)
        
        # Return average recall
        return sum(recalls) / len(recalls) if recalls else 0.0

# Use custom strategy
calculator = MetricCalculator(BalancedAccuracyStrategy())
result = calculator.calculate(y_true, y_pred)
```

---

## Factory Pattern

The Factory pattern creates metric instances based on type.

### Basic Usage

```python
from pyeval import MetricFactory, MetricType

factory = MetricFactory()

# Create metrics by type
accuracy_metric = factory.create(MetricType.ACCURACY)
f1_metric = factory.create(MetricType.F1)
precision_metric = factory.create(MetricType.PRECISION)

# Compute
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 0]

print(f"Accuracy: {accuracy_metric.compute(y_true, y_pred):.4f}")
print(f"F1: {f1_metric.compute(y_true, y_pred):.4f}")
print(f"Precision: {precision_metric.compute(y_true, y_pred):.4f}")
```

### Available Metric Types

```python
from pyeval import MetricType

# Classification
MetricType.ACCURACY
MetricType.PRECISION
MetricType.RECALL
MetricType.F1
MetricType.ROC_AUC

# Regression
MetricType.MSE
MetricType.RMSE
MetricType.MAE
MetricType.R2

# NLP
MetricType.BLEU
MetricType.ROUGE
MetricType.METEOR
```

### Registering Custom Metrics

```python
from pyeval import MetricFactory, MetricType, Metric

class CustomMetric(Metric):
    def compute(self, y_true, y_pred, **kwargs):
        # Custom computation
        return custom_score

# Register with factory
factory = MetricFactory()
factory.register('custom', CustomMetric)

# Use registered metric
metric = factory.create('custom')
```

---

## Composite Pattern

The Composite pattern groups multiple metrics for batch computation.

### Basic Usage

```python
from pyeval import CompositeMetric, SingleMetric
from pyeval import accuracy_score, precision_score, recall_score, f1_score

# Create composite
classification_metrics = CompositeMetric('classification')

# Add individual metrics
classification_metrics.add(SingleMetric('accuracy', accuracy_score))
classification_metrics.add(SingleMetric('precision', precision_score))
classification_metrics.add(SingleMetric('recall', recall_score))
classification_metrics.add(SingleMetric('f1', f1_score))

# Compute all at once
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 0]

results = classification_metrics.compute(y_true, y_pred)
print(results)
# {'accuracy': 0.8, 'precision': 0.667, 'recall': 0.667, 'f1': 0.667}
```

### Nested Composites

```python
from pyeval import CompositeMetric, SingleMetric

# Create nested structure
all_metrics = CompositeMetric('all')

# Classification group
classification = CompositeMetric('classification')
classification.add(SingleMetric('accuracy', accuracy_score))
classification.add(SingleMetric('f1', f1_score))

# Regression group
regression = CompositeMetric('regression')
regression.add(SingleMetric('mse', mean_squared_error))
regression.add(SingleMetric('r2', r2_score))

# Add groups to parent
all_metrics.add(classification)
all_metrics.add(regression)

# Compute returns nested results
results = all_metrics.compute(y_true, y_pred)
```

---

## Builder Pattern

The Builder pattern constructs complex metric configurations step by step.

### Basic Usage

```python
from pyeval import MetricBuilder

# Build metric configuration
config = (
    MetricBuilder()
    .with_metric('accuracy')
    .with_metric('f1', average='weighted')
    .with_metric('precision', average='macro')
    .with_threshold(0.5)
    .with_callbacks([ProgressCallback()])
    .build()
)

# Execute
results = config.compute(y_true, y_pred)
```

### Fluent Configuration

```python
from pyeval import EvaluationBuilder

evaluation = (
    EvaluationBuilder('classification')
    .add_classification_metrics()
    .add_roc_auc()
    .with_confidence_intervals(n_bootstrap=1000)
    .with_statistical_tests()
    .save_to('results.json')
    .build()
)

report = evaluation.run(y_true, y_pred)
```

---

## Observer Pattern

The Observer pattern allows monitoring metric computation events.

### Basic Usage

```python
from pyeval import MetricObserver, MetricSubject

class LoggingObserver(MetricObserver):
    def on_compute_start(self, metric_name, **kwargs):
        print(f"Starting: {metric_name}")
    
    def on_compute_end(self, metric_name, result, **kwargs):
        print(f"Completed: {metric_name} = {result}")
    
    def on_error(self, metric_name, error, **kwargs):
        print(f"Error in {metric_name}: {error}")

# Create subject and attach observer
subject = MetricSubject()
subject.attach(LoggingObserver())

# Compute with observation
result = subject.compute_with_observation('accuracy', y_true, y_pred)
```

### Multiple Observers

```python
class TimingObserver(MetricObserver):
    def __init__(self):
        self.start_times = {}
    
    def on_compute_start(self, metric_name, **kwargs):
        self.start_times[metric_name] = time.time()
    
    def on_compute_end(self, metric_name, result, **kwargs):
        elapsed = time.time() - self.start_times[metric_name]
        print(f"{metric_name} took {elapsed:.4f}s")

class ThresholdObserver(MetricObserver):
    def __init__(self, thresholds):
        self.thresholds = thresholds
    
    def on_compute_end(self, metric_name, result, **kwargs):
        if metric_name in self.thresholds:
            threshold = self.thresholds[metric_name]
            if result < threshold:
                print(f"WARNING: {metric_name} ({result}) below threshold ({threshold})")

# Attach multiple observers
subject.attach(TimingObserver())
subject.attach(ThresholdObserver({'accuracy': 0.9, 'f1': 0.85}))
```

---

## Chain of Responsibility

Process metrics through a chain of handlers.

```python
from pyeval import MetricHandler, ValidationHandler, ComputationHandler, LoggingHandler

# Create chain
chain = (
    ValidationHandler()
    .set_next(ComputationHandler())
    .set_next(LoggingHandler())
)

# Process through chain
result = chain.handle({
    'y_true': y_true,
    'y_pred': y_pred,
    'metric': 'accuracy'
})
```

---

## Best Practices

### Choosing the Right Pattern

| Use Case | Recommended Pattern |
|----------|---------------------|
| Switch algorithms at runtime | Strategy |
| Create metrics by type | Factory |
| Group related metrics | Composite |
| Complex configuration | Builder |
| Event monitoring | Observer |
| Sequential processing | Chain of Responsibility |

### Combining Patterns

```python
from pyeval import (
    MetricFactory, CompositeMetric, MetricCalculator,
    AccuracyStrategy, MetricSubject, LoggingObserver
)

# Factory creates metrics
factory = MetricFactory()

# Composite groups them
composite = CompositeMetric('evaluation')
composite.add(factory.create(MetricType.ACCURACY))
composite.add(factory.create(MetricType.F1))

# Observer monitors execution
subject = MetricSubject()
subject.attach(LoggingObserver())

# Strategy allows switching
calculator = MetricCalculator(composite)

# Execute with full infrastructure
results = subject.compute_with_observation('evaluation', y_true, y_pred)
```
