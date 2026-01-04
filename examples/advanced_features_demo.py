#!/usr/bin/env python3
"""
Advanced Features Demo for PyEval
=================================

This example demonstrates all the advanced features added to pyeval:
- Decorators
- Design Patterns
- Validators
- Callbacks
- Pipelines
- Functional Utilities
- Aggregators
"""

from pyeval import (
    # Core metrics
    accuracy_score, precision_score, recall_score, f1_score,
    
    # Decorators
    timed, memoize, require_same_length, retry, logged, deprecated,
    compose, pipe, MetricRegistry,
    
    # Design Patterns
    MetricCalculator, AccuracyStrategy, F1Strategy, PrecisionStrategy,
    MetricFactory, MetricType,
    CompositeMetric, SingleMetric,
    EvaluationPipelineBuilder,
    Event, EventData, EventObserver,
    
    # Validators
    TypeValidator, ListValidator, NumericValidator, SchemaValidator, FieldSchema,
    AllOf, AnyOf, validate_predictions, validate_probabilities,
    
    # Callbacks
    CallbackManager, LoggingCallback, ProgressCallback, 
    ThresholdCallback, HistoryCallback, EarlyStoppingCallback,
    
    # Pipeline
    Pipeline, create_classification_pipeline,
    
    # Functional
    Result, Option, curry,
    combine_metrics, threshold_metric, average_metric,
    
    # Aggregators
    MetricAggregator, CrossValidationAggregator, EnsembleAggregator,
    MeanAggregator, MedianAggregator
)


def divider(title: str):
    """Print a section divider."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# Sample data
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]


# =============================================================================
# 1. DECORATORS
# =============================================================================
divider("1. DECORATORS")

# @timed decorator
@timed
def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

print("\n@timed decorator:")
acc = compute_accuracy(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")

# @memoize decorator
@memoize
def expensive_f1(y_true, y_pred):
    print("  Computing F1 (first call only)...")
    return f1_score(y_true, y_pred)

print("\n@memoize decorator:")
f1_1 = expensive_f1(tuple(y_true), tuple(y_pred))
f1_2 = expensive_f1(tuple(y_true), tuple(y_pred))  # Cached
print(f"F1 Score: {f1_1:.4f}")

# @require_same_length decorator
@require_same_length
def safe_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

print("\n@require_same_length decorator:")
try:
    safe_accuracy([1, 0, 1], [1, 0])  # Will raise ValueError
except ValueError as e:
    print(f"Caught expected error: {e}")

# compose function
print("\ncompose() function:")
double = lambda x: x * 2
add_one = lambda x: x + 1
to_str = str
process = compose(double, add_one, to_str)  # double(5)=10 -> add_one(10)=11 -> str(11)="11"
result = process(5)
print(f"compose(double, add_one, str)(5) = {result}")

# MetricRegistry - using decorator pattern
print("\nMetricRegistry:")
registry = MetricRegistry()

@registry.register('custom_accuracy')
def my_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

@registry.register('custom_f1')
def my_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

print(f"Registered metrics: {registry.list_metrics()}")
result = registry.compute('custom_accuracy', y_true, y_pred)
print(f"Custom accuracy: {result:.4f}")


# =============================================================================
# 2. DESIGN PATTERNS
# =============================================================================
divider("2. DESIGN PATTERNS")

# Strategy Pattern
print("\nStrategy Pattern:")
calculator = MetricCalculator(AccuracyStrategy())
print(f"Using Accuracy Strategy: {calculator.calculate(y_true, y_pred):.4f}")

calculator.set_strategy(F1Strategy())
print(f"Using F1 Strategy: {calculator.calculate(y_true, y_pred):.4f}")

calculator.set_strategy(PrecisionStrategy())
print(f"Using Precision Strategy: {calculator.calculate(y_true, y_pred):.4f}")

# Factory Pattern
print("\nFactory Pattern:")
factory = MetricFactory()
for metric_type in [MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, MetricType.F1]:
    metric = factory.create(metric_type)
    score = metric.compute(y_true, y_pred)
    print(f"  {metric_type.value}: {score:.4f}")

# Composite Pattern
print("\nComposite Pattern:")
classification_suite = CompositeMetric("classification")
classification_suite.add(SingleMetric("accuracy", accuracy_score))
classification_suite.add(SingleMetric("precision", precision_score))
classification_suite.add(SingleMetric("recall", recall_score))
classification_suite.add(SingleMetric("f1", f1_score))

results = classification_suite.compute(y_true, y_pred)
for name, value in results.items():
    print(f"  {name}: {value:.4f}")

# Observer Pattern
print("\nObserver Pattern:")
from pyeval import MetricSubject, LoggingObserver, ProgressObserver, Event as EventEnum, EventData

subject = MetricSubject()
subject.attach(LoggingObserver())

# Notify of metric computation
subject.notify(EventData(
    event=EventEnum.METRIC_END,
    metric_name='accuracy',
    value=0.7
))


# =============================================================================
# 3. VALIDATORS
# =============================================================================
divider("3. VALIDATORS")

# Type validation
print("\nType Validators:")
int_validator = TypeValidator(int)
print(f"Is 42 an int? {int_validator.validate(42).is_valid}")
print(f"Is 'hello' an int? {int_validator.validate('hello').is_valid}")

# List validation
print("\nList Validators:")
list_validator = ListValidator(element_type=int, min_length=1, max_length=10)
result = list_validator.validate([1, 2, 3, 4, 5])
print(f"[1,2,3,4,5] valid? {result.is_valid}")

result = list_validator.validate([])
print(f"[] valid? {result.is_valid}, error: {result.errors}")

# Numeric validation
print("\nNumeric Validators:")
prob_validator = NumericValidator(min_value=0, max_value=1)
print(f"0.5 valid probability? {prob_validator.validate(0.5).is_valid}")
print(f"1.5 valid probability? {prob_validator.validate(1.5).is_valid}")

# Schema validation
print("\nSchema Validation:")
schema = SchemaValidator([
    FieldSchema('y_true', ListValidator(min_length=1)),
    FieldSchema('y_pred', ListValidator(min_length=1)),
    FieldSchema('threshold', NumericValidator(0, 1), required=False, default=0.5)
])

data = {'y_true': [1, 0, 1], 'y_pred': [1, 1, 0]}
result = schema.validate(data)
print(f"Schema valid: {result.is_valid}")

# Probability validation
print("\nProbability Validation:")
try:
    validate_probabilities([0.2, 0.3, 0.5], require_sum_one=True)
    print("Probabilities [0.2, 0.3, 0.5] are valid")
except ValueError as e:
    print(f"Error: {e}")


# =============================================================================
# 4. CALLBACKS
# =============================================================================
divider("4. CALLBACKS")

# Import callback context
from pyeval import CallbackContext, CallbackEvent

# History callback with proper context
print("\nHistory Callback:")
history = HistoryCallback()

# Create context for evaluation
ctx = CallbackContext(
    event=CallbackEvent.ON_EVALUATION_START,
    metric_name='',
    metric_value=0
)
history.on_evaluation_start(ctx)

# Log metrics
for metric, value in [('accuracy', 0.85), ('f1', 0.82), ('accuracy', 0.87)]:
    ctx = CallbackContext(
        event=CallbackEvent.ON_METRIC_END,
        metric_name=metric,
        metric_value=value
    )
    history.on_metric_end(ctx)

ctx = CallbackContext(event=CallbackEvent.ON_EVALUATION_END, metric_name='', metric_value=0)
history.on_evaluation_end(ctx)

print(f"History entries: {len(history.history)}")

# Threshold callback
print("\nThreshold Callback:")
threshold = ThresholdCallback({
    'accuracy': {'min': 0.8},
    'f1': {'min': 0.75, 'max': 0.95}
})
ctx = CallbackContext(event=CallbackEvent.ON_METRIC_END, metric_name='accuracy', metric_value=0.75)
threshold.on_metric_end(ctx)  # Below threshold
ctx = CallbackContext(event=CallbackEvent.ON_METRIC_END, metric_name='f1', metric_value=0.8)
threshold.on_metric_end(ctx)  # Within range
print(f"Breaches: {threshold.breaches}")

# Callback Manager
print("\nCallback Manager:")
manager = CallbackManager()
manager.add_callback(LoggingCallback())
manager.add_callback(ProgressCallback(total_metrics=3))

# Simulate evaluation
ctx = CallbackContext(event=CallbackEvent.ON_EVALUATION_START, metric_name='', metric_value=0)
manager.dispatch(ctx)
for metric in ['accuracy', 'precision', 'recall']:
    ctx = CallbackContext(event=CallbackEvent.ON_METRIC_START, metric_name=metric, metric_value=0)
    manager.dispatch(ctx)
    ctx = CallbackContext(event=CallbackEvent.ON_METRIC_END, metric_name=metric, metric_value=0.85)
    manager.dispatch(ctx)
ctx = CallbackContext(event=CallbackEvent.ON_EVALUATION_END, metric_name='', metric_value=0)
manager.dispatch(ctx)


# =============================================================================
# 5. PIPELINES
# =============================================================================
divider("5. PIPELINES")

# Custom pipeline
print("\nCustom Pipeline:")
pipeline = (
    Pipeline()
    .validate(lambda x: len(x[0]) == len(x[1]), "Length mismatch")
    .add_metric('accuracy', accuracy_score)
    .add_metric('precision', precision_score)
    .add_metric('recall', recall_score)
    .add_metric('f1', f1_score)
    .aggregate('mean', lambda r: sum(r.values()) / len(r))
)

# Run with return_details=True to get PipelineResult
result = pipeline.run(y_true, y_pred, return_details=True)
print(f"Metrics computed: {list(result.metrics.keys())}")
print(f"Aggregations: {result.aggregated}")
print(f"Duration: {result.duration:.4f}s")

# Or run for simple dict result
simple_result = pipeline.run(y_true, y_pred)
print(f"Simple result: {simple_result}")

# Pre-built classification pipeline
print("\nClassification Pipeline (preset):")
cls_pipeline = create_classification_pipeline()
result = cls_pipeline.run(y_true, y_pred)
print(f"Available metrics: {list(result.keys())}")


# =============================================================================
# 6. FUNCTIONAL UTILITIES
# =============================================================================
divider("6. FUNCTIONAL UTILITIES")

# Result monad
print("\nResult Monad:")
def safe_divide(a, b) -> Result:
    if b == 0:
        return Result.failure("Division by zero")
    return Result.success(a / b)

result = safe_divide(10, 2)
print(f"10/2 = {result.value}")

result = safe_divide(10, 0)
print(f"10/0 = {result.error}")

# Chain operations
result = (
    Result.success(10)
    .map(lambda x: x * 2)
    .map(lambda x: x + 5)
)
print(f"Result chain (10 * 2 + 5): {result.value}")

# Option monad
print("\nOption Monad:")
opt = Option.from_nullable(0.95)
doubled = opt.map(lambda x: x * 100)
print(f"Option(0.95).map(*100) = {doubled.get_or_else(0)}")

opt = Option.from_nullable(None)
print(f"Option(None).get_or_else(0) = {opt.get_or_else(0)}")

# Currying
print("\nCurrying:")
@curry
def weighted_sum(a, b, c, weights):
    return a * weights[0] + b * weights[1] + c * weights[2]

partial_sum = weighted_sum(1)(2)(3)
final = partial_sum([0.2, 0.3, 0.5])
print(f"weighted_sum(1)(2)(3)([0.2, 0.3, 0.5]) = {final}")

# Combine metrics
print("\nCombine Metrics:")
combined = combine_metrics(accuracy_score, precision_score, recall_score)
results = combined(y_true, y_pred)
for name, value in results.items():
    print(f"  {name}: {value:.4f}")

# Threshold metric
print("\nThreshold Metric:")
high_accuracy = threshold_metric(accuracy_score, 0.65)
is_high = high_accuracy(y_true, y_pred)
print(f"Accuracy >= 0.65? {is_high}")


# =============================================================================
# 7. AGGREGATORS
# =============================================================================
divider("7. AGGREGATORS")

# Metric Aggregator
print("\nMetric Aggregator:")
aggregator = MetricAggregator()

# Simulate 3 evaluation runs
for i in range(3):
    aggregator.add_results({
        'accuracy': 0.85 + i * 0.02,
        'f1': 0.82 + i * 0.01
    })

stats = aggregator.get_statistics()
print("Statistics:")
for metric, metric_stats in stats.items():
    print(f"  {metric}: mean={metric_stats['mean']:.4f}, std={metric_stats['std']:.4f}")

# Cross-validation Aggregator
print("\nCross-Validation Aggregator:")
cv = CrossValidationAggregator(n_folds=5)

# Simulate 5-fold cross-validation
for fold in range(5):
    cv.add_fold_result(fold, {
        'accuracy': 0.82 + fold * 0.02,
        'f1': 0.80 + fold * 0.015
    })

cv_results = cv.get_summary()
print(f"CV Results:")
for metric, agg in cv_results.items():
    print(f"  {metric}: {agg['mean']:.4f} (±{agg['std']:.4f})")

# Print summary table
print("\n" + cv.summary_table())

# Ensemble Aggregator
print("\nEnsemble Aggregator:")
ensemble = EnsembleAggregator(strategy='majority_vote')

# Simulate 3 model predictions
ensemble.add_predictions('model_1', [1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
ensemble.add_predictions('model_2', [1, 1, 1, 0, 0, 1, 0, 1, 1, 0])
ensemble.add_predictions('model_3', [1, 0, 1, 1, 0, 0, 0, 0, 1, 1])

final_preds = ensemble.aggregate()
print(f"Ensemble predictions: {final_preds}")
print(f"Number of models: {len(ensemble._predictions)}")


# =============================================================================
# SUMMARY
# =============================================================================
divider("SUMMARY")
print("""
PyEval Advanced Features demonstrated:

✅ Decorators: @timed, @memoize, @require_same_length, compose, MetricRegistry
✅ Design Patterns: Strategy, Factory, Composite, Observer
✅ Validators: Type, List, Numeric, Schema, Probability
✅ Callbacks: History, Threshold, Progress, Manager
✅ Pipelines: Fluent API, Presets, Validation, Aggregation
✅ Functional: Result/Option monads, curry, combine_metrics
✅ Aggregators: MetricAggregator, CrossValidation, Ensemble

All features work with ZERO external dependencies! 🚀
""")
