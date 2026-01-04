# Functional Utilities

PyEval provides functional programming utilities for composing evaluation functions.

---

## Function Composition

### compose

Compose multiple functions into a single function.

```python
from pyeval import compose

# Compose functions: right to left execution
pipeline = compose(
    lambda x: x * 2,      # Third: multiply by 2
    lambda x: x + 10,     # Second: add 10
    lambda x: x ** 2      # First: square
)

result = pipeline(5)  # 5² = 25 → 25 + 10 = 35 → 35 * 2 = 70
```

### pipe

Pipe functions in left to right order.

```python
from pyeval import pipe

# Pipe functions: left to right execution
pipeline = pipe(
    lambda x: x ** 2,     # First: square
    lambda x: x + 10,     # Second: add 10
    lambda x: x * 2       # Third: multiply by 2
)

result = pipeline(5)  # 5² = 25 → 25 + 10 = 35 → 35 * 2 = 70
```

---

## Currying and Partial Application

### curry

Convert a function to curried form.

```python
from pyeval import curry

@curry
def weighted_sum(a, b, weight_a, weight_b):
    return a * weight_a + b * weight_b

# Partial application
with_weights = weighted_sum(weight_a=0.3, weight_b=0.7)
result = with_weights(10, 20)  # 10 * 0.3 + 20 * 0.7 = 17.0

# Step by step
f = weighted_sum(10)
g = f(20)
h = g(0.3)
result = h(0.7)  # 17.0
```

### partial

Create a partial function with pre-filled arguments.

```python
from pyeval import partial

def f1_score(y_true, y_pred, average='binary', zero_division=0):
    # ... implementation
    pass

# Create specialized versions
binary_f1 = partial(f1_score, average='binary')
macro_f1 = partial(f1_score, average='macro')
weighted_f1 = partial(f1_score, average='weighted', zero_division=1)

# Use them
result = binary_f1(y_true, y_pred)
result = macro_f1(y_true, y_pred)
```

---

## Higher-Order Functions

### map_over

Apply a function over collections.

```python
from pyeval import map_over, accuracy_score

# Apply metric to multiple datasets
datasets = [
    ([1, 0, 1], [1, 0, 1]),
    ([1, 1, 0], [1, 0, 0]),
    ([0, 0, 1], [0, 1, 1]),
]

results = map_over(
    lambda data: accuracy_score(data[0], data[1]),
    datasets
)
# [1.0, 0.333, 0.667]
```

### filter_results

Filter results based on conditions.

```python
from pyeval import filter_results

results = {
    'accuracy': 0.85,
    'precision': 0.72,
    'recall': 0.91,
    'f1': 0.80,
    'specificity': 0.65
}

# Filter to only high-performing metrics
high_performers = filter_results(lambda v: v >= 0.8, results)
# {'accuracy': 0.85, 'recall': 0.91, 'f1': 0.80}
```

### reduce_results

Reduce results to a single value.

```python
from pyeval import reduce_results

results = {
    'accuracy': 0.85,
    'precision': 0.72,
    'recall': 0.91,
    'f1': 0.80
}

# Compute average
average = reduce_results(
    lambda acc, v: acc + v,
    results,
    initial=0
) / len(results)
# 0.82
```

---

## Function Factories

### create_metric

Create a metric function with configuration.

```python
from pyeval import create_metric

# Create configured accuracy
accuracy = create_metric(
    'accuracy',
    normalize=True,
    sample_weight=None
)

result = accuracy(y_true, y_pred)
```

### create_threshold_checker

Create a threshold checking function.

```python
from pyeval import create_threshold_checker

# Create checker with thresholds
check_quality = create_threshold_checker({
    'accuracy': {'min': 0.8, 'max': 1.0},
    'precision': {'min': 0.75},
    'recall': {'min': 0.75},
    'f1': {'min': 0.8}
})

results = {
    'accuracy': 0.85,
    'precision': 0.72,  # Below threshold!
    'recall': 0.91,
    'f1': 0.80
}

report = check_quality(results)
# {
#     'passed': False,
#     'failures': ['precision: 0.72 < 0.75'],
#     'warnings': []
# }
```

### create_aggregator

Create result aggregation functions.

```python
from pyeval import create_aggregator

# Weighted aggregator
weighted_avg = create_aggregator(
    'weighted_average',
    weights={
        'accuracy': 0.1,
        'precision': 0.2,
        'recall': 0.2,
        'f1': 0.5
    }
)

results = {
    'accuracy': 0.85,
    'precision': 0.72,
    'recall': 0.91,
    'f1': 0.80
}

score = weighted_avg(results)
# 0.85 * 0.1 + 0.72 * 0.2 + 0.91 * 0.2 + 0.80 * 0.5 = 0.811
```

---

## Memoization

### memoize

Cache function results for repeated calls.

```python
from pyeval import memoize

@memoize
def expensive_metric(y_true, y_pred):
    """Compute an expensive metric."""
    # ... expensive computation
    return result

# First call computes
result1 = expensive_metric(y_true, y_pred)  # Slow

# Second call uses cache
result2 = expensive_metric(y_true, y_pred)  # Fast!
```

### memoize_with_ttl

Cache with time-to-live expiration.

```python
from pyeval import memoize_with_ttl

@memoize_with_ttl(seconds=60)
def api_metric(y_true, y_pred):
    """Metric that calls an external API."""
    # ... API call
    return result

# Results cached for 60 seconds
```

---

## Error Handling

### safe_call

Safely call functions with error handling.

```python
from pyeval import safe_call

result = safe_call(
    risky_metric,
    y_true, y_pred,
    default=0.0,
    on_error=lambda e: print(f"Error: {e}")
)
# Returns 0.0 if error, otherwise metric value
```

### try_metrics

Try multiple metrics, return first successful.

```python
from pyeval import try_metrics

result = try_metrics(
    [metric_1, metric_2, metric_3],
    y_true, y_pred,
    default=None
)
# Returns result of first metric that succeeds
```

### retry

Retry function calls with backoff.

```python
from pyeval import retry

@retry(max_attempts=3, delay=1.0, backoff=2.0)
def flaky_metric(y_true, y_pred):
    """A metric that might fail temporarily."""
    # ... implementation that might fail
    return result

# Will retry up to 3 times with exponential backoff
```

---

## Monads

### Maybe

Handle optional values gracefully.

```python
from pyeval import Maybe

def safe_divide(a, b):
    if b == 0:
        return Maybe.nothing()
    return Maybe.just(a / b)

result = (
    Maybe.just(10)
    .bind(lambda x: safe_divide(x, 2))
    .bind(lambda x: safe_divide(x, 0))  # Returns Nothing
    .map(lambda x: x * 100)
    .get_or_else(0)  # Returns 0 because Nothing
)
```

### Either

Handle success/failure with context.

```python
from pyeval import Either, Left, Right

def compute_metric(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return Left("Length mismatch")
    if len(y_true) == 0:
        return Left("Empty inputs")
    return Right(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true))

result = (
    compute_metric(y_true, y_pred)
    .map(lambda x: round(x, 4))
    .map(lambda x: f"Accuracy: {x}")
    .fold(
        on_left=lambda err: f"Error: {err}",
        on_right=lambda val: val
    )
)
```

### Result

Rust-style Result type.

```python
from pyeval import Result, Ok, Err

def validate_inputs(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return Err(ValueError("Length mismatch"))
    return Ok((y_true, y_pred))

def compute_accuracy(inputs):
    y_true, y_pred = inputs
    return Ok(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true))

result = (
    validate_inputs(y_true, y_pred)
    .and_then(compute_accuracy)
    .map(lambda x: round(x, 4))
    .unwrap_or(0.0)
)
```

---

## Lazy Evaluation

### lazy

Create lazily evaluated values.

```python
from pyeval import lazy

# Metric not computed until needed
lazy_accuracy = lazy(accuracy_score, y_true, y_pred)

# ... later ...
result = lazy_accuracy.value  # Computed now
result = lazy_accuracy.value  # Cached
```

### LazyPipeline

Build lazily evaluated pipelines.

```python
from pyeval import LazyPipeline

# Build pipeline without executing
pipeline = (
    LazyPipeline()
    .add_metric('accuracy', accuracy_score)
    .add_metric('f1', f1_score)
    .filter(lambda k, v: v > 0.7)
    .map_values(lambda v: round(v, 4))
)

# Execute when needed
results = pipeline.evaluate(y_true, y_pred)
```

---

## Complete Example

```python
from pyeval import (
    pipe, compose, curry, partial, memoize,
    create_metric, create_threshold_checker, create_aggregator,
    Either, Right, Left,
    accuracy_score, precision_score, recall_score, f1_score
)

# Create curried metric functions
@curry
def compute_with_config(config, y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if config.get('round', False):
        metrics = {k: round(v, config.get('decimals', 4)) 
                   for k, v in metrics.items()}
    
    return metrics

# Create configured evaluator
my_evaluator = compute_with_config({
    'round': True,
    'decimals': 3
})

# Create threshold checker
check_thresholds = create_threshold_checker({
    'accuracy': {'min': 0.8},
    'f1': {'min': 0.75}
})

# Create aggregator
weighted_score = create_aggregator('weighted_average', weights={
    'accuracy': 0.1,
    'precision': 0.2,
    'recall': 0.2,
    'f1': 0.5
})

# Build evaluation pipeline
def validate_inputs(data):
    y_true, y_pred = data
    if len(y_true) != len(y_pred):
        return Left("Length mismatch")
    if len(y_true) == 0:
        return Left("Empty inputs")
    return Right(data)

def compute_metrics(data):
    y_true, y_pred = data
    return Right(my_evaluator(y_true, y_pred))

def add_aggregate(metrics):
    metrics['weighted_score'] = weighted_score(
        {k: v for k, v in metrics.items() if k != 'weighted_score'}
    )
    return Right(metrics)

def check_quality(metrics):
    report = check_thresholds(metrics)
    if not report['passed']:
        return Left(f"Quality check failed: {report['failures']}")
    return Right(metrics)

# Run the pipeline
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

result = (
    validate_inputs((y_true, y_pred))
    .and_then(compute_metrics)
    .and_then(add_aggregate)
    .and_then(check_quality)
    .fold(
        on_left=lambda err: {'status': 'error', 'message': err},
        on_right=lambda metrics: {'status': 'success', 'metrics': metrics}
    )
)

print(result)
# {
#     'status': 'success',
#     'metrics': {
#         'accuracy': 0.75,
#         'precision': 0.667,
#         'recall': 0.8,
#         'f1': 0.727,
#         'weighted_score': 0.738
#     }
# }
```

---

## Best Practices

1. **Use composition** - Build complex functions from simple ones
2. **Prefer immutability** - Don't mutate input data
3. **Use Either/Result** - Handle errors explicitly
4. **Memoize expensive computations** - Cache repeated calculations
5. **Be explicit about side effects** - Use monads for clarity
