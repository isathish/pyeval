# Decorators

PyEval provides useful decorators for metric functions.

---

## Performance Decorators

### @timed

Measure and report function execution time.

```python
from pyeval import timed

@timed
def compute_metrics(y_true, y_pred):
    """Computes multiple metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

result = compute_metrics(y_true, y_pred)
# Output: compute_metrics took 0.0023s

# Access timing info
print(result)  # Normal return value
```

### @timed with custom output

```python
from pyeval import timed

@timed(output='return')  # Include timing in return
def compute_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

result, elapsed = compute_metrics(y_true, y_pred)
print(f"Accuracy: {result}, Time: {elapsed:.4f}s")
```

---

### @memoize

Cache function results for identical inputs.

```python
from pyeval import memoize

@memoize
def expensive_metric(y_true, y_pred):
    """Expensive computation that benefits from caching."""
    # Simulated expensive operation
    result = complex_computation(y_true, y_pred)
    return result

# First call - computes
result1 = expensive_metric([1,0,1], [1,0,0])  # Takes time

# Second call with same args - returns cached
result2 = expensive_metric([1,0,1], [1,0,0])  # Instant

# Different args - computes again
result3 = expensive_metric([1,1,1], [1,0,1])  # Takes time
```

### @memoize with max size

```python
from pyeval import memoize

@memoize(maxsize=100)  # Keep at most 100 cached results
def metric_function(y_true, y_pred):
    return compute(y_true, y_pred)
```

---

### @lru_cache_decorator

LRU (Least Recently Used) caching.

```python
from pyeval import lru_cache_decorator

@lru_cache_decorator(maxsize=128)
def cached_metric(data_tuple):  # Note: args must be hashable
    return expensive_computation(data_tuple)
```

---

## Validation Decorators

### @require_same_length

Validate that input sequences have the same length.

```python
from pyeval import require_same_length

@require_same_length
def custom_accuracy(y_true, y_pred):
    """Requires y_true and y_pred to have same length."""
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

# Works fine
result = custom_accuracy([1, 0, 1], [1, 0, 0])

# Raises ValueError
result = custom_accuracy([1, 0, 1], [1, 0])  
# ValueError: y_true and y_pred must have the same length
```

### @require_non_empty

Validate that inputs are non-empty.

```python
from pyeval import require_non_empty

@require_non_empty
def mean_metric(values):
    """Requires non-empty input."""
    return sum(values) / len(values)

# Works
result = mean_metric([1, 2, 3])

# Raises ValueError
result = mean_metric([])  # ValueError: Input cannot be empty
```

### @validate_range

Validate that values are within a range.

```python
from pyeval import validate_range

@validate_range(min_val=0, max_val=1)
def process_probabilities(probs):
    """Requires all values between 0 and 1."""
    return probs

# Works
result = process_probabilities([0.1, 0.5, 0.9])

# Raises ValueError  
result = process_probabilities([0.1, 1.5, 0.9])
# ValueError: Values must be between 0 and 1
```

---

## Error Handling Decorators

### @retry

Retry function on failure.

```python
from pyeval import retry

@retry(max_attempts=3, delay=0.1)
def unreliable_metric(data):
    """May fail occasionally."""
    result = external_service_call(data)
    return result

# Will retry up to 3 times with 0.1s delay between attempts
result = unreliable_metric(data)
```

### @retry with exponential backoff

```python
from pyeval import retry

@retry(max_attempts=5, delay=0.1, backoff=2)
def api_metric(data):
    """Calls external API that may be rate-limited."""
    return api_call(data)

# Delays: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
```

### @retry with specific exceptions

```python
from pyeval import retry

@retry(max_attempts=3, exceptions=(TimeoutError, ConnectionError))
def network_metric(data):
    """Only retry on network errors."""
    return fetch_from_server(data)
```

---

### @fallback

Provide fallback value on error.

```python
from pyeval import fallback

@fallback(default_value=0.0)
def safe_metric(y_true, y_pred):
    """Returns 0.0 if computation fails."""
    return risky_computation(y_true, y_pred)

result = safe_metric(y_true, y_pred)  # Returns 0.0 on error
```

### @fallback with handler

```python
from pyeval import fallback

def error_handler(func, error, *args, **kwargs):
    print(f"Error in {func.__name__}: {error}")
    return -1.0

@fallback(handler=error_handler)
def metric_with_handler(y_true, y_pred):
    return computation(y_true, y_pred)
```

---

## Logging Decorators

### @logged

Log function calls and results.

```python
from pyeval import logged

@logged
def tracked_metric(y_true, y_pred):
    """Logs all calls."""
    return accuracy_score(y_true, y_pred)

result = tracked_metric([1,0,1], [1,0,0])
# Log: tracked_metric called with args=([1,0,1], [1,0,0])
# Log: tracked_metric returned 0.667
```

### @logged with custom logger

```python
import logging
from pyeval import logged

logger = logging.getLogger('metrics')

@logged(logger=logger, level=logging.DEBUG)
def debug_metric(y_true, y_pred):
    return f1_score(y_true, y_pred)
```

---

### @deprecated

Mark functions as deprecated.

```python
from pyeval import deprecated

@deprecated(message="Use accuracy_score instead", version="2.0")
def old_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

result = old_accuracy([1,0], [1,0])
# Warning: old_accuracy is deprecated since version 2.0. Use accuracy_score instead
```

---

## Type Decorators

### @enforce_types

Enforce argument types at runtime.

```python
from pyeval import enforce_types

@enforce_types
def typed_metric(y_true: list, y_pred: list, threshold: float = 0.5) -> float:
    """Enforces type annotations."""
    return compute(y_true, y_pred, threshold)

# Works
result = typed_metric([1,0], [1,0], 0.5)

# Raises TypeError
result = typed_metric("not a list", [1,0])
# TypeError: y_true must be list, got str
```

---

## Combining Decorators

Decorators can be stacked (applied bottom-up):

```python
from pyeval import timed, memoize, require_same_length, logged

@logged
@timed
@memoize
@require_same_length
def comprehensive_metric(y_true, y_pred):
    """
    1. Validates lengths (require_same_length)
    2. Caches results (memoize)
    3. Times execution (timed)
    4. Logs call (logged)
    """
    return expensive_computation(y_true, y_pred)
```

Order matters! Inner decorators execute first:
1. `require_same_length` - validates
2. `memoize` - checks cache
3. `timed` - measures time
4. `logged` - logs

---

## Creating Custom Decorators

```python
from functools import wraps

def threshold_check(min_score=0.0):
    """Custom decorator to validate metric scores."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if result < min_score:
                raise ValueError(f"Score {result} below minimum {min_score}")
            return result
        return wrapper
    return decorator

@threshold_check(min_score=0.5)
def quality_metric(y_true, y_pred):
    return f1_score(y_true, y_pred)
```

---

## Best Practices

1. **Order decorators carefully** - validation before caching before timing
2. **Use `@memoize` for expensive computations** - significant speedup
3. **Use `@retry` for external calls** - improved reliability
4. **Use `@require_same_length`** - catches common errors early
5. **Use `@timed` during development** - identify bottlenecks
6. **Use `@logged` sparingly** - can produce verbose output
