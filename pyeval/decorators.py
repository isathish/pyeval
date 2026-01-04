"""
PyEval Decorators - Custom Decorators for Evaluation
=====================================================

Provides decorators for:
- Timing and profiling metrics
- Caching results
- Validation
- Retry logic
- Logging
"""

import functools
import time
from typing import Callable, Any, Dict, Optional, List, TypeVar
from dataclasses import dataclass
from datetime import datetime

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Timing Decorators
# =============================================================================


def timed(func: F) -> F:
    """
    Decorator to measure execution time of a function.

    Example:
        @timed
        def compute_metrics(data):
            ...

        result = compute_metrics(data)
        # Prints: compute_metrics took 0.1234 seconds
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"{func.__name__} took {elapsed:.4f} seconds")
        return result

    return wrapper


def timed_result(func: F) -> F:
    """
    Decorator that returns both result and execution time.

    Example:
        @timed_result
        def compute_metrics(data):
            return metrics

        result, elapsed_time = compute_metrics(data)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start

    return wrapper


class Timer:
    """
    Context manager for timing code blocks.

    Example:
        with Timer() as t:
            result = compute_heavy_metrics(data)
        print(f"Took {t.elapsed:.4f} seconds")
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.start: float = 0
        self.end: float = 0
        self.elapsed: float = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        if self.name:
            print(f"{self.name}: {self.elapsed:.4f} seconds")


# =============================================================================
# Caching Decorators
# =============================================================================


def memoize(func: F) -> F:
    """
    Simple memoization decorator for caching function results.

    Example:
        @memoize
        def expensive_metric(data):
            ...
    """
    cache: Dict[str, Any] = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from args
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper


def lru_cache(maxsize: int = 128):
    """
    LRU (Least Recently Used) cache decorator.

    Example:
        @lru_cache(maxsize=100)
        def compute_similarity(text1, text2):
            ...
    """

    def decorator(func: F) -> F:
        cache: Dict[str, Any] = {}
        order: List[str] = []

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))

            if key in cache:
                # Move to end (most recently used)
                order.remove(key)
                order.append(key)
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result
            order.append(key)

            # Evict if over capacity
            while len(cache) > maxsize:
                oldest = order.pop(0)
                del cache[oldest]

            return result

        wrapper.cache = cache
        wrapper.clear_cache = lambda: (cache.clear(), order.clear())
        return wrapper

    return decorator


# =============================================================================
# Validation Decorators
# =============================================================================


def validate_inputs(*validators):
    """
    Decorator to validate function inputs.

    Example:
        def is_list(x): return isinstance(x, list)
        def same_length(x, y): return len(x) == len(y)

        @validate_inputs(is_list, is_list, same_length)
        def f1_score(y_true, y_pred):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i, (validator, arg) in enumerate(zip(validators, args)):
                if callable(validator):
                    if validator.__code__.co_argcount == 1:
                        if not validator(arg):
                            raise ValueError(
                                f"Validation failed for argument {i}: {validator.__name__}"
                            )
                    elif validator.__code__.co_argcount == 2 and i > 0:
                        if not validator(args[i - 1], arg):
                            raise ValueError(
                                f"Validation failed for arguments {i-1} and {i}: {validator.__name__}"
                            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_same_length(func: F) -> F:
    """
    Decorator to ensure first two arguments have same length.

    Example:
        @require_same_length
        def accuracy_score(y_true, y_pred):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) >= 2:
            try:
                if len(args[0]) != len(args[1]):
                    raise ValueError(
                        f"Arguments must have same length: {len(args[0])} != {len(args[1])}"
                    )
            except TypeError:
                pass  # Arguments don't support len()
        return func(*args, **kwargs)

    return wrapper


def require_non_empty(func: F) -> F:
    """
    Decorator to ensure arguments are non-empty.

    Example:
        @require_non_empty
        def mean(values):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for i, arg in enumerate(args):
            try:
                if len(arg) == 0:
                    raise ValueError(f"Argument {i} cannot be empty")
            except TypeError:
                pass
        return func(*args, **kwargs)

    return wrapper


def check_range(min_val: float = None, max_val: float = None, arg_index: int = 0):
    """
    Decorator to check if numeric argument is within range.

    Example:
        @check_range(min_val=0, max_val=1, arg_index=0)
        def beta_score(beta):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if arg_index < len(args):
                val = args[arg_index]
                if min_val is not None and val < min_val:
                    raise ValueError(f"Argument {arg_index} must be >= {min_val}")
                if max_val is not None and val > max_val:
                    raise ValueError(f"Argument {arg_index} must be <= {max_val}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Retry Decorators
# =============================================================================


def retry(max_attempts: int = 3, delay: float = 0.1, exceptions: tuple = (Exception,)):
    """
    Decorator to retry function on failure.

    Example:
        @retry(max_attempts=3, delay=0.5)
        def fetch_remote_metrics():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


def fallback(default_value: Any):
    """
    Decorator to return default value on exception.

    Example:
        @fallback(default_value=0.0)
        def safe_metric(data):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                return default_value

        return wrapper

    return decorator


# =============================================================================
# Logging Decorators
# =============================================================================


@dataclass
class LogEntry:
    """Log entry for function calls."""

    function: str
    args: tuple
    kwargs: dict
    result: Any
    timestamp: str
    elapsed: float
    error: Optional[str] = None


class MetricLogger:
    """Singleton logger for metric function calls."""

    _instance = None
    _logs: List[LogEntry] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._logs = []
        return cls._instance

    def log(self, entry: LogEntry):
        self._logs.append(entry)

    def get_logs(self) -> List[LogEntry]:
        return self._logs

    def clear(self):
        self._logs = []

    def summary(self) -> str:
        lines = ["=== Metric Logger Summary ==="]
        for entry in self._logs:
            status = "✓" if entry.error is None else "✗"
            lines.append(f"{status} {entry.function}: {entry.elapsed:.4f}s")
        return "\n".join(lines)


def logged(func: F) -> F:
    """
    Decorator to log function calls with timing.

    Example:
        @logged
        def accuracy_score(y_true, y_pred):
            ...
    """
    logger = MetricLogger()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        error = None
        result = None

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error = str(e)
            raise
        finally:
            elapsed = time.perf_counter() - start
            entry = LogEntry(
                function=func.__name__,
                args=args,
                kwargs=kwargs,
                result=result,
                timestamp=datetime.now().isoformat(),
                elapsed=elapsed,
                error=error,
            )
            logger.log(entry)

        return result

    return wrapper


def deprecated(message: str = ""):
    """
    Decorator to mark functions as deprecated.

    Example:
        @deprecated("Use new_metric instead")
        def old_metric():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import warnings

            msg = f"{func.__name__} is deprecated."
            if message:
                msg += f" {message}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Composition Decorators
# =============================================================================


def compose(*functions):
    """
    Compose multiple functions into one.

    Example:
        process = compose(tokenize, lowercase, remove_punctuation)
        result = process(text)
    """

    def composed(x):
        for func in functions:
            x = func(x)
        return x

    return composed


def pipe(value: Any, *functions):
    """
    Pipe a value through multiple functions.

    Example:
        result = pipe(
            text,
            tokenize,
            lowercase,
            compute_bleu
        )
    """
    for func in functions:
        value = func(value)
    return value


def partial_metric(metric_func: Callable, **fixed_kwargs):
    """
    Create a partial metric function with fixed parameters.

    Example:
        macro_f1 = partial_metric(f1_score, average='macro')
        score = macro_f1(y_true, y_pred)
    """

    @functools.wraps(metric_func)
    def wrapper(*args, **kwargs):
        merged_kwargs = {**fixed_kwargs, **kwargs}
        return metric_func(*args, **merged_kwargs)

    return wrapper


# =============================================================================
# Batch Processing Decorators
# =============================================================================


def vectorize(func: F) -> F:
    """
    Decorator to apply function element-wise to lists.

    Example:
        @vectorize
        def normalize(x):
            return x / max_val

        result = normalize([1, 2, 3])  # Returns [0.33, 0.66, 1.0]
    """

    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        if isinstance(data, (list, tuple)):
            return [func(item, *args, **kwargs) for item in data]
        return func(data, *args, **kwargs)

    return wrapper


def batch_process(batch_size: int = 100):
    """
    Decorator to process data in batches.

    Example:
        @batch_process(batch_size=1000)
        def compute_embeddings(texts):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(data, *args, **kwargs):
            if not isinstance(data, list):
                return func(data, *args, **kwargs)

            results = []
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                batch_result = func(batch, *args, **kwargs)
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
            return results

        return wrapper

    return decorator


# =============================================================================
# Type Coercion Decorators
# =============================================================================


def ensure_list(func: F) -> F:
    """
    Decorator to convert inputs to lists if needed.

    Example:
        @ensure_list
        def mean(values):
            return sum(values) / len(values)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, (tuple, set)):
                new_args.append(list(arg))
            else:
                new_args.append(arg)
        return func(*new_args, **kwargs)

    return wrapper


def ensure_float(func: F) -> F:
    """
    Decorator to ensure numeric results are floats.

    Example:
        @ensure_float
        def accuracy(correct, total):
            return correct / total
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, (int, float)):
            return float(result)
        return result

    return wrapper


# =============================================================================
# Metric Registration
# =============================================================================


class MetricRegistry:
    """
    Registry for custom metrics.

    Example:
        registry = MetricRegistry()

        @registry.register("custom_f1")
        def my_f1_score(y_true, y_pred):
            ...

        score = registry.compute("custom_f1", y_true, y_pred)
    """

    def __init__(self):
        self._metrics: Dict[str, Callable] = {}
        self._metadata: Dict[str, Dict] = {}

    def register(self, name: str, category: str = "custom", description: str = ""):
        """Register a metric function."""

        def decorator(func: F) -> F:
            self._metrics[name] = func
            self._metadata[name] = {
                "category": category,
                "description": description or func.__doc__,
                "name": name,
            }
            return func

        return decorator

    def compute(self, name: str, *args, **kwargs) -> Any:
        """Compute a registered metric."""
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not registered")
        return self._metrics[name](*args, **kwargs)

    def list_metrics(self, category: str = None) -> List[str]:
        """List registered metrics."""
        if category:
            return [
                name
                for name, meta in self._metadata.items()
                if meta["category"] == category
            ]
        return list(self._metrics.keys())

    def get_info(self, name: str) -> Dict:
        """Get metadata for a metric."""
        return self._metadata.get(name, {})

    def unregister(self, name: str):
        """Remove a metric from registry."""
        self._metrics.pop(name, None)
        self._metadata.pop(name, None)


# Global registry instance
metric_registry = MetricRegistry()


# =============================================================================
# Export all
# =============================================================================

__all__ = [
    # Timing
    "timed",
    "timed_result",
    "Timer",
    # Caching
    "memoize",
    "lru_cache",
    # Validation
    "validate_inputs",
    "require_same_length",
    "require_non_empty",
    "check_range",
    # Retry
    "retry",
    "fallback",
    # Logging
    "LogEntry",
    "MetricLogger",
    "logged",
    "deprecated",
    # Composition
    "compose",
    "pipe",
    "partial_metric",
    # Batch
    "vectorize",
    "batch_process",
    # Type coercion
    "ensure_list",
    "ensure_float",
    # Registry
    "MetricRegistry",
    "metric_registry",
]
