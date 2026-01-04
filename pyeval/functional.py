"""
PyEval Functional - Functional Programming Utilities
=====================================================

Provides functional programming utilities for:
- Higher-order functions
- Monads for error handling
- Function composition
- Currying and partial application
"""

from typing import Any, Callable, Generic, List, Optional, TypeVar, Tuple
from dataclasses import dataclass
from functools import reduce

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")


# =============================================================================
# Result Monad - Error Handling
# =============================================================================


@dataclass
class Result(Generic[T, E]):
    """
    Result monad for functional error handling.

    Example:
        def safe_divide(a: float, b: float) -> Result[float, str]:
            if b == 0:
                return Result.failure("Division by zero")
            return Result.success(a / b)

        result = safe_divide(10, 2)
        if result.is_success:
            print(result.value)
        else:
            print(result.error)
    """

    value: Optional[T] = None
    error: Optional[E] = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    @property
    def is_failure(self) -> bool:
        return self.error is not None

    @classmethod
    def success(cls, value: T) -> "Result[T, E]":
        return cls(value=value, error=None)

    @classmethod
    def failure(cls, error: E) -> "Result[T, E]":
        return cls(value=None, error=error)

    def map(self, func: Callable[[T], U]) -> "Result[U, E]":
        """Apply function to value if success."""
        if self.is_success:
            try:
                return Result.success(func(self.value))
            except Exception as e:
                return Result.failure(str(e))
        return Result.failure(self.error)

    def flat_map(self, func: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        """Apply function that returns Result."""
        if self.is_success:
            return func(self.value)
        return Result.failure(self.error)

    def get_or_else(self, default: T) -> T:
        """Get value or default if failure."""
        return self.value if self.is_success else default

    def get_or_raise(self) -> T:
        """Get value or raise exception if failure."""
        if self.is_failure:
            raise ValueError(self.error)
        return self.value


def try_catch(func: Callable[..., T]) -> Callable[..., Result[T, str]]:
    """
    Decorator to wrap function in Result monad.

    Example:
        @try_catch
        def risky_computation(x):
            return 1 / x

        result = risky_computation(0)  # Result.failure("division by zero")
    """

    def wrapper(*args, **kwargs) -> Result[T, str]:
        try:
            return Result.success(func(*args, **kwargs))
        except Exception as e:
            return Result.failure(str(e))

    return wrapper


# =============================================================================
# Option Monad - Optional Values
# =============================================================================


@dataclass
class Option(Generic[T]):
    """
    Option monad for handling optional values.

    Example:
        def find_metric(name: str) -> Option[float]:
            metrics = {'accuracy': 0.95}
            if name in metrics:
                return Option.some(metrics[name])
            return Option.none()

        value = find_metric('accuracy').get_or_else(0.0)
    """

    _value: Optional[T] = None
    _has_value: bool = False

    @property
    def is_some(self) -> bool:
        return self._has_value

    @property
    def is_none(self) -> bool:
        return not self._has_value

    @classmethod
    def some(cls, value: T) -> "Option[T]":
        return cls(_value=value, _has_value=True)

    @classmethod
    def none(cls) -> "Option[T]":
        return cls(_value=None, _has_value=False)

    @classmethod
    def from_nullable(cls, value: Optional[T]) -> "Option[T]":
        """Create Option from nullable value."""
        if value is None:
            return cls.none()
        return cls.some(value)

    def map(self, func: Callable[[T], U]) -> "Option[U]":
        """Apply function if value exists."""
        if self.is_some:
            return Option.some(func(self._value))
        return Option.none()

    def flat_map(self, func: Callable[[T], "Option[U]"]) -> "Option[U]":
        """Apply function that returns Option."""
        if self.is_some:
            return func(self._value)
        return Option.none()

    def filter(self, predicate: Callable[[T], bool]) -> "Option[T]":
        """Filter value based on predicate."""
        if self.is_some and predicate(self._value):
            return self
        return Option.none()

    def get_or_else(self, default: T) -> T:
        """Get value or default."""
        return self._value if self.is_some else default

    def get_or_raise(self, error_msg: str = "No value present") -> T:
        """Get value or raise exception."""
        if self.is_none:
            raise ValueError(error_msg)
        return self._value


# =============================================================================
# Function Composition
# =============================================================================


def compose(*funcs: Callable) -> Callable:
    """
    Compose multiple functions (right to left).

    Example:
        f = compose(str, lambda x: x + 1, lambda x: x * 2)
        f(3)  # str((3 * 2) + 1) = "7"
    """

    def composed(x):
        return reduce(lambda acc, f: f(acc), reversed(funcs), x)

    return composed


def pipe(*funcs: Callable) -> Callable:
    """
    Pipe functions (left to right).

    Example:
        f = pipe(lambda x: x * 2, lambda x: x + 1, str)
        f(3)  # str((3 * 2) + 1) = "7"
    """

    def piped(x):
        return reduce(lambda acc, f: f(acc), funcs, x)

    return piped


def identity(x: T) -> T:
    """Identity function."""
    return x


def constant(value: T) -> Callable[..., T]:
    """Create function that always returns the same value."""
    return lambda *args, **kwargs: value


# =============================================================================
# Currying and Partial Application
# =============================================================================


def curry(func: Callable) -> Callable:
    """
    Curry a function.

    Example:
        @curry
        def add(a, b, c):
            return a + b + c

        add(1)(2)(3)  # 6
        add(1, 2)(3)  # 6
    """
    import inspect

    sig = inspect.signature(func)
    num_params = len(sig.parameters)

    def curried(*args):
        if len(args) >= num_params:
            return func(*args[:num_params])
        return lambda *more: curried(*args, *more)

    return curried


def partial(func: Callable, *args, **kwargs) -> Callable:
    """
    Partial application of a function.

    Example:
        def metric(y_true, y_pred, average='micro'):
            ...

        macro_metric = partial(metric, average='macro')
        score = macro_metric(y_true, y_pred)
    """

    def applied(*more_args, **more_kwargs):
        return func(*args, *more_args, **{**kwargs, **more_kwargs})

    return applied


def flip(func: Callable[[T, U], Any]) -> Callable[[U, T], Any]:
    """
    Flip the first two arguments of a function.

    Example:
        def divide(a, b):
            return a / b

        flip(divide)(2, 10)  # 10 / 2 = 5
    """
    return lambda x, y, *args, **kwargs: func(y, x, *args, **kwargs)


# =============================================================================
# Higher-Order Functions
# =============================================================================


def map_list(func: Callable[[T], U], lst: List[T]) -> List[U]:
    """Map function over list."""
    return [func(x) for x in lst]


def filter_list(predicate: Callable[[T], bool], lst: List[T]) -> List[T]:
    """Filter list by predicate."""
    return [x for x in lst if predicate(x)]


def reduce_list(
    func: Callable[[T, T], T], lst: List[T], initial: Optional[T] = None
) -> T:
    """Reduce list with function."""
    if initial is not None:
        return reduce(func, lst, initial)
    return reduce(func, lst)


def fold_left(func: Callable[[U, T], U], initial: U, lst: List[T]) -> U:
    """Left fold over list."""
    return reduce(func, lst, initial)


def fold_right(func: Callable[[T, U], U], initial: U, lst: List[T]) -> U:
    """Right fold over list."""
    return reduce(lambda acc, x: func(x, acc), reversed(lst), initial)


def zip_with(func: Callable[[T, U], Any], lst1: List[T], lst2: List[U]) -> List[Any]:
    """Zip two lists with a function."""
    return [func(a, b) for a, b in zip(lst1, lst2)]


def flat_map(func: Callable[[T], List[U]], lst: List[T]) -> List[U]:
    """Map and flatten."""
    result = []
    for item in lst:
        result.extend(func(item))
    return result


def group_by(key_func: Callable[[T], Any], lst: List[T]) -> dict:
    """Group items by key function."""
    result = {}
    for item in lst:
        key = key_func(item)
        if key not in result:
            result[key] = []
        result[key].append(item)
    return result


def partition(predicate: Callable[[T], bool], lst: List[T]) -> Tuple[List[T], List[T]]:
    """Partition list by predicate."""
    true_list = []
    false_list = []
    for item in lst:
        if predicate(item):
            true_list.append(item)
        else:
            false_list.append(item)
    return true_list, false_list


def take(n: int, lst: List[T]) -> List[T]:
    """Take first n elements."""
    return lst[:n]


def drop(n: int, lst: List[T]) -> List[T]:
    """Drop first n elements."""
    return lst[n:]


def take_while(predicate: Callable[[T], bool], lst: List[T]) -> List[T]:
    """Take while predicate is true."""
    result = []
    for item in lst:
        if not predicate(item):
            break
        result.append(item)
    return result


def drop_while(predicate: Callable[[T], bool], lst: List[T]) -> List[T]:
    """Drop while predicate is true."""
    for i, item in enumerate(lst):
        if not predicate(item):
            return lst[i:]
    return []


# =============================================================================
# Metric-Specific Functional Utilities
# =============================================================================


def apply_metric(metric_func: Callable, pairs: List[Tuple[List, List]]) -> List[float]:
    """
    Apply metric function to multiple prediction pairs.

    Example:
        results = apply_metric(accuracy_score, [
            ([1, 0, 1], [1, 0, 1]),
            ([1, 1, 1], [1, 0, 1])
        ])
    """
    return [metric_func(y_true, y_pred) for y_true, y_pred in pairs]


def combine_metrics(*metric_funcs: Callable) -> Callable:
    """
    Combine multiple metric functions into one.

    Example:
        combined = combine_metrics(accuracy_score, f1_score)
        results = combined(y_true, y_pred)
        # Returns {'accuracy_score': 0.95, 'f1_score': 0.92}
    """

    def combined_metric(y_true: List, y_pred: List) -> dict:
        return {func.__name__: func(y_true, y_pred) for func in metric_funcs}

    return combined_metric


def threshold_metric(
    metric_func: Callable, threshold: float, above: bool = True
) -> Callable:
    """
    Create metric that returns boolean based on threshold.

    Example:
        high_accuracy = threshold_metric(accuracy_score, 0.9)
        is_high = high_accuracy(y_true, y_pred)  # True if accuracy > 0.9
    """

    def thresholded(y_true: List, y_pred: List) -> bool:
        value = metric_func(y_true, y_pred)
        if above:
            return value > threshold
        return value < threshold

    return thresholded


def average_metric(metric_func: Callable, samples: List[Tuple[List, List]]) -> float:
    """
    Compute average metric across multiple samples.

    Example:
        avg = average_metric(accuracy_score, samples)
    """
    if not samples:
        return 0.0
    scores = apply_metric(metric_func, samples)
    return sum(scores) / len(scores)


def weighted_average_metric(
    metric_func: Callable, samples: List[Tuple[List, List]], weights: List[float]
) -> float:
    """
    Compute weighted average metric.

    Example:
        avg = weighted_average_metric(accuracy_score, samples, weights)
    """
    if not samples or not weights:
        return 0.0
    scores = apply_metric(metric_func, samples)
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / sum(weights)


# =============================================================================
# Export all
# =============================================================================

__all__ = [
    # Monads
    "Result",
    "Option",
    "try_catch",
    # Composition
    "compose",
    "pipe",
    "identity",
    "constant",
    # Currying
    "curry",
    "partial",
    "flip",
    # Higher-order functions
    "map_list",
    "filter_list",
    "reduce_list",
    "fold_left",
    "fold_right",
    "zip_with",
    "flat_map",
    "group_by",
    "partition",
    "take",
    "drop",
    "take_while",
    "drop_while",
    # Metric utilities
    "apply_metric",
    "combine_metrics",
    "threshold_metric",
    "average_metric",
    "weighted_average_metric",
]
