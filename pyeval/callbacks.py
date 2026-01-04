"""
PyEval Callbacks - Callback System for Metric Computation
==========================================================

Provides a comprehensive callback system for:
- Pre/post computation hooks
- Progress tracking
- Metric aggregation
- Error handling
- Threshold alerts
"""

from abc import ABC
from typing import Any, Callable, Dict, List, Optional, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

T = TypeVar('T')


# =============================================================================
# Callback Events
# =============================================================================

class CallbackEvent(Enum):
    """Events that callbacks can respond to."""
    ON_EVALUATION_START = "on_evaluation_start"
    ON_EVALUATION_END = "on_evaluation_end"
    ON_METRIC_START = "on_metric_start"
    ON_METRIC_END = "on_metric_end"
    ON_BATCH_START = "on_batch_start"
    ON_BATCH_END = "on_batch_end"
    ON_ERROR = "on_error"
    ON_THRESHOLD_BREACH = "on_threshold_breach"


@dataclass
class CallbackContext:
    """Context information passed to callbacks."""
    event: CallbackEvent
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    batch_index: Optional[int] = None
    total_batches: Optional[int] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Base Callback
# =============================================================================

class Callback(ABC):
    """
    Abstract base class for callbacks.
    
    Implement specific event handlers in subclasses.
    """
    
    def on_evaluation_start(self, context: CallbackContext):
        """Called when evaluation begins."""
        pass
    
    def on_evaluation_end(self, context: CallbackContext):
        """Called when evaluation ends."""
        pass
    
    def on_metric_start(self, context: CallbackContext):
        """Called before computing a metric."""
        pass
    
    def on_metric_end(self, context: CallbackContext):
        """Called after computing a metric."""
        pass
    
    def on_batch_start(self, context: CallbackContext):
        """Called before processing a batch."""
        pass
    
    def on_batch_end(self, context: CallbackContext):
        """Called after processing a batch."""
        pass
    
    def on_error(self, context: CallbackContext):
        """Called when an error occurs."""
        pass
    
    def on_threshold_breach(self, context: CallbackContext):
        """Called when a metric breaches threshold."""
        pass


# =============================================================================
# Callback Manager
# =============================================================================

class CallbackManager:
    """
    Manages multiple callbacks and dispatches events.
    
    Example:
        manager = CallbackManager()
        manager.add_callback(LoggingCallback())
        manager.add_callback(ProgressCallback())
        
        manager.dispatch(CallbackContext(
            event=CallbackEvent.ON_METRIC_END,
            metric_name='accuracy',
            metric_value=0.95
        ))
    """
    
    def __init__(self):
        self._callbacks: List[Callback] = []
    
    def add_callback(self, callback: Callback):
        """Add a callback."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callback):
        """Remove a callback."""
        self._callbacks.remove(callback)
    
    def clear_callbacks(self):
        """Remove all callbacks."""
        self._callbacks.clear()
    
    def dispatch(self, context: CallbackContext):
        """Dispatch event to all callbacks."""
        handler_name = context.event.value
        
        for callback in self._callbacks:
            handler = getattr(callback, handler_name, None)
            if handler and callable(handler):
                try:
                    handler(context)
                except Exception as e:
                    # Create error context
                    error_context = CallbackContext(
                        event=CallbackEvent.ON_ERROR,
                        error=e,
                        metadata={'original_event': context.event}
                    )
                    # Try to call error handler
                    if hasattr(callback, 'on_error'):
                        callback.on_error(error_context)


# =============================================================================
# Built-in Callbacks
# =============================================================================

class LoggingCallback(Callback):
    """
    Callback that logs evaluation events.
    
    Example:
        callback = LoggingCallback(verbose=True)
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logs: List[str] = []
    
    def _log(self, message: str):
        self.logs.append(message)
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def on_evaluation_start(self, context: CallbackContext):
        self._log("Evaluation started")
    
    def on_evaluation_end(self, context: CallbackContext):
        self._log("Evaluation completed")
    
    def on_metric_start(self, context: CallbackContext):
        self._log(f"Computing {context.metric_name}...")
    
    def on_metric_end(self, context: CallbackContext):
        self._log(f"{context.metric_name}: {context.metric_value}")
    
    def on_error(self, context: CallbackContext):
        self._log(f"ERROR: {context.error}")


class ProgressCallback(Callback):
    """
    Callback that displays progress.
    
    Example:
        callback = ProgressCallback(total_metrics=5)
    """
    
    def __init__(self, total_metrics: int = 0, show_bar: bool = True):
        self.total_metrics = total_metrics
        self.show_bar = show_bar
        self.completed = 0
    
    def on_evaluation_start(self, context: CallbackContext):
        self.completed = 0
        if 'total_metrics' in context.metadata:
            self.total_metrics = context.metadata['total_metrics']
    
    def on_metric_end(self, context: CallbackContext):
        self.completed += 1
        if self.total_metrics > 0:
            progress = self.completed / self.total_metrics
            if self.show_bar:
                bar_length = 30
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r[{bar}] {progress*100:.1f}%", end='', flush=True)
            else:
                print(f"Progress: {self.completed}/{self.total_metrics}")
    
    def on_evaluation_end(self, context: CallbackContext):
        if self.show_bar:
            print()  # New line after progress bar


class ThresholdCallback(Callback):
    """
    Callback that alerts when metrics breach thresholds.
    
    Example:
        callback = ThresholdCallback({
            'accuracy': {'min': 0.8},
            'loss': {'max': 0.5}
        })
    """
    
    def __init__(self, thresholds: Dict[str, Dict[str, float]],
                 alert_handler: Callable[[str, float, Dict], None] = None):
        self.thresholds = thresholds
        self.alert_handler = alert_handler or self._default_alert
        self.breaches: List[Dict] = []
    
    def _default_alert(self, metric_name: str, value: float, threshold: Dict):
        print(f"⚠️ ALERT: {metric_name}={value} breached threshold {threshold}")
    
    def on_metric_end(self, context: CallbackContext):
        if context.metric_name not in self.thresholds:
            return
        
        threshold = self.thresholds[context.metric_name]
        value = context.metric_value
        breached = False
        
        if 'min' in threshold and value < threshold['min']:
            breached = True
        if 'max' in threshold and value > threshold['max']:
            breached = True
        
        if breached:
            self.breaches.append({
                'metric': context.metric_name,
                'value': value,
                'threshold': threshold,
                'timestamp': context.timestamp
            })
            self.alert_handler(context.metric_name, value, threshold)


class HistoryCallback(Callback):
    """
    Callback that records metric history.
    
    Example:
        callback = HistoryCallback()
        # After evaluation
        history = callback.get_history()
    """
    
    def __init__(self):
        self.history: List[Dict] = []
        self._current_evaluation: Dict = {}
    
    def on_evaluation_start(self, context: CallbackContext):
        self._current_evaluation = {
            'start_time': context.timestamp,
            'metrics': {},
            'metadata': context.metadata
        }
    
    def on_metric_end(self, context: CallbackContext):
        self._current_evaluation['metrics'][context.metric_name] = {
            'value': context.metric_value,
            'timestamp': context.timestamp
        }
    
    def on_evaluation_end(self, context: CallbackContext):
        self._current_evaluation['end_time'] = context.timestamp
        self.history.append(self._current_evaluation)
        self._current_evaluation = {}
    
    def get_history(self) -> List[Dict]:
        return self.history
    
    def get_metric_trend(self, metric_name: str) -> List[float]:
        """Get trend of a specific metric across evaluations."""
        values = []
        for eval_record in self.history:
            if metric_name in eval_record['metrics']:
                values.append(eval_record['metrics'][metric_name]['value'])
        return values


class EarlyStoppingCallback(Callback):
    """
    Callback that signals early stopping based on metric performance.
    
    Example:
        callback = EarlyStoppingCallback(
            metric='loss',
            mode='min',
            patience=5
        )
    """
    
    def __init__(self, metric: str, mode: str = 'min', patience: int = 5,
                 min_delta: float = 0.0):
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def on_metric_end(self, context: CallbackContext):
        if context.metric_name != self.metric:
            return
        
        value = context.metric_value
        improved = False
        
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping triggered: {self.metric} did not improve for {self.patience} evaluations")


class AggregationCallback(Callback):
    """
    Callback that aggregates metrics over multiple evaluations.
    
    Example:
        callback = AggregationCallback()
        # After multiple evaluations
        stats = callback.get_statistics()
    """
    
    def __init__(self):
        self._values: Dict[str, List[float]] = {}
    
    def on_metric_end(self, context: CallbackContext):
        name = context.metric_name
        if name not in self._values:
            self._values[name] = []
        self._values[name].append(context.metric_value)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated statistics for all metrics."""
        stats = {}
        for name, values in self._values.items():
            if not values:
                continue
            n = len(values)
            mean = sum(values) / n
            variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
            stats[name] = {
                'count': n,
                'mean': mean,
                'std': variance ** 0.5,
                'min': min(values),
                'max': max(values)
            }
        return stats
    
    def reset(self):
        """Reset aggregated values."""
        self._values.clear()


class CompositeCallback(Callback):
    """
    Combines multiple callbacks into one.
    
    Example:
        callback = CompositeCallback([
            LoggingCallback(verbose=True),
            ProgressCallback(total_metrics=5),
            HistoryCallback()
        ])
    """
    
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks
    
    def on_evaluation_start(self, context: CallbackContext):
        for cb in self.callbacks:
            cb.on_evaluation_start(context)
    
    def on_evaluation_end(self, context: CallbackContext):
        for cb in self.callbacks:
            cb.on_evaluation_end(context)
    
    def on_metric_start(self, context: CallbackContext):
        for cb in self.callbacks:
            cb.on_metric_start(context)
    
    def on_metric_end(self, context: CallbackContext):
        for cb in self.callbacks:
            cb.on_metric_end(context)
    
    def on_batch_start(self, context: CallbackContext):
        for cb in self.callbacks:
            cb.on_batch_start(context)
    
    def on_batch_end(self, context: CallbackContext):
        for cb in self.callbacks:
            cb.on_batch_end(context)
    
    def on_error(self, context: CallbackContext):
        for cb in self.callbacks:
            cb.on_error(context)


# =============================================================================
# Lambda Callback
# =============================================================================

class LambdaCallback(Callback):
    """
    Callback that uses lambda functions for event handlers.
    
    Example:
        callback = LambdaCallback(
            on_metric_end=lambda ctx: print(f"{ctx.metric_name}: {ctx.metric_value}")
        )
    """
    
    def __init__(self,
                 on_evaluation_start: Callable[[CallbackContext], None] = None,
                 on_evaluation_end: Callable[[CallbackContext], None] = None,
                 on_metric_start: Callable[[CallbackContext], None] = None,
                 on_metric_end: Callable[[CallbackContext], None] = None,
                 on_batch_start: Callable[[CallbackContext], None] = None,
                 on_batch_end: Callable[[CallbackContext], None] = None,
                 on_error: Callable[[CallbackContext], None] = None):
        self._on_evaluation_start = on_evaluation_start
        self._on_evaluation_end = on_evaluation_end
        self._on_metric_start = on_metric_start
        self._on_metric_end = on_metric_end
        self._on_batch_start = on_batch_start
        self._on_batch_end = on_batch_end
        self._on_error = on_error
    
    def on_evaluation_start(self, context: CallbackContext):
        if self._on_evaluation_start:
            self._on_evaluation_start(context)
    
    def on_evaluation_end(self, context: CallbackContext):
        if self._on_evaluation_end:
            self._on_evaluation_end(context)
    
    def on_metric_start(self, context: CallbackContext):
        if self._on_metric_start:
            self._on_metric_start(context)
    
    def on_metric_end(self, context: CallbackContext):
        if self._on_metric_end:
            self._on_metric_end(context)
    
    def on_batch_start(self, context: CallbackContext):
        if self._on_batch_start:
            self._on_batch_start(context)
    
    def on_batch_end(self, context: CallbackContext):
        if self._on_batch_end:
            self._on_batch_end(context)
    
    def on_error(self, context: CallbackContext):
        if self._on_error:
            self._on_error(context)


# =============================================================================
# Callback with Evaluation
# =============================================================================

class CallbackEvaluator:
    """
    Evaluator that integrates callbacks.
    
    Example:
        evaluator = CallbackEvaluator()
        evaluator.add_callback(LoggingCallback(verbose=True))
        evaluator.add_callback(ThresholdCallback({'accuracy': {'min': 0.8}}))
        
        results = evaluator.evaluate(
            y_true=[1, 0, 1, 1],
            y_pred=[1, 0, 0, 1],
            metrics={'accuracy': accuracy_score, 'f1': f1_score}
        )
    """
    
    def __init__(self):
        self.callback_manager = CallbackManager()
    
    def add_callback(self, callback: Callback):
        """Add a callback."""
        self.callback_manager.add_callback(callback)
    
    def evaluate(self, y_true: List, y_pred: List,
                 metrics: Dict[str, Callable]) -> Dict[str, float]:
        """Evaluate with callbacks."""
        results = {}
        
        # Dispatch evaluation start
        self.callback_manager.dispatch(CallbackContext(
            event=CallbackEvent.ON_EVALUATION_START,
            metadata={'total_metrics': len(metrics)}
        ))
        
        try:
            for name, metric_func in metrics.items():
                # Dispatch metric start
                self.callback_manager.dispatch(CallbackContext(
                    event=CallbackEvent.ON_METRIC_START,
                    metric_name=name
                ))
                
                try:
                    value = metric_func(y_true, y_pred)
                    results[name] = value
                    
                    # Dispatch metric end
                    self.callback_manager.dispatch(CallbackContext(
                        event=CallbackEvent.ON_METRIC_END,
                        metric_name=name,
                        metric_value=value
                    ))
                except Exception as e:
                    # Dispatch error
                    self.callback_manager.dispatch(CallbackContext(
                        event=CallbackEvent.ON_ERROR,
                        metric_name=name,
                        error=e
                    ))
                    raise
        finally:
            # Dispatch evaluation end
            self.callback_manager.dispatch(CallbackContext(
                event=CallbackEvent.ON_EVALUATION_END,
                metadata={'results': results}
            ))
        
        return results


# =============================================================================
# Export all
# =============================================================================

__all__ = [
    # Events
    'CallbackEvent',
    'CallbackContext',
    
    # Base
    'Callback',
    'CallbackManager',
    
    # Built-in Callbacks
    'LoggingCallback',
    'ProgressCallback',
    'ThresholdCallback',
    'HistoryCallback',
    'EarlyStoppingCallback',
    'AggregationCallback',
    'CompositeCallback',
    'LambdaCallback',
    
    # Evaluator
    'CallbackEvaluator',
]
