"""Tests for PyEval Design Patterns, Decorators, and Advanced Features"""

import unittest
import time
import math


class TestDecorators(unittest.TestCase):
    """Test decorator functionality."""

    def test_timed_decorator(self):
        """Test timing decorator."""
        from pyeval.decorators import timed

        @timed
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()
        self.assertEqual(result, 42)

    def test_memoize_decorator(self):
        """Test memoization."""
        from pyeval.decorators import memoize

        call_count = [0]

        @memoize
        def expensive_func(x):
            call_count[0] += 1
            return x * 2

        # First call
        result1 = expensive_func(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count[0], 1)

        # Second call - should use cache
        result2 = expensive_func(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count[0], 1)  # Should not increment

        # Different argument
        result3 = expensive_func(10)
        self.assertEqual(result3, 20)
        self.assertEqual(call_count[0], 2)

    def test_lru_cache_decorator(self):
        """Test LRU cache with max size."""
        from pyeval.decorators import lru_cache

        @lru_cache(maxsize=2)
        def func(x):
            return x * 2

        func(1)  # Cache: {1}
        func(2)  # Cache: {1, 2}
        func(3)  # Cache: {2, 3} - 1 evicted

        self.assertEqual(len(func.cache), 2)

    def test_require_same_length(self):
        """Test same length validation."""
        from pyeval.decorators import require_same_length

        @require_same_length
        def compare(a, b):
            return list(zip(a, b))

        # Valid case
        result = compare([1, 2, 3], [4, 5, 6])
        self.assertEqual(len(result), 3)

        # Invalid case
        with self.assertRaises(ValueError):
            compare([1, 2], [1, 2, 3])

    def test_require_non_empty(self):
        """Test non-empty validation."""
        from pyeval.decorators import require_non_empty

        @require_non_empty
        def process(data):
            return sum(data)

        # Valid case
        self.assertEqual(process([1, 2, 3]), 6)

        # Invalid case
        with self.assertRaises(ValueError):
            process([])

    def test_retry_decorator(self):
        """Test retry logic."""
        from pyeval.decorators import retry

        attempts = [0]

        @retry(max_attempts=3, delay=0.01)
        def flaky_function():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Temporary error")
            return "success"

        result = flaky_function()
        self.assertEqual(result, "success")
        self.assertEqual(attempts[0], 3)

    def test_fallback_decorator(self):
        """Test fallback on error."""
        from pyeval.decorators import fallback

        @fallback(default_value=0.0)
        def risky_function():
            raise ValueError("Error")

        result = risky_function()
        self.assertEqual(result, 0.0)

    def test_metric_registry(self):
        """Test metric registration."""
        from pyeval.decorators import MetricRegistry

        registry = MetricRegistry()

        @registry.register("custom_accuracy", category="classification")
        def custom_accuracy(y_true, y_pred):
            return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

        result = registry.compute("custom_accuracy", [1, 0, 1], [1, 0, 1])
        self.assertEqual(result, 1.0)

        self.assertIn("custom_accuracy", registry.list_metrics())

    def test_compose_function(self):
        """Test function composition."""
        from pyeval.decorators import compose

        # compose in decorators applies left to right
        double = lambda x: x * 2
        add_one = lambda x: x + 1
        to_str = lambda x: str(x)

        composed = compose(double, add_one, to_str)
        result = composed(5)  # to_str(add_one(double(5))) = "11"
        self.assertEqual(result, "11")

    def test_vectorize_decorator(self):
        """Test vectorization."""
        from pyeval.decorators import vectorize

        @vectorize
        def square(x):
            return x**2

        result = square([1, 2, 3])
        self.assertEqual(result, [1, 4, 9])


class TestPatterns(unittest.TestCase):
    """Test design patterns."""

    def test_strategy_pattern(self):
        """Test metric strategy pattern."""
        from pyeval.patterns import MetricCalculator, AccuracyStrategy, F1Strategy

        calculator = MetricCalculator(AccuracyStrategy())

        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 1, 0, 0]

        accuracy = calculator.calculate(y_true, y_pred)
        self.assertAlmostEqual(accuracy, 0.8)

        # Switch strategy
        calculator.set_strategy(F1Strategy())
        f1 = calculator.calculate(y_true, y_pred)
        self.assertGreater(f1, 0)

    def test_factory_pattern(self):
        """Test metric factory."""
        from pyeval.patterns import MetricFactory, MetricType

        factory = MetricFactory()

        accuracy_metric = factory.create(MetricType.ACCURACY)
        result = accuracy_metric.compute([1, 0, 1], [1, 0, 1])
        self.assertEqual(result, 1.0)

        self.assertIn(MetricType.ACCURACY, factory.available_metrics())

    def test_builder_pattern(self):
        """Test pipeline builder."""
        from pyeval.patterns import EvaluationPipelineBuilder

        def mock_accuracy(y_true, y_pred):
            return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

        pipeline = (
            EvaluationPipelineBuilder().add_metric("accuracy", mock_accuracy).build()
        )

        results = pipeline.run([1, 0, 1], [1, 0, 1])
        self.assertEqual(results["accuracy"], 1.0)

    def test_observer_pattern(self):
        """Test observer pattern."""
        from pyeval.patterns import MetricSubject, LoggingObserver, Event, EventData

        subject = MetricSubject()
        observer = LoggingObserver()
        subject.attach(observer)

        subject.notify(
            EventData(event=Event.METRIC_END, metric_name="accuracy", value=0.95)
        )

        self.assertEqual(len(observer.logs), 1)

    def test_composite_pattern(self):
        """Test composite metric."""
        from pyeval.patterns import CompositeMetric, SingleMetric

        def accuracy(y_true, y_pred):
            return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

        def count(y_true, y_pred):
            return float(len(y_true))

        composite = CompositeMetric("all_metrics")
        composite.add(SingleMetric("accuracy", accuracy))
        composite.add(SingleMetric("count", count))

        results = composite.compute([1, 0, 1], [1, 0, 1])
        self.assertEqual(results["accuracy"], 1.0)
        self.assertEqual(results["count"], 3.0)

    def test_validation_chain(self):
        """Test chain of responsibility."""
        from pyeval.patterns import create_validation_chain

        validator = create_validation_chain()

        # Valid input
        result = validator.handle([1, 2, 3], [1, 2, 3])
        self.assertTrue(result)

        # Invalid input - different lengths
        with self.assertRaises(ValueError):
            validator.handle([1, 2], [1, 2, 3])


class TestValidators(unittest.TestCase):
    """Test validation utilities."""

    def test_type_validator(self):
        """Test type validation."""
        from pyeval.validators import TypeValidator

        validator = TypeValidator(list, tuple)

        result = validator.validate([1, 2, 3])
        self.assertTrue(result.is_valid)

        result = validator.validate("string")
        self.assertFalse(result.is_valid)

    def test_list_validator(self):
        """Test list validation."""
        from pyeval.validators import ListValidator

        validator = ListValidator(element_type=int, min_length=1, max_length=5)

        result = validator.validate([1, 2, 3])
        self.assertTrue(result.is_valid)

        result = validator.validate([])
        self.assertFalse(result.is_valid)

        result = validator.validate([1, "2", 3])
        self.assertFalse(result.is_valid)

    def test_numeric_validator(self):
        """Test numeric validation."""
        from pyeval.validators import NumericValidator

        validator = NumericValidator(min_value=0, max_value=1)

        result = validator.validate(0.5)
        self.assertTrue(result.is_valid)

        result = validator.validate(1.5)
        self.assertFalse(result.is_valid)

    def test_prediction_validator(self):
        """Test prediction validation."""
        from pyeval.validators import PredictionValidator

        validator = PredictionValidator()

        result = validator.validate({"y_true": [1, 0, 1], "y_pred": [1, 0, 1]})
        self.assertTrue(result.is_valid)

        result = validator.validate({"y_true": [1, 0], "y_pred": [1, 0, 1]})
        self.assertFalse(result.is_valid)

    def test_composite_validators(self):
        """Test composite validators."""
        from pyeval.validators import AllOf, TypeValidator, ListValidator

        validator = AllOf(TypeValidator(list), ListValidator(min_length=1))

        result = validator.validate([1, 2, 3])
        self.assertTrue(result.is_valid)

        result = validator.validate([])
        self.assertFalse(result.is_valid)

    def test_schema_validator(self):
        """Test schema validation."""
        from pyeval.validators import (
            SchemaValidator,
            FieldSchema,
            ListValidator,
            NumericValidator,
        )

        schema = SchemaValidator(
            [
                FieldSchema("y_true", ListValidator(min_length=1)),
                FieldSchema(
                    "threshold", NumericValidator(0, 1), required=False, default=0.5
                ),
            ]
        )

        result = schema.validate({"y_true": [1, 0, 1]})
        self.assertTrue(result.is_valid)

        # Test with defaults
        data = schema.with_defaults({"y_true": [1, 0, 1]})
        self.assertEqual(data["threshold"], 0.5)


class TestCallbacks(unittest.TestCase):
    """Test callback system."""

    def test_callback_manager(self):
        """Test callback manager."""
        from pyeval.callbacks import (
            CallbackManager,
            LoggingCallback,
            CallbackContext,
            CallbackEvent,
        )

        manager = CallbackManager()
        callback = LoggingCallback(verbose=False)
        manager.add_callback(callback)

        manager.dispatch(
            CallbackContext(
                event=CallbackEvent.ON_METRIC_END,
                metric_name="accuracy",
                metric_value=0.95,
            )
        )

        self.assertEqual(len(callback.logs), 1)

    def test_threshold_callback(self):
        """Test threshold alerts."""
        from pyeval.callbacks import ThresholdCallback, CallbackContext, CallbackEvent

        callback = ThresholdCallback({"accuracy": {"min": 0.8}})

        # Above threshold
        callback.on_metric_end(
            CallbackContext(
                event=CallbackEvent.ON_METRIC_END,
                metric_name="accuracy",
                metric_value=0.9,
            )
        )
        self.assertEqual(len(callback.breaches), 0)

        # Below threshold
        callback.on_metric_end(
            CallbackContext(
                event=CallbackEvent.ON_METRIC_END,
                metric_name="accuracy",
                metric_value=0.7,
            )
        )
        self.assertEqual(len(callback.breaches), 1)

    def test_history_callback(self):
        """Test history recording."""
        from pyeval.callbacks import HistoryCallback, CallbackContext, CallbackEvent

        callback = HistoryCallback()

        callback.on_evaluation_start(
            CallbackContext(event=CallbackEvent.ON_EVALUATION_START)
        )
        callback.on_metric_end(
            CallbackContext(
                event=CallbackEvent.ON_METRIC_END,
                metric_name="accuracy",
                metric_value=0.95,
            )
        )
        callback.on_evaluation_end(
            CallbackContext(event=CallbackEvent.ON_EVALUATION_END)
        )

        history = callback.get_history()
        self.assertEqual(len(history), 1)
        self.assertIn("accuracy", history[0]["metrics"])

    def test_aggregation_callback(self):
        """Test aggregation."""
        from pyeval.callbacks import AggregationCallback, CallbackContext, CallbackEvent

        callback = AggregationCallback()

        for value in [0.9, 0.92, 0.88]:
            callback.on_metric_end(
                CallbackContext(
                    event=CallbackEvent.ON_METRIC_END,
                    metric_name="accuracy",
                    metric_value=value,
                )
            )

        stats = callback.get_statistics()
        self.assertIn("accuracy", stats)
        self.assertEqual(stats["accuracy"]["count"], 3)
        self.assertAlmostEqual(stats["accuracy"]["mean"], 0.9, places=2)


class TestPipeline(unittest.TestCase):
    """Test pipeline functionality."""

    def test_basic_pipeline(self):
        """Test basic pipeline execution."""
        from pyeval.pipeline import Pipeline

        def mock_accuracy(y_true, y_pred):
            return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

        pipeline = (
            Pipeline()
            .validate(lambda x: len(x[0]) == len(x[1]), "Length mismatch")
            .add_metric("accuracy", mock_accuracy)
        )

        results = pipeline.run([1, 0, 1], [1, 0, 1])
        self.assertEqual(results["accuracy"], 1.0)

    def test_pipeline_with_aggregation(self):
        """Test pipeline with aggregation."""
        from pyeval.pipeline import Pipeline

        def metric1(y_true, y_pred):
            return 0.9

        def metric2(y_true, y_pred):
            return 0.8

        pipeline = (
            Pipeline()
            .add_metric("m1", metric1)
            .add_metric("m2", metric2)
            .aggregate("mean", lambda r: sum(r.values()) / len(r))
        )

        results = pipeline.run([1], [1])
        self.assertEqual(results["m1"], 0.9)
        self.assertEqual(results["m2"], 0.8)
        self.assertAlmostEqual(results["mean"], 0.85)

    def test_pipeline_result_details(self):
        """Test detailed pipeline results."""
        from pyeval.pipeline import Pipeline

        def mock_metric(y_true, y_pred):
            return 0.95

        pipeline = Pipeline().add_metric("test", mock_metric)
        result = pipeline.run([1], [1], return_details=True)

        self.assertEqual(result.metrics["test"], 0.95)
        self.assertGreater(result.duration, 0)


class TestFunctional(unittest.TestCase):
    """Test functional utilities."""

    def test_result_monad(self):
        """Test Result monad."""
        from pyeval.functional import Result

        # Success case
        result = Result.success(42)
        self.assertTrue(result.is_success)
        self.assertEqual(result.value, 42)

        # Failure case
        result = Result.failure("error")
        self.assertTrue(result.is_failure)
        self.assertEqual(result.error, "error")

        # Map
        result = Result.success(10).map(lambda x: x * 2)
        self.assertEqual(result.value, 20)

    def test_option_monad(self):
        """Test Option monad."""
        from pyeval.functional import Option

        # Some case
        opt = Option.some(42)
        self.assertTrue(opt.is_some)
        self.assertEqual(opt.get_or_else(0), 42)

        # None case
        opt = Option.none()
        self.assertTrue(opt.is_none)
        self.assertEqual(opt.get_or_else(0), 0)

        # Map
        opt = Option.some(10).map(lambda x: x * 2)
        self.assertEqual(opt.get_or_else(0), 20)

    def test_compose_pipe(self):
        """Test compose and pipe."""
        from pyeval.functional import compose, pipe

        # compose applies right to left
        f = compose(str, lambda x: x + 1, lambda x: x * 2)
        result = f(5)  # str((5*2)+1) = "11"

        # pipe returns a function (left to right composition)
        p = pipe(lambda x: x * 2, lambda x: x + 1, str)
        result2 = p(5)  # str((5*2)+1) = "11"

        self.assertEqual(result, "11")
        self.assertEqual(result2, "11")

    def test_curry(self):
        """Test currying."""
        from pyeval.functional import curry

        @curry
        def add(a, b, c):
            return a + b + c

        self.assertEqual(add(1)(2)(3), 6)
        self.assertEqual(add(1, 2)(3), 6)
        self.assertEqual(add(1, 2, 3), 6)

    def test_higher_order_functions(self):
        """Test higher-order functions."""
        from pyeval.functional import map_list, filter_list, reduce_list, group_by

        self.assertEqual(map_list(lambda x: x * 2, [1, 2, 3]), [2, 4, 6])
        self.assertEqual(filter_list(lambda x: x > 1, [1, 2, 3]), [2, 3])
        self.assertEqual(reduce_list(lambda a, b: a + b, [1, 2, 3]), 6)

        grouped = group_by(lambda x: x % 2, [1, 2, 3, 4])
        self.assertEqual(grouped[0], [2, 4])
        self.assertEqual(grouped[1], [1, 3])

    def test_metric_utilities(self):
        """Test metric utilities."""
        from pyeval.functional import combine_metrics, threshold_metric

        def acc(y_true, y_pred):
            return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

        def count(y_true, y_pred):
            return float(len(y_true))

        combined = combine_metrics(acc, count)
        result = combined([1, 0, 1], [1, 0, 1])

        self.assertEqual(result["acc"], 1.0)
        self.assertEqual(result["count"], 3.0)

        # Threshold
        high_acc = threshold_metric(acc, 0.8)
        self.assertTrue(high_acc([1, 0, 1], [1, 0, 1]))
        self.assertFalse(high_acc([1, 0, 1], [0, 0, 0]))


class TestAggregators(unittest.TestCase):
    """Test aggregation utilities."""

    def test_basic_aggregators(self):
        """Test basic aggregators."""
        from pyeval.aggregators import (
            MeanAggregator,
            MedianAggregator,
            MinAggregator,
            MaxAggregator,
        )

        values = [1, 2, 3, 4, 5]

        self.assertEqual(MeanAggregator().aggregate(values), 3.0)
        self.assertEqual(MedianAggregator().aggregate(values), 3.0)
        self.assertEqual(MinAggregator().aggregate(values), 1.0)
        self.assertEqual(MaxAggregator().aggregate(values), 5.0)

    def test_statistical_aggregators(self):
        """Test statistical aggregators."""
        from pyeval.aggregators import StdAggregator, PercentileAggregator

        values = [1, 2, 3, 4, 5]

        std = StdAggregator().aggregate(values)
        self.assertGreater(std, 0)

        p50 = PercentileAggregator(50).aggregate(values)
        self.assertEqual(p50, 3.0)

    def test_metric_aggregator(self):
        """Test multi-metric aggregator."""
        from pyeval.aggregators import MetricAggregator

        aggregator = MetricAggregator()

        aggregator.add_result("accuracy", 0.9)
        aggregator.add_result("accuracy", 0.92)
        aggregator.add_result("accuracy", 0.88)

        stats = aggregator.get_statistics()

        self.assertEqual(stats["accuracy"]["count"], 3)
        self.assertAlmostEqual(stats["accuracy"]["mean"], 0.9, places=2)

    def test_cross_validation_aggregator(self):
        """Test cross-validation aggregator."""
        from pyeval.aggregators import CrossValidationAggregator

        cv = CrossValidationAggregator(n_folds=3)

        cv.add_fold_result(0, {"accuracy": 0.9, "f1": 0.88})
        cv.add_fold_result(1, {"accuracy": 0.92, "f1": 0.90})
        cv.add_fold_result(2, {"accuracy": 0.88, "f1": 0.86})

        summary = cv.get_summary()

        self.assertIn("accuracy", summary)
        self.assertEqual(summary["accuracy"]["folds"], 3)

    def test_ensemble_aggregator(self):
        """Test ensemble aggregation."""
        from pyeval.aggregators import EnsembleAggregator

        ensemble = EnsembleAggregator(strategy="majority_vote")

        ensemble.add_predictions("model1", [1, 0, 1, 1])
        ensemble.add_predictions("model2", [1, 1, 1, 0])
        ensemble.add_predictions("model3", [0, 0, 1, 1])

        final = ensemble.aggregate()

        self.assertEqual(final[0], 1)  # 1, 1, 0 -> 1
        self.assertEqual(final[2], 1)  # 1, 1, 1 -> 1


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_full_pipeline_with_callbacks(self):
        """Test full pipeline with callbacks."""
        from pyeval.pipeline import Pipeline
        from pyeval.aggregators import MetricAggregator

        def mock_accuracy(y_true, y_pred):
            return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

        aggregator = MetricAggregator()

        pipeline = (
            Pipeline()
            .add_metric("accuracy", mock_accuracy)
            .add_hook(
                "after_metric",
                lambda name, val: aggregator.add_result(name, val) if val else None,
            )
        )

        results = pipeline.run([1, 0, 1], [1, 0, 1])
        self.assertEqual(results["accuracy"], 1.0)

    def test_decorator_with_validator(self):
        """Test decorators with validators."""
        from pyeval.decorators import require_same_length, memoize
        from pyeval.validators import validate_predictions

        @memoize
        @require_same_length
        def cached_accuracy(y_true, y_pred):
            return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

        # Should work
        result = cached_accuracy([1, 0, 1], [1, 0, 1])
        self.assertEqual(result, 1.0)

        # Should use cache
        result2 = cached_accuracy([1, 0, 1], [1, 0, 1])
        self.assertEqual(result2, 1.0)


if __name__ == "__main__":
    unittest.main()
