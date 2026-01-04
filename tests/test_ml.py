"""
Unit Tests for PyEval ML Metrics
"""

import sys

sys.path.insert(0, "..")

from pyeval.ml import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
    ClassificationMetrics,
    RegressionMetrics,
)


class TestClassificationMetrics:
    """Tests for classification metrics."""

    def test_accuracy_score_perfect(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1]
        assert accuracy_score(y_true, y_pred) == 1.0

    def test_accuracy_score_partial(self):
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 0, 1, 1]
        assert accuracy_score(y_true, y_pred) == 0.6

    def test_precision_score_binary(self):
        y_true = [1, 0, 1, 1, 0, 1]
        y_pred = [1, 1, 1, 0, 0, 1]
        # TP=3, FP=1, precision = 3/4 = 0.75
        assert precision_score(y_true, y_pred) == 0.75

    def test_recall_score_binary(self):
        y_true = [1, 0, 1, 1, 0, 1]
        y_pred = [1, 1, 1, 0, 0, 1]
        # TP=3, FN=1, recall = 3/4 = 0.75
        assert recall_score(y_true, y_pred) == 0.75

    def test_f1_score_binary(self):
        y_true = [1, 0, 1, 1, 0, 1]
        y_pred = [1, 1, 1, 0, 0, 1]
        # precision = 0.75, recall = 0.75, f1 = 0.75
        assert f1_score(y_true, y_pred) == 0.75

    def test_confusion_matrix_binary(self):
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 0, 1, 1]
        cm = confusion_matrix(y_true, y_pred)
        # [[TN, FP], [FN, TP]] = [[1, 1], [1, 2]]
        assert cm[0][0] == 1  # TN
        assert cm[0][1] == 1  # FP
        assert cm[1][0] == 1  # FN
        assert cm[1][1] == 2  # TP

    def test_classification_metrics_compute(self):
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 0, 1, 0]
        metrics = ClassificationMetrics.compute(y_true, y_pred)
        assert metrics.accuracy == 0.8
        assert isinstance(metrics.precision, float)
        assert isinstance(metrics.recall, float)
        assert isinstance(metrics.f1, float)


class TestRegressionMetrics:
    """Tests for regression metrics."""

    def test_mse_perfect(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        assert mean_squared_error(y_true, y_pred) == 0.0

    def test_mse_basic(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 3.5]
        # MSE = ((0.5)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.25
        assert mean_squared_error(y_true, y_pred) == 0.25

    def test_rmse_basic(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 3.5]
        assert root_mean_squared_error(y_true, y_pred) == 0.5

    def test_mae_basic(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 3.5]
        # MAE = (0.5 + 0.5 + 0.5) / 3 = 0.5
        assert mean_absolute_error(y_true, y_pred) == 0.5

    def test_r2_perfect(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        assert r2_score(y_true, y_pred) == 1.0

    def test_regression_metrics_compute(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 2.1, 2.9, 4.2]
        metrics = RegressionMetrics.compute(y_true, y_pred)
        assert isinstance(metrics.mse, float)
        assert isinstance(metrics.rmse, float)
        assert isinstance(metrics.mae, float)
        assert isinstance(metrics.r2, float)


class TestClusteringMetrics:
    """Tests for clustering metrics."""

    def test_silhouette_score_basic(self):
        # Well-separated clusters
        X = [[0, 0], [0.1, 0.1], [0.2, 0.2], [5, 5], [5.1, 5.1], [5.2, 5.2]]
        labels = [0, 0, 0, 1, 1, 1]
        score = silhouette_score(X, labels)
        assert score > 0.5  # Should be high for well-separated clusters


def run_tests():
    """Run all tests."""
    import traceback

    test_classes = [
        TestClassificationMetrics,
        TestRegressionMetrics,
        TestClusteringMetrics,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {test_class.__name__}.{method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {test_class.__name__}.{method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {test_class.__name__}.{method_name}: {e}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*40}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
