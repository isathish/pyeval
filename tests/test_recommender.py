"""
Unit Tests for PyEval Recommender Metrics
"""

import sys

sys.path.insert(0, "..")

from pyeval.recommender import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    ndcg_at_k,
    dcg_at_k,
    average_precision,
    mean_average_precision,
    hit_rate,
    reciprocal_rank,
    mean_reciprocal_rank,
    RecommenderMetrics,
)


class TestPrecisionAtK:
    """Tests for Precision@K."""

    def test_precision_at_k_perfect(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 2, 3, 4, 5]
        assert precision_at_k(recommended, relevant, k=5) == 1.0

    def test_precision_at_k_partial(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5, 7, 9]
        # 3 relevant in top 5
        assert precision_at_k(recommended, relevant, k=5) == 0.6

    def test_precision_at_k_none(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [6, 7, 8, 9, 10]
        assert precision_at_k(recommended, relevant, k=5) == 0.0

    def test_precision_at_3(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [2, 4, 6]
        # 1 relevant in top 3
        assert precision_at_k(recommended, relevant, k=3) == 1 / 3


class TestRecallAtK:
    """Tests for Recall@K."""

    def test_recall_at_k_perfect(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 2, 3]
        assert recall_at_k(recommended, relevant, k=5) == 1.0

    def test_recall_at_k_partial(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5, 7, 9]
        # 3 out of 5 relevant found
        assert recall_at_k(recommended, relevant, k=5) == 0.6

    def test_recall_at_k_small(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 6, 7, 8]
        # 1 out of 4 relevant found
        assert recall_at_k(recommended, relevant, k=5) == 0.25


class TestNDCG:
    """Tests for NDCG@K."""

    def test_ndcg_perfect(self):
        recommended = [1, 2, 3]
        relevant = [1, 2, 3]
        assert ndcg_at_k(recommended, relevant, k=3) == 1.0

    def test_ndcg_partial(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [3, 5, 7]
        score = ndcg_at_k(recommended, relevant, k=5)
        assert 0 < score < 1.0

    def test_dcg_basic(self):
        relevances = [3, 2, 1, 0]
        dcg = dcg_at_k(relevances, k=4)
        assert dcg > 0


class TestAveragePrecision:
    """Tests for Average Precision."""

    def test_ap_perfect(self):
        recommended = [1, 2, 3]
        relevant = [1, 2, 3]
        assert average_precision(recommended, relevant) == 1.0

    def test_ap_partial(self):
        recommended = [1, 4, 2, 5, 3]
        relevant = [1, 2, 3]
        ap = average_precision(recommended, relevant)
        assert 0 < ap < 1.0

    def test_map_basic(self):
        recommendations = [[1, 2, 3], [4, 5, 6]]
        relevants = [[1, 3], [5]]
        map_score = mean_average_precision(recommendations, relevants)
        assert 0 < map_score <= 1.0


class TestHitRate:
    """Tests for Hit Rate."""

    def test_hit_rate_hit(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [3, 10, 20]
        assert hit_rate(recommended, relevant, k=5) == 1.0

    def test_hit_rate_miss(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [10, 20, 30]
        assert hit_rate(recommended, relevant, k=5) == 0.0


class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_rr_first(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 10, 20]
        assert reciprocal_rank(recommended, relevant) == 1.0

    def test_rr_third(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [3, 10, 20]
        assert reciprocal_rank(recommended, relevant) == 1 / 3

    def test_mrr_basic(self):
        recommendations = [[1, 2, 3], [4, 5, 6]]
        relevants = [[1], [6]]
        mrr = mean_reciprocal_rank(recommendations, relevants)
        # (1/1 + 1/3) / 2 = 2/3
        assert abs(mrr - 2 / 3) < 0.01


class TestRecommenderMetrics:
    """Tests for RecommenderMetrics class."""

    def test_compute(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5, 7]
        metrics = RecommenderMetrics.compute(recommended, relevant, k=5)
        assert isinstance(metrics.precision_at_k, float)
        assert isinstance(metrics.recall_at_k, float)
        assert isinstance(metrics.ndcg_at_k, float)
        assert isinstance(metrics.hit_rate, float)

    def test_compute_batch(self):
        recommendations = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        relevants = [[1, 3], [7, 9, 11]]
        metrics = RecommenderMetrics.compute_batch(recommendations, relevants, k=5)
        assert isinstance(metrics.precision_at_k, float)
        assert isinstance(metrics.map_score, float)


def run_tests():
    """Run all tests."""
    import traceback

    test_classes = [
        TestPrecisionAtK,
        TestRecallAtK,
        TestNDCG,
        TestAveragePrecision,
        TestHitRate,
        TestMRR,
        TestRecommenderMetrics,
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
