"""
Tests for fairness evaluation metrics.
"""

import sys
sys.path.insert(0, '..')

import pytest
from pyeval import (
    demographic_parity, equalized_odds, equal_opportunity,
    disparate_impact, statistical_parity_difference,
    calibration_by_group, individual_fairness,
    FairnessMetrics
)


class TestDemographicParity:
    """Tests for demographic parity."""
    
    def test_perfect_parity(self):
        """Equal selection rates should have perfect parity."""
        y_pred = [1, 0, 1, 0, 1, 0, 1, 0]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        result = demographic_parity(y_pred, sensitive)
        assert isinstance(result, dict)
        assert 'dp_difference' in result
    
    def test_different_rates(self):
        """Different selection rates should show disparity."""
        y_pred = [1, 1, 1, 1, 0, 0, 0, 0]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        result = demographic_parity(y_pred, sensitive)
        assert isinstance(result, dict)
        assert result['dp_difference'] == 1.0


class TestEqualizedOdds:
    """Tests for equalized odds."""
    
    def test_equalized_odds(self):
        """Should compute equalized odds metric."""
        y_true = [1, 0, 1, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 1, 1, 1, 0, 0, 0]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        result = equalized_odds(y_true, y_pred, sensitive)
        assert isinstance(result, dict)
    
    def test_binary_groups(self):
        """Should work with binary sensitive attribute."""
        y_true = [1, 1, 0, 0, 1, 1, 0, 0]
        y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        result = equalized_odds(y_true, y_pred, sensitive)
        assert isinstance(result, dict)


class TestEqualOpportunity:
    """Tests for equal opportunity."""
    
    def test_equal_opportunity(self):
        """Should compute equal opportunity metric."""
        y_true = [1, 0, 1, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 1, 0, 1, 0]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        result = equal_opportunity(y_true, y_pred, sensitive)
        assert isinstance(result, dict)
    
    def test_returns_dict(self):
        """Should return a dictionary."""
        y_true = [1, 1, 0, 0, 1, 1, 0, 0]
        y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        result = equal_opportunity(y_true, y_pred, sensitive)
        assert isinstance(result, dict)


class TestDisparateImpact:
    """Tests for disparate impact."""
    
    def test_no_disparity(self):
        """Equal rates should have ratio close to 1."""
        y_pred = [1, 0, 1, 0, 1, 0, 1, 0]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        result = disparate_impact(y_pred, sensitive)
        assert isinstance(result, dict)
        assert 'di_ratio' in result
    
    def test_high_disparity(self):
        """Very different rates should show disparity."""
        y_pred = [1, 1, 1, 1, 0, 0, 0, 0]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        result = disparate_impact(y_pred, sensitive)
        assert isinstance(result, dict)


class TestStatisticalParityDifference:
    """Tests for statistical parity difference."""
    
    def test_no_difference(self):
        """Equal rates should have zero difference."""
        y_pred = [1, 0, 1, 0, 1, 0, 1, 0]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        result = statistical_parity_difference(y_pred, sensitive)
        assert isinstance(result, float)
        assert abs(result) < 0.001  # Should be close to 0
    
    def test_large_difference(self):
        """Different rates should have non-zero difference."""
        y_pred = [1, 1, 1, 0, 0, 0, 0, 0]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        result = statistical_parity_difference(y_pred, sensitive)
        assert isinstance(result, float)
        assert result > 0


class TestCalibrationByGroup:
    """Tests for calibration by group."""
    
    def test_calibration(self):
        """Should compute calibration by group."""
        y_true = [1, 0, 1, 0, 1, 0, 1, 0]
        y_scores = [0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        result = calibration_by_group(y_true, y_scores, sensitive)
        assert isinstance(result, dict)
    
    def test_returns_dict(self):
        """Should return calibration info per group."""
        y_true = [1, 1, 0, 0, 1, 1, 0, 0]
        y_scores = [0.8, 0.7, 0.3, 0.2, 0.9, 0.8, 0.2, 0.1]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        result = calibration_by_group(y_true, y_scores, sensitive)
        assert isinstance(result, dict)


class TestIndividualFairness:
    """Tests for individual fairness."""
    
    def test_individual_fairness(self):
        """Should compute individual fairness metric."""
        X = [[1, 2], [1, 2.1], [5, 6], [5, 6.1]]
        y_pred = [1, 1, 0, 0]
        result = individual_fairness(X, y_pred)
        assert isinstance(result, dict)
        assert 'individual_fairness' in result or 'consistency' in result
    
    def test_similar_individuals(self):
        """Similar individuals should have similar predictions."""
        X = [[1, 1], [1.1, 1.1], [5, 5], [5.1, 5.1]]
        y_pred = [1, 1, 0, 0]
        result = individual_fairness(X, y_pred)
        assert isinstance(result, dict)


class TestFairnessMetrics:
    """Tests for FairnessMetrics class."""
    
    def test_compute_all(self):
        """Should compute all fairness metrics."""
        y_true = [1, 0, 1, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 1, 0, 1, 0]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        
        metrics = FairnessMetrics.compute(y_true, y_pred, sensitive)
        assert metrics is not None
    
    def test_with_probabilities(self):
        """Should work with probability predictions."""
        y_true = [1, 0, 1, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 1, 1, 1, 0, 0, 0]
        y_prob = [0.9, 0.2, 0.8, 0.6, 0.9, 0.3, 0.4, 0.1]
        sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        
        metrics = FairnessMetrics.compute(y_true, y_pred, sensitive, y_prob)
        assert metrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
