"""
Tests for Evaluator and EvaluationReport classes.
"""

import sys
sys.path.insert(0, '..')

import pytest
from pyeval import (
    Evaluator, EvaluationReport
)


class TestEvaluationReport:
    """Tests for EvaluationReport."""
    
    def test_create_report(self):
        """Should create report with metrics."""
        report = EvaluationReport(
            name="Test Report",
            category="classification",
            metrics={'accuracy': 0.95, 'f1': 0.92}
        )
        assert report.name == "Test Report"
        assert report.category == "classification"
        assert report.metrics['accuracy'] == 0.95
    
    def test_summary(self):
        """Should generate summary string."""
        report = EvaluationReport(
            name="Test",
            category="classification",
            metrics={'accuracy': 0.95}
        )
        summary = report.summary()
        assert isinstance(summary, str)
        assert 'Test' in summary or 'accuracy' in summary.lower()
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        report = EvaluationReport(
            name="Test",
            category="classification",
            metrics={'accuracy': 0.95}
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert 'name' in d
        assert 'metrics' in d
    
    def test_report_with_metadata(self):
        """Should support metadata."""
        report = EvaluationReport(
            name="Test",
            category="classification",
            metrics={'accuracy': 0.95},
            metadata={'model': 'RandomForest', 'version': '1.0'}
        )
        assert report.metadata['model'] == 'RandomForest'


class TestEvaluator:
    """Tests for Evaluator class."""
    
    def test_create_evaluator(self):
        """Should create evaluator."""
        evaluator = Evaluator("Test Experiment")
        assert evaluator.name == "Test Experiment"
    
    def test_evaluate_classification(self):
        """Should evaluate classification metrics."""
        evaluator = Evaluator("Test")
        y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
        
        report = evaluator.evaluate_classification(y_true, y_pred, name="Binary")
        
        assert isinstance(report, EvaluationReport)
        assert 'accuracy' in report.metrics
        assert 'precision' in report.metrics
        assert 'recall' in report.metrics
        assert 'f1' in report.metrics
    
    def test_evaluate_regression(self):
        """Should evaluate regression metrics."""
        evaluator = Evaluator("Test")
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.1, 2.9, 4.2, 4.8]
        
        report = evaluator.evaluate_regression(y_true, y_pred, name="Regressor")
        
        assert isinstance(report, EvaluationReport)
        assert 'mse' in report.metrics
        assert 'rmse' in report.metrics
        assert 'mae' in report.metrics
        assert 'r2' in report.metrics
    
    def test_evaluate_generation(self):
        """Should evaluate text generation metrics."""
        evaluator = Evaluator("Test")
        references = ["The cat sat on the mat", "Hello world"]
        hypotheses = ["A cat is on the mat", "Hello there"]
        
        report = evaluator.evaluate_generation(references, hypotheses, name="Generator")
        
        assert isinstance(report, EvaluationReport)
        assert 'bleu' in report.metrics or 'rouge' in str(report.metrics).lower()
    
    def test_multiple_evaluations(self):
        """Should handle multiple evaluations."""
        evaluator = Evaluator("Multi-Test")
        
        # Classification
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 0]
        report1 = evaluator.evaluate_classification(y_true, y_pred, name="Model A")
        
        # Another classification
        y_pred2 = [1, 1, 1, 0]
        report2 = evaluator.evaluate_classification(y_true, y_pred2, name="Model B")
        
        assert len(evaluator.reports) == 2
        assert evaluator.reports[0].name == "Model A"
        assert evaluator.reports[1].name == "Model B"
    
    def test_compare_reports(self):
        """Should compare multiple reports."""
        evaluator = Evaluator("Comparison")
        
        y_true = [1, 0, 1, 0, 1, 0]
        y_pred1 = [1, 0, 1, 0, 0, 0]
        y_pred2 = [1, 0, 0, 0, 1, 1]
        
        evaluator.evaluate_classification(y_true, y_pred1, name="Model A")
        evaluator.evaluate_classification(y_true, y_pred2, name="Model B")
        
        comparison = evaluator.compare_reports()
        assert isinstance(comparison, str)
    
    def test_get_best_report(self):
        """Should get best report by metric."""
        evaluator = Evaluator("Best Selection")
        
        y_true = [1, 0, 1, 0, 1, 0]
        y_pred1 = [1, 0, 1, 0, 1, 0]  # Perfect
        y_pred2 = [0, 0, 0, 0, 0, 0]  # All zeros
        
        evaluator.evaluate_classification(y_true, y_pred1, name="Good Model")
        evaluator.evaluate_classification(y_true, y_pred2, name="Bad Model")
        
        best = evaluator.get_best_report(metric='accuracy')
        assert best.name == "Good Model"
    
    def test_clear_reports(self):
        """Should clear all reports."""
        evaluator = Evaluator("Clear Test")
        y_true = [1, 0, 1]
        y_pred = [1, 0, 0]
        
        evaluator.evaluate_classification(y_true, y_pred, name="Test")
        assert len(evaluator.reports) == 1
        
        evaluator.clear_reports()
        assert len(evaluator.reports) == 0
    
    def test_export_reports(self):
        """Should export reports to dict."""
        evaluator = Evaluator("Export Test")
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 0]
        
        evaluator.evaluate_classification(y_true, y_pred, name="Test")
        
        exported = evaluator.export_reports()
        assert isinstance(exported, list)
        assert len(exported) == 1
        assert exported[0]['name'] == "Test"


class TestEvaluatorIntegration:
    """Integration tests for Evaluator."""
    
    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        # Create evaluator
        evaluator = Evaluator("Full Workflow Test")
        
        # Sample data
        y_true_cls = [1, 0, 1, 1, 0, 1, 0, 0]
        y_pred_cls1 = [1, 0, 0, 1, 0, 1, 1, 0]
        y_pred_cls2 = [1, 0, 1, 1, 0, 0, 0, 0]
        
        y_true_reg = [1.0, 2.0, 3.0, 4.0]
        y_pred_reg = [1.1, 2.2, 2.8, 4.1]
        
        refs = ["hello world"]
        hyps = ["hello there"]
        
        # Run evaluations
        evaluator.evaluate_classification(y_true_cls, y_pred_cls1, name="Classifier v1")
        evaluator.evaluate_classification(y_true_cls, y_pred_cls2, name="Classifier v2")
        evaluator.evaluate_regression(y_true_reg, y_pred_reg, name="Regressor")
        evaluator.evaluate_generation(refs, hyps, name="Generator")
        
        # Verify
        assert len(evaluator.reports) == 4
        
        # Get comparison
        comparison = evaluator.compare_reports()
        assert isinstance(comparison, str)
        
        # Get best classification model
        best = evaluator.get_best_report(metric='f1', category='classification')
        assert best is not None


class TestEvaluatorEdgeCases:
    """Edge case tests for Evaluator."""
    
    def test_empty_evaluator(self):
        """Should handle empty evaluator."""
        evaluator = Evaluator("Empty")
        comparison = evaluator.compare_reports()
        assert isinstance(comparison, str)
    
    def test_single_sample(self):
        """Should handle single sample."""
        evaluator = Evaluator("Single")
        y_true = [1]
        y_pred = [1]
        
        report = evaluator.evaluate_classification(y_true, y_pred, name="Single")
        assert report.metrics['accuracy'] == 1.0
    
    def test_all_same_class(self):
        """Should handle all same class."""
        evaluator = Evaluator("Same Class")
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        
        report = evaluator.evaluate_classification(y_true, y_pred, name="All Ones")
        assert report.metrics['accuracy'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
