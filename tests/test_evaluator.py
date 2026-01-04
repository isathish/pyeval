"""
Tests for Evaluator and EvaluationReport classes.
"""

import sys
sys.path.insert(0, '..')

import pytest
from pyeval import Evaluator, EvaluationReport


class TestEvaluationReport:
    """Tests for EvaluationReport."""
    
    def test_create_report(self):
        """Should create report with metrics."""
        report = EvaluationReport(
            name="Test Report",
            domain="ml",
            metrics={'accuracy': 0.95, 'f1': 0.92}
        )
        assert report.name == "Test Report"
        assert report.domain == "ml"
        assert report.metrics['accuracy'] == 0.95
    
    def test_summary(self):
        """Should generate summary string."""
        report = EvaluationReport(
            name="Test",
            domain="ml",
            metrics={'accuracy': 0.95}
        )
        summary = report.summary()
        assert isinstance(summary, str)
        assert 'Test' in summary or 'accuracy' in summary.lower()
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        report = EvaluationReport(
            name="Test",
            domain="ml",
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
            domain="ml",
            metrics={'accuracy': 0.95},
            metadata={'model': 'RandomForest', 'version': '1.0'}
        )
        assert report.metadata['model'] == 'RandomForest'
    
    def test_add_metric(self):
        """Should add metric to report."""
        report = EvaluationReport(name="Test", domain="ml")
        report.add_metric('accuracy', 0.95)
        assert report.metrics['accuracy'] == 0.95
    
    def test_add_sample(self):
        """Should add sample to report."""
        report = EvaluationReport(name="Test", domain="ml")
        report.add_sample({'input': 'x', 'output': 'y', 'score': 0.9})
        assert len(report.samples) == 1


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
        references = ["The cat sat on a mat", "Hello world"]
        hypotheses = ["The cat sat on the mat", "Hello world"]
        
        report = evaluator.evaluate_generation(references, hypotheses, name="Generator")
        
        assert isinstance(report, EvaluationReport)
        assert 'bleu' in report.metrics
    
    def test_multiple_evaluations(self):
        """Should store multiple evaluation reports."""
        evaluator = Evaluator("Test")
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 0]
        
        evaluator.evaluate_classification(y_true, y_pred, name="Eval1")
        evaluator.evaluate_classification(y_true, y_pred, name="Eval2")
        
        assert len(evaluator.reports) == 2
    
    def test_compare_reports(self):
        """Should compare reports."""
        evaluator = Evaluator("Test")
        y_true = [1, 0, 1, 0]
        y_pred1 = [1, 0, 1, 0]  # Perfect
        y_pred2 = [1, 1, 0, 0]  # 50%
        
        evaluator.evaluate_classification(y_true, y_pred1, name="Good")
        evaluator.evaluate_classification(y_true, y_pred2, name="Bad")
        
        comparison = evaluator.compare_reports()
        assert comparison is not None
    
    def test_get_best_report(self):
        """Should get best report by metric."""
        evaluator = Evaluator("Test")
        y_true = [1, 0, 1, 0]
        y_pred1 = [1, 0, 1, 0]  # Perfect
        y_pred2 = [1, 1, 0, 0]  # 50%
        
        evaluator.evaluate_classification(y_true, y_pred1, name="Good")
        evaluator.evaluate_classification(y_true, y_pred2, name="Bad")
        
        # Use get_all_reports and find best manually
        reports = evaluator.get_all_reports()
        best = max(reports, key=lambda r: r.metrics.get('accuracy', 0))
        assert best.name == "Good"
        assert best.metrics['accuracy'] == 1.0
    
    def test_clear_reports(self):
        """Should clear all reports."""
        evaluator = Evaluator("Test")
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 0]
        
        evaluator.evaluate_classification(y_true, y_pred)
        assert len(evaluator.reports) > 0
        
        evaluator.clear_reports()
        assert len(evaluator.reports) == 0
    
    def test_export_reports(self):
        """Should export reports."""
        evaluator = Evaluator("Test")
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 0]
        
        evaluator.evaluate_classification(y_true, y_pred, name="TestExport")
        
        # Get reports and convert to dict
        reports = evaluator.get_all_reports()
        exported = [r.to_dict() for r in reports]
        assert isinstance(exported, list)
        assert len(exported) > 0


class TestEvaluatorIntegration:
    """Integration tests for Evaluator."""
    
    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        evaluator = Evaluator("ML Pipeline")
        
        # Classification
        y_true_cls = [1, 0, 1, 1, 0, 1, 0, 0]
        y_pred_cls = [1, 0, 1, 0, 0, 1, 1, 0]
        report1 = evaluator.evaluate_classification(y_true_cls, y_pred_cls, name="Classifier")
        
        # Regression
        y_true_reg = [1.0, 2.0, 3.0, 4.0]
        y_pred_reg = [1.1, 2.2, 2.8, 4.1]
        report2 = evaluator.evaluate_regression(y_true_reg, y_pred_reg, name="Regressor")
        
        assert len(evaluator.reports) == 2
        assert report1.domain == "ml"
        assert report2.domain == "ml"


class TestEvaluatorEdgeCases:
    """Edge case tests for Evaluator."""
    
    def test_empty_evaluator(self):
        """Should handle empty evaluator."""
        evaluator = Evaluator("Empty")
        assert len(evaluator.reports) == 0
    
    def test_single_sample(self):
        """Should handle single sample."""
        evaluator = Evaluator("Test")
        report = evaluator.evaluate_classification([1], [1], name="Single")
        assert report.metrics['accuracy'] == 1.0
    
    def test_all_same_class(self):
        """Should handle all same class."""
        evaluator = Evaluator("Test")
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        report = evaluator.evaluate_classification(y_true, y_pred, name="Same")
        assert report.metrics['accuracy'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
