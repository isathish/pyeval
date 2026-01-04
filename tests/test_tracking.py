"""
Tests for experiment tracking.
"""

import sys
sys.path.insert(0, '..')

import pytest
from pyeval import ExperimentTracker


class TestExperimentTracker:
    """Tests for ExperimentTracker."""
    
    def test_create_tracker(self):
        """Should create tracker."""
        tracker = ExperimentTracker("test_experiments")
        assert tracker.project_name == "test_experiments"
    
    def test_create_run(self):
        """Should create experiment run."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("experiment_1")
        
        assert run.experiment_name == "experiment_1"
        assert run.run_id is not None
    
    def test_log_params(self):
        """Should log parameters."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("exp")
        
        run.log_params({
            'learning_rate': 0.01,
            'epochs': 100,
            'batch_size': 32
        })
        
        assert run.parameters['learning_rate'] == 0.01
        assert run.parameters['epochs'] == 100
    
    def test_log_metrics(self):
        """Should log metrics."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("exp")
        
        run.log_metrics({
            'accuracy': 0.95,
            'loss': 0.05,
            'f1_score': 0.92
        })
        
        assert run.metrics['accuracy'] == 0.95
        assert run.metrics['loss'] == 0.05
    
    def test_log_single_metric(self):
        """Should log single metric."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("exp")
        
        run.log_metric('accuracy', 0.95)
        run.log_metric('loss', 0.05)
        
        assert run.metrics['accuracy'] == 0.95
        assert run.metrics['loss'] == 0.05
    
    def test_set_status(self):
        """Should set run status."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("exp")
        
        run.set_status("running")
        assert run.status == "running"
        
        run.set_status("completed")
        assert run.status == "completed"
    
    def test_save_run(self):
        """Should save run to tracker."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("exp")
        run.log_metrics({'accuracy': 0.9})
        run.set_status("completed")
        
        tracker.save_run(run)
        
        assert len(tracker.runs) == 1
        assert tracker.runs[0].experiment_name == "exp"
    
    def test_multiple_runs(self):
        """Should handle multiple runs."""
        tracker = ExperimentTracker("test")
        
        for i in range(5):
            run = tracker.create_run(f"exp_{i}")
            run.log_params({'lr': 0.01 * (i + 1)})
            run.log_metrics({'accuracy': 0.8 + i * 0.02})
            run.set_status("completed")
            tracker.save_run(run)
        
        assert len(tracker.runs) == 5
    
    def test_get_best_run(self):
        """Should get best run by metric."""
        tracker = ExperimentTracker("test")
        
        # Create runs with different accuracies
        for i, acc in enumerate([0.85, 0.92, 0.88, 0.95, 0.80]):
            run = tracker.create_run(f"exp_{i}")
            run.log_metrics({'accuracy': acc})
            run.set_status("completed")
            tracker.save_run(run)
        
        best = tracker.get_best_run('accuracy')
        assert best.metrics['accuracy'] == 0.95
        assert best.experiment_name == "exp_3"
    
    def test_get_best_run_minimize(self):
        """Should get best run with minimize."""
        tracker = ExperimentTracker("test")
        
        # Create runs with different losses
        for i, loss in enumerate([0.15, 0.08, 0.12, 0.05, 0.20]):
            run = tracker.create_run(f"exp_{i}")
            run.log_metrics({'loss': loss})
            run.set_status("completed")
            tracker.save_run(run)
        
        best = tracker.get_best_run('loss', maximize=False)
        assert best.metrics['loss'] == 0.05
    
    def test_compare_runs(self):
        """Should compare all runs."""
        tracker = ExperimentTracker("test")
        
        for i in range(3):
            run = tracker.create_run(f"exp_{i}")
            run.log_params({'lr': 0.01 * (i + 1)})
            run.log_metrics({'accuracy': 0.8 + i * 0.05, 'f1': 0.75 + i * 0.05})
            run.set_status("completed")
            tracker.save_run(run)
        
        comparison = tracker.compare_runs()
        assert isinstance(comparison, str)
        assert 'exp_0' in comparison or 'accuracy' in comparison
    
    def test_filter_runs(self):
        """Should filter runs by criteria."""
        tracker = ExperimentTracker("test")
        
        for i in range(5):
            run = tracker.create_run(f"exp_{i}")
            run.log_params({'lr': 0.01 if i < 3 else 0.001})
            run.log_metrics({'accuracy': 0.8 + i * 0.02})
            run.set_status("completed" if i % 2 == 0 else "failed")
            tracker.save_run(run)
        
        # Filter by status
        completed = tracker.filter_runs(status="completed")
        assert len(completed) == 3
    
    def test_get_run_by_name(self):
        """Should get run by experiment name."""
        tracker = ExperimentTracker("test")
        
        run = tracker.create_run("special_exp")
        run.log_metrics({'accuracy': 0.99})
        tracker.save_run(run)
        
        found = tracker.get_run_by_name("special_exp")
        assert found is not None
        assert found.metrics['accuracy'] == 0.99
    
    def test_delete_run(self):
        """Should delete run."""
        tracker = ExperimentTracker("test")
        
        run = tracker.create_run("to_delete")
        tracker.save_run(run)
        
        assert len(tracker.runs) == 1
        
        tracker.delete_run("to_delete")
        assert len(tracker.runs) == 0
    
    def test_run_tags(self):
        """Should support run tags."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("tagged_exp")
        
        run.add_tag("baseline")
        run.add_tag("v1")
        
        assert "baseline" in run.tags
        assert "v1" in run.tags
    
    def test_run_artifacts(self):
        """Should log artifacts."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("artifact_exp")
        
        run.log_artifact("model.pkl", "/path/to/model.pkl")
        run.log_artifact("config.json", "/path/to/config.json")
        
        assert "model.pkl" in run.artifacts
        assert run.artifacts["model.pkl"] == "/path/to/model.pkl"
    
    def test_run_notes(self):
        """Should support run notes."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("noted_exp")
        
        run.add_note("This is a baseline experiment")
        run.add_note("Using default hyperparameters")
        
        assert len(run.notes) == 2
    
    def test_summary_statistics(self):
        """Should compute summary statistics."""
        tracker = ExperimentTracker("test")
        
        for acc in [0.80, 0.85, 0.90, 0.85, 0.82]:
            run = tracker.create_run("exp")
            run.log_metrics({'accuracy': acc})
            tracker.save_run(run)
        
        stats = tracker.summary_statistics('accuracy')
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert abs(stats['mean'] - 0.844) < 0.01


class TestRunLifecycle:
    """Tests for run lifecycle management."""
    
    def test_run_duration(self):
        """Should track run duration."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("timed_exp")
        
        run.start()
        # Simulate some work
        run.end()
        
        assert run.duration is not None or run.end_time is not None
    
    def test_run_timestamps(self):
        """Should record timestamps."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("stamped_exp")
        
        assert run.created_at is not None or hasattr(run, 'start_time')


class TestTrackerPersistence:
    """Tests for tracker persistence (if implemented)."""
    
    def test_export_to_dict(self):
        """Should export runs to dict."""
        tracker = ExperimentTracker("test")
        
        run = tracker.create_run("export_exp")
        run.log_params({'lr': 0.01})
        run.log_metrics({'accuracy': 0.9})
        tracker.save_run(run)
        
        exported = tracker.export_runs()
        assert isinstance(exported, list)
        assert len(exported) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
