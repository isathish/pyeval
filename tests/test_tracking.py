"""
Tests for experiment tracking.
"""

import sys

sys.path.insert(0, "..")

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

        run.log_params({"learning_rate": 0.01, "epochs": 100, "batch_size": 32})

        assert run.parameters["learning_rate"] == 0.01
        assert run.parameters["epochs"] == 100

    def test_log_metrics(self):
        """Should log metrics."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("exp")

        run.log_metrics({"accuracy": 0.95, "loss": 0.05, "f1_score": 0.92})

        assert run.metrics["accuracy"] == 0.95
        assert run.metrics["loss"] == 0.05

    def test_log_single_metric(self):
        """Should log single metric."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("exp")

        run.log_metric("accuracy", 0.95)
        run.log_metric("loss", 0.05)

        assert run.metrics["accuracy"] == 0.95
        assert run.metrics["loss"] == 0.05

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
        run.log_metrics({"accuracy": 0.9})
        run.set_status("completed")

        tracker.save_run(run)

        assert len(tracker.runs) == 1
        assert tracker.runs[0].experiment_name == "exp"

    def test_multiple_runs(self):
        """Should handle multiple runs."""
        tracker = ExperimentTracker("test")

        for i in range(5):
            run = tracker.create_run(f"exp_{i}")
            run.log_params({"lr": 0.01 * (i + 1)})
            run.log_metrics({"accuracy": 0.8 + i * 0.02})
            run.set_status("completed")
            tracker.save_run(run)

        assert len(tracker.runs) == 5

    def test_get_best_run(self):
        """Should get best run by metric."""
        tracker = ExperimentTracker("test")

        # Create runs with different accuracies
        for i, acc in enumerate([0.85, 0.92, 0.88, 0.95, 0.80]):
            run = tracker.create_run(f"exp_{i}")
            run.log_metrics({"accuracy": acc})
            run.set_status("completed")
            tracker.save_run(run)

        best = tracker.get_best_run("accuracy")
        assert best.metrics["accuracy"] == 0.95
        assert best.experiment_name == "exp_3"

    def test_get_best_run_minimize(self):
        """Should get best run with minimize."""
        tracker = ExperimentTracker("test")

        # Create runs with different losses
        for i, loss in enumerate([0.15, 0.08, 0.12, 0.05, 0.20]):
            run = tracker.create_run(f"exp_{i}")
            run.log_metrics({"loss": loss})
            run.set_status("completed")
            tracker.save_run(run)

        best = tracker.get_best_run("loss", higher_is_better=False)
        assert best.metrics["loss"] == 0.05

    def test_compare_runs(self):
        """Should compare all runs."""
        tracker = ExperimentTracker("test")

        for i in range(3):
            run = tracker.create_run(f"exp_{i}")
            run.log_params({"lr": 0.01 * (i + 1)})
            run.log_metrics({"accuracy": 0.8 + i * 0.05, "f1": 0.75 + i * 0.05})
            run.set_status("completed")
            tracker.save_run(run)

        comparison = tracker.compare_runs()
        assert isinstance(comparison, str)
        assert "exp_0" in comparison or "accuracy" in comparison

    def test_get_runs_by_experiment(self):
        """Should filter runs by experiment name."""
        tracker = ExperimentTracker("test")

        for name in ["exp_a", "exp_b", "exp_a", "exp_c", "exp_a"]:
            run = tracker.create_run(name)
            run.log_metrics({"accuracy": 0.9})
            run.set_status("completed")
            tracker.save_run(run)

        exp_a_runs = tracker.get_runs_by_experiment("exp_a")
        assert len(exp_a_runs) == 3

    def test_get_run_by_id(self):
        """Should get run by ID."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("exp")
        run.log_metrics({"accuracy": 0.9})
        tracker.save_run(run)

        found = tracker.get_run(run.run_id)
        assert found is not None
        assert found.run_id == run.run_id

    def test_list_experiments(self):
        """Should list all experiment names."""
        tracker = ExperimentTracker("test")

        for name in ["exp_a", "exp_b", "exp_a", "exp_c"]:
            run = tracker.create_run(name)
            tracker.save_run(run)

        experiments = tracker.list_experiments()
        assert set(experiments) == {"exp_a", "exp_b", "exp_c"}

    def test_run_tags(self):
        """Should support run tags."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("exp", tags=["baseline", "v1"])

        assert "baseline" in run.tags
        assert "v1" in run.tags

        run.add_tag("production")
        assert "production" in run.tags

    def test_get_runs_by_tag(self):
        """Should filter runs by tag."""
        tracker = ExperimentTracker("test")

        run1 = tracker.create_run("exp1", tags=["baseline"])
        run2 = tracker.create_run("exp2", tags=["experimental"])
        run3 = tracker.create_run("exp3", tags=["baseline", "v2"])

        tracker.save_run(run1)
        tracker.save_run(run2)
        tracker.save_run(run3)

        baseline_runs = tracker.get_runs_by_tag("baseline")
        assert len(baseline_runs) == 2

    def test_run_to_dict(self):
        """Should convert run to dictionary."""
        tracker = ExperimentTracker("test")
        run = tracker.create_run("exp")
        run.log_params({"lr": 0.01})
        run.log_metrics({"accuracy": 0.95})
        run.add_tag("test")

        d = run.to_dict()

        assert d["experiment_name"] == "exp"
        assert d["parameters"]["lr"] == 0.01
        assert d["metrics"]["accuracy"] == 0.95
        assert "test" in d["tags"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
