"""
Experiment Tracking - Pure Python Implementation
=================================================

Simple experiment tracking for ML/AI model evaluation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class ExperimentRun:
    """A single experiment run."""
    
    run_id: str
    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    status: str = "running"  # running, completed, failed
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self.parameters[key] = value
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self.parameters.update(params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric."""
        if step is not None:
            key = f"{key}_step_{step}"
        self.metrics[key] = value
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple metrics."""
        self.metrics.update(metrics)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the run."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def set_status(self, status: str) -> None:
        """Set run status."""
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'tags': self.tags,
            'notes': self.notes,
            'status': self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRun':
        """Create from dictionary."""
        return cls(
            run_id=data['run_id'],
            experiment_name=data['experiment_name'],
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            parameters=data.get('parameters', {}),
            metrics=data.get('metrics', {}),
            tags=data.get('tags', []),
            notes=data.get('notes', ''),
            status=data.get('status', 'completed')
        )


class ExperimentTracker:
    """
    Simple experiment tracking system.
    
    Example usage:
        tracker = ExperimentTracker("my_project")
        
        # Create a new run
        run = tracker.create_run("classification_experiment")
        run.log_params({'model': 'random_forest', 'n_estimators': 100})
        run.log_metrics({'accuracy': 0.95, 'f1': 0.93})
        run.set_status('completed')
        tracker.save_run(run)
        
        # Compare experiments
        print(tracker.compare_runs())
        
        # Get best run
        best = tracker.get_best_run('accuracy')
    """
    
    def __init__(self, project_name: str, tracking_dir: Optional[str] = None):
        """
        Initialize experiment tracker.
        
        Args:
            project_name: Name of the project
            tracking_dir: Directory to store tracking data (optional)
        """
        self.project_name = project_name
        self.tracking_dir = tracking_dir
        self.runs: List[ExperimentRun] = []
        self._run_counter = 0
        
        # Load existing runs if tracking_dir exists
        if tracking_dir and os.path.exists(tracking_dir):
            self._load_runs()
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        self._run_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}_{self._run_counter}"
    
    def create_run(self, experiment_name: str, 
                   tags: Optional[List[str]] = None) -> ExperimentRun:
        """
        Create a new experiment run.
        
        Args:
            experiment_name: Name of the experiment
            tags: Optional tags for the run
            
        Returns:
            New ExperimentRun object
        """
        run = ExperimentRun(
            run_id=self._generate_run_id(),
            experiment_name=experiment_name,
            tags=tags or []
        )
        return run
    
    def save_run(self, run: ExperimentRun) -> None:
        """Save a completed run."""
        self.runs.append(run)
        
        # Persist to disk if tracking_dir is set
        if self.tracking_dir:
            self._save_runs()
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a run by ID."""
        for run in self.runs:
            if run.run_id == run_id:
                return run
        return None
    
    def get_runs_by_experiment(self, experiment_name: str) -> List[ExperimentRun]:
        """Get all runs for an experiment."""
        return [r for r in self.runs if r.experiment_name == experiment_name]
    
    def get_runs_by_tag(self, tag: str) -> List[ExperimentRun]:
        """Get all runs with a specific tag."""
        return [r for r in self.runs if tag in r.tags]
    
    def list_experiments(self) -> List[str]:
        """List all unique experiment names."""
        return list(set(r.experiment_name for r in self.runs))
    
    def get_best_run(self, metric: str, 
                     experiment_name: Optional[str] = None,
                     higher_is_better: bool = True) -> Optional[ExperimentRun]:
        """
        Get the best run by a specific metric.
        
        Args:
            metric: Metric name to compare
            experiment_name: Filter by experiment name (optional)
            higher_is_better: Whether higher values are better
            
        Returns:
            Best run or None
        """
        runs = self.runs
        if experiment_name:
            runs = self.get_runs_by_experiment(experiment_name)
        
        valid_runs = [r for r in runs if metric in r.metrics]
        
        if not valid_runs:
            return None
        
        if higher_is_better:
            return max(valid_runs, key=lambda r: r.metrics[metric])
        else:
            return min(valid_runs, key=lambda r: r.metrics[metric])
    
    def compare_runs(self, run_ids: Optional[List[str]] = None,
                     experiment_name: Optional[str] = None) -> str:
        """
        Generate comparison of runs.
        
        Args:
            run_ids: Specific run IDs to compare
            experiment_name: Filter by experiment name
            
        Returns:
            Formatted comparison string
        """
        runs = self.runs
        
        if run_ids:
            runs = [r for r in runs if r.run_id in run_ids]
        elif experiment_name:
            runs = self.get_runs_by_experiment(experiment_name)
        
        if not runs:
            return "No runs to compare."
        
        # Collect all metrics and params
        all_metrics = set()
        all_params = set()
        for run in runs:
            all_metrics.update(run.metrics.keys())
            all_params.update(run.parameters.keys())
        
        # Build comparison table
        lines = [
            f"\n{'='*80}",
            f"RUN COMPARISON - {self.project_name}",
            f"{'='*80}",
        ]
        
        # Run info
        lines.append(f"\n{'Run ID':<25} | {'Experiment':<20} | {'Status':<10}")
        lines.append("-" * 60)
        for run in runs:
            lines.append(f"{run.run_id:<25} | {run.experiment_name:<20} | {run.status:<10}")
        
        # Parameters
        if all_params:
            lines.append("\n--- Parameters ---")
            lines.append(f"{'Parameter':<25} | " + " | ".join(f"{r.run_id[:15]:<15}" for r in runs))
            lines.append("-" * 80)
            
            for param in sorted(all_params):
                values = []
                for run in runs:
                    val = run.parameters.get(param, 'N/A')
                    val_str = str(val)[:15]
                    values.append(f"{val_str:<15}")
                lines.append(f"{param:<25} | " + " | ".join(values))
        
        # Metrics
        if all_metrics:
            lines.append("\n--- Metrics ---")
            lines.append(f"{'Metric':<25} | " + " | ".join(f"{r.run_id[:15]:<15}" for r in runs))
            lines.append("-" * 80)
            
            for metric in sorted(all_metrics):
                values = []
                for run in runs:
                    val = run.metrics.get(metric, 'N/A')
                    if isinstance(val, float):
                        values.append(f"{val:<15.4f}")
                    else:
                        values.append(f"{str(val):<15}")
                lines.append(f"{metric:<25} | " + " | ".join(values))
        
        lines.append(f"{'='*80}\n")
        return '\n'.join(lines)
    
    def summary(self) -> str:
        """Generate a summary of all tracked experiments."""
        lines = [
            f"\n{'='*60}",
            f"EXPERIMENT TRACKER SUMMARY: {self.project_name}",
            f"{'='*60}",
            f"Total Runs: {len(self.runs)}",
            f"Experiments: {len(self.list_experiments())}",
        ]
        
        for exp_name in self.list_experiments():
            exp_runs = self.get_runs_by_experiment(exp_name)
            completed = sum(1 for r in exp_runs if r.status == 'completed')
            lines.append(f"\n  {exp_name}:")
            lines.append(f"    Total Runs: {len(exp_runs)}")
            lines.append(f"    Completed: {completed}")
            
            # Get common metrics
            if exp_runs:
                metrics = set()
                for run in exp_runs:
                    metrics.update(run.metrics.keys())
                
                for metric in sorted(metrics):
                    values = [r.metrics.get(metric) for r in exp_runs 
                             if metric in r.metrics]
                    if values:
                        from pyeval.utils.math_ops import mean, std
                        avg = mean(values)
                        stddev = std(values) if len(values) > 1 else 0
                        lines.append(f"    {metric}: {avg:.4f} ± {stddev:.4f}")
        
        lines.append(f"{'='*60}\n")
        return '\n'.join(lines)
    
    def _save_runs(self) -> None:
        """Save runs to disk."""
        if not self.tracking_dir:
            return
        
        os.makedirs(self.tracking_dir, exist_ok=True)
        filepath = os.path.join(self.tracking_dir, f"{self.project_name}_runs.json")
        
        data = {
            'project_name': self.project_name,
            'saved_at': datetime.now().isoformat(),
            'runs': [r.to_dict() for r in self.runs]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_runs(self) -> None:
        """Load runs from disk."""
        if not self.tracking_dir:
            return
        
        filepath = os.path.join(self.tracking_dir, f"{self.project_name}_runs.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.runs = [ExperimentRun.from_dict(r) for r in data.get('runs', [])]
            self._run_counter = len(self.runs)
    
    def export(self, filepath: str) -> None:
        """Export all runs to JSON file."""
        data = {
            'project_name': self.project_name,
            'exported_at': datetime.now().isoformat(),
            'runs': [r.to_dict() for r in self.runs]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentTracker':
        """Load tracker from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tracker = cls(data['project_name'])
        tracker.runs = [ExperimentRun.from_dict(r) for r in data.get('runs', [])]
        tracker._run_counter = len(tracker.runs)
        
        return tracker
    
    def delete_run(self, run_id: str) -> bool:
        """Delete a run by ID."""
        for i, run in enumerate(self.runs):
            if run.run_id == run_id:
                self.runs.pop(i)
                if self.tracking_dir:
                    self._save_runs()
                return True
        return False
    
    def clear_all(self) -> None:
        """Clear all runs."""
        self.runs = []
        self._run_counter = 0
        if self.tracking_dir:
            self._save_runs()


# =============================================================================
# Metric Logger (Context Manager)
# =============================================================================

class MetricLogger:
    """
    Context manager for logging metrics during training/evaluation.
    
    Example:
        tracker = ExperimentTracker("my_project")
        
        with MetricLogger(tracker, "training_run") as logger:
            logger.log_param('epochs', 10)
            for epoch in range(10):
                loss = train_epoch()
                logger.log_metric('loss', loss, step=epoch)
            logger.log_metric('final_accuracy', evaluate())
    """
    
    def __init__(self, tracker: ExperimentTracker, experiment_name: str,
                 tags: Optional[List[str]] = None):
        self.tracker = tracker
        self.experiment_name = experiment_name
        self.tags = tags
        self.run: Optional[ExperimentRun] = None
    
    def __enter__(self) -> ExperimentRun:
        """Start the run."""
        self.run = self.tracker.create_run(self.experiment_name, self.tags)
        return self.run
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the run and save."""
        if self.run:
            if exc_type is not None:
                self.run.set_status('failed')
                self.run.notes = f"Error: {exc_val}"
            else:
                self.run.set_status('completed')
            
            self.tracker.save_run(self.run)
        
        return False  # Don't suppress exceptions
