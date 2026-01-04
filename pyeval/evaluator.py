"""
Unified Evaluator API - Pure Python Implementation
===================================================

A unified interface for evaluating models across different domains
with comprehensive reporting and experiment tracking.
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json


# =============================================================================
# Evaluation Report
# =============================================================================

@dataclass
class EvaluationReport:
    """
    Container for evaluation results with detailed reporting.
    """
    
    name: str
    domain: str  # ml, nlp, llm, rag, fairness, speech, recommender
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    samples: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric to the report."""
        self.metrics[name] = value
    
    def add_sample(self, sample: Dict[str, Any]) -> None:
        """Add a sample result to the report."""
        self.samples.append(sample)
    
    def summary(self) -> str:
        """Generate a text summary of the evaluation."""
        lines = [
            f"\n{'='*60}",
            f"EVALUATION REPORT: {self.name}",
            f"{'='*60}",
            f"Domain: {self.domain.upper()}",
            f"Timestamp: {self.timestamp}",
            f"\n--- Metrics ---"
        ]
        
        for name, value in sorted(self.metrics.items()):
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.4f}")
            else:
                lines.append(f"  {name}: {value}")
        
        if self.metadata:
            lines.append(f"\n--- Metadata ---")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        
        lines.append(f"{'='*60}\n")
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'name': self.name,
            'domain': self.domain,
            'timestamp': self.timestamp,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'samples': self.samples
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationReport':
        """Create report from dictionary."""
        return cls(
            name=data['name'],
            domain=data['domain'],
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            metrics=data.get('metrics', {}),
            metadata=data.get('metadata', {}),
            samples=data.get('samples', [])
        )


# =============================================================================
# Unified Evaluator
# =============================================================================

class Evaluator:
    """
    Unified evaluation interface for all domains.
    
    Example usage:
        evaluator = Evaluator()
        
        # ML Classification
        report = evaluator.evaluate_classification(
            y_true=[1, 0, 1, 1, 0],
            y_pred=[1, 0, 0, 1, 1]
        )
        
        # NLP Generation
        report = evaluator.evaluate_generation(
            references=["The cat sat on the mat"],
            hypotheses=["A cat is on the mat"]
        )
        
        # RAG Pipeline
        report = evaluator.evaluate_rag(
            queries=["What is Python?"],
            contexts=[["Python is a programming language"]],
            responses=["Python is a programming language"],
            ground_truths=["Python is a high-level programming language"]
        )
    """
    
    def __init__(self, name: str = "Evaluation"):
        self.name = name
        self.reports: List[EvaluationReport] = []
    
    # =========================================================================
    # ML Classification
    # =========================================================================
    
    def evaluate_classification(self, y_true: List[int], y_pred: List[int],
                                y_prob: Optional[List[float]] = None,
                                average: str = 'binary',
                                name: str = "Classification") -> EvaluationReport:
        """
        Evaluate classification model.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for ROC-AUC)
            average: Averaging method ('binary', 'macro', 'micro', 'weighted')
            name: Name for the evaluation report
            
        Returns:
            EvaluationReport with classification metrics
        """
        from pyeval.ml import (accuracy_score, precision_score, recall_score, 
                               f1_score, roc_auc_score, ClassificationMetrics)
        
        report = EvaluationReport(name=name, domain='ml')
        
        # Core metrics
        report.add_metric('accuracy', accuracy_score(y_true, y_pred))
        report.add_metric('precision', precision_score(y_true, y_pred, average=average))
        report.add_metric('recall', recall_score(y_true, y_pred, average=average))
        report.add_metric('f1', f1_score(y_true, y_pred, average=average))
        
        # ROC-AUC if probabilities provided
        if y_prob is not None:
            report.add_metric('roc_auc', roc_auc_score(y_true, y_prob))
        
        report.metadata['samples'] = len(y_true)
        report.metadata['average'] = average
        
        self.reports.append(report)
        return report
    
    # =========================================================================
    # ML Regression
    # =========================================================================
    
    def evaluate_regression(self, y_true: List[float], y_pred: List[float],
                            name: str = "Regression") -> EvaluationReport:
        """
        Evaluate regression model.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            name: Name for the evaluation report
            
        Returns:
            EvaluationReport with regression metrics
        """
        from pyeval.ml import (mean_squared_error, root_mean_squared_error,
                               mean_absolute_error, r2_score, RegressionMetrics)
        
        report = EvaluationReport(name=name, domain='ml')
        
        report.add_metric('mse', mean_squared_error(y_true, y_pred))
        report.add_metric('rmse', root_mean_squared_error(y_true, y_pred))
        report.add_metric('mae', mean_absolute_error(y_true, y_pred))
        report.add_metric('r2', r2_score(y_true, y_pred))
        
        report.metadata['samples'] = len(y_true)
        
        self.reports.append(report)
        return report
    
    # =========================================================================
    # NLP Generation
    # =========================================================================
    
    def evaluate_generation(self, references: List[str], hypotheses: List[str],
                            name: str = "Generation") -> EvaluationReport:
        """
        Evaluate text generation quality.
        
        Args:
            references: Reference texts
            hypotheses: Generated texts
            name: Name for the evaluation report
            
        Returns:
            EvaluationReport with NLP metrics
        """
        from pyeval.nlp import (sentence_bleu, rouge_score, meteor_score,
                                distinct_n, NLPMetrics)
        
        report = EvaluationReport(name=name, domain='nlp')
        
        # Calculate average metrics
        bleu_scores = []
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        meteor_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            bleu_scores.append(sentence_bleu(ref, hyp))
            
            rouge = rouge_score(ref, hyp)
            for key in rouge_scores:
                rouge_scores[key].append(rouge.get(key, {}).get('f1', 0))
            
            meteor_result = meteor_score(ref, hyp)
            meteor_scores.append(meteor_result.get('meteor', 0) if isinstance(meteor_result, dict) else meteor_result)
        
        from pyeval.utils.math_ops import mean
        
        report.add_metric('bleu', mean(bleu_scores))
        report.add_metric('rouge1_f1', mean(rouge_scores['rouge1']))
        report.add_metric('rouge2_f1', mean(rouge_scores['rouge2']))
        report.add_metric('rougeL_f1', mean(rouge_scores['rougeL']))
        report.add_metric('meteor', mean(meteor_scores))
        
        # Diversity metrics
        report.add_metric('distinct_1', distinct_n(hypotheses, n=1))
        report.add_metric('distinct_2', distinct_n(hypotheses, n=2))
        
        report.metadata['samples'] = len(references)
        
        self.reports.append(report)
        return report
    
    # =========================================================================
    # LLM Evaluation
    # =========================================================================
    
    def evaluate_llm(self, queries: List[str], responses: List[str],
                     contexts: Optional[List[str]] = None,
                     name: str = "LLM") -> EvaluationReport:
        """
        Evaluate LLM responses.
        
        Args:
            queries: User queries/prompts
            responses: LLM responses
            contexts: Optional source contexts
            name: Name for the evaluation report
            
        Returns:
            EvaluationReport with LLM evaluation metrics
        """
        from pyeval.llm import (toxicity_score, coherence_score, 
                                fluency_score, LLMMetrics)
        
        report = EvaluationReport(name=name, domain='llm')
        
        from pyeval.utils.math_ops import mean
        
        toxicity_scores = [toxicity_score(resp) for resp in responses]
        coherence_scores = [coherence_score(resp, query) 
                          for query, resp in zip(queries, responses)]
        fluency_scores = [fluency_score(resp) for resp in responses]
        
        report.add_metric('toxicity', mean(toxicity_scores))
        report.add_metric('coherence', mean(coherence_scores))
        report.add_metric('fluency', mean(fluency_scores))
        report.add_metric('avg_response_length', 
                         mean([len(r.split()) for r in responses]))
        
        # If contexts provided, check faithfulness
        if contexts:
            from pyeval.llm import faithfulness_score
            faith_scores = [faithfulness_score(resp, ctx) 
                          for resp, ctx in zip(responses, contexts)]
            report.add_metric('faithfulness', mean(faith_scores))
        
        report.metadata['samples'] = len(responses)
        
        self.reports.append(report)
        return report
    
    # =========================================================================
    # RAG Evaluation
    # =========================================================================
    
    def evaluate_rag(self, queries: List[str], 
                     contexts: List[List[str]], 
                     responses: List[str],
                     ground_truths: Optional[List[str]] = None,
                     name: str = "RAG") -> EvaluationReport:
        """
        Evaluate RAG pipeline.
        
        Args:
            queries: User queries
            contexts: Retrieved contexts for each query
            responses: Generated responses
            ground_truths: Optional ground truth answers
            name: Name for the evaluation report
            
        Returns:
            EvaluationReport with RAG metrics
        """
        from pyeval.rag import (context_relevance, groundedness_score,
                                rag_faithfulness, RAGMetrics)
        
        report = EvaluationReport(name=name, domain='rag')
        
        from pyeval.utils.math_ops import mean
        
        # Context relevance
        ctx_rel_scores = [context_relevance(query, ctx) 
                         for query, ctx in zip(queries, contexts)]
        report.add_metric('context_relevance', mean(ctx_rel_scores))
        
        # Groundedness
        ground_scores = []
        for resp, ctx in zip(responses, contexts):
            ctx_text = ' '.join(ctx)
            ground_scores.append(groundedness_score(resp, ctx_text))
        report.add_metric('groundedness', mean(ground_scores))
        
        # Faithfulness
        faith_scores = []
        for resp, ctx in zip(responses, contexts):
            faith_scores.append(rag_faithfulness(resp, ctx))
        report.add_metric('faithfulness', mean(faith_scores))
        
        # Answer correctness if ground truths provided
        if ground_truths:
            from pyeval.rag import answer_correctness
            correct_scores = [answer_correctness(resp, gt) 
                             for resp, gt in zip(responses, ground_truths)]
            report.add_metric('answer_correctness', mean(correct_scores))
        
        report.metadata['samples'] = len(queries)
        report.metadata['avg_contexts_per_query'] = mean([len(c) for c in contexts])
        
        self.reports.append(report)
        return report
    
    # =========================================================================
    # Fairness Evaluation
    # =========================================================================
    
    def evaluate_fairness(self, y_true: List[int], y_pred: List[int],
                          sensitive_features: List[Any],
                          name: str = "Fairness") -> EvaluationReport:
        """
        Evaluate model fairness.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive_features: Sensitive attribute for each sample
            name: Name for the evaluation report
            
        Returns:
            EvaluationReport with fairness metrics
        """
        from pyeval.fairness import (demographic_parity, equalized_odds,
                                     disparate_impact, FairnessMetrics)
        
        report = EvaluationReport(name=name, domain='fairness')
        
        report.add_metric('demographic_parity_diff', 
                         demographic_parity(y_pred, sensitive_features))
        report.add_metric('equalized_odds_diff', 
                         equalized_odds(y_true, y_pred, sensitive_features))
        report.add_metric('disparate_impact', 
                         disparate_impact(y_pred, sensitive_features))
        
        report.metadata['samples'] = len(y_true)
        report.metadata['unique_groups'] = len(set(sensitive_features))
        
        self.reports.append(report)
        return report
    
    # =========================================================================
    # Speech Evaluation
    # =========================================================================
    
    def evaluate_speech(self, references: List[str], hypotheses: List[str],
                        name: str = "Speech") -> EvaluationReport:
        """
        Evaluate speech recognition.
        
        Args:
            references: Ground truth transcriptions
            hypotheses: Model transcriptions
            name: Name for the evaluation report
            
        Returns:
            EvaluationReport with speech metrics
        """
        from pyeval.speech import (word_error_rate, character_error_rate,
                                   sentence_error_rate, SpeechMetrics)
        
        report = EvaluationReport(name=name, domain='speech')
        
        from pyeval.utils.math_ops import mean
        
        wer_scores = [word_error_rate(ref, hyp) 
                     for ref, hyp in zip(references, hypotheses)]
        cer_scores = [character_error_rate(ref, hyp) 
                     for ref, hyp in zip(references, hypotheses)]
        
        report.add_metric('wer', mean(wer_scores))
        report.add_metric('cer', mean(cer_scores))
        report.add_metric('ser', sentence_error_rate(references, hypotheses))
        
        report.metadata['samples'] = len(references)
        
        self.reports.append(report)
        return report
    
    # =========================================================================
    # Recommender Evaluation
    # =========================================================================
    
    def evaluate_recommender(self, recommendations: List[List[Any]], 
                             relevants: List[List[Any]],
                             k: int = 10,
                             name: str = "Recommender") -> EvaluationReport:
        """
        Evaluate recommender system.
        
        Args:
            recommendations: Recommended items for each user
            relevants: Relevant items for each user
            k: Number of top recommendations to consider
            name: Name for the evaluation report
            
        Returns:
            EvaluationReport with recommender metrics
        """
        from pyeval.recommender import (mean_precision_at_k, mean_recall_at_k,
                                        mean_ndcg_at_k, mean_average_precision,
                                        mean_hit_rate, mean_reciprocal_rank,
                                        RecommenderMetrics)
        
        report = EvaluationReport(name=name, domain='recommender')
        
        report.add_metric(f'precision@{k}', 
                         mean_precision_at_k(recommendations, relevants, k))
        report.add_metric(f'recall@{k}', 
                         mean_recall_at_k(recommendations, relevants, k))
        report.add_metric(f'ndcg@{k}', 
                         mean_ndcg_at_k(recommendations, relevants, k))
        report.add_metric('map', mean_average_precision(recommendations, relevants))
        report.add_metric(f'hit_rate@{k}', 
                         mean_hit_rate(recommendations, relevants, k))
        report.add_metric('mrr', mean_reciprocal_rank(recommendations, relevants))
        
        report.metadata['k'] = k
        report.metadata['num_users'] = len(recommendations)
        
        self.reports.append(report)
        return report
    
    # =========================================================================
    # Custom Evaluation
    # =========================================================================
    
    def evaluate_custom(self, metrics: Dict[str, Callable[..., float]],
                        data: Dict[str, Any],
                        domain: str = "custom",
                        name: str = "Custom") -> EvaluationReport:
        """
        Run custom evaluation with user-defined metrics.
        
        Args:
            metrics: Dict mapping metric names to functions
            data: Data to pass to metric functions
            domain: Domain name
            name: Name for the evaluation report
            
        Returns:
            EvaluationReport with custom metrics
        """
        report = EvaluationReport(name=name, domain=domain)
        
        for metric_name, metric_fn in metrics.items():
            try:
                value = metric_fn(**data)
                report.add_metric(metric_name, value)
            except Exception as e:
                report.metadata[f'{metric_name}_error'] = str(e)
        
        self.reports.append(report)
        return report
    
    # =========================================================================
    # Report Management
    # =========================================================================
    
    def get_all_reports(self) -> List[EvaluationReport]:
        """Get all evaluation reports."""
        return self.reports
    
    def get_reports_by_domain(self, domain: str) -> List[EvaluationReport]:
        """Get reports filtered by domain."""
        return [r for r in self.reports if r.domain == domain]
    
    def compare_reports(self, report_names: Optional[List[str]] = None) -> str:
        """
        Generate comparison of reports.
        
        Args:
            report_names: Names of reports to compare (all if None)
            
        Returns:
            Formatted comparison string
        """
        reports = self.reports
        if report_names:
            reports = [r for r in reports if r.name in report_names]
        
        if not reports:
            return "No reports to compare."
        
        # Collect all metric names
        all_metrics = set()
        for report in reports:
            all_metrics.update(report.metrics.keys())
        
        # Build comparison table
        lines = [
            f"\n{'='*80}",
            f"COMPARISON: {', '.join(r.name for r in reports)}",
            f"{'='*80}",
            f"{'Metric':<25} | " + " | ".join(f"{r.name:<15}" for r in reports)
        ]
        lines.append("-" * 80)
        
        for metric in sorted(all_metrics):
            values = []
            for report in reports:
                val = report.metrics.get(metric, 'N/A')
                if isinstance(val, float):
                    values.append(f"{val:<15.4f}")
                else:
                    values.append(f"{str(val):<15}")
            lines.append(f"{metric:<25} | " + " | ".join(values))
        
        lines.append(f"{'='*80}\n")
        return '\n'.join(lines)
    
    def export_all(self, filepath: str) -> None:
        """Export all reports to JSON file."""
        data = {
            'evaluator_name': self.name,
            'exported_at': datetime.now().isoformat(),
            'reports': [r.to_dict() for r in self.reports]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear_reports(self) -> None:
        """Clear all stored reports."""
        self.reports = []


# =============================================================================
# Experiment Tracker
# =============================================================================

@dataclass
class Experiment:
    """Container for a single experiment."""
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter."""
        self.parameters[key] = value
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self.parameters.update(params)
    
    def log_metric(self, key: str, value: float) -> None:
        """Log a metric."""
        self.metrics[key] = value
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple metrics."""
        self.metrics.update(metrics)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'artifacts': self.artifacts,
            'tags': self.tags
        }


class ExperimentTracker:
    """
    Simple experiment tracking system.
    
    Example usage:
        tracker = ExperimentTracker("my_project")
        
        with tracker.start_experiment("experiment_1") as exp:
            exp.log_params({'learning_rate': 0.01, 'epochs': 100})
            # ... run training ...
            exp.log_metrics({'accuracy': 0.95, 'loss': 0.05})
        
        tracker.compare_experiments()
    """
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.experiments: List[Experiment] = []
        self._active_experiment: Optional[Experiment] = None
    
    def start_experiment(self, name: str, 
                         description: str = "") -> Experiment:
        """
        Start a new experiment.
        
        Args:
            name: Experiment name
            description: Optional description
            
        Returns:
            Experiment context manager
        """
        exp = Experiment(name=name, description=description)
        self._active_experiment = exp
        return exp
    
    def end_experiment(self) -> None:
        """End the current experiment and save it."""
        if self._active_experiment:
            self.experiments.append(self._active_experiment)
            self._active_experiment = None
    
    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get an experiment by name."""
        for exp in self.experiments:
            if exp.name == name:
                return exp
        return None
    
    def list_experiments(self) -> List[str]:
        """List all experiment names."""
        return [exp.name for exp in self.experiments]
    
    def compare_experiments(self, names: Optional[List[str]] = None) -> str:
        """
        Compare experiments.
        
        Args:
            names: Experiment names to compare (all if None)
            
        Returns:
            Formatted comparison string
        """
        experiments = self.experiments
        if names:
            experiments = [e for e in experiments if e.name in names]
        
        if not experiments:
            return "No experiments to compare."
        
        # Collect all metric and param names
        all_metrics = set()
        all_params = set()
        for exp in experiments:
            all_metrics.update(exp.metrics.keys())
            all_params.update(exp.parameters.keys())
        
        lines = [
            f"\n{'='*80}",
            f"EXPERIMENT COMPARISON: {self.project_name}",
            f"{'='*80}",
            f"\n--- Parameters ---",
            f"{'Parameter':<25} | " + " | ".join(f"{e.name:<15}" for e in experiments)
        ]
        lines.append("-" * 80)
        
        for param in sorted(all_params):
            values = []
            for exp in experiments:
                val = exp.parameters.get(param, 'N/A')
                values.append(f"{str(val):<15}")
            lines.append(f"{param:<25} | " + " | ".join(values))
        
        lines.append(f"\n--- Metrics ---")
        lines.append(f"{'Metric':<25} | " + " | ".join(f"{e.name:<15}" for e in experiments))
        lines.append("-" * 80)
        
        for metric in sorted(all_metrics):
            values = []
            for exp in experiments:
                val = exp.metrics.get(metric, 'N/A')
                if isinstance(val, float):
                    values.append(f"{val:<15.4f}")
                else:
                    values.append(f"{str(val):<15}")
            lines.append(f"{metric:<25} | " + " | ".join(values))
        
        lines.append(f"{'='*80}\n")
        return '\n'.join(lines)
    
    def get_best_experiment(self, metric: str, 
                            higher_is_better: bool = True) -> Optional[Experiment]:
        """
        Get the best experiment by a specific metric.
        
        Args:
            metric: Metric name to compare
            higher_is_better: Whether higher values are better
            
        Returns:
            Best experiment or None
        """
        valid_experiments = [e for e in self.experiments 
                           if metric in e.metrics]
        
        if not valid_experiments:
            return None
        
        if higher_is_better:
            return max(valid_experiments, key=lambda e: e.metrics[metric])
        else:
            return min(valid_experiments, key=lambda e: e.metrics[metric])
    
    def export(self, filepath: str) -> None:
        """Export all experiments to JSON file."""
        data = {
            'project_name': self.project_name,
            'exported_at': datetime.now().isoformat(),
            'experiments': [e.to_dict() for e in self.experiments]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentTracker':
        """Load experiments from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tracker = cls(data['project_name'])
        
        for exp_data in data['experiments']:
            exp = Experiment(
                name=exp_data['name'],
                description=exp_data.get('description', ''),
                created_at=exp_data.get('created_at', ''),
                parameters=exp_data.get('parameters', {}),
                metrics=exp_data.get('metrics', {}),
                artifacts=exp_data.get('artifacts', {}),
                tags=exp_data.get('tags', [])
            )
            tracker.experiments.append(exp)
        
        return tracker
