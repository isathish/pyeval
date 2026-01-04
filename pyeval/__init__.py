"""
PyEval - A Comprehensive Python Evaluation Package
===================================================

A pure Python evaluation library for ML, NLP, LLM, RAG, Fairness, Speech, 
and Recommendation systems - no third-party dependencies required.

Features:
---------
- Classical ML Metrics (Classification, Regression, Clustering)
- NLP Metrics (BLEU, ROUGE, METEOR)
- LLM Evaluation (Hallucination, Relevancy, Faithfulness, Toxicity)
- RAG Evaluation (Context Relevance, Answer Correctness)
- Fairness Metrics (Demographic Parity, Equalized Odds)
- Speech Metrics (WER, CER)
- Recommender Metrics (Precision@K, Recall@K, NDCG)
- Experiment Tracking & Reporting

Author: PyEval Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "PyEval Team"

# Core metrics
from pyeval.ml import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
    ClassificationMetrics,
    RegressionMetrics,
    ClusteringMetrics,
)

from pyeval.nlp import (
    bleu_score,
    rouge_score,
    meteor_score,
    NLPMetrics,
)

from pyeval.llm import (
    hallucination_score,
    answer_relevancy,
    faithfulness_score,
    toxicity_score,
    coherence_score,
    LLMMetrics,
)

from pyeval.rag import (
    context_relevance,
    answer_correctness,
    retrieval_precision,
    retrieval_recall,
    RAGMetrics,
)

from pyeval.fairness import (
    demographic_parity,
    equalized_odds,
    disparate_impact,
    statistical_parity_difference,
    FairnessMetrics,
)

from pyeval.speech import (
    word_error_rate,
    character_error_rate,
    SpeechMetrics,
)

from pyeval.recommender import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_average_precision,
    hit_rate,
    RecommenderMetrics,
)

# Unified evaluator
from pyeval.evaluator import Evaluator, EvaluationReport

# Experiment tracking
from pyeval.tracking import ExperimentTracker

__all__ = [
    # Version
    "__version__",
    # ML Metrics
    "accuracy_score",
    "precision_score", 
    "recall_score",
    "f1_score",
    "roc_auc_score",
    "confusion_matrix",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "silhouette_score",
    "ClassificationMetrics",
    "RegressionMetrics",
    "ClusteringMetrics",
    # NLP Metrics
    "bleu_score",
    "rouge_score",
    "meteor_score",
    "NLPMetrics",
    # LLM Metrics
    "hallucination_score",
    "answer_relevancy",
    "faithfulness_score",
    "toxicity_score",
    "coherence_score",
    "LLMMetrics",
    # RAG Metrics
    "context_relevance",
    "answer_correctness",
    "retrieval_precision",
    "retrieval_recall",
    "RAGMetrics",
    # Fairness Metrics
    "demographic_parity",
    "equalized_odds",
    "disparate_impact",
    "statistical_parity_difference",
    "FairnessMetrics",
    # Speech Metrics
    "word_error_rate",
    "character_error_rate",
    "SpeechMetrics",
    # Recommender Metrics
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "mean_average_precision",
    "hit_rate",
    "RecommenderMetrics",
    # Evaluator
    "Evaluator",
    "EvaluationReport",
    # Tracking
    "ExperimentTracker",
]
