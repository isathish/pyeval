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
- Design Patterns (Strategy, Factory, Builder, Observer)
- Decorators (Timing, Caching, Validation, Logging)
- Callbacks (Progress, Thresholds, History)
- Validators (Type, Range, Schema)
- Pipelines (Fluent API for evaluation workflows)
- Functional Utilities (Monads, Composition, Currying)
- Aggregators (Statistical, Cross-Validation, Ensemble)

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

# Decorators
from pyeval.decorators import (
    timed,
    timed_result,
    Timer,
    memoize,
    lru_cache,
    validate_inputs,
    require_same_length,
    require_non_empty,
    check_range,
    retry,
    fallback,
    logged,
    deprecated,
    compose,
    pipe,
    partial_metric,
    vectorize,
    batch_process,
    ensure_list,
    ensure_float,
    MetricRegistry,
    metric_registry,
    MetricLogger,
)

# Design Patterns
from pyeval.patterns import (
    MetricStrategy,
    AccuracyStrategy,
    PrecisionStrategy,
    RecallStrategy,
    F1Strategy,
    MetricCalculator,
    MetricType,
    MetricFactory,
    metric_factory,
    EvaluationPipelineBuilder,
    EvaluationPipeline,
    Event,
    EventData,
    EventObserver,
    LoggingObserver,
    ProgressObserver,
    MetricSubject,
    MetricComponent,
    SingleMetric,
    CompositeMetric,
    ValidationHandler,
    create_validation_chain,
    EvaluationTemplate,
    ClassificationEvaluation,
)

# Validators
from pyeval.validators import (
    ValidationResult,
    Validator,
    TypeValidator,
    ListValidator,
    DictValidator,
    NumericValidator,
    ProbabilityValidator,
    PositiveValidator,
    IntegerValidator,
    StringValidator,
    NonEmptyStringValidator,
    AllOf,
    AnyOf,
    Optional as OptionalValidator,
    PredictionValidator,
    ProbabilityArrayValidator,
    ConfusionMatrixValidator,
    validate_predictions,
    validate_probabilities,
    validate_range,
    FieldSchema,
    SchemaValidator,
)

# Callbacks
from pyeval.callbacks import (
    CallbackEvent,
    CallbackContext,
    Callback,
    CallbackManager,
    LoggingCallback,
    ProgressCallback,
    ThresholdCallback,
    HistoryCallback,
    EarlyStoppingCallback,
    AggregationCallback,
    CompositeCallback,
    LambdaCallback,
    CallbackEvaluator,
)

# Pipelines
from pyeval.pipeline import (
    StepType,
    StepResult,
    Pipeline,
    PipelineResult,
    create_classification_pipeline,
    create_regression_pipeline,
    create_nlp_pipeline,
    PipelineRegistry,
    pipeline_registry,
)

# Functional utilities
from pyeval.functional import (
    Result,
    Option,
    try_catch,
    curry,
    identity,
    constant,
    flip,
    map_list,
    filter_list,
    reduce_list,
    fold_left,
    fold_right,
    zip_with,
    flat_map,
    group_by,
    partition,
    take,
    drop,
    take_while,
    drop_while,
    apply_metric,
    combine_metrics,
    threshold_metric,
    average_metric,
    weighted_average_metric,
)

# Aggregators
from pyeval.aggregators import (
    Aggregator,
    MeanAggregator,
    WeightedMeanAggregator,
    MedianAggregator,
    MinAggregator,
    MaxAggregator,
    SumAggregator,
    StdAggregator,
    VarianceAggregator,
    PercentileAggregator,
    GeometricMeanAggregator,
    HarmonicMeanAggregator,
    TrimmedMeanAggregator,
    WinsorizedMeanAggregator,
    AggregationResult,
    MetricAggregator,
    FoldResult,
    CrossValidationAggregator,
    EnsembleAggregator,
    create_aggregator,
)

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
    
    # Decorators
    "timed",
    "timed_result",
    "Timer",
    "memoize",
    "lru_cache",
    "validate_inputs",
    "require_same_length",
    "require_non_empty",
    "check_range",
    "retry",
    "fallback",
    "logged",
    "deprecated",
    "compose",
    "pipe",
    "partial_metric",
    "vectorize",
    "batch_process",
    "ensure_list",
    "ensure_float",
    "MetricRegistry",
    "metric_registry",
    "MetricLogger",
    
    # Design Patterns
    "MetricStrategy",
    "AccuracyStrategy",
    "PrecisionStrategy",
    "RecallStrategy",
    "F1Strategy",
    "MetricCalculator",
    "MetricType",
    "MetricFactory",
    "metric_factory",
    "EvaluationPipelineBuilder",
    "EvaluationPipeline",
    "Event",
    "EventData",
    "EventObserver",
    "LoggingObserver",
    "ProgressObserver",
    "MetricSubject",
    "MetricComponent",
    "SingleMetric",
    "CompositeMetric",
    "ValidationHandler",
    "create_validation_chain",
    "EvaluationTemplate",
    "ClassificationEvaluation",
    
    # Validators
    "ValidationResult",
    "Validator",
    "TypeValidator",
    "ListValidator",
    "DictValidator",
    "NumericValidator",
    "ProbabilityValidator",
    "PositiveValidator",
    "IntegerValidator",
    "StringValidator",
    "NonEmptyStringValidator",
    "AllOf",
    "AnyOf",
    "OptionalValidator",
    "PredictionValidator",
    "ProbabilityArrayValidator",
    "ConfusionMatrixValidator",
    "validate_predictions",
    "validate_probabilities",
    "validate_range",
    "FieldSchema",
    "SchemaValidator",
    
    # Callbacks
    "CallbackEvent",
    "CallbackContext",
    "Callback",
    "CallbackManager",
    "LoggingCallback",
    "ProgressCallback",
    "ThresholdCallback",
    "HistoryCallback",
    "EarlyStoppingCallback",
    "AggregationCallback",
    "CompositeCallback",
    "LambdaCallback",
    "CallbackEvaluator",
    
    # Pipelines
    "StepType",
    "StepResult",
    "Pipeline",
    "PipelineResult",
    "create_classification_pipeline",
    "create_regression_pipeline",
    "create_nlp_pipeline",
    "PipelineRegistry",
    "pipeline_registry",
    
    # Functional utilities
    "Result",
    "Option",
    "try_catch",
    "curry",
    "identity",
    "constant",
    "flip",
    "map_list",
    "filter_list",
    "reduce_list",
    "fold_left",
    "fold_right",
    "zip_with",
    "flat_map",
    "group_by",
    "partition",
    "take",
    "drop",
    "take_while",
    "drop_while",
    "apply_metric",
    "combine_metrics",
    "threshold_metric",
    "average_metric",
    "weighted_average_metric",
    
    # Aggregators
    "Aggregator",
    "MeanAggregator",
    "WeightedMeanAggregator",
    "MedianAggregator",
    "MinAggregator",
    "MaxAggregator",
    "SumAggregator",
    "StdAggregator",
    "VarianceAggregator",
    "PercentileAggregator",
    "GeometricMeanAggregator",
    "HarmonicMeanAggregator",
    "TrimmedMeanAggregator",
    "WinsorizedMeanAggregator",
    "AggregationResult",
    "MetricAggregator",
    "FoldResult",
    "CrossValidationAggregator",
    "EnsembleAggregator",
    "create_aggregator",
]
