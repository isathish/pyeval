# PyEval 📊

**A Comprehensive Pure Python Evaluation Framework**

PyEval is a complete evaluation library for Machine Learning, NLP, LLM, RAG, Fairness, Speech, and Recommender systems — all without any third-party dependencies.

## ✨ Features

| Category | Metrics |
|----------|---------|
| **ML Classification** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, **Balanced Accuracy, Log Loss, Brier Score, Hamming Loss, Jaccard, Top-K Accuracy, Expected Calibration Error** |
| **ML Regression** | MSE, RMSE, MAE, MAPE, R², **MSLE, Symmetric MAPE, Huber Loss, Quantile Loss, Normalized RMSE** |
| **Clustering** | Silhouette Score, Davies-Bouldin, Calinski-Harabasz, **Adjusted Rand Index, Normalized Mutual Info, Homogeneity, Completeness, V-Measure, Fowlkes-Mallows** |
| **NLP** | BLEU, ROUGE (1/2/L), METEOR, TER, Distinct-N, **chrF, Text Entropy, Repetition Ratio, Compression Ratio, Coverage, Density, Lexical Diversity** |
| **LLM Evaluation** | Toxicity, Hallucination, Relevancy, Faithfulness, Coherence, **Bias Detection, Instruction Following, Multi-Turn Coherence, Summarization Quality, Response Diversity** |
| **RAG Evaluation** | Context Relevance, Answer Correctness, Groundedness, **Context Entity Recall, Answer Attribution, Context Utilization, Question-Answer Relevance, RAG Pipeline Score** |
| **Fairness** | Demographic Parity, Equalized Odds, Disparate Impact |
| **Speech** | WER, CER, MER, WIL, SER, **Slot Error Rate, Intent Accuracy, Phoneme Error Rate, Diarization Error Rate, Keyword Spotting, MOS, Fluency** |
| **Recommender** | Precision@K, Recall@K, NDCG, MAP, Hit Rate, MRR, **Serendipity, Gini Index, Inter-List Diversity, Entropy Diversity, Ranking Correlation** |
| **Statistical Tests** | **Bootstrap CI, Paired t-test, Independent t-test, Wilcoxon, Mann-Whitney U, McNemar, Cohen's d, Hedges' g, Spearman** |
| **Visualization** | **ASCII Confusion Matrix, Classification Report, Bar Charts, Histograms, ROC/PR Curves, Sparklines** |

### 🛠️ Advanced Features

| Feature | Description |
|---------|-------------|
| **Design Patterns** | Strategy, Factory, Builder, Observer, Composite, Chain of Responsibility |
| **Decorators** | @timed, @memoize, @lru_cache, @retry, @fallback, @logged, @deprecated |
| **Validators** | Type, List, Numeric, Schema validators with composition |
| **Callbacks** | Progress, Threshold alerts, History tracking, Early stopping |
| **Pipelines** | Fluent API for building evaluation workflows |
| **Functional** | Result/Option monads, curry, compose, pipe, higher-order functions |
| **Aggregators** | Statistical, Cross-validation, Ensemble aggregation |

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pyeval.git
cd pyeval

# No dependencies to install!
# Just import and use
```

## 📖 Quick Start

### ML Classification

```python
from pyeval import accuracy_score, precision_score, recall_score, f1_score

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
```

### NLP Metrics (BLEU, ROUGE)

```python
from pyeval import bleu_score, rouge_score, meteor_score

reference = "The quick brown fox jumps over the lazy dog"
hypothesis = "A fast brown fox leaps over a lazy dog"

# BLEU (pass reference strings, candidate string)
bleu = bleu_score([reference], hypothesis)
print(f"BLEU: {bleu['bleu']:.4f}")

# ROUGE
rouge = rouge_score(reference, hypothesis)
print(f"ROUGE-1 F1: {rouge['rouge1']['f1']:.4f}")
print(f"ROUGE-L F1: {rouge['rougeL']['f1']:.4f}")

# METEOR (single reference string, candidate string)
meteor = meteor_score(reference, hypothesis)
print(f"METEOR: {meteor['meteor']:.4f}")
```

### LLM Evaluation

```python
from pyeval import (
    toxicity_score, coherence_score, 
    answer_relevancy, faithfulness_score,
    hallucination_score
)

query = "What is the capital of France?"
response = "The capital of France is Paris."
context = "France is a country in Europe. Its capital is Paris."

# All LLM functions return dictionaries with detailed metrics
toxicity = toxicity_score(response)
print(f"Toxicity:     {toxicity['toxicity']:.4f}")

coherence = coherence_score(response)
print(f"Coherence:    {coherence['coherence']:.4f}")

relevancy = answer_relevancy(query, response)
print(f"Relevancy:    {relevancy['relevancy']:.4f}")

faithful = faithfulness_score(response, context)
print(f"Faithfulness: {faithful['faithfulness']:.4f}")

hallucination = hallucination_score(response, context)
print(f"Hallucination: {hallucination['hallucination_score']:.4f}")
```

### RAG Evaluation

```python
from pyeval import (
    context_relevance, answer_correctness,
    retrieval_precision, retrieval_recall,
    RAGMetrics
)

query = "What are the benefits of exercise?"
contexts = [
    "Exercise improves cardiovascular health.",
    "Physical activity reduces stress."
]
response = "Exercise improves heart health and reduces stress."
ground_truth = "Exercise is good for heart health and mental well-being."

# Individual metrics (return dictionaries)
ctx_rel = context_relevance(query, contexts)
print(f"Context Relevance: {ctx_rel['overall_relevance']:.4f}")

ans_corr = answer_correctness(response, ground_truth)
print(f"Answer Correctness: {ans_corr['correctness']:.4f}")

# All-in-one (signature: question, answer, contexts, ground_truth_answer)
metrics = RAGMetrics.compute(query, response, contexts, ground_truth)
print(f"Context Relevance: {metrics.context_relevance:.4f}")
```

### Fairness Metrics

```python
from pyeval import demographic_parity, equalized_odds, disparate_impact

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

# All fairness functions return dictionaries with detailed metrics
dp = demographic_parity(y_pred, sensitive)
print(f"Demographic Parity: {dp['dp_difference']:.4f}")

eo = equalized_odds(y_true, y_pred, sensitive)
print(f"Equalized Odds:     {eo['eo_difference']:.4f}")

di = disparate_impact(y_pred, sensitive)
print(f"Disparate Impact:   {di['di_ratio']:.4f}")
```

### Speech Recognition

```python
from pyeval import word_error_rate, character_error_rate

reference = "the quick brown fox jumps over the lazy dog"
hypothesis = "the quick brown fox jumps over lazy dog"

# Speech metrics return dictionaries with error counts
wer = word_error_rate(reference, hypothesis)
print(f"WER: {wer['wer']:.4f}")

cer = character_error_rate(reference, hypothesis)
print(f"CER: {cer['cer']:.4f}")
```

### Recommender Systems

```python
from pyeval import precision_at_k, recall_at_k, ndcg_at_k

recommended = [101, 203, 45, 67, 89, 12, 34, 56]
relevant = [45, 89, 78, 123]

print(f"Precision@5: {precision_at_k(recommended, relevant, k=5):.4f}")
print(f"Recall@5:    {recall_at_k(recommended, relevant, k=5):.4f}")
print(f"NDCG@5:      {ndcg_at_k(recommended, relevant, k=5):.4f}")
```

### Statistical Testing

```python
from pyeval import (
    bootstrap_confidence_interval, paired_t_test, cohens_d,
    mcnemar_test, correlation_coefficient
)

# Bootstrap confidence interval
data = [0.85, 0.87, 0.86, 0.88, 0.84, 0.89, 0.87]
ci = bootstrap_confidence_interval(data, 'mean', confidence=0.95)
print(f"Mean: {ci['point_estimate']:.3f} CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

# Compare two models with paired t-test
model1_scores = [0.85, 0.87, 0.86, 0.88, 0.84]
model2_scores = [0.88, 0.89, 0.87, 0.90, 0.86]
result = paired_t_test(model1_scores, model2_scores)
print(f"t-statistic: {result['t_statistic']:.3f}, p-value: {result['p_value']:.4f}")

# Effect size
d = cohens_d(model1_scores, model2_scores)
print(f"Cohen's d: {d:.3f}")  # 0.2=small, 0.5=medium, 0.8=large

# McNemar test for classifier comparison
contingency = [[45, 15], [5, 35]]  # [[both correct, only A], [only B, both wrong]]
result = mcnemar_test(contingency)
print(f"McNemar chi²: {result['chi_square']:.3f}, p-value: {result['p_value']:.4f}")
```

### ASCII Visualizations

```python
from pyeval import (
    confusion_matrix_display, classification_report_display,
    horizontal_bar_chart, sparkline, progress_bar
)

# Display confusion matrix
matrix = [[45, 5], [10, 40]]
print(confusion_matrix_display(matrix, labels=['Negative', 'Positive']))

# Classification report
y_true = [0, 0, 1, 1, 2, 2]
y_pred = [0, 0, 1, 2, 2, 2]
print(classification_report_display(y_true, y_pred))

# Bar chart for metrics
metrics = {'Precision': 0.89, 'Recall': 0.85, 'F1': 0.87, 'Accuracy': 0.88}
print(horizontal_bar_chart(metrics, title="Model Performance"))

# Sparkline for training progress
losses = [0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26]
print(f"Training loss: {sparkline(losses)}")

# Progress bar
print(progress_bar(75, 100, prefix="Evaluation"))
```

## 🔧 Unified Evaluator

Use the unified `Evaluator` class for streamlined evaluation across domains:

```python
from pyeval import Evaluator

evaluator = Evaluator("My Evaluation")

# Classification
report = evaluator.evaluate_classification(y_true, y_pred)
print(report.summary())

# NLP Generation
report = evaluator.evaluate_generation(references, hypotheses)
print(report.summary())

# RAG Pipeline
report = evaluator.evaluate_rag(queries, contexts, responses, ground_truths)
print(report.summary())

# Compare all reports
print(evaluator.compare_reports())
```

## 📈 Experiment Tracking

Track your ML experiments:

```python
from pyeval import ExperimentTracker

tracker = ExperimentTracker("my_project")

# Create and log experiment run
run = tracker.create_run("experiment_v1")
run.log_params({"learning_rate": 0.01, "epochs": 100})
run.log_metrics({"accuracy": 0.95, "f1": 0.93})
run.set_status("completed")
tracker.save_run(run)

# Compare experiments
print(tracker.compare_runs())

# Get best run
best = tracker.get_best_run("accuracy")
```

## �️ Advanced Features

### Decorators

```python
from pyeval import timed, memoize, require_same_length, retry, logged

# Time function execution
@timed
def compute_metrics(data):
    return process(data)

# Cache results
@memoize
def expensive_metric(y_true, y_pred):
    return complex_computation(y_true, y_pred)

# Validate inputs
@require_same_length
def accuracy(y_true, y_pred):
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

# Retry on failure
@retry(max_attempts=3, delay=0.1)
def flaky_operation():
    return external_api_call()

# Log function calls
@logged
def tracked_metric(y_true, y_pred):
    return compute(y_true, y_pred)
```

### Design Patterns - Strategy Pattern

```python
from pyeval import MetricCalculator, AccuracyStrategy, F1Strategy

calculator = MetricCalculator(AccuracyStrategy())
accuracy = calculator.calculate(y_true, y_pred)

# Switch strategy
calculator.set_strategy(F1Strategy())
f1 = calculator.calculate(y_true, y_pred)
```

### Design Patterns - Factory Pattern

```python
from pyeval import MetricFactory, MetricType

factory = MetricFactory()
metric = factory.create(MetricType.ACCURACY)
score = metric.compute(y_true, y_pred)
```

### Design Patterns - Composite Pattern

```python
from pyeval import CompositeMetric, SingleMetric

# Compose multiple metrics
classification_metrics = CompositeMetric('classification')
classification_metrics.add(SingleMetric('accuracy', accuracy_score))
classification_metrics.add(SingleMetric('f1', f1_score))

# Compute all at once
results = classification_metrics.compute(y_true, y_pred)
# {'accuracy': 0.95, 'f1': 0.92}
```

### Validators

```python
from pyeval import (
    validate_predictions, validate_probabilities,
    TypeValidator, ListValidator, SchemaValidator, FieldSchema
)

# Quick validation
validate_predictions(y_true, y_pred)  # Raises if invalid
validate_probabilities([0.1, 0.3, 0.6], require_sum_one=True)

# Composable validators
validator = ListValidator(element_type=int, min_length=1, max_length=100)
result = validator.validate([1, 2, 3])
if result.is_valid:
    print("Valid!")

# Schema validation
schema = SchemaValidator([
    FieldSchema('y_true', ListValidator(min_length=1)),
    FieldSchema('threshold', NumericValidator(0, 1), required=False, default=0.5)
])
```

### Callbacks

```python
from pyeval import (
    CallbackManager, LoggingCallback, ProgressCallback,
    ThresholdCallback, HistoryCallback, EarlyStoppingCallback
)

# Progress tracking
progress = ProgressCallback(total_metrics=5, show_bar=True)

# Threshold alerts
threshold = ThresholdCallback({
    'accuracy': {'min': 0.8},
    'loss': {'max': 0.5}
})

# History recording
history = HistoryCallback()

# Early stopping
early_stop = EarlyStoppingCallback(metric='loss', mode='min', patience=5)
```

### Pipelines

```python
from pyeval import Pipeline, create_classification_pipeline

# Fluent pipeline builder
pipeline = (
    Pipeline()
    .validate(lambda x: len(x[0]) == len(x[1]), "Length mismatch")
    .add_metric('accuracy', accuracy_score)
    .add_metric('f1', f1_score)
    .aggregate('mean', lambda r: sum(r.values()) / len(r))
)

results = pipeline.run(y_true, y_pred)

# Or use presets
classification_pipeline = create_classification_pipeline()
results = classification_pipeline.run(y_true, y_pred)
```

### Functional Utilities

```python
from pyeval import (
    Result, Option, curry, compose, pipe,
    combine_metrics, threshold_metric, average_metric
)

# Result monad for error handling
def safe_divide(a, b) -> Result:
    if b == 0:
        return Result.failure("Division by zero")
    return Result.success(a / b)

result = safe_divide(10, 2)
if result.is_success:
    print(result.value)  # 5.0

# Option monad for nullable values
opt = Option.from_nullable(get_metric("accuracy"))
value = opt.map(lambda x: x * 100).get_or_else(0)

# Function composition
process = compose(str, lambda x: x + 1, lambda x: x * 2)
result = process(5)  # "11"

# Currying
@curry
def add(a, b, c):
    return a + b + c

add(1)(2)(3)  # 6

# Combine metrics
combined = combine_metrics(accuracy_score, f1_score)
results = combined(y_true, y_pred)
# {'accuracy_score': 0.95, 'f1_score': 0.92}

# Threshold check
high_accuracy = threshold_metric(accuracy_score, 0.9)
is_high = high_accuracy(y_true, y_pred)  # True/False
```

### Aggregators

```python
from pyeval import (
    MetricAggregator, CrossValidationAggregator, EnsembleAggregator,
    MeanAggregator, MedianAggregator, PercentileAggregator
)

# Multi-metric aggregation
aggregator = MetricAggregator()
for fold_results in cross_val_results:
    aggregator.add_results(fold_results)
stats = aggregator.get_statistics()

# Cross-validation
cv = CrossValidationAggregator(n_folds=5)
for fold_idx, (y_true, y_pred) in enumerate(cv_splits):
    cv.add_fold_result(fold_idx, {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    })
print(cv.summary_table())

# Ensemble aggregation
ensemble = EnsembleAggregator(strategy='majority_vote')
ensemble.add_predictions('model1', predictions1)
ensemble.add_predictions('model2', predictions2)
final_predictions = ensemble.aggregate()
```

## 📁 Package Structure

```
pyeval/
├── __init__.py          # Main package exports (300+ exports)
├── evaluator.py         # Unified Evaluator & Report
├── tracking.py          # Experiment Tracking
├── decorators.py        # @timed, @memoize, @retry, @logged, etc.
├── patterns.py          # Strategy, Factory, Builder, Observer patterns
├── validators.py        # Type, Schema, Prediction validators
├── callbacks.py         # Progress, Threshold, History callbacks
├── pipeline.py          # Fluent pipeline builder
├── functional.py        # Result/Option monads, curry, compose
├── aggregators.py       # Statistical, CV, Ensemble aggregators
├── ml/
│   └── __init__.py      # Classification, Regression, Clustering (40+ metrics)
├── nlp/
│   └── __init__.py      # BLEU, ROUGE, METEOR, TER, chrF (15+ metrics)
├── llm/
│   └── __init__.py      # Toxicity, Hallucination, Bias, Coherence (15+ metrics)
├── rag/
│   └── __init__.py      # Context Relevance, Groundedness, Attribution (15+ metrics)
├── fairness/
│   └── __init__.py      # Demographic Parity, Equalized Odds (10+ metrics)
├── speech/
│   └── __init__.py      # WER, CER, MER, SER, MOS (20+ metrics)
├── recommender/
│   └── __init__.py      # Precision@K, NDCG, MAP, Serendipity (25+ metrics)
└── utils/
    ├── math_ops.py      # Mathematical operations + Statistical tests
    ├── text_ops.py      # Text processing utilities
    ├── data_ops.py      # Data manipulation utilities
    └── viz_ops.py       # ASCII visualizations (Confusion matrix, charts, sparklines)
```

## 🎯 Metric Classes

Each domain provides a convenience class for computing all metrics at once:

| Class | Domain |
|-------|--------|
| `ClassificationMetrics` | Binary/Multiclass classification |
| `RegressionMetrics` | Regression tasks |
| `ClusteringMetrics` | Clustering evaluation |
| `NLPMetrics` | Text generation quality |
| `LLMMetrics` | LLM response quality |
| `RAGMetrics` | RAG pipeline evaluation |
| `FairnessMetrics` | Model fairness |
| `SpeechMetrics` | Speech recognition |
| `RecommenderMetrics` | Recommendation systems |

## 🧪 Running Tests

```bash
cd pyeval
python -m pytest tests/ -v
# 275 tests passing
```

## 📄 License

MIT License

---

**PyEval** - Evaluate everything, depend on nothing. 🚀