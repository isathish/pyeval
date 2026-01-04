# 🎯 PyEval

<div align="center">

**A Comprehensive Pure Python Evaluation Framework**

*Evaluate everything, depend on nothing.*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg?style=for-the-badge)](.)
[![Exports](https://img.shields.io/badge/exports-327+-orange.svg?style=for-the-badge)](.)
[![Tests](https://img.shields.io/badge/tests-302%20passing-success.svg?style=for-the-badge)](.)

</div>

---

PyEval is a **zero-dependency** evaluation library for Machine Learning, NLP, LLM, RAG, Fairness, Speech, and Recommender systems. Every metric is implemented in pure Python — no NumPy, no scikit-learn, no external packages required.

## 🌟 Why PyEval?

| Feature | Description |
|---------|-------------|
| 🚫 **Zero Dependencies** | Pure Python implementation — works anywhere Python runs |
| 📦 **327+ Public APIs** | Comprehensive coverage across all evaluation domains |
| 🧪 **302 Tests** | Thoroughly tested and production-ready |
| 🔧 **Unified Interface** | Consistent API design across all metric types |
| 📊 **Built-in Visualizations** | ASCII charts, confusion matrices, sparklines |
| 🏗️ **Design Patterns** | Strategy, Factory, Composite, Pipeline, Observer patterns |
| ⚡ **Functional Utilities** | Result/Option monads, curry, compose, pipe |
| 🎯 **Type Safe** | Comprehensive input validation and error handling |

---

## 📊 Complete Feature Matrix

### Core Evaluation Metrics

| Category | Metrics | Count |
|----------|---------|:-----:|
| **ML Classification** | Accuracy, Precision, Recall, F1, ROC-AUC, Specificity, MCC, Cohen's Kappa, Log Loss, Brier Score, Balanced Accuracy, Hamming Loss, Jaccard, Top-K Accuracy, ROC Curve, PR Curve, ECE | **40+** |
| **ML Regression** | MSE, RMSE, MAE, MAPE, R², MSLE, Symmetric MAPE, Huber Loss, Quantile Loss, Explained Variance, Normalized RMSE | **15+** |
| **Clustering** | Silhouette Score, Davies-Bouldin, Calinski-Harabasz, ARI, NMI, Homogeneity, Completeness, V-Measure, Fowlkes-Mallows, Purity | **12+** |

### Text & Language Metrics

| Category | Metrics | Count |
|----------|---------|:-----:|
| **NLP Generation** | BLEU (1-4), ROUGE (1/2/L/S), METEOR, TER, chrF, Distinct-N, Text Entropy, Perplexity, Repetition Ratio, Compression Ratio, Coverage, Density, Lexical Diversity | **20+** |
| **LLM Evaluation** | Toxicity, Hallucination, Relevancy, Faithfulness, Coherence, Bias Detection, Instruction Following, Multi-Turn Coherence, Summarization Quality, Response Diversity, Safety | **15+** |
| **RAG Pipelines** | Context Relevance, Answer Correctness, Groundedness, Context Entity Recall, Answer Attribution, Context Utilization, QA Relevance, Faithfulness, Retrieval F1, MRR, Pipeline Score | **20+** |

### Specialized Domains

| Category | Metrics | Count |
|----------|---------|:-----:|
| **Fairness** | Demographic Parity, Equalized Odds, Disparate Impact, TPR Difference, FPR Difference, Calibration, Predictive Parity, Individual Fairness, Counterfactual Fairness | **15+** |
| **Speech Recognition** | WER, CER, MER, WIL, SER, Slot Error Rate, Intent Accuracy, Phoneme Error Rate, Diarization Error, Keyword Spotting, MOS, Fluency | **20+** |
| **Recommender Systems** | Precision@K, Recall@K, NDCG@K, MAP, Hit Rate, MRR, Serendipity, Novelty, Diversity, Coverage, Gini Index, Inter-List Diversity, Entropy Diversity | **25+** |

### Statistical & Visualization

| Category | Metrics | Count |
|----------|---------|:-----:|
| **Statistical Tests** | Bootstrap CI, Paired t-test, Independent t-test, Wilcoxon, Mann-Whitney U, McNemar, Cohen's d, Hedges' g, Spearman, Pearson, Permutation Test | **15+** |
| **Visualizations** | ASCII Confusion Matrix, Classification Report, Horizontal Bar Charts, Histograms, ROC Curve Display, PR Curve Display, Sparklines, Progress Bars | **10+** |

### Advanced Features

| Feature | Components |
|---------|------------|
| **Design Patterns** | Strategy, Factory, Builder, Observer, Composite, Chain of Responsibility, Singleton |
| **Decorators** | `@timed`, `@memoize`, `@lru_cache`, `@retry`, `@fallback`, `@logged`, `@deprecated`, `@require_same_length` |
| **Validators** | TypeValidator, ListValidator, NumericValidator, SchemaValidator, PredictionValidator |
| **Callbacks** | ProgressCallback, ThresholdCallback, HistoryCallback, EarlyStoppingCallback, LoggingCallback |
| **Pipelines** | Fluent API, Validation chains, Metric composition, Aggregation |
| **Functional** | Result monad, Option monad, `curry`, `compose`, `pipe`, `combine_metrics` |
| **Aggregators** | MetricAggregator, CrossValidationAggregator, EnsembleAggregator, BootstrapAggregator |

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pyeval.git
cd pyeval

# That's it! No dependencies to install.
# Just import and use
```

Or install via pip (when published):
```bash
pip install pyeval
```

---

## 📖 Quick Start

### ML Classification

```python
from pyeval import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef,
    specificity_score, cohen_kappa_score, balanced_accuracy
)

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision:         {precision_score(y_true, y_pred):.4f}")
print(f"Recall:            {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score:          {f1_score(y_true, y_pred):.4f}")
print(f"Specificity:       {specificity_score(y_true, y_pred):.4f}")
print(f"MCC:               {matthews_corrcoef(y_true, y_pred):.4f}")
print(f"Cohen's Kappa:     {cohen_kappa_score(y_true, y_pred):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy(y_true, y_pred):.4f}")
```

### ML Regression

```python
from pyeval import (
    mean_squared_error, root_mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score, explained_variance_score
)

y_true = [3.0, -0.5, 2.0, 7.0, 4.5]
y_pred = [2.5, 0.0, 2.1, 7.8, 4.0]

print(f"MSE:                {mean_squared_error(y_true, y_pred):.4f}")
print(f"RMSE:               {root_mean_squared_error(y_true, y_pred):.4f}")
print(f"MAE:                {mean_absolute_error(y_true, y_pred):.4f}")
print(f"MAPE:               {mean_absolute_percentage_error(y_true, y_pred):.4f}")
print(f"R²:                 {r2_score(y_true, y_pred):.4f}")
print(f"Explained Variance: {explained_variance_score(y_true, y_pred):.4f}")
```

### NLP Evaluation

```python
from pyeval import bleu_score, rouge_score, meteor_score, ter_score, distinct_n

reference = "The quick brown fox jumps over the lazy dog"
hypothesis = "A fast brown fox leaps over the lazy dog"

# Text generation metrics
print(f"BLEU:      {bleu_score(reference, hypothesis):.4f}")
print(f"ROUGE-L:   {rouge_score(reference, hypothesis, rouge_type='l')['f']:.4f}")
print(f"METEOR:    {meteor_score(reference, hypothesis):.4f}")
print(f"TER:       {ter_score(reference, hypothesis):.4f}")
print(f"Distinct-2:{distinct_n([hypothesis], n=2):.4f}")
```

### LLM Evaluation

```python
from pyeval import (
    toxicity_score, hallucination_score, coherence_score,
    bias_detection_score, instruction_following_score
)

prompt = "Explain quantum computing in simple terms"
response = "Quantum computing uses quantum bits that can be both 0 and 1..."
context = "Quantum computing is a type of computation..."

# LLM quality metrics
print(f"Toxicity:      {toxicity_score(response)['toxicity_score']:.4f}")
print(f"Hallucination: {hallucination_score(response, context)['hallucination_score']:.4f}")
print(f"Coherence:     {coherence_score(response)['coherence_score']:.4f}")
print(f"Bias:          {bias_detection_score(response)['bias_score']:.4f}")
```

### RAG Evaluation

```python
from pyeval import (
    context_relevance, answer_correctness, groundedness_score,
    context_entity_recall, retrieval_f1, retrieval_mrr
)

query = "What is machine learning?"
context = "Machine learning is a subset of AI that enables systems to learn from data."
response = "Machine learning is an AI technique that allows computers to learn from data."
ground_truth = "Machine learning is a type of artificial intelligence."

# RAG pipeline metrics
print(f"Context Relevance: {context_relevance(query, context):.4f}")
print(f"Groundedness:      {groundedness_score(response, context):.4f}")
print(f"Answer Correct:    {answer_correctness(response, ground_truth):.4f}")

# Retrieval metrics
retrieved = [[1, 3, 5], [2, 4, 1]]  # Retrieved doc IDs
relevant = [[1, 2], [1, 4]]         # Relevant doc IDs
print(f"Retrieval F1:      {retrieval_f1(retrieved, relevant):.4f}")
```

### Fairness Evaluation

```python
from pyeval import (
    demographic_parity, equalized_odds, disparate_impact,
    true_positive_rate_difference, false_positive_rate_difference
)

y_true = [1, 1, 0, 0, 1, 1, 0, 0]
y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

# Fairness metrics return detailed dictionaries
dp = demographic_parity(y_pred, sensitive)
print(f"Demographic Parity:     {dp['dp_difference']:.4f}")

eo = equalized_odds(y_true, y_pred, sensitive)
print(f"Equalized Odds:         {eo['eo_difference']:.4f}")

di = disparate_impact(y_pred, sensitive)
print(f"Disparate Impact Ratio: {di['di_ratio']:.4f}")

tpr_diff = true_positive_rate_difference(y_true, y_pred, sensitive)
print(f"TPR Difference:         {tpr_diff['tpr_difference']:.4f}")
```

### Speech Recognition

```python
from pyeval import word_error_rate, character_error_rate, slot_error_rate

reference = "the quick brown fox jumps over the lazy dog"
hypothesis = "the quick brown fox jumps over lazy dog"

# Speech metrics return detailed error breakdowns
wer = word_error_rate(reference, hypothesis)
print(f"WER: {wer['wer']:.4f} (S:{wer['substitutions']}, I:{wer['insertions']}, D:{wer['deletions']})")

cer = character_error_rate(reference, hypothesis)
print(f"CER: {cer['cer']:.4f}")
```

### Recommender Systems

```python
from pyeval import (
    precision_at_k, recall_at_k, ndcg_at_k, 
    mean_average_precision, mean_reciprocal_rank
)

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
    mcnemar_test, correlation_coefficient, wilcoxon_signed_rank
)

# Bootstrap confidence interval
data = [0.85, 0.87, 0.86, 0.88, 0.84, 0.89, 0.87]
ci = bootstrap_confidence_interval(data, 'mean', confidence=0.95)
print(f"Mean: {ci['point_estimate']:.3f} CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

# Compare two models
model1_scores = [0.85, 0.87, 0.86, 0.88, 0.84]
model2_scores = [0.88, 0.89, 0.87, 0.90, 0.86]

result = paired_t_test(model1_scores, model2_scores)
print(f"Paired t-test: t={result['t_statistic']:.3f}, p={result['p_value']:.4f}")

d = cohens_d(model1_scores, model2_scores)
print(f"Cohen's d: {d:.3f} (0.2=small, 0.5=medium, 0.8=large)")

# McNemar test for classifier comparison
contingency = [[45, 15], [5, 35]]  # [[both correct, only A], [only B, both wrong]]
result = mcnemar_test(contingency)
print(f"McNemar χ²: {result['chi_square']:.3f}, p={result['p_value']:.4f}")
```

### ASCII Visualizations

```python
from pyeval import (
    confusion_matrix_display, classification_report_display,
    horizontal_bar_chart, sparkline, progress_bar
)

# Confusion Matrix
matrix = [[45, 5], [10, 40]]
print(confusion_matrix_display(matrix, labels=['Negative', 'Positive']))
# Output:
#              Predicted
#              Negative  Positive
# Actual Negative    45         5
#        Positive    10        40

# Classification Report
y_true = [0, 0, 1, 1, 2, 2, 0, 1, 2]
y_pred = [0, 0, 1, 2, 2, 2, 0, 1, 1]
print(classification_report_display(y_true, y_pred))

# Bar Chart
metrics = {'Precision': 0.89, 'Recall': 0.85, 'F1': 0.87, 'Accuracy': 0.88}
print(horizontal_bar_chart(metrics, title="Model Performance"))
# Output:
# ═══ Model Performance ═══
# Precision ████████████████████████████████████████░░░░░░░░░░ 0.890
# Recall    ██████████████████████████████████████░░░░░░░░░░░░ 0.850
# F1        ███████████████████████████████████████░░░░░░░░░░░ 0.870
# Accuracy  ████████████████████████████████████████░░░░░░░░░░ 0.880

# Sparkline for training progress
losses = [0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26]
print(f"Training loss: {sparkline(losses)}")  # ▇▅▃▂▂▂▁▁

# Progress Bar
print(progress_bar(75, 100, prefix="Evaluation"))  # Evaluation [████████████████████░░░░░░░░░░] 75%
```

---

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

---

## 📈 Experiment Tracking

Track your ML experiments:

```python
from pyeval import ExperimentTracker

tracker = ExperimentTracker("my_project")

# Create and log experiment run
run = tracker.create_run("experiment_v1")
run.log_params({"learning_rate": 0.01, "epochs": 100, "batch_size": 32})
run.log_metrics({"accuracy": 0.95, "f1": 0.93, "loss": 0.15})
run.set_status("completed")
tracker.save_run(run)

# Compare experiments
print(tracker.compare_runs())

# Get best run
best = tracker.get_best_run("accuracy")
print(f"Best accuracy: {best.metrics['accuracy']}")
```

---

## 🏗️ Advanced Features

### Decorators

```python
from pyeval import timed, memoize, require_same_length, retry, logged

@timed
def compute_metrics(data):
    """Automatically times function execution"""
    return process(data)

@memoize
def expensive_metric(y_true, y_pred):
    """Caches results for identical inputs"""
    return complex_computation(y_true, y_pred)

@require_same_length
def custom_accuracy(y_true, y_pred):
    """Validates input lengths match"""
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

@retry(max_attempts=3, delay=0.1)
def flaky_operation():
    """Retries on failure"""
    return external_api_call()

@logged
def tracked_metric(y_true, y_pred):
    """Logs function calls and results"""
    return compute(y_true, y_pred)
```

### Design Patterns

#### Strategy Pattern
```python
from pyeval import MetricCalculator, AccuracyStrategy, F1Strategy

calculator = MetricCalculator(AccuracyStrategy())
accuracy = calculator.calculate(y_true, y_pred)

# Switch strategy at runtime
calculator.set_strategy(F1Strategy())
f1 = calculator.calculate(y_true, y_pred)
```

#### Factory Pattern
```python
from pyeval import MetricFactory, MetricType

factory = MetricFactory()
metric = factory.create(MetricType.ACCURACY)
score = metric.compute(y_true, y_pred)
```

#### Composite Pattern
```python
from pyeval import CompositeMetric, SingleMetric

classification_metrics = CompositeMetric('classification')
classification_metrics.add(SingleMetric('accuracy', accuracy_score))
classification_metrics.add(SingleMetric('f1', f1_score))

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
validate_predictions(y_true, y_pred)  # Raises ValueError if invalid
validate_probabilities([0.1, 0.3, 0.6], require_sum_one=True)

# Composable validators
validator = ListValidator(element_type=(int, float), min_length=1, max_length=100)
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

# Progress tracking with visual feedback
progress = ProgressCallback(total_metrics=5, show_bar=True)

# Threshold alerts when metrics cross boundaries
threshold = ThresholdCallback({
    'accuracy': {'min': 0.8},
    'loss': {'max': 0.5}
})

# History recording for analysis
history = HistoryCallback()

# Early stopping for iterative evaluation
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
    .add_metric('precision', precision_score)
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
    combine_metrics, threshold_metric
)

# Result monad for safe error handling
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

# Combine metrics into single function
combined = combine_metrics(accuracy_score, f1_score, precision_score)
results = combined(y_true, y_pred)
# {'accuracy_score': 0.95, 'f1_score': 0.92, 'precision_score': 0.89}

# Threshold check
high_accuracy = threshold_metric(accuracy_score, 0.9)
is_high = high_accuracy(y_true, y_pred)  # True/False
```

### Aggregators

```python
from pyeval import (
    MetricAggregator, CrossValidationAggregator, EnsembleAggregator
)

# Cross-validation aggregation
cv = CrossValidationAggregator(n_folds=5)
for fold_idx, (y_true_fold, y_pred_fold) in enumerate(cv_splits):
    cv.add_fold_result(fold_idx, {
        'accuracy': accuracy_score(y_true_fold, y_pred_fold),
        'f1': f1_score(y_true_fold, y_pred_fold)
    })

print(cv.summary_table())
# ╔═══════════╦══════════════╦════════════════╗
# ║ Metric    ║ Mean ± Std   ║ [Min, Max]     ║
# ╠═══════════╬══════════════╬════════════════╣
# ║ accuracy  ║ 0.92 ± 0.02  ║ [0.89, 0.95]   ║
# ║ f1        ║ 0.90 ± 0.03  ║ [0.86, 0.93]   ║
# ╚═══════════╩══════════════╩════════════════╝

# Ensemble aggregation
ensemble = EnsembleAggregator(strategy='majority_vote')
ensemble.add_predictions('model1', predictions1)
ensemble.add_predictions('model2', predictions2)
ensemble.add_predictions('model3', predictions3)
final_predictions = ensemble.aggregate()
```

---

## 🎯 Metric Classes

Each domain provides a convenience class for computing all metrics at once:

| Class | Domain | Metrics Computed |
|-------|--------|------------------|
| `ClassificationMetrics` | Binary/Multiclass classification | All classification metrics |
| `RegressionMetrics` | Regression tasks | MSE, RMSE, MAE, R², etc. |
| `ClusteringMetrics` | Clustering evaluation | Silhouette, ARI, NMI, etc. |
| `NLPMetrics` | Text generation quality | BLEU, ROUGE, METEOR, etc. |
| `LLMMetrics` | LLM response quality | Toxicity, coherence, etc. |
| `RAGMetrics` | RAG pipeline evaluation | Relevance, groundedness, etc. |
| `FairnessMetrics` | Model fairness | DP, EO, DI, etc. |
| `SpeechMetrics` | Speech recognition | WER, CER, MER, etc. |
| `RecommenderMetrics` | Recommendation systems | P@K, NDCG, MAP, etc. |

```python
from pyeval import ClassificationMetrics

# Compute all classification metrics at once
cm = ClassificationMetrics()
results = cm.compute(y_true, y_pred)
print(results)
# {'accuracy': 0.95, 'precision': 0.94, 'recall': 0.93, 'f1': 0.935, ...}
```

---

## 📁 Package Structure

```
pyeval/
├── __init__.py          # Main package exports (327+ public APIs)
├── evaluator.py         # Unified Evaluator & Report classes
├── tracking.py          # Experiment Tracking system
├── decorators.py        # @timed, @memoize, @retry, @logged, etc.
├── patterns.py          # Strategy, Factory, Builder, Observer patterns
├── validators.py        # Type, Schema, Prediction validators
├── callbacks.py         # Progress, Threshold, History callbacks
├── pipeline.py          # Fluent pipeline builder
├── functional.py        # Result/Option monads, curry, compose
├── aggregators.py       # Statistical, CV, Ensemble aggregators
│
├── ml/                  # Machine Learning metrics
│   └── __init__.py      # Classification, Regression, Clustering (40+ metrics)
│
├── nlp/                 # Natural Language Processing metrics
│   └── __init__.py      # BLEU, ROUGE, METEOR, TER, chrF (20+ metrics)
│
├── llm/                 # Large Language Model metrics
│   └── __init__.py      # Toxicity, Hallucination, Coherence (15+ metrics)
│
├── rag/                 # Retrieval Augmented Generation metrics
│   └── __init__.py      # Context Relevance, Groundedness (20+ metrics)
│
├── fairness/            # Fairness evaluation metrics
│   └── __init__.py      # Demographic Parity, Equalized Odds (15+ metrics)
│
├── speech/              # Speech recognition metrics
│   └── __init__.py      # WER, CER, MER, SER, MOS (20+ metrics)
│
├── recommender/         # Recommender system metrics
│   └── __init__.py      # Precision@K, NDCG, MAP, Diversity (25+ metrics)
│
└── utils/               # Utility modules
    ├── math_ops.py      # Mathematical operations & statistical tests
    ├── text_ops.py      # Text processing utilities
    ├── data_ops.py      # Data manipulation utilities
    └── viz_ops.py       # ASCII visualizations
```

---

## 🧪 Running Tests

```bash
cd pyeval
python -m pytest tests/ -v

# Output: 302 tests passing ✓
```

Run specific test categories:
```bash
# ML metrics only
python -m pytest tests/test_ml_metrics.py -v

# NLP metrics only
python -m pytest tests/test_nlp_metrics.py -v

# All tests with coverage
python -m pytest tests/ -v --cov=pyeval
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**PyEval** — *Evaluate everything, depend on nothing.* 🚀

Made with ❤️ in pure Python

</div>
