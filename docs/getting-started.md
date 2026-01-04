# Getting Started

This guide will help you get up and running with PyEval in minutes.

## Installation

=== "pip (recommended)"

    ```bash
    pip install pyeval
    ```

=== "From Source"

    ```bash
    git clone https://github.com/yourusername/pyeval.git
    cd pyeval
    pip install -e .
    ```

=== "Development"

    ```bash
    git clone https://github.com/yourusername/pyeval.git
    cd pyeval
    pip install -e ".[dev]"  # Includes test dependencies
    ```

### Verify Installation

```python
import pyeval
print(f"PyEval version: {pyeval.__version__}")
print(f"Available exports: {len([x for x in dir(pyeval) if not x.startswith('_')])}")
# Output: PyEval version: 1.0.0
# Output: Available exports: 327
```

## Requirements

- **Python 3.12+**
- **No external dependencies!**

!!! info "Pure Python"
    PyEval is a pure Python library with zero dependencies. It works anywhere Python runs — edge devices, serverless functions, restricted environments.

---

## Quick Start Examples

### ML Classification

```python
from pyeval import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef,
    ClassificationMetrics
)

# Sample data
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

# Individual metrics
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
print(f"MCC:       {matthews_corrcoef(y_true, y_pred):.4f}")

# Or compute all at once
cm = ClassificationMetrics()
results = cm.compute(y_true, y_pred)
print(results)
```

### ML Regression

```python
from pyeval import (
    mean_squared_error, root_mean_squared_error, 
    mean_absolute_error, r2_score
)

y_true = [3.0, -0.5, 2.0, 7.0, 4.5]
y_pred = [2.5, 0.0, 2.1, 7.8, 4.0]

print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
print(f"R²:   {r2_score(y_true, y_pred):.4f}")
```

### NLP Generation

```python
from pyeval import bleu_score, rouge_score, meteor_score

reference = "The quick brown fox jumps over the lazy dog"
hypothesis = "A fast brown fox leaps over the lazy dog"

print(f"BLEU:    {bleu_score(reference, hypothesis):.4f}")
print(f"ROUGE-L: {rouge_score(reference, hypothesis, rouge_type='l')['f']:.4f}")
print(f"METEOR:  {meteor_score(reference, hypothesis):.4f}")
```

### LLM Evaluation

```python
from pyeval import toxicity_score, coherence_score, hallucination_score

response = "Machine learning is a subset of AI that enables computers to learn from data."
context = "Machine learning is a type of artificial intelligence."

print(f"Toxicity:      {toxicity_score(response)['toxicity_score']:.4f}")
print(f"Coherence:     {coherence_score(response)['coherence_score']:.4f}")
print(f"Hallucination: {hallucination_score(response, context)['hallucination_score']:.4f}")
```

### RAG Evaluation

```python
from pyeval import context_relevance, groundedness_score, answer_correctness

query = "What is machine learning?"
context = "Machine learning is a subset of AI that enables systems to learn from data."
response = "Machine learning is an AI technique that allows computers to learn from data."
ground_truth = "Machine learning is a type of artificial intelligence."

print(f"Context Relevance: {context_relevance(query, context):.4f}")
print(f"Groundedness:      {groundedness_score(response, context):.4f}")
print(f"Answer Correct:    {answer_correctness(response, ground_truth):.4f}")
```

### Fairness Evaluation

```python
from pyeval import demographic_parity, equalized_odds, disparate_impact

y_true = [1, 1, 0, 0, 1, 1, 0, 0]
y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

dp = demographic_parity(y_pred, sensitive)
print(f"Demographic Parity: {dp['dp_difference']:.4f}")

eo = equalized_odds(y_true, y_pred, sensitive)
print(f"Equalized Odds:     {eo['eo_difference']:.4f}")
```

---

## Core Concepts

### Metric Functions

All metric functions in PyEval follow a consistent pattern:

```python
def metric_function(y_true, y_pred, **kwargs) -> float | dict:
    """
    Compute a specific metric.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        **kwargs: Additional metric-specific parameters
    
    Returns:
        Either a single float score or a dictionary with detailed metrics
    """
```

### Metric Classes

For computing multiple related metrics at once, use metric classes:

```python
from pyeval import ClassificationMetrics, RegressionMetrics, NLPMetrics

# Classification
cm = ClassificationMetrics()
results = cm.compute(y_true, y_pred)

# Regression
rm = RegressionMetrics()
results = rm.compute(y_true, y_pred)

# NLP
nm = NLPMetrics()
results = nm.compute(references, hypotheses)
```

### Evaluator Class

For complex evaluation pipelines, use the unified `Evaluator`:

```python
from pyeval import Evaluator

evaluator = Evaluator("My Evaluation")

# Add and run evaluations
report = evaluator.evaluate_classification(y_true, y_pred)
print(report.summary())

# Compare multiple reports
print(evaluator.compare_reports())
```

---

## Next Steps

- 📊 Explore [ML Metrics](api/ml.md) for classification, regression, and clustering
- 📝 Learn about [NLP Metrics](api/nlp.md) for text generation
- 🤖 Discover [LLM Metrics](api/llm.md) for language model evaluation
- 🔍 Check out [RAG Metrics](api/rag.md) for retrieval-augmented generation
- ⚖️ Review [Fairness Metrics](api/fairness.md) for bias detection
- 🎯 See [Advanced Features](advanced/patterns.md) for design patterns and utilities
