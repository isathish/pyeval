# Getting Started

This guide will help you get up and running with PyEval in minutes.

## Installation

### Using pip (recommended)

```bash
pip install pyeval
```

### From Source

```bash
git clone https://github.com/isathish/pyeval.git
cd pyeval
pip install -e .
```

### Development Setup

```bash
git clone https://github.com/isathish/pyeval.git
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

---

## Requirements

- **Python 3.12+**
- **No external dependencies!**

> **Note:** PyEval is a pure Python library with zero dependencies. It works anywhere Python runs — edge devices, serverless functions, restricted environments.

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

### Recommender Systems

```python
from pyeval import precision_at_k, recall_at_k, ndcg_at_k, mrr

# User preferences (1 = relevant, 0 = not relevant)
actual = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
predicted_ranking = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

print(f"Precision@5: {precision_at_k(actual, predicted_ranking, k=5):.4f}")
print(f"Recall@5:    {recall_at_k(actual, predicted_ranking, k=5):.4f}")
print(f"NDCG@5:      {ndcg_at_k(actual, predicted_ranking, k=5):.4f}")
print(f"MRR:         {mrr(actual, predicted_ranking):.4f}")
```

### Speech Recognition

```python
from pyeval import word_error_rate, character_error_rate

reference = "the quick brown fox jumps over the lazy dog"
hypothesis = "the quick brown fox jumped over a lazy dog"

print(f"WER: {word_error_rate(reference, hypothesis):.4f}")
print(f"CER: {character_error_rate(reference, hypothesis):.4f}")
```

---

## Built-in Visualization

PyEval includes ASCII-based visualization that works anywhere:

### Confusion Matrix

```python
from pyeval import confusion_matrix

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

# Display ASCII confusion matrix
cm = confusion_matrix(y_true, y_pred, display=True)
```

Output:
```
         Predicted
         0    1
       ┌────┬────┐
    0  │ 3  │ 1  │
Actual ├────┼────┤
    1  │ 1  │ 3  │
       └────┴────┘
```

### Sparklines

```python
from pyeval import sparkline

values = [1, 4, 2, 8, 5, 7, 3, 9, 6]
print(sparkline(values))
# Output: ▁▃▂█▄▆▂▇▅
```

### Progress Bars

```python
from pyeval import progress_bar

# Simple progress
print(progress_bar(75, 100))
# Output: [████████████████████████████░░░░░░░░░░] 75%
```

---

## API Design Principles

### Consistent Patterns

All PyEval functions follow consistent patterns:

```python
# Pattern 1: Compare y_true vs y_pred
score = metric(y_true, y_pred)

# Pattern 2: Evaluate single input
score = metric(text)

# Pattern 3: Evaluate with context
score = metric(text, context)

# Pattern 4: Return detailed results
result = metric(y_true, y_pred, detailed=True)
```

### Return Types

| Return Type | Example | When Used |
|-------------|---------|-----------|
| `float` | `0.8571` | Single metric score |
| `dict` | `{'precision': 0.8, 'recall': 0.7}` | Multiple related scores |
| `list` | `[0.8, 0.7, 0.9]` | Per-class or per-sample scores |

---

## Common Workflows

### Model Comparison

```python
from pyeval import accuracy_score, f1_score, matthews_corrcoef

y_true = [1, 0, 1, 1, 0, 1, 0, 0]

models = {
    'Model A': [1, 0, 0, 1, 0, 1, 1, 0],
    'Model B': [1, 0, 1, 1, 0, 1, 0, 1],
    'Model C': [1, 1, 1, 1, 0, 0, 0, 0],
}

print("Model Comparison")
print("-" * 40)
for name, y_pred in models.items():
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")
```

### Batch Evaluation

```python
from pyeval import bleu_score

references = [
    "The cat sat on the mat",
    "Hello world",
    "Machine learning is fascinating"
]

hypotheses = [
    "A cat was sitting on the mat",
    "Hello there world",
    "ML is really interesting"
]

scores = [bleu_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
avg_bleu = sum(scores) / len(scores)
print(f"Average BLEU: {avg_bleu:.4f}")
```

---

## Next Steps

- **[API Overview](api/overview.md)** — Quick reference for all 327+ APIs
- **[ML Examples](examples/ml.md)** — Detailed machine learning examples
- **[NLP Examples](examples/nlp.md)** — Text evaluation examples
- **[LLM Examples](examples/llm.md)** — Large language model evaluation
- **[RAG Examples](examples/rag.md)** — Retrieval-augmented generation

---

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you're using Python 3.12+:

```bash
python --version  # Should be 3.12 or higher
```

### Package Conflicts

PyEval has no dependencies, so conflicts are unlikely. If you have issues, try a clean virtual environment:

```bash
python -m venv pyeval_env
source pyeval_env/bin/activate  # On Windows: pyeval_env\Scripts\activate
pip install pyeval
```

### Getting Help

- **[GitHub Issues](https://github.com/isathish/pyeval/issues)** — Report bugs or request features
- **[Contributing Guide](contributing.md)** — How to contribute
