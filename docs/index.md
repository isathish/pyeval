# PyEval

<div align="center">

![PyEval](https://img.shields.io/badge/PyEval-v1.0.0-blue?style=for-the-badge)

**A Comprehensive Pure Python Evaluation Framework**

*Evaluate everything, depend on nothing.*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](.)
[![Tests](https://img.shields.io/badge/tests-302%20passed-success.svg)](.)

</div>

---

## What is PyEval?

PyEval is a **zero-dependency** evaluation library for Machine Learning, NLP, LLM, RAG, Fairness, Speech, and Recommender systems. Every metric is implemented in **pure Python** — no NumPy, no scikit-learn, no external packages required.

### Why Choose PyEval?

| Feature | Description |
|---------|-------------|
| 🚫 **Zero Dependencies** | Works anywhere Python runs — edge devices, serverless, restricted environments |
| 📦 **327+ Public APIs** | The most comprehensive evaluation library available |
| 🔧 **Unified Interface** | Consistent API design across all domains |
| 🧪 **Battle-Tested** | 302 tests ensure reliability and correctness |
| 📊 **Built-in Viz** | ASCII charts, confusion matrices, sparklines included |

---

## Quick Start

### Installation

**Using pip (recommended):**

```bash
pip install pyeval
```

**From source:**

```bash
git clone https://github.com/isathish/pyeval.git
cd pyeval
pip install -e .
```

### 30-Second Example

```python
from pyeval import accuracy_score, f1_score, bleu_score, confusion_matrix

# ML Classification
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 0, 1, 0, 1]

print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")  # 83.33%
print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")       # 0.8571

# NLP Generation
ref = "The quick brown fox jumps over the lazy dog"
hyp = "A fast brown fox leaps over the lazy dog"
print(f"BLEU: {bleu_score(ref, hyp):.4f}")              # 0.4234

# Confusion Matrix (ASCII visualization!)
print(confusion_matrix(y_true, y_pred, display=True))
```

---

## Documentation Overview

### Getting Started

New to PyEval? Start with the [Getting Started Guide](getting-started.md) for installation and basic usage.

### API Reference

| Module | Description | Link |
|--------|-------------|------|
| **ML Metrics** | Classification, Regression, Clustering — 40+ metrics | [View API](api/ml.md) |
| **NLP Metrics** | BLEU, ROUGE, METEOR, TER, BERTScore, and more | [View API](api/nlp.md) |
| **LLM Evaluation** | Toxicity, Hallucination, Coherence, Bias detection | [View API](api/llm.md) |
| **RAG Evaluation** | Context Relevance, Groundedness, Faithfulness | [View API](api/rag.md) |
| **Fairness Metrics** | Demographic Parity, Equalized Odds, Disparate Impact | [View API](api/fairness.md) |
| **Speech Metrics** | WER, CER, MER, and speech quality evaluation | [View API](api/speech.md) |
| **Recommender Metrics** | Precision@K, NDCG, MAP, MRR, Diversity, Coverage | [View API](api/recommender.md) |
| **Statistical Utilities** | Hypothesis testing, confidence intervals, distributions | [View API](api/statistical.md) |
| **Visualization** | ASCII charts, sparklines, progress bars | [View API](api/visualization.md) |

### Advanced Features

| Feature | Description | Link |
|---------|-------------|------|
| **Pipelines** | Chain evaluation steps together | [Learn More](advanced/pipelines.md) |
| **Decorators** | Add validation, logging, retry logic | [Learn More](advanced/decorators.md) |
| **Validators** | Type checking and data validation | [Learn More](advanced/validators.md) |
| **Design Patterns** | Reusable patterns for evaluation | [Learn More](advanced/patterns.md) |
| **Functional Utilities** | Map, filter, reduce for metrics | [Learn More](advanced/functional.md) |

---

## Feature Highlights

### Complete Domain Coverage

```
┌─────────────────────────────────────────────────────────────────┐
│                        PyEval v1.0.0                            │
├─────────────────────────────────────────────────────────────────┤
│  ML            │  NLP           │  LLM           │  RAG         │
│  ├─ classify   │  ├─ bleu       │  ├─ toxicity   │  ├─ context  │
│  ├─ regress    │  ├─ rouge      │  ├─ coherence  │  ├─ ground   │
│  ├─ cluster    │  ├─ meteor     │  ├─ hallucin   │  ├─ faithful │
│  └─ rank       │  └─ ter        │  └─ bias       │  └─ answer   │
├─────────────────────────────────────────────────────────────────┤
│  Fairness      │  Speech        │  Recommender   │  Utilities   │
│  ├─ parity     │  ├─ wer        │  ├─ precision  │  ├─ stats    │
│  ├─ equality   │  ├─ cer        │  ├─ ndcg       │  ├─ viz      │
│  └─ calibrate  │  └─ mer        │  └─ diversity  │  └─ valid    │
└─────────────────────────────────────────────────────────────────┘
```

### Zero Dependencies

PyEval works in environments where other libraries can't:

- **Edge Devices** — Raspberry Pi, microcontrollers, IoT
- **Serverless** — AWS Lambda, Azure Functions, Google Cloud Functions
- **Restricted Environments** — Air-gapped systems, secure facilities
- **Embedded Systems** — No package manager required
- **Minimal Docker** — Tiny container images

### Consistent API Design

All metrics follow the same patterns:

```python
# Pattern 1: Simple comparison
score = metric_function(y_true, y_pred)

# Pattern 2: With options
score = metric_function(y_true, y_pred, **options)

# Pattern 3: Batch processing  
scores = [metric_function(t, p) for t, p in zip(true_batch, pred_batch)]
```

---

## Quick Examples by Domain

### Machine Learning

```python
from pyeval import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score
)

# Classification
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1:        {f1_score(y_true, y_pred):.4f}")
```

### Natural Language Processing

```python
from pyeval import bleu_score, rouge_score, meteor_score

reference = "The cat sat on the mat"
hypothesis = "A cat was sitting on the mat"

print(f"BLEU:   {bleu_score(reference, hypothesis):.4f}")
print(f"ROUGE:  {rouge_score(reference, hypothesis)}")
print(f"METEOR: {meteor_score(reference, hypothesis):.4f}")
```

### LLM Evaluation

```python
from pyeval import (
    toxicity_score, coherence_score, 
    hallucination_score, readability_score
)

text = "This is a sample generated response from an LLM."
context = "Information about the topic being discussed."

print(f"Toxicity:     {toxicity_score(text):.4f}")
print(f"Coherence:    {coherence_score(text):.4f}")
print(f"Readability:  {readability_score(text):.4f}")
```

### RAG Evaluation

```python
from pyeval import (
    context_relevance_score, groundedness_score,
    answer_relevance_score, faithfulness_score
)

query = "What is machine learning?"
context = "Machine learning is a subset of AI that enables systems to learn."
answer = "Machine learning is an AI technique for learning from data."

print(f"Context Relevance: {context_relevance_score(query, context):.4f}")
print(f"Groundedness:      {groundedness_score(answer, context):.4f}")
print(f"Answer Relevance:  {answer_relevance_score(query, answer):.4f}")
```

---

## Comparison with Other Libraries

| Feature | PyEval | scikit-learn | Evaluate (HF) | TorchMetrics |
|---------|--------|--------------|---------------|--------------|
| **Dependencies** | None | NumPy, SciPy | 15+ packages | PyTorch |
| **ML Metrics** | ✅ 40+ | ✅ 30+ | ⚠️ Limited | ✅ 25+ |
| **NLP Metrics** | ✅ 20+ | ❌ | ✅ 20+ | ⚠️ Limited |
| **LLM Metrics** | ✅ 15+ | ❌ | ⚠️ Limited | ❌ |
| **RAG Metrics** | ✅ 10+ | ❌ | ❌ | ❌ |
| **Fairness** | ✅ 10+ | ❌ | ❌ | ❌ |
| **Speech** | ✅ 5+ | ❌ | ⚠️ Limited | ❌ |
| **Recommender** | ✅ 10+ | ❌ | ❌ | ❌ |
| **Edge Deploy** | ✅ | ❌ | ❌ | ❌ |
| **Serverless** | ✅ | ⚠️ | ⚠️ | ❌ |

---

## Get Involved

- **[GitHub Repository](https://github.com/isathish/pyeval)** — Star us, report issues, contribute
- **[Contributing Guide](contributing.md)** — How to contribute to PyEval
- **[Changelog](changelog.md)** — What's new in each release

---

## License

PyEval is released under the **MIT License**. See the [LICENSE](https://github.com/isathish/pyeval/blob/main/LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for the ML community**

*Zero dependencies. Maximum evaluation coverage.*

</div>
