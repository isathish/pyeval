# PyEval Documentation

<div align="center">

**A Comprehensive Pure Python Evaluation Framework**

*Evaluate everything, depend on nothing.*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](.)

</div>

---

## Welcome to PyEval

PyEval is a **zero-dependency** evaluation library for Machine Learning, NLP, LLM, RAG, Fairness, Speech, and Recommender systems. Every metric is implemented in pure Python — no NumPy, no scikit-learn, no external packages required.

## Quick Navigation

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting Started](getting-started.md)**

    ---

    Installation, quick start guide, and basic usage examples

-   :material-chart-bar: **[ML Metrics](api/ml.md)**

    ---

    Classification, Regression, and Clustering metrics

-   :material-text: **[NLP Metrics](api/nlp.md)**

    ---

    BLEU, ROUGE, METEOR, TER, and more

-   :material-robot: **[LLM Metrics](api/llm.md)**

    ---

    Toxicity, Hallucination, Coherence evaluation

-   :material-database-search: **[RAG Metrics](api/rag.md)**

    ---

    Context Relevance, Groundedness, Faithfulness

-   :material-scale-balance: **[Fairness Metrics](api/fairness.md)**

    ---

    Demographic Parity, Equalized Odds, Disparate Impact

-   :material-microphone: **[Speech Metrics](api/speech.md)**

    ---

    WER, CER, MER, and speech quality metrics

-   :material-star: **[Recommender Metrics](api/recommender.md)**

    ---

    Precision@K, NDCG, MAP, Diversity metrics

</div>

## Key Features

| Feature | Description |
|---------|-------------|
| 🚫 **Zero Dependencies** | Pure Python implementation — works anywhere Python runs |
| 📦 **327+ Public APIs** | Comprehensive coverage across all evaluation domains |
| 🧪 **302 Tests** | Thoroughly tested and production-ready |
| 🔧 **Unified Interface** | Consistent API design across all metric types |
| 📊 **Built-in Visualizations** | ASCII charts, confusion matrices, sparklines |
| 🏗️ **Design Patterns** | Strategy, Factory, Composite, Pipeline patterns |

## Installation

```bash
pip install pyeval
```

Or clone from source:

```bash
git clone https://github.com/yourusername/pyeval.git
cd pyeval
pip install -e .
```

## Quick Example

```python
from pyeval import accuracy_score, precision_score, f1_score

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
```

## What's New in v1.0.0

- ✅ **327+ evaluation metrics** across 9 domains
- ✅ **Zero dependencies** — pure Python implementation
- ✅ **Statistical testing** utilities (t-test, bootstrap, McNemar)
- ✅ **ASCII visualizations** (confusion matrix, charts, sparklines)
- ✅ **Design patterns** (Strategy, Factory, Composite, Pipeline)
- ✅ **Functional utilities** (Result/Option monads, curry, compose)

## License

PyEval is released under the [MIT License](https://opensource.org/licenses/MIT).
