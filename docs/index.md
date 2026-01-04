# PyEval Documentation

<div align="center">

<img src="https://img.shields.io/badge/PyEval-v1.0.0-blue?style=for-the-badge" alt="Version">

**A Comprehensive Pure Python Evaluation Framework**

*Evaluate everything, depend on nothing.*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](.)
[![Tests](https://img.shields.io/badge/tests-302%20passed-success.svg)](.)

</div>

---

## 🎯 What is PyEval?

PyEval is a **zero-dependency** evaluation library for Machine Learning, NLP, LLM, RAG, Fairness, Speech, and Recommender systems. Every metric is implemented in **pure Python** — no NumPy, no scikit-learn, no external packages required.

!!! success "Why Choose PyEval?"
    - **🚫 Zero Dependencies** — Works anywhere Python runs (edge devices, serverless, restricted environments)
    - **📦 327+ Public APIs** — Most comprehensive evaluation library available
    - **🔧 Unified Interface** — Consistent API design across all domains
    - **🧪 Battle-Tested** — 302 tests ensure reliability and correctness
    - **📊 Built-in Viz** — ASCII charts, confusion matrices, sparklines included

---

## 🚀 Quick Start

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

## 📚 Documentation Overview

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    New to PyEval? Start here for installation and basic usage.

    [:octicons-arrow-right-24: Quick Start Guide](getting-started.md)

-   :material-chart-bar:{ .lg .middle } **ML Metrics**

    ---

    Classification, Regression, Clustering — 40+ metrics.

    [:octicons-arrow-right-24: ML API Reference](api/ml.md)

-   :material-text:{ .lg .middle } **NLP Metrics**

    ---

    BLEU, ROUGE, METEOR, TER, BERTScore, and more.

    [:octicons-arrow-right-24: NLP API Reference](api/nlp.md)

-   :material-robot:{ .lg .middle } **LLM Evaluation**

    ---

    Toxicity, Hallucination, Coherence, Bias detection.

    [:octicons-arrow-right-24: LLM API Reference](api/llm.md)

-   :material-database-search:{ .lg .middle } **RAG Evaluation**

    ---

    Context Relevance, Groundedness, Faithfulness metrics.

    [:octicons-arrow-right-24: RAG API Reference](api/rag.md)

-   :material-scale-balance:{ .lg .middle } **Fairness Metrics**

    ---

    Demographic Parity, Equalized Odds, Disparate Impact.

    [:octicons-arrow-right-24: Fairness API Reference](api/fairness.md)

-   :material-microphone:{ .lg .middle } **Speech Metrics**

    ---

    WER, CER, MER, and speech quality evaluation.

    [:octicons-arrow-right-24: Speech API Reference](api/speech.md)

-   :material-star:{ .lg .middle } **Recommender Metrics**

    ---

    Precision@K, NDCG, MAP, MRR, Diversity, Coverage.

    [:octicons-arrow-right-24: Recommender API Reference](api/recommender.md)

</div>

---

## 🏗️ Advanced Features

<div class="grid cards" markdown>

-   :material-pipe:{ .lg .middle } **[Pipelines](advanced/pipelines.md)**

    ---

    Build evaluation workflows with fluent API.

-   :material-cog:{ .lg .middle } **[Design Patterns](advanced/patterns.md)**

    ---

    Strategy, Factory, Composite patterns for extensibility.

-   :material-function:{ .lg .middle } **[Functional API](advanced/functional.md)**

    ---

    Higher-order functions for metric composition.

-   :material-shield-check:{ .lg .middle } **[Validators](advanced/validators.md)**

    ---

    Input validation and schema enforcement.

-   :material-decorator:{ .lg .middle } **[Decorators](advanced/decorators.md)**

    ---

    Caching, timing, retry logic for metrics.

</div>

---

## 📊 Feature Comparison

| Feature | PyEval | scikit-learn | evaluate (HF) | torchmetrics |
|---------|:------:|:------------:|:-------------:|:------------:|
| Zero Dependencies | ✅ | ❌ | ❌ | ❌ |
| ML Metrics | ✅ | ✅ | ⚠️ | ✅ |
| NLP Metrics | ✅ | ❌ | ✅ | ⚠️ |
| LLM Metrics | ✅ | ❌ | ⚠️ | ❌ |
| RAG Metrics | ✅ | ❌ | ⚠️ | ❌ |
| Fairness Metrics | ✅ | ❌ | ❌ | ⚠️ |
| Speech Metrics | ✅ | ❌ | ✅ | ⚠️ |
| Recommender Metrics | ✅ | ❌ | ❌ | ⚠️ |
| Built-in Viz | ✅ | ⚠️ | ❌ | ❌ |
| Edge/Serverless Ready | ✅ | ❌ | ❌ | ❌ |

---

## 💡 Use Cases

### When to Use PyEval

- **Edge Deployment** — Evaluate models on IoT devices, mobile, or embedded systems
- **Serverless Functions** — No cold start penalty from heavy dependencies
- **Restricted Environments** — Corporate policies prohibiting certain packages
- **CI/CD Pipelines** — Fast, lightweight evaluation in automated workflows
- **Educational Projects** — Learn evaluation concepts without dependency complexity
- **Multi-Domain Evaluation** — One library for ML, NLP, LLM, and more

---

## 📈 What's New in v1.0.0

- ✅ **327+ evaluation metrics** across 9 domains
- ✅ **Zero dependencies** — pure Python implementation
- ✅ **Statistical testing** utilities (t-test, bootstrap, McNemar)
- ✅ **ASCII visualizations** (confusion matrix, charts, sparklines)
- ✅ **Design patterns** (Strategy, Factory, Composite, Pipeline)
- ✅ **Functional utilities** (Result/Option monads, curry, compose)

---

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](contributing.md) for details.

---

## 📜 License

PyEval is released under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">

**Made with ❤️ for the ML community**

[GitHub](https://github.com/yourusername/pyeval) · [PyPI](https://pypi.org/project/pyeval/) · [Documentation](https://yourusername.github.io/pyeval)

</div>
