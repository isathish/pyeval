# PyEval 📊

**A Comprehensive Pure Python Evaluation Framework**

PyEval is a complete evaluation library for Machine Learning, NLP, LLM, RAG, Fairness, Speech, and Recommender systems — all without any third-party dependencies.

## ✨ Features

| Category | Metrics |
|----------|---------|
| **ML Classification** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| **ML Regression** | MSE, RMSE, MAE, MAPE, R² |
| **Clustering** | Silhouette Score, Davies-Bouldin, Calinski-Harabasz |
| **NLP** | BLEU, ROUGE (1/2/L), METEOR, TER, Distinct-N |
| **LLM Evaluation** | Toxicity, Hallucination, Relevancy, Faithfulness, Coherence |
| **RAG Evaluation** | Context Relevance, Answer Correctness, Groundedness |
| **Fairness** | Demographic Parity, Equalized Odds, Disparate Impact |
| **Speech** | WER, CER, MER, WIL, SER |
| **Recommender** | Precision@K, Recall@K, NDCG, MAP, Hit Rate, MRR |

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

# BLEU
bleu = bleu_score([reference.split()], hypothesis.split())
print(f"BLEU: {bleu:.4f}")

# ROUGE
rouge = rouge_score(reference, hypothesis)
print(f"ROUGE-1 F1: {rouge['rouge1']['f1']:.4f}")
print(f"ROUGE-L F1: {rouge['rougeL']['f1']:.4f}")

# METEOR
meteor = meteor_score([reference], hypothesis)
print(f"METEOR: {meteor:.4f}")
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

print(f"Toxicity:     {toxicity_score(response):.4f}")
print(f"Coherence:    {coherence_score(response, query):.4f}")
print(f"Relevancy:    {answer_relevancy(response, query):.4f}")
print(f"Faithfulness: {faithfulness_score(response, context):.4f}")
print(f"Hallucination: {hallucination_score(response, context):.4f}")
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

# Individual metrics
print(f"Context Relevance: {context_relevance(query, contexts):.4f}")
print(f"Answer Correctness: {answer_correctness(response, ground_truth):.4f}")

# All-in-one
metrics = RAGMetrics.compute(query, contexts, response, ground_truth)
print(f"Context Relevance: {metrics.context_relevance:.4f}")
```

### Fairness Metrics

```python
from pyeval import demographic_parity, equalized_odds, disparate_impact

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

print(f"Demographic Parity: {demographic_parity(y_pred, sensitive):.4f}")
print(f"Equalized Odds:     {equalized_odds(y_true, y_pred, sensitive):.4f}")
print(f"Disparate Impact:   {disparate_impact(y_pred, sensitive):.4f}")
```

### Speech Recognition

```python
from pyeval import word_error_rate, character_error_rate

reference = "the quick brown fox jumps over the lazy dog"
hypothesis = "the quick brown fox jumps over lazy dog"

print(f"WER: {word_error_rate(reference, hypothesis):.4f}")
print(f"CER: {character_error_rate(reference, hypothesis):.4f}")
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

## 📁 Package Structure

```
pyeval/
├── __init__.py          # Main package exports
├── evaluator.py         # Unified Evaluator & Report
├── tracking.py          # Experiment Tracking
├── ml/
│   └── __init__.py      # Classification, Regression, Clustering
├── nlp/
│   └── __init__.py      # BLEU, ROUGE, METEOR, TER
├── llm/
│   └── __init__.py      # Toxicity, Hallucination, Relevancy
├── rag/
│   └── __init__.py      # Context Relevance, Groundedness
├── fairness/
│   └── __init__.py      # Demographic Parity, Equalized Odds
├── speech/
│   └── __init__.py      # WER, CER, MER
├── recommender/
│   └── __init__.py      # Precision@K, NDCG, MAP
└── utils/
    ├── math_ops.py      # Mathematical operations
    ├── text_ops.py      # Text processing utilities
    └── data_ops.py      # Data manipulation utilities
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

## 🧪 Running Examples

```bash
cd pyeval
python examples/basic_usage.py
```

## 📄 License

MIT License

---

**PyEval** - Evaluate everything, depend on nothing. 🚀