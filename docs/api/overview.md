# API Overview

This page provides a quick reference for all metrics available in PyEval.

---

## 📊 Quick Reference Table

| Domain | Metrics Count | Key Functions |
|--------|---------------|---------------|
| [ML](ml.md) | 40+ | `accuracy_score`, `f1_score`, `roc_auc_score` |
| [NLP](nlp.md) | 15+ | `bleu_score`, `rouge_score`, `meteor_score` |
| [LLM](llm.md) | 12+ | `toxicity_score`, `hallucination_score`, `coherence_score` |
| [RAG](rag.md) | 15+ | `context_relevance`, `groundedness_score`, `faithfulness_score` |
| [Fairness](fairness.md) | 10+ | `demographic_parity`, `equalized_odds`, `disparate_impact` |
| [Speech](speech.md) | 10+ | `word_error_rate`, `character_error_rate` |
| [Recommender](recommender.md) | 15+ | `precision_at_k`, `ndcg_at_k`, `mean_reciprocal_rank` |

---

## 🎯 ML Metrics

### Classification

| Metric | Function | Returns |
|--------|----------|---------|
| Accuracy | `accuracy_score(y_true, y_pred)` | `float` |
| Precision | `precision_score(y_true, y_pred)` | `float` |
| Recall | `recall_score(y_true, y_pred)` | `float` |
| F1 Score | `f1_score(y_true, y_pred)` | `float` |
| ROC AUC | `roc_auc_score(y_true, y_prob)` | `float` |
| Confusion Matrix | `confusion_matrix(y_true, y_pred)` | `list[list]` |
| Matthews Corr. | `matthews_corrcoef(y_true, y_pred)` | `float` |
| Cohen's Kappa | `cohen_kappa_score(y_true, y_pred)` | `float` |

### Regression

| Metric | Function | Returns |
|--------|----------|---------|
| MSE | `mean_squared_error(y_true, y_pred)` | `float` |
| RMSE | `root_mean_squared_error(y_true, y_pred)` | `float` |
| MAE | `mean_absolute_error(y_true, y_pred)` | `float` |
| R² | `r2_score(y_true, y_pred)` | `float` |
| MAPE | `mean_absolute_percentage_error(y_true, y_pred)` | `float` |

---

## 📝 NLP Metrics

| Metric | Function | Returns |
|--------|----------|---------|
| BLEU | `bleu_score(reference, hypothesis)` | `float` |
| ROUGE | `rouge_score(reference, hypothesis)` | `dict` |
| METEOR | `meteor_score(reference, hypothesis)` | `float` |
| TER | `ter_score(reference, hypothesis)` | `float` |
| chrF | `chrf_score(reference, hypothesis)` | `float` |
| BERTScore | `bert_score(reference, hypothesis)` | `dict` |

---

## 🤖 LLM Metrics

| Metric | Function | Returns |
|--------|----------|---------|
| Toxicity | `toxicity_score(text)` | `dict` |
| Coherence | `coherence_score(text)` | `dict` |
| Hallucination | `hallucination_score(response, context)` | `dict` |
| Bias Detection | `bias_detection_score(text)` | `dict` |
| Fluency | `fluency_score(text)` | `dict` |
| Factuality | `factuality_score(text, facts)` | `dict` |

---

## 🔍 RAG Metrics

| Metric | Function | Returns |
|--------|----------|---------|
| Context Relevance | `context_relevance(query, context)` | `float` |
| Groundedness | `groundedness_score(response, context)` | `float` |
| Faithfulness | `faithfulness_score(response, context)` | `float` |
| Answer Correctness | `answer_correctness(response, ground_truth)` | `float` |
| Retrieval Precision | `retrieval_precision(retrieved, relevant)` | `float` |
| Retrieval Recall | `retrieval_recall(retrieved, relevant)` | `float` |

---

## ⚖️ Fairness Metrics

| Metric | Function | Returns |
|--------|----------|---------|
| Demographic Parity | `demographic_parity(y_pred, sensitive)` | `dict` |
| Equalized Odds | `equalized_odds(y_true, y_pred, sensitive)` | `dict` |
| Equal Opportunity | `equal_opportunity(y_true, y_pred, sensitive)` | `dict` |
| Disparate Impact | `disparate_impact(y_pred, sensitive)` | `dict` |

---

## 🎤 Speech Metrics

| Metric | Function | Returns |
|--------|----------|---------|
| WER | `word_error_rate(reference, hypothesis)` | `float` |
| CER | `character_error_rate(reference, hypothesis)` | `float` |
| MER | `match_error_rate(reference, hypothesis)` | `float` |
| SER | `sentence_error_rate(references, hypotheses)` | `float` |

---

## ⭐ Recommender Metrics

| Metric | Function | Returns |
|--------|----------|---------|
| Precision@K | `precision_at_k(recommendations, relevant, k)` | `float` |
| Recall@K | `recall_at_k(recommendations, relevant, k)` | `float` |
| NDCG@K | `ndcg_at_k(recommendations, relevant, k)` | `float` |
| MAP | `mean_average_precision(recommendations, relevant)` | `float` |
| MRR | `mean_reciprocal_rank(recommendations, relevant)` | `float` |
| Hit Rate | `hit_rate(recommendations, relevant, k)` | `float` |
| Coverage | `coverage(recommendations, catalog)` | `float` |
| Diversity | `diversity(recommendations)` | `float` |

---

## 🔧 Utility Classes

### Metric Classes

```python
from pyeval import (
    ClassificationMetrics,
    RegressionMetrics,
    NLPMetrics,
    LLMMetrics,
    RAGMetrics,
    FairnessMetrics,
    SpeechMetrics,
    RecommenderMetrics
)
```

### Evaluator

```python
from pyeval import Evaluator

evaluator = Evaluator("My Evaluation")
report = evaluator.evaluate_classification(y_true, y_pred)
```

### Pipeline

```python
from pyeval import Pipeline

pipeline = (
    Pipeline()
    .add_metric('accuracy', accuracy_score)
    .add_metric('f1', f1_score)
)
results = pipeline.run(y_true, y_pred)
```
