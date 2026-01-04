# Quick Examples

Get started with PyEval in seconds with these copy-paste examples.

---

## 🚀 One-Liner Examples

### Classification

```python
from pyeval import accuracy_score, f1_score
accuracy_score([1, 0, 1, 1], [1, 0, 0, 1])  # 0.75
f1_score([1, 0, 1, 1], [1, 0, 0, 1])        # 0.8
```

### Regression

```python
from pyeval import mean_squared_error, r2_score
mean_squared_error([3.0, -0.5, 2.0], [2.5, 0.0, 2.0])  # 0.1667
r2_score([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.1, 7.8])  # 0.948
```

### NLP

```python
from pyeval import bleu_score, rouge_score
bleu_score("the cat sat on mat", "the cat is on the mat")  # 0.52
rouge_score("the cat sat on mat", "the cat is on the mat")  # {'1': {...}, '2': {...}, 'l': {...}}
```

### LLM

```python
from pyeval import toxicity_score, coherence_score
toxicity_score("This is a helpful response")  # {'toxicity_score': 0.02, 'is_toxic': False}
coherence_score("ML is AI. It learns from data.")  # {'coherence_score': 0.85}
```

---

## 📊 Complete Examples

### Example 1: Binary Classification

```python
from pyeval import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    ClassificationMetrics
)

# Data
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
y_prob = [0.9, 0.1, 0.4, 0.8, 0.2, 0.9, 0.6, 0.1, 0.85, 0.3]

# Individual metrics
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")   # 0.7000
print(f"Precision: {precision_score(y_true, y_pred):.4f}") # 0.6667
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")    # 0.6667
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")        # 0.6667
print(f"ROC AUC:   {roc_auc_score(y_true, y_prob):.4f}")   # 0.8500

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix: {cm}")
# [[3, 1], [2, 4]] => TN=3, FP=1, FN=2, TP=4

# All at once
metrics = ClassificationMetrics()
results = metrics.compute(y_true, y_pred)
print(results)
```

### Example 2: Multi-Class Classification

```python
from pyeval import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Multi-class data (3 classes: 0, 1, 2)
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 2, 0, 1, 2]

# Accuracy (same for all)
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")  # 0.5556

# Macro average (treat all classes equally)
print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
print(f"Recall (macro):    {recall_score(y_true, y_pred, average='macro'):.4f}")
print(f"F1 (macro):        {f1_score(y_true, y_pred, average='macro'):.4f}")

# Weighted average (by class frequency)
print(f"F1 (weighted):     {f1_score(y_true, y_pred, average='weighted'):.4f}")
```

### Example 3: Text Generation Quality

```python
from pyeval import bleu_score, rouge_score, meteor_score, ter_score

reference = "The quick brown fox jumps over the lazy dog"
hypothesis = "A fast brown fox leaps over the lazy dog"

# BLEU (0-1, higher is better)
bleu = bleu_score(reference, hypothesis)
print(f"BLEU: {bleu:.4f}")

# ROUGE (precision, recall, f1 for n-grams)
rouge = rouge_score(reference, hypothesis)
print(f"ROUGE-1 F1: {rouge['1']['f']:.4f}")
print(f"ROUGE-2 F1: {rouge['2']['f']:.4f}")
print(f"ROUGE-L F1: {rouge['l']['f']:.4f}")

# METEOR
meteor = meteor_score(reference, hypothesis)
print(f"METEOR: {meteor:.4f}")

# TER (0-∞, lower is better)
ter = ter_score(reference, hypothesis)
print(f"TER: {ter:.4f}")
```

### Example 4: LLM Response Evaluation

```python
from pyeval import (
    toxicity_score,
    coherence_score,
    hallucination_score,
    bias_detection_score,
    LLMMetrics
)

response = """
Machine learning is a subset of artificial intelligence 
that enables computers to learn from data. It has 
applications in image recognition, natural language 
processing, and recommendation systems.
"""

context = """
Machine learning is a type of AI that allows software 
to learn from data. Common applications include image 
classification and NLP.
"""

# Toxicity check
tox = toxicity_score(response)
print(f"Toxic: {tox['is_toxic']}, Score: {tox['toxicity_score']:.4f}")

# Coherence check
coh = coherence_score(response)
print(f"Coherence: {coh['coherence_score']:.4f}")

# Hallucination detection
hall = hallucination_score(response, context)
print(f"Hallucination: {hall['hallucination_score']:.4f}")

# All LLM metrics at once
metrics = LLMMetrics()
results = metrics.compute(response, context=context)
```

### Example 5: RAG Pipeline Evaluation

```python
from pyeval import (
    context_relevance,
    groundedness_score,
    faithfulness_score,
    answer_correctness,
    RAGMetrics
)

query = "What is machine learning?"
context = """
Machine learning is a subset of artificial intelligence (AI) 
that enables systems to automatically learn and improve from 
experience without being explicitly programmed. ML algorithms 
build mathematical models based on training data.
"""
response = "Machine learning is an AI technique that allows computers to learn from data."
ground_truth = "Machine learning is a type of artificial intelligence."

# Individual metrics
print(f"Context Relevance: {context_relevance(query, context):.4f}")
print(f"Groundedness:      {groundedness_score(response, context):.4f}")
print(f"Faithfulness:      {faithfulness_score(response, context):.4f}")
print(f"Correctness:       {answer_correctness(response, ground_truth):.4f}")

# All RAG metrics at once
metrics = RAGMetrics()
results = metrics.compute(
    query=query,
    context=context,
    response=response,
    ground_truth=ground_truth
)
```

### Example 6: Fairness Evaluation

```python
from pyeval import (
    demographic_parity,
    equalized_odds,
    disparate_impact,
    FairnessMetrics
)

# Predictions and sensitive attribute
y_true = [1, 1, 0, 0, 1, 1, 0, 0]
y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

# Demographic Parity (selection rate difference)
dp = demographic_parity(y_pred, sensitive)
print(f"DP Difference: {dp['dp_difference']:.4f}")
print(f"Group A rate:  {dp['rates']['A']:.4f}")
print(f"Group B rate:  {dp['rates']['B']:.4f}")

# Equalized Odds (TPR/FPR equality)
eo = equalized_odds(y_true, y_pred, sensitive)
print(f"EO Difference: {eo['eo_difference']:.4f}")

# Disparate Impact (ratio of selection rates)
di = disparate_impact(y_pred, sensitive)
print(f"Disparate Impact Ratio: {di['di_ratio']:.4f}")
# Rule: ratio should be > 0.8 (80% rule)
```

### Example 7: Recommender System Evaluation

```python
from pyeval import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    mean_average_precision,
    RecommenderMetrics
)

# User recommendations (lists of item IDs)
recommendations = [
    [1, 5, 3, 8, 2],  # User 1: recommended items
    [4, 2, 7, 1, 9],  # User 2
    [3, 6, 1, 8, 5],  # User 3
]

# Ground truth relevant items
relevant = [
    [1, 2, 10],       # User 1: actually relevant items
    [4, 7, 9],        # User 2
    [6, 8],           # User 3
]

k = 3  # Evaluate top-3 recommendations

print(f"Precision@{k}: {precision_at_k(recommendations, relevant, k):.4f}")
print(f"Recall@{k}:    {recall_at_k(recommendations, relevant, k):.4f}")
print(f"NDCG@{k}:      {ndcg_at_k(recommendations, relevant, k):.4f}")
print(f"MRR:           {mean_reciprocal_rank(recommendations, relevant):.4f}")
print(f"MAP:           {mean_average_precision(recommendations, relevant):.4f}")
```

### Example 8: Speech Recognition Evaluation

```python
from pyeval import (
    word_error_rate,
    character_error_rate,
    sentence_error_rate,
    SpeechMetrics
)

# Reference (ground truth) and hypothesis (ASR output)
references = [
    "the quick brown fox jumps over the lazy dog",
    "hello world how are you today",
    "speech recognition is amazing"
]
hypotheses = [
    "the quick brown fox jumped over a lazy dog",
    "hello world how are you",
    "speech recognition is amazing"
]

# Per-utterance WER
for ref, hyp in zip(references, hypotheses):
    wer = word_error_rate(ref, hyp)
    cer = character_error_rate(ref, hyp)
    print(f"WER: {wer:.4f}, CER: {cer:.4f}")

# Corpus-level
ser = sentence_error_rate(references, hypotheses)
print(f"Sentence Error Rate: {ser:.4f}")

# All metrics
metrics = SpeechMetrics()
results = metrics.compute(references, hypotheses)
```

---

## 🔄 Using Pipelines

```python
from pyeval import Pipeline, accuracy_score, precision_score, recall_score, f1_score

# Create evaluation pipeline
pipeline = (
    Pipeline()
    .add_metric('accuracy', accuracy_score)
    .add_metric('precision', precision_score)
    .add_metric('recall', recall_score)
    .add_metric('f1', f1_score)
)

# Run on data
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 0]

results = pipeline.run(y_true, y_pred)
print(results)
# {'accuracy': 0.8, 'precision': 0.667, 'recall': 0.667, 'f1': 0.667}
```

---

## 📈 Visualization

```python
from pyeval import confusion_matrix, classification_report

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

# ASCII confusion matrix
cm = confusion_matrix(y_true, y_pred, display=True)

# Classification report
report = classification_report(y_true, y_pred)
print(report)
```
