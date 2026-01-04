# ML Examples

Comprehensive examples for Machine Learning evaluation with PyEval.

---

## 🎯 Binary Classification

### Complete Evaluation Workflow

```python
from pyeval import (
    # Metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    balanced_accuracy_score,
    log_loss,
    
    # Utilities
    ClassificationMetrics,
    classification_report
)

# Sample data
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
y_prob = [0.9, 0.1, 0.4, 0.8, 0.2, 0.95, 0.6, 0.15, 0.85, 0.3, 0.1, 0.9, 0.2, 0.88, 0.55]

# === Basic Metrics ===
print("=== Basic Metrics ===")
print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
print(f"Precision:         {precision_score(y_true, y_pred):.4f}")
print(f"Recall:            {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score:          {f1_score(y_true, y_pred):.4f}")

# === Advanced Metrics ===
print("\n=== Advanced Metrics ===")
print(f"F0.5 Score:        {fbeta_score(y_true, y_pred, beta=0.5):.4f}")  # Precision-weighted
print(f"F2 Score:          {fbeta_score(y_true, y_pred, beta=2):.4f}")    # Recall-weighted
print(f"MCC:               {matthews_corrcoef(y_true, y_pred):.4f}")
print(f"Cohen's Kappa:     {cohen_kappa_score(y_true, y_pred):.4f}")

# === Probability-based Metrics ===
print("\n=== Probability Metrics ===")
print(f"ROC AUC:           {roc_auc_score(y_true, y_prob):.4f}")
print(f"PR AUC:            {average_precision_score(y_true, y_prob):.4f}")
print(f"Log Loss:          {log_loss(y_true, y_prob):.4f}")

# === Confusion Matrix ===
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_true, y_pred)
print(f"TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

# === All-in-One ===
print("\n=== Classification Report ===")
report = classification_report(y_true, y_pred)
print(report)
```

---

## 🎨 Multi-Class Classification

```python
from pyeval import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    top_k_accuracy_score
)

# 5-class problem
y_true = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2]
y_pred = [0, 2, 2, 3, 4, 0, 1, 1, 3, 3, 1, 1, 2]

# Top-k probabilities (for top-k accuracy)
y_prob = [
    [0.9, 0.05, 0.02, 0.02, 0.01],  # Class 0
    [0.1, 0.3, 0.4, 0.1, 0.1],      # Class 1 (wrong pred)
    [0.1, 0.1, 0.6, 0.1, 0.1],      # Class 2
    # ... more samples
]

print("=== Multi-Class Accuracy ===")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

print("\n=== Averaging Methods ===")
# Micro: Global calculation (good for imbalanced classes)
print(f"Micro Precision: {precision_score(y_true, y_pred, average='micro'):.4f}")

# Macro: Average per class (treats all classes equally)
print(f"Macro Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
print(f"Macro Recall:    {recall_score(y_true, y_pred, average='macro'):.4f}")
print(f"Macro F1:        {f1_score(y_true, y_pred, average='macro'):.4f}")

# Weighted: Average weighted by support
print(f"Weighted F1:     {f1_score(y_true, y_pred, average='weighted'):.4f}")

print("\n=== Per-Class Metrics ===")
labels = [0, 1, 2, 3, 4]
for label in labels:
    # One-vs-rest for each class
    y_true_binary = [1 if y == label else 0 for y in y_true]
    y_pred_binary = [1 if y == label else 0 for y in y_pred]
    p = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    r = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    print(f"Class {label}: P={p:.3f}, R={r:.3f}, F1={f:.3f}")

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_true, y_pred)
for row in cm:
    print(row)
```

---

## 📈 Regression

```python
from pyeval import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    explained_variance_score,
    max_error,
    median_absolute_error,
    mean_squared_log_error,
    RegressionMetrics
)

# Sample regression data
y_true = [3.0, -0.5, 2.0, 7.0, 4.5, 6.2, 1.8, 5.5, 3.3, 8.1]
y_pred = [2.5, 0.0, 2.1, 7.8, 4.0, 5.9, 2.1, 5.2, 3.8, 7.5]

print("=== Error Metrics ===")
print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
print(f"MedAE:{median_absolute_error(y_true, y_pred):.4f}")
print(f"Max:  {max_error(y_true, y_pred):.4f}")

print("\n=== Relative Metrics ===")
# Filter positive values for MAPE and MSLE
y_true_pos = [abs(y) + 0.01 for y in y_true]  # Ensure positive
y_pred_pos = [abs(y) + 0.01 for y in y_pred]
print(f"MAPE: {mean_absolute_percentage_error(y_true_pos, y_pred_pos):.4f}")
print(f"MSLE: {mean_squared_log_error(y_true_pos, y_pred_pos):.4f}")

print("\n=== Goodness of Fit ===")
print(f"R²:   {r2_score(y_true, y_pred):.4f}")
print(f"Explained Variance: {explained_variance_score(y_true, y_pred):.4f}")

# All at once
print("\n=== All Metrics ===")
metrics = RegressionMetrics()
results = metrics.compute(y_true, y_pred)
for name, value in results.items():
    print(f"{name}: {value:.4f}")
```

---

## 🔵 Clustering

```python
from pyeval import (
    adjusted_rand_index,
    normalized_mutual_info,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score,
    davies_bouldin_index,
    calinski_harabasz_index
)

# Clustering results (labels_true = ground truth, labels_pred = predicted)
labels_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
labels_pred = [0, 0, 1, 1, 1, 1, 2, 2, 2]

# Data points (for intrinsic metrics)
X = [
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],  # Cluster 0
    [5.0, 5.0], [5.2, 5.1], [4.8, 5.2],  # Cluster 1
    [9.0, 1.0], [9.2, 0.8], [8.8, 1.1],  # Cluster 2
]

print("=== External Metrics (with ground truth) ===")
print(f"ARI:         {adjusted_rand_index(labels_true, labels_pred):.4f}")
print(f"NMI:         {normalized_mutual_info(labels_true, labels_pred):.4f}")
print(f"Homogeneity: {homogeneity_score(labels_true, labels_pred):.4f}")
print(f"Completeness:{completeness_score(labels_true, labels_pred):.4f}")
print(f"V-Measure:   {v_measure_score(labels_true, labels_pred):.4f}")
print(f"FMI:         {fowlkes_mallows_score(labels_true, labels_pred):.4f}")

print("\n=== Internal Metrics (without ground truth) ===")
print(f"Silhouette:       {silhouette_score(X, labels_pred):.4f}")
print(f"Davies-Bouldin:   {davies_bouldin_index(X, labels_pred):.4f}")
print(f"Calinski-Harabasz:{calinski_harabasz_index(X, labels_pred):.4f}")
```

---

## 📊 Model Comparison

```python
from pyeval import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    Evaluator
)

# Ground truth
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]

# Multiple model predictions
models = {
    'Logistic Regression': [1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    'Random Forest':       [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
    'SVM':                 [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    'Neural Network':      [1, 0, 1, 1, 0, 0, 0, 0, 1, 1],
}

# Compare models
print("Model Comparison:")
print("-" * 50)
print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10}")
print("-" * 50)

for name, y_pred in models.items():
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{name:<20} {acc:>10.4f} {f1:>10.4f}")

# Find best model
best_model = max(models.keys(), key=lambda m: f1_score(y_true, models[m]))
print(f"\nBest model by F1: {best_model}")
```

---

## 🎲 Cross-Validation Metrics

```python
from pyeval import accuracy_score, f1_score, mean, std

# Simulated CV results (5 folds)
cv_scores = {
    'accuracy': [0.85, 0.82, 0.88, 0.84, 0.86],
    'f1': [0.83, 0.80, 0.87, 0.82, 0.85],
    'precision': [0.84, 0.81, 0.86, 0.83, 0.84],
    'recall': [0.82, 0.79, 0.88, 0.81, 0.86],
}

print("Cross-Validation Results (5 folds):")
print("-" * 45)
print(f"{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>8} {'Max':>8}")
print("-" * 45)

for metric, scores in cv_scores.items():
    print(f"{metric:<15} {mean(scores):>10.4f} {std(scores):>10.4f} {min(scores):>8.4f} {max(scores):>8.4f}")
```

---

## 🎯 Threshold Optimization

```python
from pyeval import precision_score, recall_score, f1_score

# Ground truth and probabilities
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_prob = [0.9, 0.2, 0.6, 0.8, 0.3, 0.7, 0.4, 0.1, 0.85, 0.55]

# Find optimal threshold
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("Threshold Analysis:")
print("-" * 55)
print(f"{'Threshold':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
print("-" * 55)

best_threshold = 0.5
best_f1 = 0

for thresh in thresholds:
    y_pred = [1 if p >= thresh else 0 for p in y_prob]
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"{thresh:>10.1f} {p:>12.4f} {r:>10.4f} {f:>10.4f}")
    
    if f > best_f1:
        best_f1 = f
        best_threshold = thresh

print(f"\nOptimal Threshold: {best_threshold} (F1={best_f1:.4f})")
```
