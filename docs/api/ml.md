# ML Metrics API Reference

Machine Learning metrics for Classification, Regression, and Clustering evaluation.

---

## Classification Metrics

### accuracy_score

Compute the accuracy classification score.

```python
from pyeval import accuracy_score

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 0]

score = accuracy_score(y_true, y_pred)
# Returns: 0.8
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | list | Ground truth labels |
| `y_pred` | list | Predicted labels |

**Returns:** `float` - Accuracy score between 0 and 1

---

### precision_score

Compute the precision score.

```python
from pyeval import precision_score

score = precision_score(y_true, y_pred)
score = precision_score(y_true, y_pred, average='macro')  # For multiclass
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | list | required | Ground truth labels |
| `y_pred` | list | required | Predicted labels |
| `average` | str | 'binary' | 'binary', 'micro', 'macro', 'weighted' |
| `pos_label` | int | 1 | Positive class label for binary classification |

**Returns:** `float` - Precision score between 0 and 1

---

### recall_score

Compute the recall score.

```python
from pyeval import recall_score

score = recall_score(y_true, y_pred)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | list | required | Ground truth labels |
| `y_pred` | list | required | Predicted labels |
| `average` | str | 'binary' | 'binary', 'micro', 'macro', 'weighted' |
| `pos_label` | int | 1 | Positive class label for binary classification |

**Returns:** `float` - Recall score between 0 and 1

---

### f1_score

Compute the F1 score (harmonic mean of precision and recall).

```python
from pyeval import f1_score

score = f1_score(y_true, y_pred)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | list | required | Ground truth labels |
| `y_pred` | list | required | Predicted labels |
| `average` | str | 'binary' | 'binary', 'micro', 'macro', 'weighted' |
| `beta` | float | 1.0 | Weight of recall in F-beta score |

**Returns:** `float` - F1 score between 0 and 1

---

### specificity_score

Compute the specificity (true negative rate).

```python
from pyeval import specificity_score

score = specificity_score(y_true, y_pred)
# Returns: TN / (TN + FP)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | list | Ground truth labels |
| `y_pred` | list | Predicted labels |

**Returns:** `float` - Specificity score between 0 and 1

---

### matthews_corrcoef

Compute the Matthews Correlation Coefficient (MCC).

```python
from pyeval import matthews_corrcoef

mcc = matthews_corrcoef(y_true, y_pred)
# Returns: value between -1 and 1
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | list | Ground truth labels |
| `y_pred` | list | Predicted labels |

**Returns:** `float` - MCC between -1 (worst) and 1 (best), 0 indicates random

---

### cohen_kappa_score

Compute Cohen's Kappa coefficient.

```python
from pyeval import cohen_kappa_score

kappa = cohen_kappa_score(y_true, y_pred)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | list | required | Ground truth labels |
| `y_pred` | list | required | Predicted labels |
| `weights` | str | None | None, 'linear', or 'quadratic' |

**Returns:** `float` - Kappa coefficient between -1 and 1

---

### balanced_accuracy

Compute the balanced accuracy score.

```python
from pyeval import balanced_accuracy

score = balanced_accuracy(y_true, y_pred)
# Returns: Average of recall for each class
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | list | Ground truth labels |
| `y_pred` | list | Predicted labels |

**Returns:** `float` - Balanced accuracy between 0 and 1

---

### log_loss

Compute the log loss (cross-entropy loss).

```python
from pyeval import log_loss

y_true = [1, 0, 1, 1]
y_prob = [0.9, 0.1, 0.8, 0.7]

loss = log_loss(y_true, y_prob)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | list | required | Ground truth labels |
| `y_prob` | list | required | Predicted probabilities |
| `eps` | float | 1e-15 | Small value to avoid log(0) |

**Returns:** `float` - Log loss (lower is better)

---

### roc_auc_score

Compute Area Under the ROC Curve.

```python
from pyeval import roc_auc_score

y_true = [1, 0, 1, 1, 0]
y_prob = [0.9, 0.1, 0.8, 0.7, 0.3]

auc = roc_auc_score(y_true, y_prob)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | list | Ground truth binary labels |
| `y_score` | list | Predicted scores/probabilities |

**Returns:** `float` - AUC score between 0 and 1

---

### roc_curve

Compute ROC curve points.

```python
from pyeval import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | list | Ground truth binary labels |
| `y_score` | list | Predicted scores/probabilities |

**Returns:** `tuple` - (fpr, tpr, thresholds)

---

### precision_recall_curve

Compute precision-recall curve points.

```python
from pyeval import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | list | Ground truth binary labels |
| `y_score` | list | Predicted scores/probabilities |

**Returns:** `tuple` - (precision, recall, thresholds)

---

### confusion_matrix

Compute confusion matrix.

```python
from pyeval import confusion_matrix

matrix = confusion_matrix(y_true, y_pred)
# Returns: [[TN, FP], [FN, TP]]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | list | Ground truth labels |
| `y_pred` | list | Predicted labels |

**Returns:** `list[list]` - Confusion matrix

---

## Regression Metrics

### mean_squared_error

Compute Mean Squared Error.

```python
from pyeval import mean_squared_error

y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

mse = mean_squared_error(y_true, y_pred)
```

**Returns:** `float` - MSE (lower is better)

---

### root_mean_squared_error

Compute Root Mean Squared Error.

```python
from pyeval import root_mean_squared_error

rmse = root_mean_squared_error(y_true, y_pred)
```

**Returns:** `float` - RMSE (lower is better)

---

### mean_absolute_error

Compute Mean Absolute Error.

```python
from pyeval import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

**Returns:** `float` - MAE (lower is better)

---

### mean_absolute_percentage_error

Compute Mean Absolute Percentage Error.

```python
from pyeval import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_true, y_pred)
```

**Returns:** `float` - MAPE as a ratio (lower is better)

---

### r2_score

Compute R² (coefficient of determination).

```python
from pyeval import r2_score

r2 = r2_score(y_true, y_pred)
```

**Returns:** `float` - R² score (1 is perfect, can be negative)

---

### explained_variance_score

Compute explained variance score.

```python
from pyeval import explained_variance_score

ev = explained_variance_score(y_true, y_pred)
```

**Returns:** `float` - Explained variance (1 is best)

---

## Clustering Metrics

### silhouette_score

Compute the mean Silhouette Coefficient.

```python
from pyeval import silhouette_score

X = [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
labels = [0, 0, 0, 1, 1, 1]

score = silhouette_score(X, labels)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | list[list] | Feature matrix |
| `labels` | list | Cluster labels |

**Returns:** `float` - Silhouette score between -1 and 1

---

### adjusted_rand_index

Compute Adjusted Rand Index.

```python
from pyeval import adjusted_rand_index

labels_true = [0, 0, 1, 1, 2, 2]
labels_pred = [0, 0, 1, 1, 1, 2]

ari = adjusted_rand_index(labels_true, labels_pred)
```

**Returns:** `float` - ARI between -1 and 1

---

### normalized_mutual_info

Compute Normalized Mutual Information.

```python
from pyeval import normalized_mutual_info

nmi = normalized_mutual_info(labels_true, labels_pred)
```

**Returns:** `float` - NMI between 0 and 1

---

## Metric Classes

### ClassificationMetrics

Compute all classification metrics at once.

```python
from pyeval import ClassificationMetrics

cm = ClassificationMetrics()
results = cm.compute(y_true, y_pred)

print(results)
# {
#     'accuracy': 0.85,
#     'precision': 0.82,
#     'recall': 0.88,
#     'f1': 0.85,
#     'specificity': 0.82,
#     'mcc': 0.70,
#     ...
# }
```

### RegressionMetrics

Compute all regression metrics at once.

```python
from pyeval import RegressionMetrics

rm = RegressionMetrics()
results = rm.compute(y_true, y_pred)

print(results)
# {
#     'mse': 0.25,
#     'rmse': 0.50,
#     'mae': 0.40,
#     'r2': 0.95,
#     ...
# }
```

### ClusteringMetrics

Compute all clustering metrics at once.

```python
from pyeval import ClusteringMetrics

cm = ClusteringMetrics()
results = cm.compute(X, labels_true, labels_pred)
```

---

## Complete Classification Metrics List

| Metric | Function | Description |
|--------|----------|-------------|
| Accuracy | `accuracy_score` | Overall correctness |
| Precision | `precision_score` | True positives / predicted positives |
| Recall | `recall_score` | True positives / actual positives |
| F1 Score | `f1_score` | Harmonic mean of precision and recall |
| Specificity | `specificity_score` | True negatives / actual negatives |
| MCC | `matthews_corrcoef` | Matthews Correlation Coefficient |
| Cohen's Kappa | `cohen_kappa_score` | Agreement beyond chance |
| Balanced Accuracy | `balanced_accuracy` | Average recall per class |
| Log Loss | `log_loss` | Cross-entropy loss |
| AUC-ROC | `roc_auc_score` | Area under ROC curve |
| Brier Score | `brier_score` | Mean squared error of probabilities |
| Hamming Loss | `hamming_loss` | Fraction of wrong labels |
| Jaccard Score | `jaccard_score` | Intersection over union |
| Top-K Accuracy | `top_k_accuracy` | Correct in top K predictions |
| ECE | `expected_calibration_error` | Calibration error |
