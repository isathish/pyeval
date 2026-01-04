# Visualization Utilities API Reference

ASCII-based visualization utilities for terminal-friendly output.

---

## Confusion Matrix

### confusion_matrix_display

Display a formatted confusion matrix.

```python
from pyeval import confusion_matrix_display

matrix = [[45, 5], [10, 40]]
labels = ['Negative', 'Positive']

output = confusion_matrix_display(matrix, labels=labels)
print(output)
```

**Output:**
```
                    Predicted
                    Negative  Positive
Actual  Negative         45         5
        Positive         10        40
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `matrix` | list[list] | required | Confusion matrix |
| `labels` | list | None | Class labels |
| `normalize` | bool | False | Normalize values |
| `show_percentages` | bool | False | Show as percentages |

**Returns:** `str` - Formatted confusion matrix

---

### confusion_matrix_display (multiclass)

```python
from pyeval import confusion_matrix_display

matrix = [
    [40, 3, 2],
    [5, 38, 2],
    [1, 4, 35]
]
labels = ['Cat', 'Dog', 'Bird']

print(confusion_matrix_display(matrix, labels=labels))
```

**Output:**
```
                Predicted
            Cat    Dog   Bird
Actual Cat   40      3      2
       Dog    5     38      2
       Bird   1      4     35
```

---

## Classification Report

### classification_report_display

Display a formatted classification report.

```python
from pyeval import classification_report_display

y_true = [0, 0, 1, 1, 2, 2, 0, 1, 2]
y_pred = [0, 0, 1, 2, 2, 2, 0, 1, 1]

report = classification_report_display(y_true, y_pred)
print(report)
```

**Output:**
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         3
           1       0.67      0.67      0.67         3
           2       0.67      0.67      0.67         3

    accuracy                           0.78         9
   macro avg       0.78      0.78      0.78         9
weighted avg       0.78      0.78      0.78         9
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | list | required | Ground truth labels |
| `y_pred` | list | required | Predicted labels |
| `labels` | list | None | Class labels |
| `digits` | int | 2 | Decimal places |

**Returns:** `str` - Formatted classification report

---

## Charts

### horizontal_bar_chart

Create an ASCII horizontal bar chart.

```python
from pyeval import horizontal_bar_chart

metrics = {
    'Precision': 0.89,
    'Recall': 0.85,
    'F1 Score': 0.87,
    'Accuracy': 0.88
}

chart = horizontal_bar_chart(metrics, title="Model Performance", width=50)
print(chart)
```

**Output:**
```
═══════════ Model Performance ═══════════
Precision ████████████████████████████████████████████░░░░░░ 0.890
Recall    ██████████████████████████████████████████░░░░░░░░ 0.850
F1 Score  ███████████████████████████████████████████░░░░░░░ 0.870
Accuracy  ████████████████████████████████████████████░░░░░░ 0.880
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | dict | required | Label → value mapping |
| `title` | str | None | Chart title |
| `width` | int | 50 | Bar width in characters |
| `fill_char` | str | '█' | Character for filled portion |
| `empty_char` | str | '░' | Character for empty portion |
| `show_values` | bool | True | Display numeric values |

**Returns:** `str` - ASCII bar chart

---

### vertical_bar_chart

Create an ASCII vertical bar chart.

```python
from pyeval import vertical_bar_chart

data = {'A': 45, 'B': 30, 'C': 55, 'D': 20}

chart = vertical_bar_chart(data, height=10)
print(chart)
```

**Output:**
```
     C
    ██
    ██ A
    ██ ██
    ██ ██
    ██ ██ B
    ██ ██ ██
    ██ ██ ██
    ██ ██ ██ D
    ██ ██ ██ ██
 A  B  C  D
```

**Returns:** `str` - Vertical bar chart

---

### histogram_display

Create an ASCII histogram.

```python
from pyeval import histogram_display

data = [0.2, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.6, 0.7]

hist = histogram_display(data, bins=5, width=40)
print(hist)
```

**Output:**
```
[0.20-0.30) | ████████               | 2
[0.30-0.40) | ████████████           | 3
[0.40-0.50) | ████████████████       | 4
[0.50-0.60) | ████████               | 2
[0.60-0.70] | ████                   | 1
```

**Returns:** `str` - ASCII histogram

---

## Sparklines

### sparkline

Create a compact sparkline visualization.

```python
from pyeval import sparkline

# Training loss over epochs
losses = [0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26]

line = sparkline(losses)
print(f"Training loss: {line}")
# Output: Training loss: ▇▅▃▂▂▂▁▁
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | list | required | Numeric values |
| `min_val` | float | None | Minimum for scaling |
| `max_val` | float | None | Maximum for scaling |

**Returns:** `str` - Sparkline characters (▁▂▃▄▅▆▇█)

---

### sparkline_with_stats

Create sparkline with summary statistics.

```python
from pyeval import sparkline_with_stats

accuracy_scores = [0.82, 0.85, 0.84, 0.88, 0.87, 0.89, 0.91, 0.90]

result = sparkline_with_stats(accuracy_scores)
print(result)
# Output: ▁▃▂▅▄▆█▇ (min: 0.82, max: 0.91, avg: 0.87)
```

**Returns:** `str` - Sparkline with statistics

---

## Progress Indicators

### progress_bar

Create an ASCII progress bar.

```python
from pyeval import progress_bar

# Basic progress bar
bar = progress_bar(75, 100)
print(bar)
# Output: [████████████████████████░░░░░░░░] 75%

# With prefix
bar = progress_bar(75, 100, prefix="Evaluation")
print(bar)
# Output: Evaluation [████████████████████████░░░░░░░░] 75%

# Custom width
bar = progress_bar(3, 10, width=20, prefix="Step")
print(bar)
# Output: Step [██████░░░░░░░░░░░░░░] 30%
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `current` | int | required | Current progress |
| `total` | int | required | Total items |
| `width` | int | 30 | Bar width |
| `prefix` | str | '' | Prefix text |
| `fill` | str | '█' | Fill character |
| `empty` | str | '░' | Empty character |

**Returns:** `str` - Progress bar

---

### spinner

Get spinner animation frames.

```python
from pyeval import spinner

frames = spinner()  # Returns: ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

# Usage in a loop
import time
for i in range(20):
    print(f"\r{frames[i % len(frames)]} Processing...", end='')
    time.sleep(0.1)
```

**Returns:** `list` - Animation frame characters

---

## Curve Visualizations

### roc_curve_display

Display ASCII ROC curve.

```python
from pyeval import roc_curve_display

fpr = [0.0, 0.1, 0.2, 0.4, 0.6, 1.0]
tpr = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
auc = 0.85

display = roc_curve_display(fpr, tpr, auc=auc, height=10, width=40)
print(display)
```

**Output:**
```
ROC Curve (AUC = 0.85)
1.0 ┤                                    ●
    │                              ●●●●●●
    │                        ●●●●●●
    │                  ●●●●●●
0.5 ┤            ●●●●●●
    │      ●●●●●●
    │●●●●●●
0.0 ┤●
    └────────────────────────────────────
     0.0                              1.0
                    FPR
```

**Returns:** `str` - ASCII ROC curve

---

### pr_curve_display

Display ASCII Precision-Recall curve.

```python
from pyeval import pr_curve_display

precision = [1.0, 0.9, 0.8, 0.7, 0.6]
recall = [0.0, 0.3, 0.5, 0.7, 1.0]

display = pr_curve_display(precision, recall, height=10, width=40)
print(display)
```

**Returns:** `str` - ASCII PR curve

---

## Tables

### table_display

Create formatted ASCII table.

```python
from pyeval import table_display

headers = ['Model', 'Accuracy', 'F1', 'Latency']
rows = [
    ['Model A', '0.92', '0.89', '15ms'],
    ['Model B', '0.95', '0.93', '25ms'],
    ['Model C', '0.91', '0.88', '10ms']
]

table = table_display(headers, rows)
print(table)
```

**Output:**
```
┌─────────┬──────────┬──────┬─────────┐
│ Model   │ Accuracy │ F1   │ Latency │
├─────────┼──────────┼──────┼─────────┤
│ Model A │ 0.92     │ 0.89 │ 15ms    │
│ Model B │ 0.95     │ 0.93 │ 25ms    │
│ Model C │ 0.91     │ 0.88 │ 10ms    │
└─────────┴──────────┴──────┴─────────┘
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `headers` | list | required | Column headers |
| `rows` | list[list] | required | Table rows |
| `alignment` | str | 'left' | Text alignment |

**Returns:** `str` - Formatted table

---

## Complete Visualization Functions

| Function | Description |
|----------|-------------|
| `confusion_matrix_display` | Formatted confusion matrix |
| `classification_report_display` | Classification metrics report |
| `horizontal_bar_chart` | Horizontal ASCII bars |
| `vertical_bar_chart` | Vertical ASCII bars |
| `histogram_display` | Distribution histogram |
| `sparkline` | Compact trend line |
| `sparkline_with_stats` | Sparkline with statistics |
| `progress_bar` | Progress indicator |
| `spinner` | Animation frames |
| `roc_curve_display` | ASCII ROC curve |
| `pr_curve_display` | ASCII PR curve |
| `table_display` | Formatted table |

---

## Usage Tips

### Combining Visualizations

```python
from pyeval import (
    confusion_matrix_display, classification_report_display,
    horizontal_bar_chart, sparkline
)

def evaluation_report(y_true, y_pred, training_losses):
    """Generate comprehensive visual evaluation report."""
    
    # Confusion Matrix
    print("="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    print(confusion_matrix_display(
        confusion_matrix(y_true, y_pred),
        labels=['Neg', 'Pos']
    ))
    
    # Classification Report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report_display(y_true, y_pred))
    
    # Metrics Bar Chart
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    print(horizontal_bar_chart(metrics))
    
    # Training Progress
    print("\n" + "="*50)
    print("TRAINING PROGRESS")
    print("="*50)
    print(f"Loss: {sparkline(training_losses)}")
```

### Real-time Progress Display

```python
from pyeval import progress_bar
import time

def process_with_progress(items):
    """Process items with progress visualization."""
    total = len(items)
    
    for i, item in enumerate(items):
        # Process item
        process(item)
        
        # Update progress
        bar = progress_bar(i + 1, total, prefix="Processing")
        print(f"\r{bar}", end='', flush=True)
    
    print()  # New line after completion
```
