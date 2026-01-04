# Pipelines

PyEval provides a fluent pipeline API for building evaluation workflows.

---

## Basic Pipeline

### Creating Pipelines

```python
from pyeval import Pipeline
from pyeval import accuracy_score, precision_score, recall_score, f1_score

# Create a simple pipeline
pipeline = (
    Pipeline()
    .add_metric('accuracy', accuracy_score)
    .add_metric('precision', precision_score)
    .add_metric('recall', recall_score)
    .add_metric('f1', f1_score)
)

# Run pipeline
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 0]

results = pipeline.run(y_true, y_pred)
print(results)
# {
#     'accuracy': 0.8,
#     'precision': 0.667,
#     'recall': 0.667,
#     'f1': 0.667
# }
```

---

## Pipeline Features

### Validation Steps

```python
from pyeval import Pipeline

pipeline = (
    Pipeline()
    # Add validation
    .validate(
        lambda x: len(x[0]) == len(x[1]), 
        "y_true and y_pred must have same length"
    )
    .validate(
        lambda x: len(x[0]) > 0,
        "Inputs cannot be empty"
    )
    # Add metrics
    .add_metric('accuracy', accuracy_score)
)

# Valid data works
results = pipeline.run(y_true, y_pred)

# Invalid data raises error with message
results = pipeline.run([1, 0, 1], [1, 0])  # ValueError: y_true and y_pred must have same length
```

### Preprocessing Steps

```python
from pyeval import Pipeline

def binarize(y_true, y_pred):
    """Convert probabilities to binary predictions."""
    y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
    return y_true, y_pred_binary

pipeline = (
    Pipeline()
    .preprocess(binarize)
    .add_metric('accuracy', accuracy_score)
    .add_metric('f1', f1_score)
)

y_true = [1, 0, 1, 1, 0]
y_prob = [0.9, 0.1, 0.4, 0.8, 0.3]

results = pipeline.run(y_true, y_prob)
# Automatically converts probabilities to binary before computing metrics
```

### Postprocessing Steps

```python
from pyeval import Pipeline

def round_results(results):
    """Round all results to 4 decimal places."""
    return {k: round(v, 4) for k, v in results.items()}

def add_summary(results):
    """Add summary statistics."""
    values = list(results.values())
    results['mean'] = sum(values) / len(values)
    results['min'] = min(values)
    results['max'] = max(values)
    return results

pipeline = (
    Pipeline()
    .add_metric('accuracy', accuracy_score)
    .add_metric('f1', f1_score)
    .postprocess(round_results)
    .postprocess(add_summary)
)
```

### Aggregation

```python
from pyeval import Pipeline

pipeline = (
    Pipeline()
    .add_metric('accuracy', accuracy_score)
    .add_metric('precision', precision_score)
    .add_metric('recall', recall_score)
    .add_metric('f1', f1_score)
    .aggregate('mean', lambda r: sum(r.values()) / len(r))
    .aggregate('weighted_mean', lambda r: (
        r['accuracy'] * 0.1 + 
        r['f1'] * 0.9
    ))
)

results = pipeline.run(y_true, y_pred)
# Includes 'mean' and 'weighted_mean' in results
```

---

## Preset Pipelines

### Classification Pipeline

```python
from pyeval import create_classification_pipeline

# Comprehensive classification evaluation
pipeline = create_classification_pipeline()

results = pipeline.run(y_true, y_pred)
print(results)
# {
#     'accuracy': 0.8,
#     'precision': 0.667,
#     'recall': 0.667,
#     'f1': 0.667,
#     'specificity': 1.0,
#     'mcc': 0.577,
#     ...
# }
```

### Regression Pipeline

```python
from pyeval import create_regression_pipeline

pipeline = create_regression_pipeline()

y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

results = pipeline.run(y_true, y_pred)
# {
#     'mse': 0.375,
#     'rmse': 0.612,
#     'mae': 0.5,
#     'r2': 0.948,
#     ...
# }
```

### NLP Pipeline

```python
from pyeval import create_nlp_pipeline

pipeline = create_nlp_pipeline()

reference = "The quick brown fox"
hypothesis = "A fast brown fox"

results = pipeline.run(reference, hypothesis)
# {
#     'bleu': 0.45,
#     'rouge_1': {...},
#     'rouge_l': {...},
#     'meteor': 0.72,
#     ...
# }
```

---

## Conditional Pipelines

### Branching

```python
from pyeval import Pipeline, ConditionalPipeline

# Different pipelines for different conditions
binary_pipeline = (
    Pipeline()
    .add_metric('accuracy', accuracy_score)
    .add_metric('f1', f1_score)
    .add_metric('roc_auc', roc_auc_score)
)

multiclass_pipeline = (
    Pipeline()
    .add_metric('accuracy', accuracy_score)
    .add_metric('macro_f1', lambda t, p: f1_score(t, p, average='macro'))
    .add_metric('weighted_f1', lambda t, p: f1_score(t, p, average='weighted'))
)

# Conditional based on number of classes
def is_binary(y_true, y_pred):
    return len(set(y_true)) == 2

conditional = ConditionalPipeline()
conditional.when(is_binary, binary_pipeline)
conditional.otherwise(multiclass_pipeline)

results = conditional.run(y_true, y_pred)
```

---

## Pipeline Composition

### Combining Pipelines

```python
from pyeval import Pipeline, CompositePipeline

# Create component pipelines
accuracy_pipeline = Pipeline().add_metric('accuracy', accuracy_score)
f1_pipeline = Pipeline().add_metric('f1', f1_score)

# Combine them
combined = (
    CompositePipeline()
    .add_pipeline('accuracy_metrics', accuracy_pipeline)
    .add_pipeline('f1_metrics', f1_pipeline)
)

results = combined.run(y_true, y_pred)
# {
#     'accuracy_metrics': {'accuracy': 0.8},
#     'f1_metrics': {'f1': 0.667}
# }
```

### Sequential Pipelines

```python
from pyeval import SequentialPipeline

# Run pipelines in sequence, passing results through
sequential = (
    SequentialPipeline()
    .add_stage(preprocess_pipeline)
    .add_stage(compute_pipeline)
    .add_stage(postprocess_pipeline)
)

results = sequential.run(y_true, y_pred)
```

---

## Parallel Execution

### Parallel Metrics

```python
from pyeval import ParallelPipeline

# Compute multiple expensive metrics in parallel
parallel = (
    ParallelPipeline(n_jobs=4)
    .add_metric('metric_1', expensive_metric_1)
    .add_metric('metric_2', expensive_metric_2)
    .add_metric('metric_3', expensive_metric_3)
)

results = parallel.run(y_true, y_pred)
```

---

## Pipeline Callbacks

### Adding Callbacks

```python
from pyeval import Pipeline, ProgressCallback, ThresholdCallback

progress = ProgressCallback(show_bar=True)
threshold = ThresholdCallback({
    'accuracy': {'min': 0.8},
    'f1': {'min': 0.75}
})

pipeline = (
    Pipeline()
    .add_callback(progress)
    .add_callback(threshold)
    .add_metric('accuracy', accuracy_score)
    .add_metric('f1', f1_score)
)

results = pipeline.run(y_true, y_pred)
# Shows progress bar
# Alerts if metrics below threshold
```

---

## Saving and Loading

### Export Results

```python
from pyeval import Pipeline

pipeline = (
    Pipeline()
    .add_metric('accuracy', accuracy_score)
    .add_metric('f1', f1_score)
    .save_to('results.json')  # Auto-save results
)

results = pipeline.run(y_true, y_pred)
# Results automatically saved to results.json
```

### Load Pipeline Configuration

```python
from pyeval import Pipeline

# Save pipeline config
pipeline.save_config('pipeline_config.json')

# Load pipeline config
loaded_pipeline = Pipeline.from_config('pipeline_config.json')
```

---

## Complete Example

```python
from pyeval import (
    Pipeline, 
    accuracy_score, precision_score, recall_score, f1_score,
    ProgressCallback, ThresholdCallback
)

def preprocess_data(y_true, y_pred):
    """Clean and validate data."""
    # Remove invalid entries
    valid = [(t, p) for t, p in zip(y_true, y_pred) 
             if t is not None and p is not None]
    return [t for t, p in valid], [p for t, p in valid]

def format_results(results):
    """Format results for display."""
    return {k: f"{v:.4f}" for k, v in results.items()}

# Build comprehensive pipeline
evaluation_pipeline = (
    Pipeline('Classification Evaluation')
    
    # Validation
    .validate(lambda x: len(x[0]) == len(x[1]), "Length mismatch")
    .validate(lambda x: len(x[0]) > 0, "Empty inputs")
    
    # Preprocessing
    .preprocess(preprocess_data)
    
    # Callbacks
    .add_callback(ProgressCallback(show_bar=True))
    .add_callback(ThresholdCallback({'accuracy': {'min': 0.7}}))
    
    # Metrics
    .add_metric('accuracy', accuracy_score)
    .add_metric('precision', precision_score)
    .add_metric('recall', recall_score)
    .add_metric('f1', f1_score)
    
    # Aggregation
    .aggregate('mean_score', lambda r: sum(r.values()) / len(r))
    
    # Postprocessing
    .postprocess(format_results)
    
    # Save results
    .save_to('evaluation_results.json')
)

# Run evaluation
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

results = evaluation_pipeline.run(y_true, y_pred)
print(results)
```

---

## Best Practices

1. **Validate early** - Add validation steps at the start
2. **Name your metrics** - Use descriptive names
3. **Use preset pipelines** - Don't reinvent common patterns
4. **Add callbacks for monitoring** - Track progress and thresholds
5. **Save results** - Auto-save for reproducibility
6. **Compose pipelines** - Build complex from simple
