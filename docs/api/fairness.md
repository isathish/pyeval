# Fairness Metrics API Reference

Metrics for evaluating model fairness and bias.

---

## Group Fairness Metrics

### demographic_parity

Evaluate demographic parity (statistical parity) across groups.

```python
from pyeval import demographic_parity

y_pred = [1, 0, 1, 1, 0, 0, 1, 1]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

result = demographic_parity(y_pred, sensitive)
print(result)
# {
#     'dp_difference': 0.25,
#     'is_fair': False,  # |difference| > 0.1
#     'group_rates': {'A': 0.75, 'B': 0.50},
#     'reference_group': 'A',
#     'privileged_group': 'A'
# }
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_pred` | list | required | Predicted labels |
| `sensitive_attr` | list | required | Sensitive attribute values |
| `threshold` | float | 0.1 | Fairness threshold |

**Returns:** `dict` - Demographic parity analysis

---

### equalized_odds

Evaluate equalized odds (equal TPR and FPR across groups).

```python
from pyeval import equalized_odds

y_true = [1, 1, 0, 0, 1, 1, 0, 0]
y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

result = equalized_odds(y_true, y_pred, sensitive)
print(result)
# {
#     'eo_difference': 0.15,
#     'tpr_difference': 0.0,
#     'fpr_difference': 0.25,
#     'is_fair': False,
#     'group_tpr': {'A': 0.5, 'B': 0.5},
#     'group_fpr': {'A': 0.0, 'B': 0.25}
# }
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | list | required | Ground truth labels |
| `y_pred` | list | required | Predicted labels |
| `sensitive_attr` | list | required | Sensitive attribute values |
| `threshold` | float | 0.1 | Fairness threshold |

**Returns:** `dict` - Equalized odds analysis

---

### disparate_impact

Compute disparate impact ratio.

```python
from pyeval import disparate_impact

y_pred = [1, 0, 1, 1, 0, 0, 1, 1]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

result = disparate_impact(y_pred, sensitive)
print(result)
# {
#     'di_ratio': 0.67,  # Rate_B / Rate_A
#     'is_fair': False,  # ratio < 0.8 (4/5 rule)
#     'group_rates': {'A': 0.75, 'B': 0.50},
#     'four_fifths_rule': False
# }
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_pred` | list | required | Predicted labels |
| `sensitive_attr` | list | required | Sensitive attribute values |
| `threshold` | float | 0.8 | Four-fifths rule threshold |

**Returns:** `dict` - Disparate impact analysis

---

### true_positive_rate_difference

Compute True Positive Rate (TPR) difference between groups.

```python
from pyeval import true_positive_rate_difference

y_true = [1, 1, 0, 0, 1, 1, 0, 0]
y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

result = true_positive_rate_difference(y_true, y_pred, sensitive)
print(result)
# {
#     'tpr_difference': 0.5,
#     'group_tpr': {'A': 0.5, 'B': 1.0},
#     'is_fair': False
# }
```

**Returns:** `dict` - TPR difference analysis

---

### false_positive_rate_difference

Compute False Positive Rate (FPR) difference between groups.

```python
from pyeval import false_positive_rate_difference

y_true = [1, 1, 0, 0, 1, 1, 0, 0]
y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

result = false_positive_rate_difference(y_true, y_pred, sensitive)
print(result)
# {
#     'fpr_difference': 0.5,
#     'group_fpr': {'A': 0.0, 'B': 0.5},
#     'is_fair': False
# }
```

**Returns:** `dict` - FPR difference analysis

---

## Calibration Metrics

### calibration_by_group

Evaluate prediction calibration across groups.

```python
from pyeval import calibration_by_group

y_true = [1, 0, 1, 0, 1, 0, 1, 0]
y_prob = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

result = calibration_by_group(y_true, y_prob, sensitive)
print(result)
# {
#     'calibration_difference': 0.05,
#     'group_calibration': {'A': 0.85, 'B': 0.80},
#     'is_calibrated': True
# }
```

**Returns:** `dict` - Calibration analysis per group

---

### predictive_parity

Evaluate if precision is equal across groups.

```python
from pyeval import predictive_parity

y_true = [1, 1, 0, 0, 1, 1, 0, 0]
y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

result = predictive_parity(y_true, y_pred, sensitive)
print(result)
# {
#     'pp_difference': 0.17,
#     'group_precision': {'A': 1.0, 'B': 0.67},
#     'is_fair': False
# }
```

**Returns:** `dict` - Predictive parity analysis

---

## Individual Fairness Metrics

### individual_fairness

Evaluate individual fairness (similar individuals get similar predictions).

```python
from pyeval import individual_fairness

# Features for each individual
X = [
    [0.5, 0.3],
    [0.5, 0.35],  # Similar to first
    [0.9, 0.8],
    [0.1, 0.2]
]

# Predictions
y_pred = [1, 0, 1, 0]

result = individual_fairness(X, y_pred)
print(result)
# {
#     'individual_fairness_score': 0.75,
#     'violations': 1,  # Similar inputs with different outputs
#     'consistency_score': 0.75
# }
```

**Returns:** `dict` - Individual fairness analysis

---

### counterfactual_fairness

Evaluate counterfactual fairness.

```python
from pyeval import counterfactual_fairness

# Original predictions
y_pred_original = [1, 0, 1, 0, 1]
sensitive_original = ['A', 'A', 'B', 'B', 'A']

# Counterfactual predictions (with flipped sensitive attribute)
y_pred_counterfactual = [1, 0, 0, 1, 1]

result = counterfactual_fairness(
    y_pred_original, 
    y_pred_counterfactual
)
print(result)
# {
#     'cf_score': 0.6,
#     'changed_predictions': 2,
#     'total_predictions': 5
# }
```

**Returns:** `dict` - Counterfactual fairness analysis

---

## Comprehensive Analysis

### fairness_report

Generate a comprehensive fairness report.

```python
from pyeval import fairness_report

y_true = [1, 1, 0, 0, 1, 1, 0, 0]
y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
sensitive = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

report = fairness_report(y_true, y_pred, sensitive)
print(report)
# {
#     'demographic_parity': {...},
#     'equalized_odds': {...},
#     'disparate_impact': {...},
#     'predictive_parity': {...},
#     'overall_fairness_score': 0.72,
#     'recommendations': [
#         'Consider rebalancing training data',
#         'Review features correlated with sensitive attribute'
#     ]
# }
```

---

## Metric Class

### FairnessMetrics

Compute all fairness metrics at once.

```python
from pyeval import FairnessMetrics

fm = FairnessMetrics()

results = fm.compute(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_attr=sensitive
)

print(results)
# {
#     'demographic_parity_diff': 0.25,
#     'equalized_odds_diff': 0.15,
#     'disparate_impact_ratio': 0.67,
#     'tpr_difference': 0.0,
#     'fpr_difference': 0.25,
#     'is_fair': False,
#     ...
# }
```

---

## Complete Fairness Metrics List

| Metric | Function | Description |
|--------|----------|-------------|
| Demographic Parity | `demographic_parity` | Equal positive rate across groups |
| Equalized Odds | `equalized_odds` | Equal TPR and FPR across groups |
| Disparate Impact | `disparate_impact` | Ratio of positive rates (4/5 rule) |
| TPR Difference | `true_positive_rate_difference` | True positive rate gap |
| FPR Difference | `false_positive_rate_difference` | False positive rate gap |
| Calibration | `calibration_by_group` | Prediction calibration per group |
| Predictive Parity | `predictive_parity` | Equal precision across groups |
| Individual Fairness | `individual_fairness` | Similar treatment for similar individuals |
| Counterfactual | `counterfactual_fairness` | Fairness under attribute change |

---

## Fairness Criteria Reference

| Criterion | Definition | When to Use |
|-----------|------------|-------------|
| **Demographic Parity** | P(Ŷ=1\|A=a) = P(Ŷ=1\|A=b) | When outcome should be independent of sensitive attribute |
| **Equalized Odds** | P(Ŷ=1\|Y=y,A=a) = P(Ŷ=1\|Y=y,A=b) | When error rates should be equal |
| **Disparate Impact** | P(Ŷ=1\|A=a) / P(Ŷ=1\|A=b) ≥ 0.8 | Legal compliance (US 4/5 rule) |
| **Predictive Parity** | P(Y=1\|Ŷ=1,A=a) = P(Y=1\|Ŷ=1,A=b) | When precision should be equal |
| **Individual Fairness** | Similar → Similar predictions | When individual treatment matters |

---

## Usage Tips

### Handling Multiple Sensitive Attributes

```python
from pyeval import demographic_parity

# Create intersectional groups
def intersect_attributes(attr1, attr2):
    return [f"{a}_{b}" for a, b in zip(attr1, attr2)]

gender = ['M', 'M', 'F', 'F', 'M', 'F']
race = ['A', 'B', 'A', 'B', 'A', 'B']

intersectional = intersect_attributes(gender, race)
# ['M_A', 'M_B', 'F_A', 'F_B', 'M_A', 'F_B']

result = demographic_parity(y_pred, intersectional)
```

### Setting Fairness Thresholds

```python
from pyeval import equalized_odds

# Strict threshold
result_strict = equalized_odds(y_true, y_pred, sensitive, threshold=0.05)

# Lenient threshold  
result_lenient = equalized_odds(y_true, y_pred, sensitive, threshold=0.2)
```

### Comparing Models

```python
from pyeval import FairnessMetrics

fm = FairnessMetrics()

# Evaluate multiple models
models = {
    'model_a': predictions_a,
    'model_b': predictions_b,
    'model_c': predictions_c
}

fairness_scores = {}
for name, preds in models.items():
    result = fm.compute(y_true, preds, sensitive)
    fairness_scores[name] = {
        'dp_diff': result['demographic_parity_diff'],
        'eo_diff': result['equalized_odds_diff'],
        'di_ratio': result['disparate_impact_ratio']
    }

# Find fairest model
fairest = min(fairness_scores.items(), 
              key=lambda x: abs(x[1]['dp_diff']))
print(f"Fairest model: {fairest[0]}")
```
