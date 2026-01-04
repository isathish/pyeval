# Statistical Utilities API Reference

Statistical testing and analysis utilities for model comparison.

---

## Confidence Intervals

### bootstrap_confidence_interval

Compute bootstrap confidence interval for any statistic.

```python
from pyeval import bootstrap_confidence_interval

data = [0.85, 0.87, 0.86, 0.88, 0.84, 0.89, 0.87, 0.86]

# Mean with 95% CI
result = bootstrap_confidence_interval(data, statistic='mean', confidence=0.95)
print(result)
# {
#     'point_estimate': 0.865,
#     'ci_lower': 0.852,
#     'ci_upper': 0.878,
#     'confidence': 0.95,
#     'n_bootstrap': 1000
# }

# Median with 99% CI
result = bootstrap_confidence_interval(data, statistic='median', confidence=0.99)

# Custom statistic
result = bootstrap_confidence_interval(
    data, 
    statistic=lambda x: max(x) - min(x),  # Range
    confidence=0.95
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | list | required | Sample data |
| `statistic` | str/callable | 'mean' | Statistic to compute |
| `confidence` | float | 0.95 | Confidence level |
| `n_bootstrap` | int | 1000 | Number of bootstrap samples |

**Returns:** `dict` - Point estimate and confidence interval

---

## Hypothesis Tests

### paired_t_test

Perform paired t-test for comparing two models on same data.

```python
from pyeval import paired_t_test

model1_scores = [0.85, 0.87, 0.86, 0.88, 0.84]
model2_scores = [0.88, 0.89, 0.87, 0.90, 0.86]

result = paired_t_test(model1_scores, model2_scores)
print(result)
# {
#     't_statistic': -3.24,
#     'p_value': 0.032,
#     'degrees_freedom': 4,
#     'is_significant': True,  # p < 0.05
#     'mean_difference': -0.024
# }
```

**Returns:** `dict` - Test results with significance

---

### independent_t_test

Perform independent samples t-test.

```python
from pyeval import independent_t_test

group1 = [0.85, 0.87, 0.86, 0.88, 0.84]
group2 = [0.92, 0.93, 0.91, 0.94, 0.90]

result = independent_t_test(group1, group2)
print(result)
# {
#     't_statistic': -8.45,
#     'p_value': 0.0001,
#     'degrees_freedom': 8,
#     'is_significant': True
# }
```

**Returns:** `dict` - Test results

---

### wilcoxon_signed_rank

Perform Wilcoxon signed-rank test (non-parametric paired test).

```python
from pyeval import wilcoxon_signed_rank

model1_scores = [0.85, 0.87, 0.86, 0.88, 0.84]
model2_scores = [0.88, 0.89, 0.87, 0.90, 0.86]

result = wilcoxon_signed_rank(model1_scores, model2_scores)
print(result)
# {
#     'statistic': 0.0,
#     'p_value': 0.063,
#     'is_significant': False
# }
```

**Returns:** `dict` - Test results

---

### mann_whitney_u

Perform Mann-Whitney U test (non-parametric independent test).

```python
from pyeval import mann_whitney_u

group1 = [0.85, 0.87, 0.86, 0.88, 0.84]
group2 = [0.92, 0.93, 0.91, 0.94, 0.90]

result = mann_whitney_u(group1, group2)
print(result)
# {
#     'u_statistic': 0.0,
#     'p_value': 0.008,
#     'is_significant': True
# }
```

**Returns:** `dict` - Test results

---

### mcnemar_test

Perform McNemar's test for comparing classifiers.

```python
from pyeval import mcnemar_test

# Contingency table:
# [[both correct, only model1 correct],
#  [only model2 correct, both wrong]]
contingency = [[45, 15], [5, 35]]

result = mcnemar_test(contingency)
print(result)
# {
#     'chi_square': 5.0,
#     'p_value': 0.025,
#     'is_significant': True,
#     'odds_ratio': 3.0
# }
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `contingency_table` | list[list] | 2x2 contingency table |

**Returns:** `dict` - Test results

---

### permutation_test

Perform permutation test for comparing two groups.

```python
from pyeval import permutation_test

group1 = [0.85, 0.87, 0.86, 0.88, 0.84]
group2 = [0.88, 0.89, 0.87, 0.90, 0.86]

result = permutation_test(group1, group2, n_permutations=10000)
print(result)
# {
#     'observed_difference': -0.024,
#     'p_value': 0.035,
#     'is_significant': True
# }
```

**Returns:** `dict` - Permutation test results

---

## Effect Size Measures

### cohens_d

Compute Cohen's d effect size.

```python
from pyeval import cohens_d

group1 = [0.85, 0.87, 0.86, 0.88, 0.84]
group2 = [0.88, 0.89, 0.87, 0.90, 0.86]

d = cohens_d(group1, group2)
print(f"Cohen's d: {d:.3f}")
# Interpretation: 0.2=small, 0.5=medium, 0.8=large
```

**Returns:** `float` - Cohen's d value

---

### hedges_g

Compute Hedges' g effect size (bias-corrected Cohen's d).

```python
from pyeval import hedges_g

g = hedges_g(group1, group2)
print(f"Hedges' g: {g:.3f}")
```

**Returns:** `float` - Hedges' g value

---

### glass_delta

Compute Glass's delta effect size.

```python
from pyeval import glass_delta

delta = glass_delta(group1, group2)  # Uses group2's std as denominator
```

**Returns:** `float` - Glass's delta value

---

## Correlation Measures

### correlation_coefficient

Compute correlation between two variables.

```python
from pyeval import correlation_coefficient

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Pearson correlation
r = correlation_coefficient(x, y, method='pearson')
print(f"Pearson r: {r:.3f}")

# Spearman correlation
rho = correlation_coefficient(x, y, method='spearman')
print(f"Spearman ρ: {rho:.3f}")

# Kendall correlation
tau = correlation_coefficient(x, y, method='kendall')
print(f"Kendall τ: {tau:.3f}")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | list | required | First variable |
| `y` | list | required | Second variable |
| `method` | str | 'pearson' | 'pearson', 'spearman', or 'kendall' |

**Returns:** `float` - Correlation coefficient

---

## Descriptive Statistics

### descriptive_stats

Compute comprehensive descriptive statistics.

```python
from pyeval import descriptive_stats

data = [0.85, 0.87, 0.86, 0.88, 0.84, 0.89, 0.87, 0.86]

stats = descriptive_stats(data)
print(stats)
# {
#     'mean': 0.865,
#     'median': 0.865,
#     'std': 0.016,
#     'variance': 0.00026,
#     'min': 0.84,
#     'max': 0.89,
#     'range': 0.05,
#     'q1': 0.855,
#     'q3': 0.875,
#     'iqr': 0.02,
#     'skewness': 0.0,
#     'kurtosis': -1.2,
#     'n': 8
# }
```

**Returns:** `dict` - Comprehensive statistics

---

### percentile

Compute percentile value.

```python
from pyeval import percentile

data = [0.85, 0.87, 0.86, 0.88, 0.84, 0.89, 0.87, 0.86]

p50 = percentile(data, 50)  # Median
p95 = percentile(data, 95)  # 95th percentile
```

**Returns:** `float` - Percentile value

---

## Multiple Comparison Corrections

### bonferroni_correction

Apply Bonferroni correction to p-values.

```python
from pyeval import bonferroni_correction

p_values = [0.01, 0.03, 0.04, 0.02]

corrected = bonferroni_correction(p_values)
print(corrected)
# {
#     'corrected_p_values': [0.04, 0.12, 0.16, 0.08],
#     'significant': [True, False, False, False],  # at α=0.05
#     'alpha': 0.0125  # 0.05/4
# }
```

**Returns:** `dict` - Corrected p-values

---

### holm_bonferroni

Apply Holm-Bonferroni correction (less conservative).

```python
from pyeval import holm_bonferroni

p_values = [0.01, 0.03, 0.04, 0.02]

corrected = holm_bonferroni(p_values)
```

**Returns:** `dict` - Corrected p-values with significance

---

## Complete Statistical Functions List

| Function | Description |
|----------|-------------|
| `bootstrap_confidence_interval` | Bootstrap CI for any statistic |
| `paired_t_test` | Paired samples t-test |
| `independent_t_test` | Independent samples t-test |
| `wilcoxon_signed_rank` | Non-parametric paired test |
| `mann_whitney_u` | Non-parametric independent test |
| `mcnemar_test` | Classifier comparison test |
| `permutation_test` | Permutation-based test |
| `cohens_d` | Cohen's d effect size |
| `hedges_g` | Hedges' g effect size |
| `glass_delta` | Glass's delta effect size |
| `correlation_coefficient` | Pearson/Spearman/Kendall correlation |
| `descriptive_stats` | Comprehensive descriptive statistics |
| `percentile` | Percentile computation |
| `bonferroni_correction` | Bonferroni multiple comparison |
| `holm_bonferroni` | Holm-Bonferroni correction |

---

## Usage Tips

### Choosing the Right Test

| Scenario | Recommended Test |
|----------|------------------|
| Comparing 2 models, same data | Paired t-test or Wilcoxon |
| Comparing 2 models, different data | Independent t-test or Mann-Whitney |
| Non-normal distributions | Wilcoxon or Mann-Whitney |
| Comparing classifiers | McNemar's test |
| Small sample, no assumptions | Permutation test |
| Multiple comparisons | Apply Bonferroni/Holm correction |

### Comprehensive Model Comparison

```python
from pyeval import (
    paired_t_test, wilcoxon_signed_rank, cohens_d,
    bootstrap_confidence_interval
)

def compare_models(model1_scores, model2_scores, name1="Model1", name2="Model2"):
    """Comprehensive comparison of two models."""
    
    # Statistical tests
    t_test = paired_t_test(model1_scores, model2_scores)
    wilcoxon = wilcoxon_signed_rank(model1_scores, model2_scores)
    
    # Effect size
    effect = cohens_d(model1_scores, model2_scores)
    
    # Confidence intervals
    ci1 = bootstrap_confidence_interval(model1_scores)
    ci2 = bootstrap_confidence_interval(model2_scores)
    
    # Report
    print(f"=== {name1} vs {name2} ===")
    print(f"{name1}: {ci1['point_estimate']:.4f} [{ci1['ci_lower']:.4f}, {ci1['ci_upper']:.4f}]")
    print(f"{name2}: {ci2['point_estimate']:.4f} [{ci2['ci_lower']:.4f}, {ci2['ci_upper']:.4f}]")
    print(f"Paired t-test: t={t_test['t_statistic']:.3f}, p={t_test['p_value']:.4f}")
    print(f"Wilcoxon: p={wilcoxon['p_value']:.4f}")
    print(f"Cohen's d: {effect:.3f}")
    
    if t_test['is_significant']:
        better = name2 if ci2['point_estimate'] > ci1['point_estimate'] else name1
        print(f"Conclusion: {better} is significantly better")
    else:
        print("Conclusion: No significant difference")
```
