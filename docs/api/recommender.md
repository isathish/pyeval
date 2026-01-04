# Recommender Metrics API Reference

Metrics for evaluating recommendation systems.

---

## Ranking Metrics

### precision_at_k

Compute Precision@K.

```python
from pyeval import precision_at_k

recommended = [101, 203, 45, 67, 89, 12, 34]
relevant = [45, 89, 78, 123]

p5 = precision_at_k(recommended, relevant, k=5)
# Returns: 2/5 = 0.4 (items 45 and 89 are relevant in top 5)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recommended` | list | required | Ranked list of recommended items |
| `relevant` | list | required | Set of relevant items |
| `k` | int | 10 | Number of top items to consider |

**Returns:** `float` - Precision@K score

---

### recall_at_k

Compute Recall@K.

```python
from pyeval import recall_at_k

recommended = [101, 203, 45, 67, 89, 12, 34]
relevant = [45, 89, 78, 123]

r5 = recall_at_k(recommended, relevant, k=5)
# Returns: 2/4 = 0.5 (2 out of 4 relevant items found in top 5)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recommended` | list | required | Ranked list of recommended items |
| `relevant` | list | required | Set of relevant items |
| `k` | int | 10 | Number of top items to consider |

**Returns:** `float` - Recall@K score

---

### ndcg_at_k

Compute Normalized Discounted Cumulative Gain at K.

```python
from pyeval import ndcg_at_k

recommended = [101, 203, 45, 67, 89]
relevant = [45, 89]

# Binary relevance
ndcg = ndcg_at_k(recommended, relevant, k=5)

# With graded relevance
relevance_scores = {101: 0, 203: 0, 45: 3, 67: 0, 89: 2}
ndcg = ndcg_at_k(recommended, relevant, k=5, relevance_scores=relevance_scores)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recommended` | list | required | Ranked list of recommended items |
| `relevant` | list | required | Set of relevant items |
| `k` | int | 10 | Number of top items |
| `relevance_scores` | dict | None | Item → relevance mapping |

**Returns:** `float` - NDCG@K score between 0 and 1

---

### mean_average_precision

Compute Mean Average Precision (MAP).

```python
from pyeval import mean_average_precision

# Multiple users
recommended_lists = [
    [1, 3, 5, 7],
    [2, 4, 1, 3],
    [5, 2, 3, 1]
]
relevant_lists = [
    [1, 5],
    [1, 2],
    [1, 3, 5]
]

map_score = mean_average_precision(recommended_lists, relevant_lists)
```

**Returns:** `float` - MAP score between 0 and 1

---

### mean_reciprocal_rank

Compute Mean Reciprocal Rank (MRR).

```python
from pyeval import mean_reciprocal_rank

# Multiple queries
recommended_lists = [
    [3, 1, 2],  # First relevant at position 2
    [1, 2, 3],  # First relevant at position 1
    [2, 3, 1]   # First relevant at position 3
]
relevant_lists = [
    [1],
    [1],
    [1]
]

mrr = mean_reciprocal_rank(recommended_lists, relevant_lists)
# Returns: (1/2 + 1/1 + 1/3) / 3 = 0.611
```

**Returns:** `float` - MRR score between 0 and 1

---

### hit_rate

Compute Hit Rate (HR@K).

```python
from pyeval import hit_rate

recommended = [101, 203, 45, 67, 89]
relevant = [45]

hr = hit_rate(recommended, relevant, k=5)
# Returns: 1.0 (item 45 is in top 5)
```

**Returns:** `float` - 1.0 if any relevant item in top K, else 0.0

---

## Diversity Metrics

### intra_list_diversity

Compute diversity within a recommendation list.

```python
from pyeval import intra_list_diversity

# Item features/embeddings
item_features = {
    1: [0.1, 0.2, 0.3],
    2: [0.15, 0.25, 0.35],
    3: [0.9, 0.8, 0.7],
    4: [0.5, 0.5, 0.5]
}

recommended = [1, 2, 3, 4]

diversity = intra_list_diversity(recommended, item_features)
# Higher value = more diverse recommendations
```

**Returns:** `float` - Diversity score

---

### inter_list_diversity

Compute diversity across different users' recommendations.

```python
from pyeval import inter_list_diversity

# Recommendations for multiple users
user_recommendations = {
    'user1': [1, 2, 3],
    'user2': [1, 2, 4],
    'user3': [1, 5, 6]
}

diversity = inter_list_diversity(user_recommendations)
```

**Returns:** `float` - Inter-list diversity score

---

### entropy_diversity

Compute recommendation diversity using entropy.

```python
from pyeval import entropy_diversity

# Item categories
item_categories = {
    1: 'action',
    2: 'action',
    3: 'comedy',
    4: 'drama',
    5: 'action'
}

recommended = [1, 2, 3, 4, 5]

entropy = entropy_diversity(recommended, item_categories)
```

**Returns:** `float` - Entropy-based diversity

---

### gini_index

Compute Gini index for recommendation concentration.

```python
from pyeval import gini_index

# Recommendation counts per item
recommendation_counts = [100, 50, 30, 10, 5, 3, 2]

gini = gini_index(recommendation_counts)
# Returns: 0 (equal) to 1 (concentrated)
```

**Returns:** `float` - Gini coefficient

---

## Coverage Metrics

### catalog_coverage

Compute what fraction of catalog is recommended.

```python
from pyeval import catalog_coverage

all_items = list(range(1000))  # 1000 items in catalog
recommended_items = [1, 5, 10, 20, 50, 100, 200]

coverage = catalog_coverage(recommended_items, all_items)
# Returns: 7/1000 = 0.007
```

**Returns:** `float` - Coverage ratio

---

### user_coverage

Compute what fraction of users receive recommendations.

```python
from pyeval import user_coverage

all_users = ['u1', 'u2', 'u3', 'u4', 'u5']
users_with_recs = ['u1', 'u2', 'u3']

coverage = user_coverage(users_with_recs, all_users)
# Returns: 3/5 = 0.6
```

**Returns:** `float` - User coverage ratio

---

## Novelty & Serendipity

### novelty_score

Compute novelty of recommendations.

```python
from pyeval import novelty_score

recommended = [101, 203, 45]
item_popularity = {
    101: 0.8,   # Very popular
    203: 0.05,  # Rare
    45: 0.3     # Moderately popular
}

novelty = novelty_score(recommended, item_popularity)
# Higher when recommending less popular items
```

**Returns:** `float` - Novelty score

---

### serendipity_score

Compute serendipity (unexpected but relevant) of recommendations.

```python
from pyeval import serendipity_score

recommended = [1, 2, 3, 4, 5]
relevant = [2, 4, 5]
expected = [1, 2, 3]  # Items user would expect

serendipity = serendipity_score(recommended, relevant, expected)
# Measures relevant items that weren't expected
```

**Returns:** `float` - Serendipity score

---

## Ranking Correlation

### ranking_correlation

Compute correlation between two rankings.

```python
from pyeval import ranking_correlation

ranking1 = [1, 2, 3, 4, 5]
ranking2 = [1, 3, 2, 5, 4]

corr = ranking_correlation(ranking1, ranking2, method='spearman')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ranking1` | list | required | First ranking |
| `ranking2` | list | required | Second ranking |
| `method` | str | 'spearman' | 'spearman' or 'kendall' |

**Returns:** `float` - Correlation coefficient

---

## Metric Class

### RecommenderMetrics

Compute all recommender metrics at once.

```python
from pyeval import RecommenderMetrics

rm = RecommenderMetrics()

recommended = [101, 203, 45, 67, 89]
relevant = [45, 89, 123]

results = rm.compute(recommended, relevant, k=5)

print(results)
# {
#     'precision_at_k': 0.4,
#     'recall_at_k': 0.67,
#     'ndcg_at_k': 0.65,
#     'hit_rate': 1.0,
#     'mrr': 0.33,
#     ...
# }
```

---

## Complete Recommender Metrics List

| Metric | Function | Description |
|--------|----------|-------------|
| Precision@K | `precision_at_k` | Relevant items in top K |
| Recall@K | `recall_at_k` | Coverage of relevant items |
| NDCG@K | `ndcg_at_k` | Ranking quality |
| MAP | `mean_average_precision` | Average precision |
| MRR | `mean_reciprocal_rank` | First relevant rank |
| Hit Rate | `hit_rate` | Any relevant in top K |
| ILD | `intra_list_diversity` | Within-list diversity |
| Inter-list Diversity | `inter_list_diversity` | Cross-user diversity |
| Entropy | `entropy_diversity` | Category diversity |
| Gini | `gini_index` | Concentration |
| Catalog Coverage | `catalog_coverage` | Item space coverage |
| User Coverage | `user_coverage` | User space coverage |
| Novelty | `novelty_score` | Unexpectedness |
| Serendipity | `serendipity_score` | Surprising relevance |
| Ranking Correlation | `ranking_correlation` | Ranking agreement |

---

## Usage Tips

### Evaluating at Multiple K Values

```python
from pyeval import precision_at_k, recall_at_k, ndcg_at_k

recommended = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
relevant = [3, 7, 11, 15]

k_values = [1, 3, 5, 10]

for k in k_values:
    p = precision_at_k(recommended, relevant, k=k)
    r = recall_at_k(recommended, relevant, k=k)
    n = ndcg_at_k(recommended, relevant, k=k)
    print(f"K={k}: P@K={p:.3f}, R@K={r:.3f}, NDCG@K={n:.3f}")
```

### Multi-User Evaluation

```python
from pyeval import RecommenderMetrics

rm = RecommenderMetrics()

# User → (recommended, relevant)
user_data = {
    'user1': ([1, 2, 3], [1, 4]),
    'user2': ([2, 3, 4], [2, 3]),
    'user3': ([1, 4, 5], [5]),
}

user_scores = {}
for user, (rec, rel) in user_data.items():
    user_scores[user] = rm.compute(rec, rel, k=3)

# Average across users
avg_precision = sum(s['precision_at_k'] for s in user_scores.values()) / len(user_scores)
print(f"Average P@3: {avg_precision:.3f}")
```

### Comprehensive Evaluation Report

```python
from pyeval import (
    precision_at_k, recall_at_k, ndcg_at_k,
    novelty_score, intra_list_diversity, catalog_coverage
)

def full_evaluation(recommended_lists, relevant_lists, item_features, catalog):
    """Comprehensive recommendation evaluation."""
    
    # Accuracy metrics
    precisions = [precision_at_k(r, rel) for r, rel in zip(recommended_lists, relevant_lists)]
    recalls = [recall_at_k(r, rel) for r, rel in zip(recommended_lists, relevant_lists)]
    ndcgs = [ndcg_at_k(r, rel) for r, rel in zip(recommended_lists, relevant_lists)]
    
    # Beyond-accuracy metrics
    diversities = [intra_list_diversity(r, item_features) for r in recommended_lists]
    
    # Aggregate
    all_recommended = set(item for recs in recommended_lists for item in recs)
    coverage = catalog_coverage(list(all_recommended), catalog)
    
    return {
        'avg_precision': sum(precisions) / len(precisions),
        'avg_recall': sum(recalls) / len(recalls),
        'avg_ndcg': sum(ndcgs) / len(ndcgs),
        'avg_diversity': sum(diversities) / len(diversities),
        'catalog_coverage': coverage
    }
```
