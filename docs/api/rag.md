# RAG Metrics API Reference

Metrics for evaluating Retrieval-Augmented Generation pipelines.

---

## Retrieval Metrics

### context_relevance

Evaluate relevance of retrieved context to the query.

```python
from pyeval import context_relevance

query = "What is machine learning?"
context = "Machine learning is a subset of AI that enables systems to learn from data."

score = context_relevance(query, context)
# Returns: 0.85
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | str | User query |
| `context` | str | Retrieved context |

**Returns:** `float` - Relevance score between 0 and 1

---

### retrieval_precision

Compute precision of retrieved documents.

```python
from pyeval import retrieval_precision

retrieved = [1, 3, 5, 7, 9]  # Retrieved document IDs
relevant = [1, 5, 10, 15]    # Relevant document IDs

precision = retrieval_precision(retrieved, relevant)
# Returns: 2/5 = 0.4
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `retrieved` | list | Retrieved document IDs |
| `relevant` | list | Relevant document IDs |

**Returns:** `float` - Precision score

---

### retrieval_recall

Compute recall of retrieved documents.

```python
from pyeval import retrieval_recall

retrieved = [1, 3, 5, 7, 9]
relevant = [1, 5, 10, 15]

recall = retrieval_recall(retrieved, relevant)
# Returns: 2/4 = 0.5
```

**Returns:** `float` - Recall score

---

### retrieval_f1

Compute F1 score for retrieval.

```python
from pyeval import retrieval_f1

retrieved = [[1, 3, 5], [2, 4, 6]]
relevant = [[1, 2], [4, 5]]

f1 = retrieval_f1(retrieved, relevant)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `retrieved` | list[list] | Retrieved docs per query |
| `relevant` | list[list] | Relevant docs per query |

**Returns:** `float` - Average F1 score

---

### retrieval_mrr

Compute Mean Reciprocal Rank.

```python
from pyeval import retrieval_mrr, mean_reciprocal_rank

retrieved = [[3, 1, 2], [1, 2, 3], [2, 3, 1]]
relevant = [[1], [2], [1]]

mrr = retrieval_mrr(retrieved, relevant)
# Also available as: mean_reciprocal_rank(retrieved, relevant)
```

**Returns:** `float` - MRR score between 0 and 1

---

## Generation Quality Metrics

### groundedness_score

Evaluate if response is grounded in the provided context.

```python
from pyeval import groundedness_score

context = "Python was created by Guido van Rossum in 1991."
response = "Python is a programming language created by Guido van Rossum."

score = groundedness_score(response, context)
# Returns: 0.9 (highly grounded)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `response` | str | Generated response |
| `context` | str | Source context |

**Returns:** `float` - Groundedness score between 0 and 1

---

### answer_correctness

Evaluate correctness of answer against ground truth.

```python
from pyeval import answer_correctness

response = "Machine learning is a type of artificial intelligence."
ground_truth = "Machine learning is a subset of AI."

score = answer_correctness(response, ground_truth)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `response` | str | Generated answer |
| `ground_truth` | str | Expected answer |

**Returns:** `float` - Correctness score between 0 and 1

---

### answer_relevance

Evaluate relevance of answer to the original query.

```python
from pyeval import answer_relevance

query = "What programming language should I learn first?"
response = "Python is recommended for beginners due to its simple syntax."

score = answer_relevance(query, response)
```

**Returns:** `float` - Relevance score between 0 and 1

---

### faithfulness_score

Evaluate if all claims in response are supported by context.

```python
from pyeval import faithfulness_score

context = "The Eiffel Tower is 330 meters tall and located in Paris."
response = "The Eiffel Tower, at 330 meters, stands in Paris, France."

result = faithfulness_score(response, context)
print(result)
# {
#     'faithfulness_score': 0.95,
#     'supported_claims': 2,
#     'total_claims': 2,
#     'unsupported_claims': []
# }
```

**Returns:** `dict` - Faithfulness analysis

---

## Entity & Attribution Metrics

### context_entity_recall

Measure recall of entities from context in the response.

```python
from pyeval import context_entity_recall

context = "Albert Einstein developed the theory of relativity in 1905."
response = "Einstein's theory of relativity revolutionized physics."

score = context_entity_recall(context, response)
# Checks: Einstein ✓, relativity ✓, 1905 ✗
```

**Returns:** `float` - Entity recall score

---

### answer_attribution

Evaluate if answer properly attributes information to sources.

```python
from pyeval import answer_attribution

contexts = [
    {"id": 1, "text": "Python was created by Guido van Rossum."},
    {"id": 2, "text": "Python's first version was released in 1991."}
]

response = "Python was created by Guido van Rossum in 1991."

result = answer_attribution(response, contexts)
print(result)
# {
#     'attribution_score': 1.0,
#     'attributed_to': [1, 2],
#     'unattributed_claims': []
# }
```

**Returns:** `dict` - Attribution analysis

---

### context_utilization

Measure how much of the context was used in generating the response.

```python
from pyeval import context_utilization

context = """
Machine learning is a subset of artificial intelligence.
It uses algorithms to learn from data.
ML can be supervised, unsupervised, or reinforcement learning.
"""

response = "Machine learning uses algorithms to learn from data patterns."

score = context_utilization(context, response)
```

**Returns:** `float` - Utilization score between 0 and 1

---

## Pipeline Metrics

### rag_pipeline_score

Compute overall RAG pipeline quality score.

```python
from pyeval import rag_pipeline_score

query = "What is Python?"
context = "Python is a programming language created by Guido van Rossum."
response = "Python is a popular programming language."
ground_truth = "Python is a general-purpose programming language."

result = rag_pipeline_score(
    query=query,
    context=context,
    response=response,
    ground_truth=ground_truth
)

print(result)
# {
#     'overall_score': 0.82,
#     'retrieval_score': 0.85,
#     'generation_score': 0.80,
#     'component_scores': {
#         'context_relevance': 0.85,
#         'groundedness': 0.90,
#         'answer_correctness': 0.75,
#         'faithfulness': 0.85
#     }
# }
```

**Returns:** `dict` - Comprehensive pipeline evaluation

---

### question_answer_relevance

Evaluate overall QA relevance in RAG context.

```python
from pyeval import question_answer_relevance

query = "How do neural networks learn?"
context = "Neural networks learn by adjusting weights through backpropagation."
response = "Neural networks learn by updating their weights using backpropagation."

score = question_answer_relevance(query, context, response)
```

**Returns:** `float` - QA relevance score

---

## Metric Class

### RAGMetrics

Compute all RAG metrics at once.

```python
from pyeval import RAGMetrics

rag = RAGMetrics()

results = rag.compute(
    query="What is machine learning?",
    context="Machine learning is a subset of AI...",
    response="Machine learning enables computers to learn from data.",
    ground_truth="Machine learning is a type of AI."
)

print(results)
# {
#     'context_relevance': 0.88,
#     'groundedness': 0.92,
#     'answer_correctness': 0.85,
#     'answer_relevance': 0.90,
#     'faithfulness': 0.88,
#     'context_utilization': 0.75,
#     'overall_score': 0.86
# }
```

---

## Complete RAG Metrics List

| Metric | Function | Description |
|--------|----------|-------------|
| Context Relevance | `context_relevance` | Query-context alignment |
| Retrieval Precision | `retrieval_precision` | Precision of retrieved docs |
| Retrieval Recall | `retrieval_recall` | Recall of retrieved docs |
| Retrieval F1 | `retrieval_f1` | F1 score for retrieval |
| MRR | `retrieval_mrr` | Mean Reciprocal Rank |
| Groundedness | `groundedness_score` | Response-context grounding |
| Answer Correctness | `answer_correctness` | Answer accuracy |
| Answer Relevance | `answer_relevance` | Answer-query alignment |
| Faithfulness | `faithfulness_score` | Claim support verification |
| Entity Recall | `context_entity_recall` | Entity coverage |
| Attribution | `answer_attribution` | Source attribution |
| Context Utilization | `context_utilization` | Context usage efficiency |
| Pipeline Score | `rag_pipeline_score` | Overall RAG quality |
| QA Relevance | `question_answer_relevance` | End-to-end relevance |

---

## Usage Tips

### Evaluating Multiple Queries

```python
from pyeval import RAGMetrics

rag = RAGMetrics()

test_cases = [
    {
        "query": "What is Python?",
        "context": "Python is a programming language...",
        "response": "Python is a popular language...",
        "ground_truth": "Python is a general-purpose language..."
    },
    # More test cases...
]

results = []
for case in test_cases:
    result = rag.compute(**case)
    results.append(result)

# Aggregate scores
avg_scores = {
    metric: sum(r[metric] for r in results) / len(results)
    for metric in results[0].keys()
}
```

### Custom Retrieval Evaluation

```python
from pyeval import retrieval_precision, retrieval_recall, retrieval_f1

def evaluate_retriever(queries, retrieved_docs, relevant_docs):
    """Evaluate retrieval performance across multiple queries."""
    
    metrics = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        metrics['precision'].append(retrieval_precision(retrieved, relevant))
        metrics['recall'].append(retrieval_recall(retrieved, relevant))
    
    # Average metrics
    return {
        'avg_precision': sum(metrics['precision']) / len(metrics['precision']),
        'avg_recall': sum(metrics['recall']) / len(metrics['recall']),
        'f1': retrieval_f1(retrieved_docs, relevant_docs)
    }
```

### End-to-End RAG Evaluation

```python
from pyeval import (
    context_relevance, groundedness_score, 
    answer_correctness, faithfulness_score
)

def evaluate_rag_response(query, contexts, response, ground_truth):
    """Comprehensive RAG evaluation."""
    
    # Combine contexts
    combined_context = " ".join(contexts)
    
    # Compute metrics
    relevance = context_relevance(query, combined_context)
    grounded = groundedness_score(response, combined_context)
    correct = answer_correctness(response, ground_truth)
    faithful = faithfulness_score(response, combined_context)
    
    # Overall score (weighted average)
    overall = (
        0.2 * relevance +
        0.3 * grounded +
        0.3 * correct +
        0.2 * faithful['faithfulness_score']
    )
    
    return {
        'context_relevance': relevance,
        'groundedness': grounded,
        'answer_correctness': correct,
        'faithfulness': faithful,
        'overall_score': overall
    }
```
