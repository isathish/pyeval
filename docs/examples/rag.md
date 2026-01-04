# RAG Examples

Comprehensive examples for Retrieval-Augmented Generation evaluation with PyEval.

---

## 🔍 RAG Pipeline Overview

RAG (Retrieval-Augmented Generation) combines retrieval and generation. Evaluation covers:

1. **Retrieval Quality** - Did we retrieve relevant documents?
2. **Generation Quality** - Is the answer faithful to context?
3. **End-to-End Quality** - Is the final answer correct?

```
Query → Retriever → Context → Generator → Response
         ↓                      ↓
    Retrieval Metrics    Generation Metrics
```

---

## 📊 Retrieval Evaluation

### Context Relevance

Measure how relevant the retrieved context is to the query.

```python
from pyeval import context_relevance

query = "What are the benefits of machine learning?"

# Highly relevant context
relevant_context = """
Machine learning offers numerous benefits including automated decision making,
pattern recognition, predictive analytics, and the ability to process large 
amounts of data efficiently. It enables personalization and improves over time.
"""

# Partially relevant context
partial_context = """
Machine learning is a subset of artificial intelligence. It was first 
introduced in the 1950s. Many companies use machine learning today.
"""

# Irrelevant context
irrelevant_context = """
The weather today is sunny with a high of 75 degrees. Tomorrow will 
bring clouds and possibly rain in the afternoon.
"""

print("Context Relevance Scores:")
print(f"  Relevant:   {context_relevance(query, relevant_context):.4f}")
print(f"  Partial:    {context_relevance(query, partial_context):.4f}")
print(f"  Irrelevant: {context_relevance(query, irrelevant_context):.4f}")
```

---

### Retrieval Precision & Recall

Evaluate document-level retrieval quality.

```python
from pyeval import (
    retrieval_precision,
    retrieval_recall,
    retrieval_f1,
    mean_reciprocal_rank
)

# For each query: list of retrieved doc IDs
retrieved_docs = [
    [1, 5, 3, 8, 2],    # Query 1: top-5 retrieved
    [4, 2, 7, 1, 9],    # Query 2
    [3, 6, 1, 8, 5],    # Query 3
]

# Ground truth: relevant doc IDs for each query
relevant_docs = [
    [1, 2, 10],         # Query 1: actually relevant
    [4, 7, 9],          # Query 2
    [6, 8],             # Query 3
]

print("Retrieval Metrics:")
print("-" * 40)

# Per-query metrics
for i, (ret, rel) in enumerate(zip(retrieved_docs, relevant_docs)):
    p = retrieval_precision(ret, rel)
    r = retrieval_recall(ret, rel)
    print(f"Query {i+1}: Precision={p:.4f}, Recall={r:.4f}")

# Corpus-level metrics
print(f"\nCorpus Precision: {retrieval_precision(retrieved_docs[0], relevant_docs[0]):.4f}")
print(f"Corpus Recall:    {retrieval_recall(retrieved_docs[0], relevant_docs[0]):.4f}")

# MRR - rank of first relevant document
mrr = mean_reciprocal_rank(retrieved_docs, relevant_docs)
print(f"\nMRR: {mrr:.4f}")
```

---

### Precision@K and NDCG@K

Evaluate ranking quality at different cutoffs.

```python
from pyeval import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k
)

retrieved = [[1, 5, 3, 8, 2, 7, 4, 9, 6, 10]]
relevant = [[1, 2, 3]]

print("Metrics at Different K:")
print("-" * 45)
print(f"{'K':<5} {'P@K':>10} {'R@K':>10} {'NDCG@K':>10}")
print("-" * 45)

for k in [1, 3, 5, 10]:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    n = ndcg_at_k(retrieved, relevant, k)
    print(f"{k:<5} {p:>10.4f} {r:>10.4f} {n:>10.4f}")
```

---

## 📝 Generation Evaluation

### Groundedness Score

Check if the response is grounded in the provided context.

```python
from pyeval import groundedness_score

context = """
Python was created by Guido van Rossum and released in 1991. It emphasizes
code readability with significant indentation. Python supports multiple 
programming paradigms, including procedural, object-oriented, and functional.
"""

# Well-grounded response
grounded = """
Python is a programming language released in 1991. It was created by 
Guido van Rossum and is known for its readable code style.
"""

# Partially grounded
partial = """
Python was created in 1991 by Guido van Rossum. It is the most popular
programming language in the world and is used by millions of developers.
"""

# Ungrounded
ungrounded = """
Python was created in 2000 by Microsoft. It is primarily used for
mobile app development and is known for its complex syntax.
"""

print("Groundedness Scores:")
print(f"  Grounded:   {groundedness_score(grounded, context):.4f}")
print(f"  Partial:    {groundedness_score(partial, context):.4f}")
print(f"  Ungrounded: {groundedness_score(ungrounded, context):.4f}")
```

---

### Faithfulness Score

Measure how faithful the response is to the source context.

```python
from pyeval import faithfulness_score

context = """
The company reported Q3 revenue of $5.2 billion, up 15% year-over-year.
Operating income was $1.1 billion with a margin of 21%. The company 
expects Q4 revenue between $5.5 and $5.8 billion.
"""

# Faithful summary
faithful = """
Q3 revenue reached $5.2 billion, a 15% increase from last year. 
Operating income was $1.1 billion. Q4 guidance is $5.5-5.8 billion.
"""

# Unfaithful (contains errors)
unfaithful = """
Q3 revenue was $6.2 billion, up 25% year-over-year. Operating income
reached $2 billion. The company expects Q4 revenue above $6 billion.
"""

print("Faithfulness Scores:")
print(f"  Faithful:   {faithfulness_score(faithful, context):.4f}")
print(f"  Unfaithful: {faithfulness_score(unfaithful, context):.4f}")
```

---

### Answer Correctness

Compare generated answer against ground truth.

```python
from pyeval import answer_correctness

ground_truth = "The capital of France is Paris."

# Correct answers
correct1 = "Paris is the capital of France."
correct2 = "The capital city of France is Paris."

# Partially correct
partial = "The capital of France is a major European city called Paris."

# Incorrect
wrong = "The capital of France is Lyon."

print("Answer Correctness Scores:")
print(f"  Correct 1: {answer_correctness(correct1, ground_truth):.4f}")
print(f"  Correct 2: {answer_correctness(correct2, ground_truth):.4f}")
print(f"  Partial:   {answer_correctness(partial, ground_truth):.4f}")
print(f"  Wrong:     {answer_correctness(wrong, ground_truth):.4f}")
```

---

## 🎯 End-to-End RAG Evaluation

### Using RAGMetrics Class

```python
from pyeval import RAGMetrics

# Sample RAG interaction
query = "What are the key features of Python programming language?"

context = """
Python is a high-level, interpreted programming language known for:
1. Simple and readable syntax
2. Dynamic typing
3. Extensive standard library
4. Support for multiple paradigms (OOP, functional, procedural)
5. Large ecosystem of third-party packages
6. Strong community support
"""

response = """
Python's key features include simple and readable syntax, dynamic typing,
an extensive standard library, and support for multiple programming paradigms
like object-oriented and functional programming. It also has a large ecosystem
of packages and strong community support.
"""

ground_truth = """
Python features include readable syntax, dynamic typing, extensive libraries,
multiple paradigm support, and a large package ecosystem.
"""

# Compute all RAG metrics
metrics = RAGMetrics()
results = metrics.compute(
    query=query,
    context=context,
    response=response,
    ground_truth=ground_truth
)

print("=== RAG Evaluation Results ===\n")
for metric, value in results.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")
```

---

### Evaluating Multiple RAG Samples

```python
from pyeval import (
    context_relevance,
    groundedness_score,
    faithfulness_score,
    answer_correctness,
    mean
)

# Test dataset
test_cases = [
    {
        'query': "What is machine learning?",
        'context': "Machine learning is a subset of AI that enables systems to learn from data.",
        'response': "Machine learning is an AI technique that allows computers to learn from data.",
        'ground_truth': "Machine learning is a type of artificial intelligence."
    },
    {
        'query': "Who created Python?",
        'context': "Python was created by Guido van Rossum and released in 1991.",
        'response': "Guido van Rossum created Python in 1991.",
        'ground_truth': "Guido van Rossum created Python."
    },
    {
        'query': "What is deep learning?",
        'context': "Deep learning uses neural networks with multiple layers to learn representations.",
        'response': "Deep learning is a type of ML using multi-layer neural networks.",
        'ground_truth': "Deep learning uses neural networks with many layers."
    },
]

print("RAG Evaluation Report")
print("=" * 75)
print(f"{'#':<3} {'Relevance':>12} {'Groundedness':>14} {'Faithfulness':>14} {'Correctness':>13}")
print("-" * 75)

scores = {'relevance': [], 'groundedness': [], 'faithfulness': [], 'correctness': []}

for i, tc in enumerate(test_cases):
    rel = context_relevance(tc['query'], tc['context'])
    gnd = groundedness_score(tc['response'], tc['context'])
    fai = faithfulness_score(tc['response'], tc['context'])
    cor = answer_correctness(tc['response'], tc['ground_truth'])
    
    scores['relevance'].append(rel)
    scores['groundedness'].append(gnd)
    scores['faithfulness'].append(fai)
    scores['correctness'].append(cor)
    
    print(f"{i+1:<3} {rel:>12.4f} {gnd:>14.4f} {fai:>14.4f} {cor:>13.4f}")

print("-" * 75)
print(f"{'AVG':<3} {mean(scores['relevance']):>12.4f} {mean(scores['groundedness']):>14.4f} {mean(scores['faithfulness']):>14.4f} {mean(scores['correctness']):>13.4f}")
```

---

## 🔄 RAG Pipeline Comparison

```python
from pyeval import (
    context_relevance,
    groundedness_score,
    answer_correctness,
    mean
)

# Same queries, different RAG configurations
query = "What are the benefits of exercise?"

# Configuration 1: Dense retriever + GPT-4
config1 = {
    'name': 'Dense + GPT-4',
    'context': "Exercise improves cardiovascular health, builds muscle strength, enhances mood through endorphin release, and helps maintain healthy weight.",
    'response': "Exercise benefits include improved heart health, stronger muscles, better mood from endorphins, and weight management."
}

# Configuration 2: BM25 retriever + GPT-3.5
config2 = {
    'name': 'BM25 + GPT-3.5',
    'context': "Physical activity is good for health. Exercise helps the body. Movement is important for wellness.",
    'response': "Exercise is good for your health and helps your body stay well."
}

# Configuration 3: Hybrid retriever + Claude
config3 = {
    'name': 'Hybrid + Claude',
    'context': "Regular exercise provides numerous benefits: cardiovascular improvement, muscle development, mental health benefits through endorphin release, weight control, and improved sleep quality.",
    'response': "The benefits of exercise include cardiovascular health, muscle strength, mental wellness via endorphins, weight management, and better sleep."
}

ground_truth = "Exercise improves heart health, builds muscles, boosts mood, and helps with weight."

configs = [config1, config2, config3]

print("RAG Configuration Comparison")
print("=" * 65)
print(f"{'Configuration':<20} {'Relevance':>12} {'Grounded':>12} {'Correct':>12}")
print("-" * 65)

for cfg in configs:
    rel = context_relevance(query, cfg['context'])
    gnd = groundedness_score(cfg['response'], cfg['context'])
    cor = answer_correctness(cfg['response'], ground_truth)
    
    print(f"{cfg['name']:<20} {rel:>12.4f} {gnd:>12.4f} {cor:>12.4f}")
```

---

## 📈 Production Monitoring Dashboard

```python
from pyeval import (
    context_relevance,
    groundedness_score,
    faithfulness_score,
    answer_correctness,
    mean,
    std
)

def evaluate_rag_batch(samples):
    """Evaluate a batch of RAG samples and return statistics."""
    metrics = {
        'context_relevance': [],
        'groundedness': [],
        'faithfulness': [],
        'answer_correctness': []
    }
    
    for sample in samples:
        metrics['context_relevance'].append(
            context_relevance(sample['query'], sample['context'])
        )
        metrics['groundedness'].append(
            groundedness_score(sample['response'], sample['context'])
        )
        metrics['faithfulness'].append(
            faithfulness_score(sample['response'], sample['context'])
        )
        if 'ground_truth' in sample:
            metrics['answer_correctness'].append(
                answer_correctness(sample['response'], sample['ground_truth'])
            )
    
    return metrics

# Simulated production batch
production_samples = [
    {
        'query': "What is Python?",
        'context': "Python is a programming language created by Guido van Rossum.",
        'response': "Python is a programming language created in 1991.",
        'ground_truth': "Python is a programming language."
    },
    # Add more samples...
]

# Evaluate
results = evaluate_rag_batch(production_samples)

# Generate report
print("=" * 50)
print("RAG PRODUCTION MONITORING DASHBOARD")
print("=" * 50)
print(f"Samples Evaluated: {len(production_samples)}")
print("-" * 50)

for metric, scores in results.items():
    if scores:
        avg = mean(scores)
        s = std(scores) if len(scores) > 1 else 0
        mn = min(scores)
        mx = max(scores)
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Mean: {avg:.4f} ± {s:.4f}")
        print(f"  Range: [{mn:.4f}, {mx:.4f}]")
        
        # Alerts
        if metric == 'groundedness' and avg < 0.7:
            print("  ⚠️ WARNING: Low groundedness detected!")
        if metric == 'faithfulness' and avg < 0.7:
            print("  ⚠️ WARNING: Low faithfulness detected!")
```
