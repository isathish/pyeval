# LLM Examples

Comprehensive examples for Large Language Model evaluation with PyEval.

---

## 🤖 Response Quality Evaluation

### Toxicity Detection

Detect harmful, offensive, or inappropriate content in LLM outputs.

```python
from pyeval import toxicity_score

# Clean response
clean_response = """
Machine learning is a fascinating field of study. It enables computers 
to learn from data and make predictions. Many applications benefit from ML,
including healthcare, finance, and transportation.
"""

result = toxicity_score(clean_response)
print("Clean Response Analysis:")
print(f"  Toxicity Score: {result['toxicity_score']:.4f}")
print(f"  Is Toxic: {result['is_toxic']}")
print(f"  Severity: {result['severity']}")
print(f"  Patterns Found: {result['toxic_patterns_found']}")

# Problematic response
problematic = "This is terrible and stupid. What an idiotic idea."

result = toxicity_score(problematic)
print("\nProblematic Response Analysis:")
print(f"  Toxicity Score: {result['toxicity_score']:.4f}")
print(f"  Is Toxic: {result['is_toxic']}")
print(f"  Severity: {result['severity']}")
print(f"  Patterns Found: {result['toxic_patterns_found']}")

# Custom threshold
result = toxicity_score(clean_response, threshold=0.3)
print(f"\nWith lower threshold: Is Toxic = {result['is_toxic']}")
```

---

### Coherence Evaluation

Measure logical flow and sentence connectivity.

```python
from pyeval import coherence_score

# Coherent response
coherent = """
Machine learning is a subset of artificial intelligence. It enables 
computers to learn from data. This learning process improves their 
performance over time. As a result, ML systems can make accurate 
predictions without explicit programming.
"""

result = coherence_score(coherent)
print("Coherent Response:")
print(f"  Coherence Score: {result['coherence_score']:.4f}")
print(f"  Sentence Count: {result['sentence_count']}")
print(f"  Transition Quality: {result.get('transition_quality', 'N/A')}")

# Incoherent response
incoherent = """
Machine learning uses data. The weather is nice today. Cats are 
furry animals. Python is a programming language. Pizza is delicious.
"""

result = coherence_score(incoherent)
print("\nIncoherent Response:")
print(f"  Coherence Score: {result['coherence_score']:.4f}")
print(f"  Sentence Count: {result['sentence_count']}")
```

---

### Hallucination Detection

Identify claims not supported by the provided context.

```python
from pyeval import hallucination_score

context = """
Python was created by Guido van Rossum and first released in 1991. 
It is known for its simple syntax and readability. Python supports 
multiple programming paradigms including procedural, object-oriented, 
and functional programming.
"""

# Faithful response
faithful = """
Python is a programming language created by Guido van Rossum in 1991. 
It is known for its readable syntax and supports object-oriented programming.
"""

result = hallucination_score(faithful, context)
print("Faithful Response:")
print(f"  Hallucination Score: {result['hallucination_score']:.4f}")
print(f"  Grounded Ratio: {result['grounded_ratio']:.4f}")
print(f"  Ungrounded Claims: {result.get('ungrounded_claims', [])}")

# Hallucinated response
hallucinated = """
Python was created by Guido van Rossum in 1989 at Microsoft. It is the 
fastest programming language and is primarily used for mobile app development.
"""

result = hallucination_score(hallucinated, context)
print("\nHallucinated Response:")
print(f"  Hallucination Score: {result['hallucination_score']:.4f}")
print(f"  Grounded Ratio: {result['grounded_ratio']:.4f}")
print(f"  Ungrounded Claims: {result.get('ungrounded_claims', [])}")
```

---

### Bias Detection

Identify potential biases in generated text.

```python
from pyeval import bias_detection_score

# Biased text
biased_text = """
Men are naturally better at math and science, while women excel at 
nurturing and caregiving. Older employees are typically less adaptable 
to new technology.
"""

result = bias_detection_score(biased_text)
print("Bias Analysis:")
print(f"  Overall Bias Score: {result['bias_score']:.4f}")
print(f"  Categories Detected: {result.get('categories', [])}")
print(f"  Biased Phrases: {result.get('biased_phrases', [])}")

# Neutral text
neutral_text = """
Machine learning algorithms can be applied across various domains.
Research shows that diverse teams often produce better results.
Individual capabilities vary regardless of demographic factors.
"""

result = bias_detection_score(neutral_text)
print("\nNeutral Text:")
print(f"  Overall Bias Score: {result['bias_score']:.4f}")
print(f"  Categories Detected: {result.get('categories', [])}")
```

---

### Fluency Evaluation

Measure grammatical correctness and natural language quality.

```python
from pyeval import fluency_score

# Fluent text
fluent = """
The quick brown fox jumps over the lazy dog. This sentence contains
every letter of the alphabet and demonstrates proper grammar and 
sentence structure.
"""

result = fluency_score(fluent)
print("Fluent Text:")
print(f"  Fluency Score: {result['fluency_score']:.4f}")
print(f"  Grammar Score: {result.get('grammar_score', 'N/A')}")

# Disfluent text
disfluent = """
The quick brown fox jump over lazy dog. This sentence contains
every letter alphabet and demonstrate proper grammar sentence structure.
"""

result = fluency_score(disfluent)
print("\nDisfluent Text:")
print(f"  Fluency Score: {result['fluency_score']:.4f}")
```

---

## 📊 Comprehensive LLM Evaluation

### Using LLMMetrics Class

```python
from pyeval import LLMMetrics

# Sample LLM interaction
context = """
Artificial Intelligence (AI) is intelligence demonstrated by machines.
AI research began in the 1950s. Machine learning is a subset of AI that
focuses on learning from data. Deep learning uses neural networks with
multiple layers.
"""

response = """
AI is demonstrated by machines and began in the 1950s. Machine learning,
a subset of AI, learns from data. Deep learning uses multi-layer neural networks.
This technology is transforming healthcare and transportation.
"""

# Compute all metrics
metrics = LLMMetrics()
results = metrics.compute(response, context=context)

print("=== Comprehensive LLM Evaluation ===\n")
for metric, value in results.items():
    if isinstance(value, dict):
        print(f"{metric}:")
        for k, v in value.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    elif isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")
```

---

### Evaluating Multiple Responses

```python
from pyeval import (
    toxicity_score,
    coherence_score,
    hallucination_score,
    fluency_score
)

query = "What is machine learning?"

context = """
Machine learning is a type of artificial intelligence that enables 
computers to learn from data without being explicitly programmed.
"""

responses = {
    'GPT-4': """
        Machine learning is a branch of AI that allows systems to learn 
        and improve from experience automatically. It works by analyzing 
        patterns in data.
    """,
    'Claude': """
        Machine learning enables computers to learn from data without 
        explicit programming. It's a subset of artificial intelligence 
        focused on pattern recognition.
    """,
    'LLaMA': """
        ML is AI that learns from data. Computers can improve automatically
        by finding patterns. Very useful for predictions.
    """,
}

print("LLM Response Comparison:")
print("=" * 70)
print(f"{'Model':<12} {'Toxicity':>10} {'Coherence':>11} {'Hallucin.':>11} {'Fluency':>10}")
print("-" * 70)

for model, response in responses.items():
    tox = toxicity_score(response)['toxicity_score']
    coh = coherence_score(response)['coherence_score']
    hal = hallucination_score(response, context)['hallucination_score']
    flu = fluency_score(response)['fluency_score']
    
    print(f"{model:<12} {tox:>10.4f} {coh:>11.4f} {hal:>11.4f} {flu:>10.4f}")
```

---

## 🔍 Instruction Following

### Evaluating Task Completion

```python
from pyeval import instruction_following_score

instruction = "Write a haiku about programming"

# Good response (follows instruction)
good_response = """
Code flows like water
Bugs hide in the logic maze
Debug brings the light
"""

# Bad response (doesn't follow instruction)
bad_response = """
Programming is really fun. I love writing code in Python. 
It's my favorite language because it's so easy to learn.
"""

result_good = instruction_following_score(instruction, good_response)
result_bad = instruction_following_score(instruction, bad_response)

print("Instruction Following Evaluation:")
print(f"\nGood Response:")
print(f"  Score: {result_good['score']:.4f}")
print(f"  Format Correct: {result_good.get('format_correct', 'N/A')}")

print(f"\nBad Response:")
print(f"  Score: {result_bad['score']:.4f}")
print(f"  Format Correct: {result_bad.get('format_correct', 'N/A')}")
```

---

## 📈 Production Monitoring

### Batch Evaluation Pipeline

```python
from pyeval import (
    toxicity_score,
    coherence_score,
    hallucination_score,
    LLMMetrics,
    mean
)

# Simulated production responses
production_logs = [
    {
        'query': "What is Python?",
        'context': "Python is a programming language created in 1991.",
        'response': "Python is a programming language created by Guido van Rossum."
    },
    {
        'query': "Explain ML",
        'context': "Machine learning is a subset of AI.",
        'response': "ML allows computers to learn from data automatically."
    },
    {
        'query': "What is deep learning?",
        'context': "Deep learning uses neural networks.",
        'response': "Deep learning uses multi-layer neural networks for complex patterns."
    },
]

# Evaluate batch
print("Production Monitoring Report")
print("=" * 60)

toxicity_scores = []
coherence_scores = []
hallucination_scores = []

for i, log in enumerate(production_logs):
    tox = toxicity_score(log['response'])['toxicity_score']
    coh = coherence_score(log['response'])['coherence_score']
    hal = hallucination_score(log['response'], log['context'])['hallucination_score']
    
    toxicity_scores.append(tox)
    coherence_scores.append(coh)
    hallucination_scores.append(hal)
    
    print(f"\nQuery {i+1}: {log['query'][:30]}...")
    print(f"  Toxicity: {tox:.4f} | Coherence: {coh:.4f} | Hallucination: {hal:.4f}")

# Summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("-" * 60)
print(f"Average Toxicity:      {mean(toxicity_scores):.4f}")
print(f"Average Coherence:     {mean(coherence_scores):.4f}")
print(f"Average Hallucination: {mean(hallucination_scores):.4f}")
print(f"Samples Evaluated:     {len(production_logs)}")

# Alert thresholds
if mean(toxicity_scores) > 0.3:
    print("\n⚠️ ALERT: High average toxicity detected!")
if mean(hallucination_scores) > 0.5:
    print("\n⚠️ ALERT: High hallucination rate detected!")
```

---

## 🎯 Custom Evaluation Rubric

```python
from pyeval import (
    toxicity_score,
    coherence_score,
    hallucination_score,
    fluency_score
)

def evaluate_llm_response(response, context, weights=None):
    """
    Custom rubric for LLM evaluation.
    
    Args:
        response: LLM generated response
        context: Reference context
        weights: Dict of metric weights (must sum to 1.0)
    
    Returns:
        Dict with individual scores and weighted total
    """
    if weights is None:
        weights = {
            'toxicity': 0.2,      # Weight for (1 - toxicity)
            'coherence': 0.3,
            'faithfulness': 0.3,  # 1 - hallucination
            'fluency': 0.2
        }
    
    # Get individual scores
    tox = 1 - toxicity_score(response)['toxicity_score']  # Invert: higher is better
    coh = coherence_score(response)['coherence_score']
    faith = 1 - hallucination_score(response, context)['hallucination_score']
    flu = fluency_score(response)['fluency_score']
    
    # Calculate weighted score
    weighted_total = (
        weights['toxicity'] * tox +
        weights['coherence'] * coh +
        weights['faithfulness'] * faith +
        weights['fluency'] * flu
    )
    
    return {
        'non_toxicity': tox,
        'coherence': coh,
        'faithfulness': faith,
        'fluency': flu,
        'weighted_score': weighted_total,
        'grade': 'A' if weighted_total >= 0.9 else
                 'B' if weighted_total >= 0.8 else
                 'C' if weighted_total >= 0.7 else
                 'D' if weighted_total >= 0.6 else 'F'
    }

# Example usage
context = "Python is a programming language created by Guido van Rossum in 1991."
response = "Python is a popular programming language created in 1991 by Guido van Rossum."

result = evaluate_llm_response(response, context)

print("Custom Evaluation Rubric:")
print("-" * 40)
for metric, value in result.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")
```
