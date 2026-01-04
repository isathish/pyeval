# LLM Metrics API Reference

Metrics for evaluating Large Language Model outputs.

---

## Content Quality Metrics

### toxicity_score

Evaluate toxicity in generated text using keyword and pattern matching.

```python
from pyeval import toxicity_score

response = "This is a friendly and helpful response."
result = toxicity_score(response)

print(result)
# {
#     'toxicity_score': 0.05,
#     'is_toxic': False,
#     'toxic_patterns_found': [],
#     'severity': 'none'
# }
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Text to evaluate |
| `threshold` | float | 0.5 | Toxicity threshold |

**Returns:** `dict` - Toxicity analysis results

---

### coherence_score

Evaluate the coherence and logical flow of text.

```python
from pyeval import coherence_score

response = """
Machine learning is a subset of artificial intelligence.
It enables computers to learn from data.
This learning improves their performance over time.
"""

result = coherence_score(response)
print(result['coherence_score'])  # 0.85
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | str | Text to evaluate |

**Returns:** `dict` - {'coherence_score': float, 'sentence_count': int, 'transition_quality': float}

---

### hallucination_score

Detect hallucinations by comparing response to reference context.

```python
from pyeval import hallucination_score

context = "Python was created by Guido van Rossum in 1991."
response = "Python was created by Guido van Rossum in 1989."

result = hallucination_score(response, context)
print(result)
# {
#     'hallucination_score': 0.3,
#     'grounded_ratio': 0.7,
#     'ungrounded_claims': ['1989']
# }
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `response` | str | Generated response |
| `context` | str | Reference context/facts |

**Returns:** `dict` - Hallucination analysis

---

### bias_detection_score

Detect potential bias in text.

```python
from pyeval import bias_detection_score

text = "Engineers are typically men who work long hours."

result = bias_detection_score(text)
print(result)
# {
#     'bias_score': 0.6,
#     'bias_types': ['gender'],
#     'flagged_phrases': ['typically men'],
#     'severity': 'moderate'
# }
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | str | Text to analyze |

**Returns:** `dict` - Bias detection results

---

### instruction_following_score

Evaluate how well the response follows the given instruction.

```python
from pyeval import instruction_following_score

instruction = "Explain quantum computing in simple terms"
response = "Quantum computing uses quantum bits that can be 0 and 1 at the same time..."

result = instruction_following_score(instruction, response)
print(result['instruction_following_score'])  # 0.85
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `instruction` | str | The original instruction/prompt |
| `response` | str | Generated response |

**Returns:** `dict` - Instruction following analysis

---

## Safety Metrics

### safety_score

Evaluate overall safety of generated content.

```python
from pyeval import safety_score

response = "Here's how to bake a delicious chocolate cake..."

result = safety_score(response)
print(result)
# {
#     'safety_score': 0.95,
#     'is_safe': True,
#     'risk_categories': [],
#     'confidence': 0.9
# }
```

**Returns:** `dict` - Safety evaluation results

---

### harmful_content_score

Detect harmful or dangerous content.

```python
from pyeval import harmful_content_score

text = "Always consult a doctor before taking any medication."
result = harmful_content_score(text)
```

**Returns:** `dict` - {'harmful_score': float, 'categories': list, 'is_harmful': bool}

---

## Conversation Metrics

### multi_turn_coherence

Evaluate coherence across a multi-turn conversation.

```python
from pyeval import multi_turn_coherence

conversation = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "What can I build with it?"},
    {"role": "assistant", "content": "You can build web apps, ML models, and more."}
]

result = multi_turn_coherence(conversation)
print(result['coherence_score'])  # 0.88
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `conversation` | list[dict] | List of conversation turns |

**Returns:** `dict` - Multi-turn coherence analysis

---

### response_diversity

Measure diversity across multiple responses.

```python
from pyeval import response_diversity

responses = [
    "Machine learning is great for data analysis.",
    "ML helps analyze large datasets efficiently.",
    "Using ML, we can process data at scale."
]

result = response_diversity(responses)
print(result['diversity_score'])  # 0.72
```

**Returns:** `dict` - {'diversity_score': float, 'unique_ngrams': int}

---

## Summarization Metrics

### summarization_quality

Evaluate summary quality against source document.

```python
from pyeval import summarization_quality

source = """
Machine learning is a method of data analysis that automates 
analytical model building. It is a branch of artificial intelligence 
based on the idea that systems can learn from data, identify patterns 
and make decisions with minimal human intervention.
"""

summary = "Machine learning automates data analysis using AI to find patterns."

result = summarization_quality(source, summary)
print(result)
# {
#     'quality_score': 0.82,
#     'coverage': 0.75,
#     'compression_ratio': 0.15,
#     'informativeness': 0.85
# }
```

**Returns:** `dict` - Summarization quality metrics

---

## Factuality Metrics

### factuality_score

Evaluate factual accuracy of a response.

```python
from pyeval import factuality_score

facts = [
    "The Earth orbits the Sun",
    "Water boils at 100°C at sea level",
    "Python was created in 1991"
]

response = "The Earth orbits the Sun, and water boils at 100 degrees Celsius."

result = factuality_score(response, facts)
print(result['factuality_score'])  # 0.9
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `response` | str | Text to evaluate |
| `facts` | list[str] | Known facts to verify against |

**Returns:** `dict` - Factuality analysis

---

## Metric Class

### LLMMetrics

Compute all LLM metrics at once.

```python
from pyeval import LLMMetrics

llm_metrics = LLMMetrics()

result = llm_metrics.compute(
    response="Machine learning enables computers to learn from data.",
    context="ML is a subset of AI focusing on learning algorithms.",
    instruction="Explain machine learning briefly"
)

print(result)
# {
#     'toxicity': 0.02,
#     'coherence': 0.92,
#     'hallucination': 0.15,
#     'instruction_following': 0.88,
#     'safety': 0.98,
#     ...
# }
```

---

## Complete LLM Metrics List

| Metric | Function | Description |
|--------|----------|-------------|
| Toxicity | `toxicity_score` | Harmful language detection |
| Coherence | `coherence_score` | Logical flow evaluation |
| Hallucination | `hallucination_score` | Factual grounding check |
| Bias | `bias_detection_score` | Bias pattern detection |
| Instruction Following | `instruction_following_score` | Prompt adherence |
| Safety | `safety_score` | Content safety evaluation |
| Harmful Content | `harmful_content_score` | Dangerous content detection |
| Multi-turn Coherence | `multi_turn_coherence` | Conversation consistency |
| Response Diversity | `response_diversity` | Output variation |
| Summarization Quality | `summarization_quality` | Summary evaluation |
| Factuality | `factuality_score` | Fact verification |

---

## Usage Tips

### Combining Multiple Metrics

```python
from pyeval import (
    toxicity_score, coherence_score, 
    hallucination_score, safety_score
)

def comprehensive_evaluation(response, context=None):
    results = {
        'toxicity': toxicity_score(response),
        'coherence': coherence_score(response),
        'safety': safety_score(response)
    }
    
    if context:
        results['hallucination'] = hallucination_score(response, context)
    
    # Compute overall quality score
    scores = [r.get('toxicity_score', r.get('coherence_score', r.get('safety_score', 0))) 
              for r in results.values()]
    results['overall'] = sum(scores) / len(scores)
    
    return results
```

### Batch Evaluation

```python
from pyeval import LLMMetrics

metrics = LLMMetrics()

responses = [
    "Response 1...",
    "Response 2...",
    "Response 3..."
]

results = [metrics.compute(response=r) for r in responses]

# Aggregate results
avg_toxicity = sum(r['toxicity'] for r in results) / len(results)
```
