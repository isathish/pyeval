# NLP Metrics API Reference

Natural Language Processing metrics for text generation evaluation.

---

## Text Generation Metrics

### bleu_score

Compute BLEU (Bilingual Evaluation Understudy) score.

```python
from pyeval import bleu_score

reference = "The quick brown fox jumps over the lazy dog"
hypothesis = "A fast brown fox leaps over the lazy dog"

# Default: BLEU-4
score = bleu_score(reference, hypothesis)

# Specific n-gram
score = bleu_score(reference, hypothesis, n=2)  # BLEU-2

# With smoothing
score = bleu_score(reference, hypothesis, smoothing=True)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference` | str/list | required | Reference text(s) |
| `hypothesis` | str | required | Generated text |
| `n` | int | 4 | Maximum n-gram order |
| `smoothing` | bool | False | Apply smoothing for short texts |
| `weights` | list | None | Custom n-gram weights |

**Returns:** `float` - BLEU score between 0 and 1

---

### rouge_score

Compute ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores.

```python
from pyeval import rouge_score

reference = "The quick brown fox jumps over the lazy dog"
hypothesis = "A fast brown fox leaps over the lazy dog"

# ROUGE-L (Longest Common Subsequence)
result = rouge_score(reference, hypothesis, rouge_type='l')
print(result)  # {'precision': 0.78, 'recall': 0.78, 'f': 0.78}

# ROUGE-1 (Unigrams)
result = rouge_score(reference, hypothesis, rouge_type='1')

# ROUGE-2 (Bigrams)
result = rouge_score(reference, hypothesis, rouge_type='2')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference` | str | required | Reference text |
| `hypothesis` | str | required | Generated text |
| `rouge_type` | str | 'l' | '1', '2', 'l', or 's' (skip-bigram) |

**Returns:** `dict` - {'precision': float, 'recall': float, 'f': float}

---

### meteor_score

Compute METEOR (Metric for Evaluation of Translation with Explicit ORdering) score.

```python
from pyeval import meteor_score

reference = "The quick brown fox jumps over the lazy dog"
hypothesis = "A fast brown fox leaps over the lazy dog"

score = meteor_score(reference, hypothesis)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference` | str | required | Reference text |
| `hypothesis` | str | required | Generated text |
| `alpha` | float | 0.9 | Precision weight |
| `beta` | float | 3.0 | Fragmentation penalty weight |
| `gamma` | float | 0.5 | Fragmentation penalty exponent |

**Returns:** `float` - METEOR score between 0 and 1

---

### ter_score

Compute TER (Translation Edit Rate) score.

```python
from pyeval import ter_score

reference = "The quick brown fox jumps over the lazy dog"
hypothesis = "A fast brown fox leaps over the lazy dog"

score = ter_score(reference, hypothesis)
# Returns: number of edits / reference length
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference` | str | Reference text |
| `hypothesis` | str | Generated text |

**Returns:** `float` - TER score (lower is better, can exceed 1)

---

### chrf_score

Compute chrF (character n-gram F-score).

```python
from pyeval import chrf_score

reference = "The quick brown fox"
hypothesis = "The fast brown fox"

score = chrf_score(reference, hypothesis)
score = chrf_score(reference, hypothesis, n=6, beta=2)  # chrF++
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference` | str | required | Reference text |
| `hypothesis` | str | required | Generated text |
| `n` | int | 6 | Maximum character n-gram order |
| `beta` | float | 2.0 | Recall weight (β > 1 favors recall) |

**Returns:** `float` - chrF score between 0 and 1

---

### distinct_n

Compute Distinct-N diversity score.

```python
from pyeval import distinct_n

texts = [
    "The cat sat on the mat",
    "A dog ran in the park",
    "Birds fly in the sky"
]

# Distinct unigrams
d1 = distinct_n(texts, n=1)

# Distinct bigrams
d2 = distinct_n(texts, n=2)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `texts` | list[str] | required | List of generated texts |
| `n` | int | 1 | N-gram order |

**Returns:** `float` - Ratio of unique n-grams to total n-grams

---

### text_entropy

Compute text entropy (information content).

```python
from pyeval import text_entropy

text = "The quick brown fox jumps over the lazy dog"

entropy = text_entropy(text)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Input text |
| `base` | int | 2 | Logarithm base |

**Returns:** `float` - Entropy value

---

### perplexity

Compute perplexity of text.

```python
from pyeval import perplexity

text = "The quick brown fox jumps"
log_probs = [-2.3, -1.5, -2.1, -1.8, -2.0]  # Log probabilities per token

ppl = perplexity(log_probs)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `log_probs` | list[float] | Log probabilities of each token |

**Returns:** `float` - Perplexity (lower is better)

---

### repetition_ratio

Compute the repetition ratio in text.

```python
from pyeval import repetition_ratio

text = "The cat sat. The cat slept. The cat ate."

ratio = repetition_ratio(text, n=2)  # Repeated bigrams
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Input text |
| `n` | int | 3 | N-gram size to check for repetition |

**Returns:** `float` - Ratio of repeated n-grams

---

### lexical_diversity

Compute lexical diversity (Type-Token Ratio).

```python
from pyeval import lexical_diversity

text = "The quick brown fox jumps over the lazy dog"

ttr = lexical_diversity(text)
# Returns: unique_words / total_words
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | str | Input text |

**Returns:** `float` - Type-Token Ratio between 0 and 1

---

### coverage_score

Compute how much of the reference is covered in the hypothesis.

```python
from pyeval import coverage_score

reference = "The quick brown fox jumps"
hypothesis = "A quick brown fox leaps"

score = coverage_score(reference, hypothesis)
```

**Returns:** `float` - Coverage ratio between 0 and 1

---

### density_score

Compute extractive density of summary.

```python
from pyeval import density_score

source = "The quick brown fox jumps over the lazy dog in the garden"
summary = "A brown fox jumps over a lazy dog"

score = density_score(source, summary)
```

**Returns:** `float` - Density score

---

## Metric Class

### NLPMetrics

Compute all NLP metrics at once.

```python
from pyeval import NLPMetrics

nlp = NLPMetrics()

reference = "The quick brown fox jumps over the lazy dog"
hypothesis = "A fast brown fox leaps over the lazy dog"

results = nlp.compute(reference, hypothesis)

print(results)
# {
#     'bleu': 0.45,
#     'rouge_1': {'precision': 0.78, 'recall': 0.78, 'f': 0.78},
#     'rouge_2': {'precision': 0.50, 'recall': 0.50, 'f': 0.50},
#     'rouge_l': {'precision': 0.67, 'recall': 0.67, 'f': 0.67},
#     'meteor': 0.72,
#     'ter': 0.33,
#     ...
# }
```

---

## Complete NLP Metrics List

| Metric | Function | Description |
|--------|----------|-------------|
| BLEU | `bleu_score` | N-gram precision with brevity penalty |
| ROUGE-1 | `rouge_score` | Unigram overlap |
| ROUGE-2 | `rouge_score` | Bigram overlap |
| ROUGE-L | `rouge_score` | Longest common subsequence |
| METEOR | `meteor_score` | Alignment-based with synonyms |
| TER | `ter_score` | Translation edit rate |
| chrF | `chrf_score` | Character n-gram F-score |
| Distinct-N | `distinct_n` | N-gram diversity |
| Entropy | `text_entropy` | Information content |
| Perplexity | `perplexity` | Language model quality |
| Repetition | `repetition_ratio` | Repeated n-grams ratio |
| Lexical Diversity | `lexical_diversity` | Type-token ratio |
| Coverage | `coverage_score` | Reference coverage |
| Density | `density_score` | Extractive density |

---

## Usage Tips

### Choosing the Right Metric

| Use Case | Recommended Metrics |
|----------|---------------------|
| Machine Translation | BLEU, TER, chrF, METEOR |
| Summarization | ROUGE-1, ROUGE-2, ROUGE-L, Coverage |
| Text Generation | Distinct-N, Perplexity, Repetition |
| Dialogue Systems | BLEU, Distinct-N, Lexical Diversity |

### Multiple References

For multiple reference translations, pass a list:

```python
references = [
    "The quick brown fox jumps over the lazy dog",
    "A fast brown fox leaps over the lazy dog"
]
hypothesis = "The brown fox jumps over a lazy dog"

# BLEU with multiple references
score = bleu_score(references, hypothesis)
```

### Corpus-Level Scores

For corpus-level evaluation:

```python
from pyeval import corpus_bleu

references_list = [["ref1a", "ref1b"], ["ref2a", "ref2b"]]
hypotheses_list = ["hyp1", "hyp2"]

corpus_score = corpus_bleu(references_list, hypotheses_list)
```
