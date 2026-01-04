# NLP Examples

Comprehensive examples for Natural Language Processing evaluation with PyEval.

---

## 📝 Text Generation Metrics

### BLEU Score

BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between reference and hypothesis.

```python
from pyeval import bleu_score, sentence_bleu, corpus_bleu

# Single sentence
reference = "The cat sat on the mat"
hypothesis = "The cat is on the mat"

# Default BLEU (uses 1-4 grams with smoothing)
score = bleu_score(reference, hypothesis)
print(f"BLEU: {score:.4f}")

# Sentence-level BLEU with specific n-gram weights
score_1 = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0))  # BLEU-1
score_2 = sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0))  # BLEU-2
score_4 = sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))  # BLEU-4

print(f"BLEU-1: {score_1:.4f}")
print(f"BLEU-2: {score_2:.4f}")
print(f"BLEU-4: {score_4:.4f}")

# Multiple references (pick best match)
references = [
    "The cat sat on the mat",
    "A cat was sitting on the mat",
    "There is a cat on the mat"
]
score = bleu_score(references, hypothesis)
print(f"Multi-ref BLEU: {score:.4f}")

# Corpus-level BLEU
corpus_refs = [
    ["The cat sat on the mat"],
    ["Hello world"],
    ["Machine learning is great"]
]
corpus_hyps = [
    "The cat is on the mat",
    "Hello there world",
    "Machine learning is awesome"
]
corpus_score = corpus_bleu(corpus_refs, corpus_hyps)
print(f"Corpus BLEU: {corpus_score:.4f}")
```

---

### ROUGE Score

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures recall of n-grams.

```python
from pyeval import rouge_score, rouge_n, rouge_l

reference = "The quick brown fox jumps over the lazy dog in the park"
hypothesis = "A quick brown fox jumped over the lazy dog"

# Full ROUGE scores
rouge = rouge_score(reference, hypothesis)

print("=== ROUGE-1 (unigram) ===")
print(f"  Precision: {rouge['1']['p']:.4f}")
print(f"  Recall:    {rouge['1']['r']:.4f}")
print(f"  F1:        {rouge['1']['f']:.4f}")

print("\n=== ROUGE-2 (bigram) ===")
print(f"  Precision: {rouge['2']['p']:.4f}")
print(f"  Recall:    {rouge['2']['r']:.4f}")
print(f"  F1:        {rouge['2']['f']:.4f}")

print("\n=== ROUGE-L (longest common subsequence) ===")
print(f"  Precision: {rouge['l']['p']:.4f}")
print(f"  Recall:    {rouge['l']['r']:.4f}")
print(f"  F1:        {rouge['l']['f']:.4f}")

# Individual ROUGE variants
r1 = rouge_n(reference, hypothesis, n=1)
r2 = rouge_n(reference, hypothesis, n=2)
rl = rouge_l(reference, hypothesis)
```

---

### METEOR Score

METEOR uses stemming, synonyms, and word alignment.

```python
from pyeval import meteor_score

reference = "The cat sat on the mat"
hypothesis = "A cat was sitting on the mat"

# METEOR handles synonyms and stemming
score = meteor_score(reference, hypothesis)
print(f"METEOR: {score:.4f}")

# Multiple references
references = [
    "The cat sat on the mat",
    "A cat was sitting on the mat",
]
score = meteor_score(references, hypothesis)
print(f"Multi-ref METEOR: {score:.4f}")
```

---

### TER (Translation Edit Rate)

TER measures the number of edits needed to transform hypothesis to reference.

```python
from pyeval import ter_score

reference = "The quick brown fox jumps over the lazy dog"
hypothesis = "A quick brown fox jumps over the lazy dog"

# TER (lower is better, 0 = perfect match)
ter = ter_score(reference, hypothesis)
print(f"TER: {ter:.4f}")

# TER can be > 1 if many edits needed
bad_hypothesis = "Something completely different"
ter_bad = ter_score(reference, bad_hypothesis)
print(f"Bad TER: {ter_bad:.4f}")
```

---

### chrF Score

chrF uses character n-grams (language-agnostic).

```python
from pyeval import chrf_score

reference = "The quick brown fox"
hypothesis = "The quick brown dog"

# chrF (character-level F-score)
score = chrf_score(reference, hypothesis)
print(f"chrF: {score:.4f}")

# chrF++ (with word n-grams)
score_pp = chrf_score(reference, hypothesis, word_order=2)
print(f"chrF++: {score_pp:.4f}")
```

---

### BERTScore

BERTScore uses contextual embeddings for semantic similarity.

```python
from pyeval import bert_score

reference = "The weather is beautiful today"
hypothesis = "It's a lovely day outside"

# BERTScore (semantic similarity)
score = bert_score(reference, hypothesis)
print(f"BERTScore Precision: {score['precision']:.4f}")
print(f"BERTScore Recall:    {score['recall']:.4f}")
print(f"BERTScore F1:        {score['f1']:.4f}")
```

---

## 🔤 Text Similarity

### Word-Level Metrics

```python
from pyeval import (
    jaccard_similarity,
    cosine_similarity,
    text_similarity,
    word_error_rate
)

text1 = "The quick brown fox jumps over the lazy dog"
text2 = "A fast brown fox leaps over the lazy dog"

# Jaccard similarity (word overlap)
jaccard = jaccard_similarity(text1.split(), text2.split())
print(f"Jaccard: {jaccard:.4f}")

# Text similarity (composite score)
sim = text_similarity(text1, text2)
print(f"Text Similarity: {sim:.4f}")
```

---

## 📊 Corpus Evaluation

### Evaluating a Translation System

```python
from pyeval import corpus_bleu, rouge_score, ter_score, NLPMetrics

# Corpus data
references = [
    "The cat sat on the mat.",
    "Hello, how are you today?",
    "Machine learning is transforming industries.",
    "The weather is nice today.",
    "I love programming in Python."
]

hypotheses = [
    "A cat was sitting on the mat.",
    "Hello, how are you?",
    "Machine learning is revolutionizing industries.",
    "Today the weather is nice.",
    "I enjoy coding in Python."
]

print("=== Corpus Evaluation ===\n")

# BLEU (corpus-level)
bleu = corpus_bleu([[r] for r in references], hypotheses)
print(f"Corpus BLEU: {bleu:.4f}")

# Average ROUGE
rouge_scores = []
for ref, hyp in zip(references, hypotheses):
    rouge = rouge_score(ref, hyp)
    rouge_scores.append(rouge['l']['f'])
avg_rouge = sum(rouge_scores) / len(rouge_scores)
print(f"Average ROUGE-L F1: {avg_rouge:.4f}")

# Average TER
ter_scores = [ter_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
avg_ter = sum(ter_scores) / len(ter_scores)
print(f"Average TER: {avg_ter:.4f}")

# Using NLPMetrics class
metrics = NLPMetrics()
results = metrics.compute(references, hypotheses)
print(f"\n=== All NLP Metrics ===")
for name, value in results.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
```

---

## 📝 Summarization Evaluation

```python
from pyeval import rouge_score, bleu_score

# Original document
document = """
Machine learning is a subset of artificial intelligence that provides 
systems the ability to automatically learn and improve from experience 
without being explicitly programmed. Machine learning focuses on the 
development of computer programs that can access data and use it to 
learn for themselves.
"""

# Reference summary (human-written)
reference_summary = """
Machine learning enables systems to learn from experience automatically, 
without explicit programming, by developing programs that access and learn from data.
"""

# System-generated summaries
summaries = {
    'Extractive': "Machine learning is a subset of artificial intelligence. It focuses on developing programs that access data to learn.",
    'Abstractive': "ML allows computers to learn automatically from data without being explicitly programmed.",
    'Short': "Machine learning enables automatic learning from data.",
}

print("Summarization Evaluation:")
print("=" * 60)

for name, summary in summaries.items():
    rouge = rouge_score(reference_summary, summary)
    bleu = bleu_score(reference_summary, summary)
    
    print(f"\n{name} Summary:")
    print(f"  ROUGE-1 F1: {rouge['1']['f']:.4f}")
    print(f"  ROUGE-2 F1: {rouge['2']['f']:.4f}")
    print(f"  ROUGE-L F1: {rouge['l']['f']:.4f}")
    print(f"  BLEU:       {bleu:.4f}")
```

---

## 🌐 Machine Translation Evaluation

```python
from pyeval import bleu_score, ter_score, chrf_score, meteor_score

# Source (for reference)
source = "La casa es grande y bonita."

# Reference translations
references = [
    "The house is big and beautiful.",
    "The home is large and pretty.",
]

# System translations from different MT systems
mt_outputs = {
    'Google Translate': "The house is big and beautiful.",
    'DeepL': "The house is large and beautiful.",
    'Custom NMT': "The home is big and nice.",
    'Bad System': "House big beautiful.",
}

print("Machine Translation Evaluation:")
print("=" * 70)
print(f"{'System':<20} {'BLEU':>10} {'TER':>10} {'chrF':>10} {'METEOR':>10}")
print("-" * 70)

for system, output in mt_outputs.items():
    bleu = bleu_score(references, output)
    ter = ter_score(references[0], output)
    chrf = chrf_score(references[0], output)
    meteor = meteor_score(references, output)
    
    print(f"{system:<20} {bleu:>10.4f} {ter:>10.4f} {chrf:>10.4f} {meteor:>10.4f}")
```

---

## 🔍 Detailed Analysis

### N-gram Analysis

```python
from pyeval import ngrams, get_word_frequencies

text = "the quick brown fox jumps over the lazy dog"

# Generate n-grams
unigrams = list(ngrams(text.split(), 1))
bigrams = list(ngrams(text.split(), 2))
trigrams = list(ngrams(text.split(), 3))

print("N-gram Analysis:")
print(f"Unigrams: {unigrams}")
print(f"Bigrams:  {bigrams}")
print(f"Trigrams: {trigrams}")

# Word frequencies
freq = get_word_frequencies(text)
print(f"\nWord Frequencies: {freq}")
```

### Length and Coverage Analysis

```python
from pyeval import tokenize, word_count

references = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming industries.",
]

hypotheses = [
    "A fast brown fox leaps over the lazy dog.",
    "ML is revolutionizing business.",
]

print("\nLength Analysis:")
print("-" * 50)

for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
    ref_len = word_count(ref)
    hyp_len = word_count(hyp)
    ratio = hyp_len / ref_len if ref_len > 0 else 0
    
    print(f"Pair {i+1}:")
    print(f"  Reference length: {ref_len} words")
    print(f"  Hypothesis length: {hyp_len} words")
    print(f"  Length ratio: {ratio:.2f}")
```
