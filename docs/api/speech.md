# Speech Metrics API Reference

Metrics for evaluating speech recognition and synthesis systems.

---

## Error Rate Metrics

### word_error_rate

Compute Word Error Rate (WER).

```python
from pyeval import word_error_rate

reference = "the quick brown fox jumps over the lazy dog"
hypothesis = "the quick brown fox jumps over lazy dog"

result = word_error_rate(reference, hypothesis)
print(result)
# {
#     'wer': 0.111,
#     'substitutions': 0,
#     'insertions': 0,
#     'deletions': 1,
#     'reference_length': 9
# }
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference` | str | Reference transcription |
| `hypothesis` | str | Hypothesis transcription |

**Returns:** `dict` - WER and error breakdown

**Formula:** WER = (S + I + D) / N

- S = Substitutions
- I = Insertions
- D = Deletions
- N = Reference word count

---

### character_error_rate

Compute Character Error Rate (CER).

```python
from pyeval import character_error_rate

reference = "hello world"
hypothesis = "helo word"

result = character_error_rate(reference, hypothesis)
print(result)
# {
#     'cer': 0.182,
#     'substitutions': 0,
#     'insertions': 0,
#     'deletions': 2,
#     'reference_length': 11
# }
```

**Returns:** `dict` - CER and character-level error breakdown

---

### match_error_rate

Compute Match Error Rate (MER).

```python
from pyeval import match_error_rate

reference = "the quick brown fox"
hypothesis = "the fast brown fox"

result = match_error_rate(reference, hypothesis)
print(result['mer'])  # 0.25
```

**Returns:** `dict` - MER analysis

---

### word_information_lost

Compute Word Information Lost (WIL).

```python
from pyeval import word_information_lost

reference = "the quick brown fox"
hypothesis = "the fast brown fox"

result = word_information_lost(reference, hypothesis)
print(result['wil'])
```

**Returns:** `dict` - WIL score

---

## Semantic Understanding Metrics

### slot_error_rate

Compute Slot Error Rate for NLU evaluation.

```python
from pyeval import slot_error_rate

reference_slots = {'location': 'new york', 'date': 'tomorrow'}
hypothesis_slots = {'location': 'new york', 'date': 'today'}

result = slot_error_rate(reference_slots, hypothesis_slots)
print(result)
# {
#     'ser': 0.5,
#     'correct_slots': 1,
#     'total_slots': 2,
#     'incorrect_slots': ['date']
# }
```

**Returns:** `dict` - Slot error analysis

---

### intent_accuracy

Compute intent classification accuracy.

```python
from pyeval import intent_accuracy

reference_intents = ['book_flight', 'weather', 'book_hotel']
hypothesis_intents = ['book_flight', 'weather', 'book_flight']

result = intent_accuracy(reference_intents, hypothesis_intents)
print(result)
# {
#     'accuracy': 0.667,
#     'correct': 2,
#     'total': 3,
#     'confusion': {'book_hotel': 'book_flight'}
# }
```

**Returns:** `dict` - Intent accuracy analysis

---

## Phonetic Metrics

### phoneme_error_rate

Compute Phoneme Error Rate (PER).

```python
from pyeval import phoneme_error_rate

reference_phonemes = ['HH', 'AH', 'L', 'OW']  # "hello"
hypothesis_phonemes = ['HH', 'EH', 'L', 'OW']

result = phoneme_error_rate(reference_phonemes, hypothesis_phonemes)
print(result['per'])  # 0.25
```

**Returns:** `dict` - PER analysis

---

## Speaker Metrics

### diarization_error_rate

Compute Diarization Error Rate (DER).

```python
from pyeval import diarization_error_rate

# Segments: [(start, end, speaker)]
reference_segments = [
    (0.0, 2.0, 'A'),
    (2.0, 4.0, 'B'),
    (4.0, 6.0, 'A')
]

hypothesis_segments = [
    (0.0, 1.5, 'A'),
    (1.5, 4.0, 'B'),
    (4.0, 6.0, 'A')
]

result = diarization_error_rate(reference_segments, hypothesis_segments)
print(result)
# {
#     'der': 0.083,
#     'miss_rate': 0.0,
#     'false_alarm': 0.0,
#     'confusion': 0.083
# }
```

**Returns:** `dict` - DER breakdown

---

## Keyword & Wake Word Metrics

### keyword_spotting_metrics

Evaluate keyword spotting performance.

```python
from pyeval import keyword_spotting_metrics

# True positives, false positives, false negatives
detections = {
    'true_positives': 45,
    'false_positives': 5,
    'false_negatives': 10
}

result = keyword_spotting_metrics(
    detections['true_positives'],
    detections['false_positives'],
    detections['false_negatives']
)
print(result)
# {
#     'precision': 0.90,
#     'recall': 0.818,
#     'f1': 0.857,
#     'false_rejection_rate': 0.182,
#     'false_acceptance_rate': 0.10
# }
```

**Returns:** `dict` - Keyword spotting metrics

---

## Speech Quality Metrics

### mean_opinion_score

Estimate Mean Opinion Score (MOS) for speech quality.

```python
from pyeval import mean_opinion_score

# Quality indicators
quality_features = {
    'signal_to_noise_ratio': 25.0,  # dB
    'clarity': 0.85,
    'naturalness': 0.90
}

result = mean_opinion_score(quality_features)
print(result)
# {
#     'mos': 4.2,
#     'quality_category': 'good',
#     'components': {
#         'snr_score': 4.5,
#         'clarity_score': 4.25,
#         'naturalness_score': 4.5
#     }
# }
```

**Returns:** `dict` - MOS estimation (1-5 scale)

---

### fluency_score

Evaluate speech fluency.

```python
from pyeval import fluency_score

# Fluency indicators
fluency_data = {
    'words_per_minute': 120,
    'pause_ratio': 0.15,
    'filler_words': ['um', 'uh'],
    'filler_count': 3,
    'total_words': 100
}

result = fluency_score(fluency_data)
print(result)
# {
#     'fluency_score': 0.82,
#     'speaking_rate': 'normal',
#     'filler_ratio': 0.03,
#     'pause_assessment': 'appropriate'
# }
```

**Returns:** `dict` - Fluency analysis

---

### signal_to_noise_ratio

Compute Signal-to-Noise Ratio.

```python
from pyeval import signal_to_noise_ratio

# Audio signal and noise samples
signal = [0.5, 0.8, 0.3, 0.9, 0.4]
noise = [0.01, 0.02, 0.01, 0.03, 0.01]

snr = signal_to_noise_ratio(signal, noise)
print(f"SNR: {snr:.2f} dB")
```

**Returns:** `float` - SNR in decibels

---

## Real-time Metrics

### real_time_factor

Compute Real-Time Factor (RTF).

```python
from pyeval import real_time_factor

audio_duration = 10.0  # seconds
processing_time = 2.5  # seconds

rtf = real_time_factor(audio_duration, processing_time)
print(f"RTF: {rtf:.2f}")  # 0.25 (faster than real-time)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_duration` | float | Duration of audio in seconds |
| `processing_time` | float | Time to process in seconds |

**Returns:** `float` - RTF (< 1 is faster than real-time)

---

### latency_metrics

Compute latency metrics for streaming ASR.

```python
from pyeval import latency_metrics

# Timestamps for each word
reference_times = [0.5, 1.0, 1.5, 2.0]
hypothesis_times = [0.7, 1.2, 1.6, 2.2]

result = latency_metrics(reference_times, hypothesis_times)
print(result)
# {
#     'mean_latency': 0.175,
#     'max_latency': 0.2,
#     'min_latency': 0.1,
#     'p90_latency': 0.2
# }
```

**Returns:** `dict` - Latency statistics

---

## Metric Class

### SpeechMetrics

Compute all speech metrics at once.

```python
from pyeval import SpeechMetrics

sm = SpeechMetrics()

result = sm.compute(
    reference="the quick brown fox",
    hypothesis="the fast brown fox"
)

print(result)
# {
#     'wer': 0.25,
#     'cer': 0.16,
#     'mer': 0.25,
#     'substitutions': 1,
#     'insertions': 0,
#     'deletions': 0,
#     ...
# }
```

---

## Complete Speech Metrics List

| Metric | Function | Description |
|--------|----------|-------------|
| WER | `word_error_rate` | Word-level error rate |
| CER | `character_error_rate` | Character-level error rate |
| MER | `match_error_rate` | Match error rate |
| WIL | `word_information_lost` | Information loss measure |
| SER | `slot_error_rate` | NLU slot errors |
| Intent Accuracy | `intent_accuracy` | Intent classification |
| PER | `phoneme_error_rate` | Phoneme-level errors |
| DER | `diarization_error_rate` | Speaker diarization errors |
| KWS Metrics | `keyword_spotting_metrics` | Keyword detection |
| MOS | `mean_opinion_score` | Perceptual quality |
| Fluency | `fluency_score` | Speaking fluency |
| SNR | `signal_to_noise_ratio` | Signal quality |
| RTF | `real_time_factor` | Processing speed |
| Latency | `latency_metrics` | Streaming latency |

---

## Usage Tips

### Batch Evaluation

```python
from pyeval import word_error_rate

test_cases = [
    ("hello world", "hello world"),
    ("good morning", "good moring"),
    ("how are you", "how are yo")
]

total_wer = 0
for ref, hyp in test_cases:
    result = word_error_rate(ref, hyp)
    total_wer += result['wer']

avg_wer = total_wer / len(test_cases)
print(f"Average WER: {avg_wer:.3f}")
```

### Detailed Error Analysis

```python
from pyeval import word_error_rate, character_error_rate

def detailed_analysis(reference, hypothesis):
    wer_result = word_error_rate(reference, hypothesis)
    cer_result = character_error_rate(reference, hypothesis)
    
    print(f"Reference: {reference}")
    print(f"Hypothesis: {hypothesis}")
    print(f"WER: {wer_result['wer']:.3f}")
    print(f"  - Substitutions: {wer_result['substitutions']}")
    print(f"  - Insertions: {wer_result['insertions']}")
    print(f"  - Deletions: {wer_result['deletions']}")
    print(f"CER: {cer_result['cer']:.3f}")
```

### Comparing ASR Systems

```python
from pyeval import SpeechMetrics

sm = SpeechMetrics()

systems = {
    'system_a': hypotheses_a,
    'system_b': hypotheses_b
}

results = {}
for name, hypotheses in systems.items():
    wers = []
    for ref, hyp in zip(references, hypotheses):
        result = sm.compute(ref, hyp)
        wers.append(result['wer'])
    results[name] = sum(wers) / len(wers)

best_system = min(results.items(), key=lambda x: x[1])
print(f"Best system: {best_system[0]} with WER: {best_system[1]:.3f}")
```
