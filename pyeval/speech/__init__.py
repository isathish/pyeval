"""
Speech Evaluation Metrics - Pure Python Implementation
======================================================

Evaluation metrics for speech-to-text and text-to-speech systems
including WER, CER, and related metrics.
"""

from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from pyeval.utils.text_ops import normalize_text
from pyeval.utils.math_ops import mean


# =============================================================================
# Edit Distance Operations
# =============================================================================

def _levenshtein_operations(reference: List[str], hypothesis: List[str]) -> Tuple[int, int, int, int]:
    """
    Calculate Levenshtein edit operations.
    
    Args:
        reference: Reference tokens
        hypothesis: Hypothesis tokens
        
    Returns:
        Tuple of (substitutions, deletions, insertions, correct)
    """
    r_len = len(reference)
    h_len = len(hypothesis)
    
    # DP table
    d = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    
    # Backtracking info: 0=match, 1=sub, 2=del, 3=ins
    ops = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    
    # Initialize
    for i in range(r_len + 1):
        d[i][0] = i
        if i > 0:
            ops[i][0] = 2  # Deletion
    
    for j in range(h_len + 1):
        d[0][j] = j
        if j > 0:
            ops[0][j] = 3  # Insertion
    
    # Fill DP table
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if reference[i-1] == hypothesis[j-1]:
                d[i][j] = d[i-1][j-1]
                ops[i][j] = 0  # Match
            else:
                sub = d[i-1][j-1] + 1
                del_ = d[i-1][j] + 1
                ins = d[i][j-1] + 1
                
                d[i][j] = min(sub, del_, ins)
                
                if d[i][j] == sub:
                    ops[i][j] = 1  # Substitution
                elif d[i][j] == del_:
                    ops[i][j] = 2  # Deletion
                else:
                    ops[i][j] = 3  # Insertion
    
    # Backtrack to count operations
    substitutions = 0
    deletions = 0
    insertions = 0
    correct = 0
    
    i, j = r_len, h_len
    while i > 0 or j > 0:
        op = ops[i][j]
        
        if op == 0:  # Match
            correct += 1
            i -= 1
            j -= 1
        elif op == 1:  # Substitution
            substitutions += 1
            i -= 1
            j -= 1
        elif op == 2:  # Deletion
            deletions += 1
            i -= 1
        else:  # Insertion
            insertions += 1
            j -= 1
    
    return substitutions, deletions, insertions, correct


# =============================================================================
# Word Error Rate
# =============================================================================

def word_error_rate(reference: str, hypothesis: str,
                    normalize: bool = True) -> Dict[str, Any]:
    """
    Calculate Word Error Rate (WER).
    
    WER = (S + D + I) / N
    
    Where:
        S = Substitutions
        D = Deletions
        I = Insertions
        N = Number of words in reference
    
    Args:
        reference: Reference transcription
        hypothesis: Hypothesis (ASR output)
        normalize: Normalize text before comparison
        
    Returns:
        Dictionary with WER and edit statistics
        
    Example:
        >>> ref = "hello world how are you"
        >>> hyp = "hello word how you"
        >>> result = word_error_rate(ref, hyp)
        >>> result['wer']
        0.4
    """
    # Tokenize
    if normalize:
        reference = normalize_text(reference, lowercase=True, remove_punct=True)
        hypothesis = normalize_text(hypothesis, lowercase=True, remove_punct=True)
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if not ref_words:
        return {
            'wer': 0.0 if not hyp_words else 1.0,
            'substitutions': 0,
            'deletions': 0,
            'insertions': len(hyp_words),
            'correct': 0,
            'reference_length': 0,
            'hypothesis_length': len(hyp_words)
        }
    
    # Calculate edit operations
    subs, dels, ins, correct = _levenshtein_operations(ref_words, hyp_words)
    
    # WER calculation
    n_ref = len(ref_words)
    wer = (subs + dels + ins) / n_ref
    
    return {
        'wer': wer,
        'substitutions': subs,
        'deletions': dels,
        'insertions': ins,
        'correct': correct,
        'total_errors': subs + dels + ins,
        'reference_length': n_ref,
        'hypothesis_length': len(hyp_words),
        'accuracy': correct / n_ref if n_ref > 0 else 0.0
    }


def corpus_wer(references: List[str], hypotheses: List[str],
               normalize: bool = True) -> Dict[str, Any]:
    """
    Calculate corpus-level Word Error Rate.
    
    Args:
        references: List of reference transcriptions
        hypotheses: List of hypotheses
        normalize: Normalize text before comparison
        
    Returns:
        Dictionary with corpus WER and statistics
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    total_subs = 0
    total_dels = 0
    total_ins = 0
    total_correct = 0
    total_ref_len = 0
    
    sentence_wers = []
    
    for ref, hyp in zip(references, hypotheses):
        result = word_error_rate(ref, hyp, normalize)
        
        total_subs += result['substitutions']
        total_dels += result['deletions']
        total_ins += result['insertions']
        total_correct += result['correct']
        total_ref_len += result['reference_length']
        
        sentence_wers.append(result['wer'])
    
    corpus_wer_value = (total_subs + total_dels + total_ins) / total_ref_len if total_ref_len > 0 else 0.0
    
    return {
        'wer': corpus_wer_value,
        'substitutions': total_subs,
        'deletions': total_dels,
        'insertions': total_ins,
        'correct': total_correct,
        'total_reference_words': total_ref_len,
        'num_sentences': len(references),
        'sentence_wers': sentence_wers,
        'mean_sentence_wer': mean(sentence_wers) if sentence_wers else 0.0
    }


# =============================================================================
# Character Error Rate
# =============================================================================

def character_error_rate(reference: str, hypothesis: str,
                         include_spaces: bool = False) -> Dict[str, Any]:
    """
    Calculate Character Error Rate (CER).
    
    CER = (S + D + I) / N
    
    Where N is the number of characters in reference.
    
    Args:
        reference: Reference transcription
        hypothesis: Hypothesis (ASR output)
        include_spaces: Include spaces in character count
        
    Returns:
        Dictionary with CER and edit statistics
        
    Example:
        >>> ref = "hello world"
        >>> hyp = "helo word"
        >>> result = character_error_rate(ref, hyp)
        >>> result['cer']
    """
    # Convert to character lists
    if include_spaces:
        ref_chars = list(reference.lower())
        hyp_chars = list(hypothesis.lower())
    else:
        ref_chars = list(reference.lower().replace(' ', ''))
        hyp_chars = list(hypothesis.lower().replace(' ', ''))
    
    if not ref_chars:
        return {
            'cer': 0.0 if not hyp_chars else 1.0,
            'substitutions': 0,
            'deletions': 0,
            'insertions': len(hyp_chars),
            'correct': 0,
            'reference_length': 0,
            'hypothesis_length': len(hyp_chars)
        }
    
    # Calculate edit operations
    subs, dels, ins, correct = _levenshtein_operations(ref_chars, hyp_chars)
    
    # CER calculation
    n_ref = len(ref_chars)
    cer = (subs + dels + ins) / n_ref
    
    return {
        'cer': cer,
        'substitutions': subs,
        'deletions': dels,
        'insertions': ins,
        'correct': correct,
        'total_errors': subs + dels + ins,
        'reference_length': n_ref,
        'hypothesis_length': len(hyp_chars),
        'accuracy': correct / n_ref if n_ref > 0 else 0.0
    }


def corpus_cer(references: List[str], hypotheses: List[str],
               include_spaces: bool = False) -> Dict[str, Any]:
    """
    Calculate corpus-level Character Error Rate.
    
    Args:
        references: List of reference transcriptions
        hypotheses: List of hypotheses
        include_spaces: Include spaces in character count
        
    Returns:
        Dictionary with corpus CER and statistics
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    total_subs = 0
    total_dels = 0
    total_ins = 0
    total_correct = 0
    total_ref_len = 0
    
    sentence_cers = []
    
    for ref, hyp in zip(references, hypotheses):
        result = character_error_rate(ref, hyp, include_spaces)
        
        total_subs += result['substitutions']
        total_dels += result['deletions']
        total_ins += result['insertions']
        total_correct += result['correct']
        total_ref_len += result['reference_length']
        
        sentence_cers.append(result['cer'])
    
    corpus_cer_value = (total_subs + total_dels + total_ins) / total_ref_len if total_ref_len > 0 else 0.0
    
    return {
        'cer': corpus_cer_value,
        'substitutions': total_subs,
        'deletions': total_dels,
        'insertions': total_ins,
        'correct': total_correct,
        'total_reference_chars': total_ref_len,
        'num_sentences': len(references),
        'sentence_cers': sentence_cers,
        'mean_sentence_cer': mean(sentence_cers) if sentence_cers else 0.0
    }


# =============================================================================
# Match Error Rate
# =============================================================================

def match_error_rate(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculate Match Error Rate (MER).
    
    MER = (S + D + I) / (S + D + C)
    
    MER is more interpretable than WER when hypothesis is longer than reference.
    
    Args:
        reference: Reference transcription
        hypothesis: Hypothesis transcription
        
    Returns:
        Dictionary with MER and related metrics
    """
    result = word_error_rate(reference, hypothesis)
    
    # MER denominator is (S + D + C) = total alignments
    total_alignments = result['substitutions'] + result['deletions'] + result['correct']
    
    if total_alignments == 0:
        mer = 0.0 if result['insertions'] == 0 else 1.0
    else:
        mer = result['total_errors'] / total_alignments
    
    return {
        'mer': mer,
        'wer': result['wer'],
        'total_alignments': total_alignments
    }


# =============================================================================
# Word Information Lost/Preserved
# =============================================================================

def word_information_lost(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculate Word Information Lost (WIL).
    
    WIL = 1 - (C² / (N * P))
    
    Where C = correct words, N = reference length, P = hypothesis length
    
    Args:
        reference: Reference transcription
        hypothesis: Hypothesis transcription
        
    Returns:
        Dictionary with WIL and WIP (Word Information Preserved)
    """
    result = word_error_rate(reference, hypothesis)
    
    c = result['correct']
    n = result['reference_length']
    p = result['hypothesis_length']
    
    if n == 0 or p == 0:
        wil = 1.0 if c == 0 else 0.0
    else:
        wil = 1 - (c * c) / (n * p)
    
    wip = 1 - wil  # Word Information Preserved
    
    return {
        'wil': wil,
        'wip': wip,
        'correct': c,
        'reference_length': n,
        'hypothesis_length': p
    }


# =============================================================================
# Sentence Error Rate
# =============================================================================

def sentence_error_rate(references: List[str], hypotheses: List[str],
                        normalize: bool = True) -> Dict[str, Any]:
    """
    Calculate Sentence Error Rate (SER).
    
    SER = number of incorrect sentences / total sentences
    
    Args:
        references: List of reference transcriptions
        hypotheses: List of hypotheses
        normalize: Normalize text before comparison
        
    Returns:
        Dictionary with SER and statistics
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    incorrect = 0
    
    for ref, hyp in zip(references, hypotheses):
        if normalize:
            ref = normalize_text(ref, lowercase=True, remove_punct=True)
            hyp = normalize_text(hyp, lowercase=True, remove_punct=True)
        
        if ref != hyp:
            incorrect += 1
    
    total = len(references)
    ser = incorrect / total if total > 0 else 0.0
    
    return {
        'ser': ser,
        'incorrect_sentences': incorrect,
        'correct_sentences': total - incorrect,
        'total_sentences': total,
        'accuracy': 1 - ser
    }


# =============================================================================
# Additional Speech Metrics
# =============================================================================

def real_time_factor(audio_duration: float, processing_time: float) -> float:
    """
    Calculate Real-Time Factor (RTF).
    
    RTF < 1 means faster than real-time.
    
    Args:
        audio_duration: Duration of audio in seconds
        processing_time: Time taken to process in seconds
        
    Returns:
        Real-time factor
    """
    if audio_duration <= 0:
        raise ValueError("Audio duration must be positive")
    
    return processing_time / audio_duration


def recognition_accuracy(reference: str, hypothesis: str) -> float:
    """
    Calculate simple word-level recognition accuracy.
    
    Accuracy = Correct / Reference Length
    
    Args:
        reference: Reference transcription
        hypothesis: Hypothesis transcription
        
    Returns:
        Recognition accuracy (0 to 1)
    """
    result = word_error_rate(reference, hypothesis)
    return result['accuracy']


# =============================================================================
# Alignment Visualization
# =============================================================================

def align_texts(reference: str, hypothesis: str) -> List[Dict[str, Any]]:
    """
    Generate word-level alignment between reference and hypothesis.
    
    Args:
        reference: Reference transcription
        hypothesis: Hypothesis transcription
        
    Returns:
        List of alignment operations
    """
    ref_words = normalize_text(reference, lowercase=True, remove_punct=True).split()
    hyp_words = normalize_text(hypothesis, lowercase=True, remove_punct=True).split()
    
    r_len = len(ref_words)
    h_len = len(hyp_words)
    
    # DP table
    d = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    ops = [[None] * (h_len + 1) for _ in range(r_len + 1)]
    
    for i in range(r_len + 1):
        d[i][0] = i
        if i > 0:
            ops[i][0] = ('delete', ref_words[i-1], None)
    
    for j in range(h_len + 1):
        d[0][j] = j
        if j > 0:
            ops[0][j] = ('insert', None, hyp_words[j-1])
    
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
                ops[i][j] = ('match', ref_words[i-1], hyp_words[j-1])
            else:
                costs = [
                    (d[i-1][j-1] + 1, ('substitute', ref_words[i-1], hyp_words[j-1])),
                    (d[i-1][j] + 1, ('delete', ref_words[i-1], None)),
                    (d[i][j-1] + 1, ('insert', None, hyp_words[j-1]))
                ]
                d[i][j], ops[i][j] = min(costs, key=lambda x: x[0])
    
    # Backtrack
    alignment = []
    i, j = r_len, h_len
    
    while i > 0 or j > 0:
        op = ops[i][j]
        if op is None:
            break
        
        alignment.append({
            'operation': op[0],
            'reference': op[1],
            'hypothesis': op[2]
        })
        
        if op[0] in ('match', 'substitute'):
            i -= 1
            j -= 1
        elif op[0] == 'delete':
            i -= 1
        else:  # insert
            j -= 1
    
    alignment.reverse()
    return alignment


# =============================================================================
# Advanced Speech Metrics
# =============================================================================

def slot_error_rate(reference_slots: Dict[str, str], 
                    hypothesis_slots: Dict[str, str]) -> Dict[str, Any]:
    """
    Calculate Slot Error Rate for spoken language understanding.
    
    SER = (Incorrect + Missing + Spurious) / Total Reference Slots
    
    Args:
        reference_slots: Ground truth slot-value pairs
        hypothesis_slots: Predicted slot-value pairs
        
    Returns:
        Dictionary with SER and detailed errors
    """
    ref_keys = set(reference_slots.keys())
    hyp_keys = set(hypothesis_slots.keys())
    
    # Correct slots (same key and value)
    correct = 0
    incorrect = 0
    
    for key in ref_keys & hyp_keys:
        if reference_slots[key] == hypothesis_slots[key]:
            correct += 1
        else:
            incorrect += 1
    
    # Missing slots (in ref but not in hyp)
    missing = len(ref_keys - hyp_keys)
    
    # Spurious slots (in hyp but not in ref)
    spurious = len(hyp_keys - ref_keys)
    
    total_ref = len(ref_keys)
    
    if total_ref == 0:
        ser = 0.0 if len(hyp_keys) == 0 else 1.0
    else:
        ser = (incorrect + missing + spurious) / total_ref
    
    return {
        'slot_error_rate': ser,
        'correct': correct,
        'incorrect': incorrect,
        'missing': missing,
        'spurious': spurious,
        'total_reference_slots': total_ref,
        'total_hypothesis_slots': len(hyp_keys),
        'slot_accuracy': correct / total_ref if total_ref > 0 else 0.0
    }


def intent_accuracy(reference_intents: List[str], 
                    hypothesis_intents: List[str]) -> Dict[str, float]:
    """
    Calculate intent classification accuracy for spoken language understanding.
    
    Args:
        reference_intents: Ground truth intents
        hypothesis_intents: Predicted intents
        
    Returns:
        Dictionary with intent accuracy metrics
    """
    if len(reference_intents) != len(hypothesis_intents):
        raise ValueError("Number of reference and hypothesis intents must match")
    
    correct = sum(1 for r, h in zip(reference_intents, hypothesis_intents) if r == h)
    total = len(reference_intents)
    
    return {
        'intent_accuracy': correct / total if total > 0 else 0.0,
        'correct': correct,
        'incorrect': total - correct,
        'total': total
    }


def phoneme_error_rate(reference_phonemes: List[str], 
                       hypothesis_phonemes: List[str]) -> Dict[str, Any]:
    """
    Calculate Phoneme Error Rate (PER).
    
    Args:
        reference_phonemes: List of reference phonemes
        hypothesis_phonemes: List of hypothesis phonemes
        
    Returns:
        Dictionary with PER and edit statistics
    """
    if not reference_phonemes:
        return {
            'per': 0.0 if not hypothesis_phonemes else 1.0,
            'substitutions': 0,
            'deletions': 0,
            'insertions': len(hypothesis_phonemes),
            'correct': 0,
            'reference_length': 0
        }
    
    subs, dels, ins, correct = _levenshtein_operations(
        reference_phonemes, hypothesis_phonemes
    )
    
    n_ref = len(reference_phonemes)
    per = (subs + dels + ins) / n_ref
    
    return {
        'per': per,
        'substitutions': subs,
        'deletions': dels,
        'insertions': ins,
        'correct': correct,
        'total_errors': subs + dels + ins,
        'reference_length': n_ref,
        'accuracy': correct / n_ref
    }


def diarization_error_rate(reference_segments: List[Tuple[float, float, str]],
                           hypothesis_segments: List[Tuple[float, float, str]]) -> Dict[str, float]:
    """
    Calculate Speaker Diarization Error Rate (DER).
    
    DER = (False Alarm + Miss + Speaker Error) / Total Reference Time
    
    Args:
        reference_segments: List of (start, end, speaker_id) tuples
        hypothesis_segments: List of (start, end, speaker_id) tuples
        
    Returns:
        Dictionary with DER components
    """
    if not reference_segments:
        return {
            'der': 0.0 if not hypothesis_segments else 1.0,
            'false_alarm': 0.0,
            'miss': 0.0,
            'speaker_error': 0.0
        }
    
    # Calculate total reference time
    total_ref_time = sum(end - start for start, end, _ in reference_segments)
    
    if total_ref_time <= 0:
        return {
            'der': 0.0,
            'false_alarm': 0.0,
            'miss': 0.0,
            'speaker_error': 0.0
        }
    
    # Sample at fine granularity
    sample_rate = 0.01  # 10ms
    
    # Find time range
    all_times = []
    for start, end, _ in reference_segments + hypothesis_segments:
        all_times.extend([start, end])
    
    min_time = min(all_times) if all_times else 0
    max_time = max(all_times) if all_times else 0
    
    miss_time = 0.0
    false_alarm_time = 0.0
    speaker_error_time = 0.0
    
    t = min_time
    while t < max_time:
        # Find reference speaker at time t
        ref_speaker = None
        for start, end, speaker in reference_segments:
            if start <= t < end:
                ref_speaker = speaker
                break
        
        # Find hypothesis speaker at time t
        hyp_speaker = None
        for start, end, speaker in hypothesis_segments:
            if start <= t < end:
                hyp_speaker = speaker
                break
        
        # Count errors
        if ref_speaker is not None and hyp_speaker is None:
            miss_time += sample_rate
        elif ref_speaker is None and hyp_speaker is not None:
            false_alarm_time += sample_rate
        elif ref_speaker is not None and hyp_speaker is not None and ref_speaker != hyp_speaker:
            speaker_error_time += sample_rate
        
        t += sample_rate
    
    der = (miss_time + false_alarm_time + speaker_error_time) / total_ref_time
    
    return {
        'der': der,
        'false_alarm': false_alarm_time / total_ref_time,
        'miss': miss_time / total_ref_time,
        'speaker_error': speaker_error_time / total_ref_time,
        'total_reference_time': total_ref_time
    }


def keyword_spotting_metrics(reference_keywords: List[str],
                              detected_keywords: List[str],
                              audio_duration: float = 1.0) -> Dict[str, float]:
    """
    Calculate keyword spotting metrics.
    
    Args:
        reference_keywords: List of actual keywords in audio
        detected_keywords: List of detected keywords
        audio_duration: Duration of audio in hours (for FA/hour calculation)
        
    Returns:
        Dictionary with precision, recall, F1, and false alarm rate
    """
    ref_set = set(reference_keywords)
    det_set = set(detected_keywords)
    
    true_positives = len(ref_set & det_set)
    false_positives = len(det_set - ref_set)
    false_negatives = len(ref_set - det_set)
    
    precision = true_positives / len(det_set) if det_set else 0.0
    recall = true_positives / len(ref_set) if ref_set else 0.0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # False alarms per hour
    fa_per_hour = false_positives / audio_duration if audio_duration > 0 else 0.0
    
    # Miss rate
    miss_rate = false_negatives / len(ref_set) if ref_set else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'false_alarms_per_hour': fa_per_hour,
        'miss_rate': miss_rate
    }


def mean_opinion_score(scores: List[float], 
                       confidence_interval: float = 0.95) -> Dict[str, float]:
    """
    Calculate Mean Opinion Score (MOS) statistics.
    
    MOS is typically rated on a 1-5 scale for speech quality.
    
    Args:
        scores: List of opinion scores (typically 1-5)
        confidence_interval: Confidence level for interval calculation
        
    Returns:
        Dictionary with MOS statistics
    """
    if not scores:
        return {
            'mos': 0.0,
            'std': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'count': 0
        }
    
    n = len(scores)
    mos = sum(scores) / n
    
    # Standard deviation
    variance = sum((s - mos) ** 2 for s in scores) / n
    std = variance ** 0.5
    
    # Confidence interval (using normal approximation)
    # z-score for 95% CI is approximately 1.96
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence_interval, 1.96)
    
    margin = z * std / (n ** 0.5)
    
    return {
        'mos': mos,
        'std': std,
        'ci_lower': max(1.0, mos - margin),
        'ci_upper': min(5.0, mos + margin),
        'count': n,
        'min': min(scores),
        'max': max(scores)
    }


def speech_intelligibility_index(snr_db: float) -> float:
    """
    Calculate Speech Intelligibility Index approximation based on SNR.
    
    This is a simplified approximation. Real SII requires spectral analysis.
    
    Args:
        snr_db: Signal-to-noise ratio in decibels
        
    Returns:
        SII approximation (0 to 1)
    """
    # Simple logistic approximation
    # SII ≈ 0 at SNR < -10 dB, SII ≈ 1 at SNR > 20 dB
    if snr_db < -10:
        return 0.0
    elif snr_db > 20:
        return 1.0
    else:
        # Logistic function centered at 5 dB
        return 1 / (1 + 2.718 ** (-(snr_db - 5) / 5))


def fluency_score(reference: str, hypothesis: str, 
                  expected_rate: float = 150.0) -> Dict[str, float]:
    """
    Calculate fluency-related metrics.
    
    Args:
        reference: Reference transcription
        hypothesis: Hypothesis transcription  
        expected_rate: Expected words per minute
        
    Returns:
        Dictionary with fluency metrics
    """
    hyp_words = hypothesis.split()
    
    # Hesitation markers (common disfluencies)
    hesitations = ['um', 'uh', 'er', 'ah', 'like', 'you know']
    
    hyp_lower = [w.lower() for w in hyp_words]
    hesitation_count = sum(1 for w in hyp_lower if w in hesitations)
    
    # Repetition detection
    repetitions = 0
    for i in range(1, len(hyp_words)):
        if hyp_words[i].lower() == hyp_words[i-1].lower():
            repetitions += 1
    
    # Calculate fluency score
    total_words = len(hyp_words)
    if total_words == 0:
        return {
            'fluency_score': 0.0,
            'hesitation_rate': 0.0,
            'repetition_rate': 0.0,
            'hesitation_count': 0,
            'repetition_count': 0
        }
    
    hesitation_rate = hesitation_count / total_words
    repetition_rate = repetitions / total_words
    
    # Fluency score (penalize hesitations and repetitions)
    fluency = max(0.0, 1.0 - hesitation_rate * 2 - repetition_rate * 2)
    
    return {
        'fluency_score': fluency,
        'hesitation_rate': hesitation_rate,
        'repetition_rate': repetition_rate,
        'hesitation_count': hesitation_count,
        'repetition_count': repetitions,
        'total_words': total_words
    }


# =============================================================================
# Speech Metrics Class
# =============================================================================

@dataclass
class SpeechMetrics:
    """Container for speech evaluation metrics."""
    
    wer: float = 0.0
    cer: float = 0.0
    mer: float = 0.0
    wil: float = 0.0
    accuracy: float = 0.0
    
    @classmethod
    def compute(cls, reference: str, hypothesis: str) -> 'SpeechMetrics':
        """
        Compute all speech metrics.
        
        Args:
            reference: Reference transcription
            hypothesis: Hypothesis transcription
            
        Returns:
            SpeechMetrics object
        """
        wer_result = word_error_rate(reference, hypothesis)
        cer_result = character_error_rate(reference, hypothesis)
        mer_result = match_error_rate(reference, hypothesis)
        wil_result = word_information_lost(reference, hypothesis)
        
        return cls(
            wer=wer_result['wer'],
            cer=cer_result['cer'],
            mer=mer_result['mer'],
            wil=wil_result['wil'],
            accuracy=wer_result['accuracy']
        )
    
    @classmethod
    def compute_corpus(cls, references: List[str], 
                       hypotheses: List[str]) -> 'SpeechMetrics':
        """
        Compute corpus-level speech metrics.
        
        Args:
            references: List of reference transcriptions
            hypotheses: List of hypotheses
            
        Returns:
            SpeechMetrics object with corpus-level metrics
        """
        wer_result = corpus_wer(references, hypotheses)
        cer_result = corpus_cer(references, hypotheses)
        
        # Calculate corpus-level MER and WIL
        mer_values = [match_error_rate(r, h)['mer'] 
                      for r, h in zip(references, hypotheses)]
        wil_values = [word_information_lost(r, h)['wil'] 
                      for r, h in zip(references, hypotheses)]
        accuracy_values = [recognition_accuracy(r, h) 
                          for r, h in zip(references, hypotheses)]
        
        return cls(
            wer=wer_result['wer'],
            cer=cer_result['cer'],
            mer=mean(mer_values) if mer_values else 0.0,
            wil=mean(wil_values) if wil_values else 0.0,
            accuracy=mean(accuracy_values) if accuracy_values else 0.0
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'wer': self.wer,
            'cer': self.cer,
            'mer': self.mer,
            'wil': self.wil,
            'accuracy': self.accuracy
        }
