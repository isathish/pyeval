"""
NLP Metrics - Pure Python Implementation
=========================================

Text evaluation metrics for NLP tasks including BLEU, ROUGE, METEOR.
"""

from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass
from collections import Counter
import math

from pyeval.utils.text_ops import (
    tokenize, 
    ngrams, 
    word_ngrams,
    normalize_text,
    stem,
    STOPWORDS
)
from pyeval.utils.math_ops import mean


# =============================================================================
# BLEU Score
# =============================================================================

def _modified_precision(reference_ngrams: List[Counter], 
                        candidate_ngrams: Counter) -> Tuple[int, int]:
    """
    Calculate modified precision for n-grams.
    
    Args:
        reference_ngrams: List of Counter objects for each reference
        candidate_ngrams: Counter object for candidate
        
    Returns:
        Tuple of (clipped_count, total_count)
    """
    clipped_count = 0
    total_count = 0
    
    for ngram, count in candidate_ngrams.items():
        # Maximum count in any reference
        max_ref_count = max(ref.get(ngram, 0) for ref in reference_ngrams)
        clipped_count += min(count, max_ref_count)
        total_count += count
    
    return clipped_count, total_count


def _brevity_penalty(reference_lengths: List[int], candidate_length: int) -> float:
    """
    Calculate brevity penalty for BLEU.
    
    Args:
        reference_lengths: Lengths of references
        candidate_length: Length of candidate
        
    Returns:
        Brevity penalty (0 to 1)
    """
    if candidate_length == 0:
        return 0.0
    
    # Choose closest reference length
    closest_ref_len = min(reference_lengths, 
                          key=lambda r: (abs(r - candidate_length), r))
    
    if candidate_length >= closest_ref_len:
        return 1.0
    
    return math.exp(1 - closest_ref_len / candidate_length)


def bleu_score(references: Union[List[str], List[List[str]]], 
               candidate: str,
               max_n: int = 4,
               weights: Optional[List[float]] = None,
               smoothing: bool = False) -> Dict[str, float]:
    """
    Calculate BLEU (Bilingual Evaluation Understudy) score.
    
    Args:
        references: Single reference or list of references
        candidate: Candidate translation
        max_n: Maximum n-gram order (default: 4 for BLEU-4)
        weights: Weights for each n-gram (default: uniform)
        smoothing: Apply smoothing for zero counts
        
    Returns:
        Dictionary with BLEU score and n-gram precisions
        
    Example:
        >>> refs = ["The cat sat on the mat"]
        >>> cand = "The cat is on the mat"
        >>> result = bleu_score(refs, cand)
        >>> result['bleu']
    """
    # Default weights
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    # Tokenize candidate
    candidate_tokens = tokenize(candidate, lowercase=True, remove_punct=True)
    
    # Handle references - can be list of strings or list of lists of tokens
    reference_tokens_list = []
    for ref in references:
        if isinstance(ref, str):
            reference_tokens_list.append(tokenize(ref, lowercase=True, remove_punct=True))
        elif isinstance(ref, list):
            # Already tokenized
            reference_tokens_list.append([t.lower() if isinstance(t, str) else t for t in ref])
    
    # Calculate n-gram precisions
    precisions = []
    
    for n in range(1, max_n + 1):
        candidate_ngrams = Counter(ngrams(candidate_tokens, n))
        reference_ngrams = [Counter(ngrams(ref_tokens, n)) 
                           for ref_tokens in reference_tokens_list]
        
        clipped, total = _modified_precision(reference_ngrams, candidate_ngrams)
        
        if total == 0:
            if smoothing:
                precisions.append(1.0 / (len(candidate_tokens) + 1))
            else:
                precisions.append(0.0)
        else:
            if smoothing and clipped == 0:
                clipped = 1.0 / (len(candidate_tokens) + 1)
            precisions.append(clipped / total)
    
    # Calculate geometric mean of precisions
    if any(p == 0 for p in precisions):
        bleu = 0.0
    else:
        log_precisions = sum(w * math.log(p) for w, p in zip(weights, precisions))
        bleu = math.exp(log_precisions)
    
    # Apply brevity penalty
    reference_lengths = [len(tokens) for tokens in reference_tokens_list]
    bp = _brevity_penalty(reference_lengths, len(candidate_tokens))
    
    bleu *= bp
    
    return {
        'bleu': bleu,
        'brevity_penalty': bp,
        'precisions': precisions,
        'candidate_length': len(candidate_tokens),
        'reference_length': min(reference_lengths, 
                                key=lambda r: abs(r - len(candidate_tokens)))
    }


def sentence_bleu(reference: str, candidate: str, 
                  max_n: int = 4) -> float:
    """
    Calculate sentence-level BLEU score.
    
    Args:
        reference: Reference sentence
        candidate: Candidate sentence
        max_n: Maximum n-gram order
        
    Returns:
        BLEU score (0 to 1)
    """
    result = bleu_score([reference], candidate, max_n=max_n, smoothing=True)
    return result['bleu']


def corpus_bleu(references: List[List[str]], 
                candidates: List[str],
                max_n: int = 4) -> Dict[str, float]:
    """
    Calculate corpus-level BLEU score.
    
    Args:
        references: List of reference lists for each candidate
        candidates: List of candidate translations
        max_n: Maximum n-gram order
        
    Returns:
        Dictionary with corpus BLEU and statistics
    """
    if len(references) != len(candidates):
        raise ValueError("Number of references must match number of candidates")
    
    # Aggregate n-gram counts
    total_clipped = [0] * max_n
    total_count = [0] * max_n
    total_candidate_len = 0
    total_reference_len = 0
    
    for refs, cand in zip(references, candidates):
        cand_tokens = tokenize(cand, lowercase=True, remove_punct=True)
        ref_tokens_list = [tokenize(ref, lowercase=True, remove_punct=True) for ref in refs]
        
        for n in range(1, max_n + 1):
            cand_ngrams = Counter(ngrams(cand_tokens, n))
            ref_ngrams = [Counter(ngrams(ref_tokens, n)) for ref_tokens in ref_tokens_list]
            
            clipped, count = _modified_precision(ref_ngrams, cand_ngrams)
            total_clipped[n-1] += clipped
            total_count[n-1] += count
        
        total_candidate_len += len(cand_tokens)
        
        # Choose closest reference length
        ref_lengths = [len(tokens) for tokens in ref_tokens_list]
        closest = min(ref_lengths, key=lambda r: abs(r - len(cand_tokens)))
        total_reference_len += closest
    
    # Calculate precisions
    precisions = []
    for i in range(max_n):
        if total_count[i] == 0:
            precisions.append(0.0)
        else:
            precisions.append(total_clipped[i] / total_count[i])
    
    # Calculate BLEU
    if any(p == 0 for p in precisions):
        bleu = 0.0
    else:
        weights = [1.0 / max_n] * max_n
        log_precisions = sum(w * math.log(p) for w, p in zip(weights, precisions))
        bleu = math.exp(log_precisions)
    
    # Brevity penalty
    if total_candidate_len >= total_reference_len:
        bp = 1.0
    elif total_candidate_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - total_reference_len / total_candidate_len)
    
    bleu *= bp
    
    return {
        'bleu': bleu,
        'brevity_penalty': bp,
        'precisions': precisions,
        'total_candidate_length': total_candidate_len,
        'total_reference_length': total_reference_len
    }


# =============================================================================
# ROUGE Score
# =============================================================================

def _lcs_length(s1: List[str], s2: List[str]) -> int:
    """
    Calculate length of Longest Common Subsequence.
    
    Args:
        s1: First sequence
        s2: Second sequence
        
    Returns:
        LCS length
    """
    m, n = len(s1), len(s2)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def rouge_score(reference: str, candidate: str,
                rouge_types: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores.
    
    Args:
        reference: Reference summary
        candidate: Candidate summary
        rouge_types: List of ROUGE types ('rouge1', 'rouge2', 'rougeL')
        
    Returns:
        Dictionary with ROUGE scores for each type
        
    Example:
        >>> ref = "The cat sat on the mat"
        >>> cand = "The cat is on the mat"
        >>> result = rouge_score(ref, cand)
        >>> result['rouge1']['f1']
    """
    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    ref_tokens = tokenize(reference, lowercase=True, remove_punct=True)
    cand_tokens = tokenize(candidate, lowercase=True, remove_punct=True)
    
    results = {}
    
    for rouge_type in rouge_types:
        if rouge_type == 'rouge1':
            # Unigram overlap
            ref_unigrams = Counter(ref_tokens)
            cand_unigrams = Counter(cand_tokens)
            
            overlap = sum((ref_unigrams & cand_unigrams).values())
            ref_len = sum(ref_unigrams.values())
            cand_len = sum(cand_unigrams.values())
            
        elif rouge_type == 'rouge2':
            # Bigram overlap
            ref_bigrams = Counter(ngrams(ref_tokens, 2))
            cand_bigrams = Counter(ngrams(cand_tokens, 2))
            
            overlap = sum((ref_bigrams & cand_bigrams).values())
            ref_len = sum(ref_bigrams.values())
            cand_len = sum(cand_bigrams.values())
            
        elif rouge_type == 'rougeL':
            # LCS-based
            lcs_len = _lcs_length(ref_tokens, cand_tokens)
            overlap = lcs_len
            ref_len = len(ref_tokens)
            cand_len = len(cand_tokens)
            
        else:
            # Generic n-gram ROUGE (e.g., rouge3, rouge4)
            n = int(rouge_type.replace('rouge', ''))
            ref_ngrams = Counter(ngrams(ref_tokens, n))
            cand_ngrams = Counter(ngrams(cand_tokens, n))
            
            overlap = sum((ref_ngrams & cand_ngrams).values())
            ref_len = sum(ref_ngrams.values())
            cand_len = sum(cand_ngrams.values())
        
        # Calculate precision, recall, F1
        precision = overlap / cand_len if cand_len > 0 else 0.0
        recall = overlap / ref_len if ref_len > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        results[rouge_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return results


def rouge_n(reference: str, candidate: str, n: int = 1) -> Dict[str, float]:
    """
    Calculate ROUGE-N score for specific n-gram size.
    
    Args:
        reference: Reference text
        candidate: Candidate text
        n: N-gram size
        
    Returns:
        Dictionary with precision, recall, f1
    """
    result = rouge_score(reference, candidate, [f'rouge{n}'])
    return result[f'rouge{n}']


def rouge_l(reference: str, candidate: str) -> Dict[str, float]:
    """
    Calculate ROUGE-L score (LCS-based).
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        Dictionary with precision, recall, f1
    """
    result = rouge_score(reference, candidate, ['rougeL'])
    return result['rougeL']


# =============================================================================
# METEOR Score
# =============================================================================

def _exact_match(ref_tokens: List[str], cand_tokens: List[str]) -> Tuple[Set[int], Set[int]]:
    """Find exact matches between tokens."""
    ref_matched = set()
    cand_matched = set()
    
    ref_remaining = {i: t for i, t in enumerate(ref_tokens)}
    
    for i, cand_tok in enumerate(cand_tokens):
        for j, ref_tok in list(ref_remaining.items()):
            if cand_tok == ref_tok:
                cand_matched.add(i)
                ref_matched.add(j)
                del ref_remaining[j]
                break
    
    return ref_matched, cand_matched


def _stem_match(ref_tokens: List[str], cand_tokens: List[str],
                ref_matched: Set[int], cand_matched: Set[int]) -> Tuple[Set[int], Set[int]]:
    """Find stem matches between unmatched tokens."""
    ref_remaining = {i: stem(t) for i, t in enumerate(ref_tokens) if i not in ref_matched}
    
    for i, cand_tok in enumerate(cand_tokens):
        if i in cand_matched:
            continue
        cand_stem = stem(cand_tok)
        
        for j, ref_stem in list(ref_remaining.items()):
            if cand_stem == ref_stem:
                cand_matched.add(i)
                ref_matched.add(j)
                del ref_remaining[j]
                break
    
    return ref_matched, cand_matched


def _synonym_match(ref_tokens: List[str], cand_tokens: List[str],
                   ref_matched: Set[int], cand_matched: Set[int]) -> Tuple[Set[int], Set[int]]:
    """
    Find synonym matches (simplified - uses word similarity).
    
    Note: Full implementation would require a WordNet-like resource.
    This is a simplified version using basic word similarity.
    """
    # Simple synonyms dictionary (expandable)
    SYNONYMS = {
        'good': {'great', 'excellent', 'fine', 'nice'},
        'bad': {'poor', 'terrible', 'awful'},
        'big': {'large', 'huge', 'enormous'},
        'small': {'tiny', 'little', 'mini'},
        'fast': {'quick', 'rapid', 'swift'},
        'slow': {'sluggish', 'gradual'},
        'happy': {'glad', 'joyful', 'pleased'},
        'sad': {'unhappy', 'sorrowful', 'gloomy'},
        'start': {'begin', 'commence'},
        'end': {'finish', 'conclude', 'stop'},
        'see': {'view', 'watch', 'observe'},
        'say': {'tell', 'speak', 'talk'},
    }
    
    def are_synonyms(w1: str, w2: str) -> bool:
        w1, w2 = w1.lower(), w2.lower()
        if w1 == w2:
            return True
        for base, syns in SYNONYMS.items():
            words = {base} | syns
            if w1 in words and w2 in words:
                return True
        return False
    
    ref_remaining = {i: ref_tokens[i] for i in range(len(ref_tokens)) if i not in ref_matched}
    
    for i, cand_tok in enumerate(cand_tokens):
        if i in cand_matched:
            continue
        
        for j, ref_tok in list(ref_remaining.items()):
            if are_synonyms(cand_tok, ref_tok):
                cand_matched.add(i)
                ref_matched.add(j)
                del ref_remaining[j]
                break
    
    return ref_matched, cand_matched


def _count_chunks(ref_tokens: List[str], cand_tokens: List[str],
                  ref_matched: Set[int], cand_matched: Set[int]) -> int:
    """Count the number of chunks (contiguous aligned matches)."""
    if not cand_matched:
        return 0
    
    # Build mapping from candidate to reference positions
    cand_to_ref = {}
    ref_to_cand = {}
    
    sorted_cand = sorted(cand_matched)
    sorted_ref = sorted(ref_matched)
    
    # Simple alignment (assumes order preservation)
    for c, r in zip(sorted_cand, sorted_ref):
        cand_to_ref[c] = r
        ref_to_cand[r] = c
    
    # Count chunks
    chunks = 1
    prev_ref_pos = -2
    
    for cand_pos in sorted_cand:
        ref_pos = cand_to_ref.get(cand_pos, -1)
        if ref_pos != prev_ref_pos + 1:
            if prev_ref_pos >= 0:
                chunks += 1
        prev_ref_pos = ref_pos
    
    return chunks


def meteor_score(reference: str, candidate: str,
                 alpha: float = 0.9, beta: float = 3.0, 
                 gamma: float = 0.5) -> Dict[str, float]:
    """
    Calculate METEOR (Metric for Evaluation of Translation with Explicit ORdering) score.
    
    METEOR considers exact matches, stems, synonyms, and word order.
    
    Args:
        reference: Reference text
        candidate: Candidate text
        alpha: Weight for precision (default 0.9)
        beta: Weight for fragmentation penalty (default 3.0)
        gamma: Weight for chunk penalty (default 0.5)
        
    Returns:
        Dictionary with METEOR score and components
        
    Example:
        >>> ref = "The cat sat on the mat"
        >>> cand = "The cat is on the mat"
        >>> result = meteor_score(ref, cand)
        >>> result['meteor']
    """
    ref_tokens = tokenize(reference, lowercase=True, remove_punct=True)
    cand_tokens = tokenize(candidate, lowercase=True, remove_punct=True)
    
    if not ref_tokens or not cand_tokens:
        return {
            'meteor': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f_mean': 0.0,
            'penalty': 0.0,
            'matches': 0
        }
    
    # Stage 1: Exact matches
    ref_matched, cand_matched = _exact_match(ref_tokens, cand_tokens)
    
    # Stage 2: Stem matches
    ref_matched, cand_matched = _stem_match(ref_tokens, cand_tokens, 
                                            ref_matched, cand_matched)
    
    # Stage 3: Synonym matches
    ref_matched, cand_matched = _synonym_match(ref_tokens, cand_tokens,
                                               ref_matched, cand_matched)
    
    matches = len(cand_matched)
    
    # Calculate precision and recall
    precision = matches / len(cand_tokens) if cand_tokens else 0.0
    recall = matches / len(ref_tokens) if ref_tokens else 0.0
    
    # Calculate F-mean
    if precision + recall > 0:
        f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    else:
        f_mean = 0.0
    
    # Calculate fragmentation penalty
    chunks = _count_chunks(ref_tokens, cand_tokens, ref_matched, cand_matched)
    
    if matches > 0:
        frag = chunks / matches
    else:
        frag = 0.0
    
    penalty = gamma * (frag ** beta)
    
    # Final METEOR score
    meteor = f_mean * (1 - penalty)
    
    return {
        'meteor': meteor,
        'precision': precision,
        'recall': recall,
        'f_mean': f_mean,
        'penalty': penalty,
        'matches': matches,
        'chunks': chunks
    }


# =============================================================================
# Additional NLP Metrics
# =============================================================================

def ter_score(reference: str, candidate: str) -> Dict[str, float]:
    """
    Calculate Translation Edit Rate (TER).
    
    TER = (insertions + deletions + substitutions + shifts) / reference_length
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        Dictionary with TER and edit counts
    """
    from pyeval.utils.math_ops import levenshtein_distance
    
    ref_tokens = tokenize(reference, lowercase=True, remove_punct=True)
    cand_tokens = tokenize(candidate, lowercase=True, remove_punct=True)
    
    # Calculate edit distance
    edit_dist = levenshtein_distance(' '.join(ref_tokens), ' '.join(cand_tokens))
    
    ref_len = len(ref_tokens) if ref_tokens else 1
    ter = edit_dist / ref_len
    
    return {
        'ter': ter,
        'edit_distance': edit_dist,
        'reference_length': len(ref_tokens)
    }


def distinct_n(texts: List[str], n: int = 1) -> float:
    """
    Calculate Distinct-N metric for diversity evaluation.
    
    Distinct-N = unique n-grams / total n-grams
    
    Args:
        texts: List of generated texts
        n: N-gram size
        
    Returns:
        Distinct-N score (0 to 1)
    """
    all_ngrams = []
    
    for text in texts:
        tokens = tokenize(text, lowercase=True, remove_punct=True)
        all_ngrams.extend(ngrams(tokens, n))
    
    if not all_ngrams:
        return 0.0
    
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    
    return unique_ngrams / total_ngrams


def self_bleu(texts: List[str], max_n: int = 4) -> float:
    """
    Calculate Self-BLEU for diversity evaluation.
    
    Lower Self-BLEU indicates higher diversity.
    
    Args:
        texts: List of generated texts
        max_n: Maximum n-gram order
        
    Returns:
        Average Self-BLEU score
    """
    if len(texts) < 2:
        return 0.0
    
    scores = []
    
    for i, text in enumerate(texts):
        # Use all other texts as references
        references = [texts[j] for j in range(len(texts)) if j != i]
        result = bleu_score(references, text, max_n=max_n, smoothing=True)
        scores.append(result['bleu'])
    
    return mean(scores)


def perplexity_from_logprobs(log_probs: List[float]) -> float:
    """
    Calculate perplexity from log probabilities.
    
    Perplexity = exp(-avg(log_probs))
    
    Args:
        log_probs: List of log probabilities
        
    Returns:
        Perplexity score
    """
    if not log_probs:
        return float('inf')
    
    avg_log_prob = mean(log_probs)
    return math.exp(-avg_log_prob)


def chrf_score(reference: str, candidate: str, 
               n: int = 6, beta: float = 2.0) -> Dict[str, float]:
    """
    Calculate chrF (character n-gram F-score).
    
    chrF is a character-level metric that is more robust to morphological
    variations than word-based metrics.
    
    Args:
        reference: Reference text
        candidate: Candidate text
        n: Maximum character n-gram order
        beta: Weight of recall vs precision
        
    Returns:
        Dictionary with chrF scores
    """
    def char_ngrams(text: str, order: int) -> Counter:
        """Extract character n-grams from text."""
        return Counter(text[i:i+order] for i in range(len(text) - order + 1))
    
    ref_clean = reference.lower()
    cand_clean = candidate.lower()
    
    total_precision = 0.0
    total_recall = 0.0
    n_orders = 0
    
    for order in range(1, n + 1):
        ref_ngrams = char_ngrams(ref_clean, order)
        cand_ngrams = char_ngrams(cand_clean, order)
        
        if not cand_ngrams or not ref_ngrams:
            continue
        
        # Calculate precision and recall
        overlap = sum((cand_ngrams & ref_ngrams).values())
        
        precision = overlap / sum(cand_ngrams.values()) if cand_ngrams else 0.0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0
        
        total_precision += precision
        total_recall += recall
        n_orders += 1
    
    if n_orders == 0:
        return {'chrf': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    avg_precision = total_precision / n_orders
    avg_recall = total_recall / n_orders
    
    # F-score with beta
    if avg_precision + avg_recall == 0:
        chrf = 0.0
    else:
        chrf = (1 + beta ** 2) * avg_precision * avg_recall / (beta ** 2 * avg_precision + avg_recall)
    
    return {
        'chrf': chrf,
        'precision': avg_precision,
        'recall': avg_recall
    }


def text_entropy(text: str, n: int = 1) -> float:
    """
    Calculate entropy of text based on n-gram distribution.
    
    Higher entropy indicates more diverse/unpredictable text.
    
    Args:
        text: Input text
        n: N-gram size
        
    Returns:
        Entropy value
    """
    tokens = tokenize(text, lowercase=True, remove_punct=True)
    
    if len(tokens) < n:
        return 0.0
    
    ngram_list = list(ngrams(tokens, n))
    total = len(ngram_list)
    
    if total == 0:
        return 0.0
    
    ngram_counts = Counter(ngram_list)
    
    entropy = 0.0
    for count in ngram_counts.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    
    return entropy


def repetition_ratio(text: str, n: int = 3) -> float:
    """
    Calculate repetition ratio of text.
    
    Measures how much of the text is repeated n-grams.
    Higher values indicate more repetition.
    
    Args:
        text: Input text
        n: N-gram size for detecting repetition
        
    Returns:
        Repetition ratio (0 to 1)
    """
    tokens = tokenize(text, lowercase=True, remove_punct=True)
    
    if len(tokens) < n:
        return 0.0
    
    ngram_list = list(ngrams(tokens, n))
    total = len(ngram_list)
    
    if total == 0:
        return 0.0
    
    ngram_counts = Counter(ngram_list)
    
    # Count tokens covered by repeated n-grams
    repeated_ngrams = sum(count for count in ngram_counts.values() if count > 1)
    
    return repeated_ngrams / total


def compression_ratio(original: str, summary: str) -> float:
    """
    Calculate compression ratio for summarization.
    
    Args:
        original: Original text
        summary: Summary text
        
    Returns:
        Compression ratio (summary_length / original_length)
    """
    orig_tokens = tokenize(original)
    sum_tokens = tokenize(summary)
    
    if not orig_tokens:
        return 0.0
    
    return len(sum_tokens) / len(orig_tokens)


def coverage_score(source: str, summary: str) -> float:
    """
    Calculate coverage score for summarization.
    
    Measures how much of the source content is covered in the summary.
    
    Args:
        source: Source text
        summary: Summary text
        
    Returns:
        Coverage score (0 to 1)
    """
    source_tokens = set(tokenize(source, lowercase=True, remove_punct=True))
    source_tokens -= STOPWORDS
    
    summary_tokens = set(tokenize(summary, lowercase=True, remove_punct=True))
    summary_tokens -= STOPWORDS
    
    if not source_tokens:
        return 0.0
    
    covered = len(source_tokens & summary_tokens)
    return covered / len(source_tokens)


def density_score(source: str, summary: str) -> float:
    """
    Calculate density score for extractive summarization.
    
    Higher density means summary contains more extractive fragments.
    
    Args:
        source: Source text
        summary: Summary text
        
    Returns:
        Density score
    """
    source_tokens = tokenize(source, lowercase=True)
    summary_tokens = tokenize(summary, lowercase=True)
    
    if not summary_tokens:
        return 0.0
    
    # Find longest common subsequences
    def lcs_length(s1, s2):
        if not s1 or not s2:
            return 0
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs_len = lcs_length(source_tokens, summary_tokens)
    return lcs_len / len(summary_tokens) if summary_tokens else 0.0


def word_mover_distance_approx(text1: str, text2: str) -> float:
    """
    Calculate approximate Word Mover's Distance.
    
    This is a simplified version using word overlap as a proxy.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Approximate WMD (lower is more similar)
    """
    tokens1 = set(tokenize(text1, lowercase=True, remove_punct=True)) - STOPWORDS
    tokens2 = set(tokenize(text2, lowercase=True, remove_punct=True)) - STOPWORDS
    
    if not tokens1 or not tokens2:
        return 1.0
    
    # Use Jaccard distance as approximation
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return 1 - (intersection / union) if union > 0 else 1.0


def lexical_diversity(text: str) -> Dict[str, float]:
    """
    Calculate multiple lexical diversity metrics.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with diversity metrics
    """
    tokens = tokenize(text, lowercase=True, remove_punct=True)
    
    if not tokens:
        return {
            'ttr': 0.0,
            'root_ttr': 0.0,
            'log_ttr': 0.0,
            'maas': 0.0,
            'vocabulary_size': 0,
            'total_tokens': 0
        }
    
    types = set(tokens)
    n_types = len(types)
    n_tokens = len(tokens)
    
    # Type-Token Ratio
    ttr = n_types / n_tokens
    
    # Root TTR (Guiraud's index)
    root_ttr = n_types / math.sqrt(n_tokens)
    
    # Log TTR (Herdan's C)
    log_ttr = math.log(n_types) / math.log(n_tokens) if n_tokens > 1 else 0
    
    # Maas index (lower is more diverse)
    maas = (math.log(n_tokens) - math.log(n_types)) / (math.log(n_tokens) ** 2) if n_tokens > 1 else 0
    
    return {
        'ttr': ttr,
        'root_ttr': root_ttr,
        'log_ttr': log_ttr,
        'maas': maas,
        'vocabulary_size': n_types,
        'total_tokens': n_tokens
    }


def sentence_bleu(reference: str, candidate: str, 
                  max_n: int = 4, smoothing: bool = True) -> float:
    """
    Calculate sentence-level BLEU score.
    
    Convenience wrapper for single sentence evaluation.
    
    Args:
        reference: Reference sentence
        candidate: Candidate sentence
        max_n: Maximum n-gram order
        smoothing: Apply smoothing
        
    Returns:
        BLEU score
    """
    result = bleu_score([reference], candidate, max_n=max_n, smoothing=smoothing)
    return result['bleu']


# =============================================================================
# NLP Metrics Class
# =============================================================================

@dataclass
class NLPMetrics:
    """Container for NLP evaluation metrics."""
    
    bleu: float = 0.0
    rouge1_f1: float = 0.0
    rouge2_f1: float = 0.0
    rougeL_f1: float = 0.0
    meteor: float = 0.0
    
    @classmethod
    def compute(cls, reference: str, candidate: str) -> 'NLPMetrics':
        """
        Compute all NLP metrics.
        
        Args:
            reference: Reference text
            candidate: Candidate text
            
        Returns:
            NLPMetrics object
        """
        bleu_result = bleu_score([reference], candidate, smoothing=True)
        rouge_result = rouge_score(reference, candidate)
        meteor_result = meteor_score(reference, candidate)
        
        return cls(
            bleu=bleu_result['bleu'],
            rouge1_f1=rouge_result['rouge1']['f1'],
            rouge2_f1=rouge_result['rouge2']['f1'],
            rougeL_f1=rouge_result['rougeL']['f1'],
            meteor=meteor_result['meteor']
        )
    
    @classmethod
    def compute_batch(cls, references: List[str], 
                      candidates: List[str]) -> 'NLPMetrics':
        """
        Compute average NLP metrics over a batch.
        
        Args:
            references: List of reference texts
            candidates: List of candidate texts
            
        Returns:
            NLPMetrics object with averaged scores
        """
        if len(references) != len(candidates):
            raise ValueError("Number of references must match number of candidates")
        
        metrics_list = [cls.compute(ref, cand) 
                        for ref, cand in zip(references, candidates)]
        
        return cls(
            bleu=mean([m.bleu for m in metrics_list]),
            rouge1_f1=mean([m.rouge1_f1 for m in metrics_list]),
            rouge2_f1=mean([m.rouge2_f1 for m in metrics_list]),
            rougeL_f1=mean([m.rougeL_f1 for m in metrics_list]),
            meteor=mean([m.meteor for m in metrics_list])
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'bleu': self.bleu,
            'rouge1_f1': self.rouge1_f1,
            'rouge2_f1': self.rouge2_f1,
            'rougeL_f1': self.rougeL_f1,
            'meteor': self.meteor
        }
