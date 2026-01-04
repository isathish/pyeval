"""
Mathematical Operations - Pure Python Implementation
====================================================

Core mathematical functions without external dependencies.
"""

import math
from typing import List, Union, Optional, Tuple

Number = Union[int, float]


def mean(values: List[Number]) -> float:
    """
    Calculate the arithmetic mean of a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        The arithmetic mean
        
    Example:
        >>> mean([1, 2, 3, 4, 5])
        3.0
    """
    if not values:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(values) / len(values)


def variance(values: List[Number], ddof: int = 0) -> float:
    """
    Calculate the variance of a list of values.
    
    Args:
        values: List of numeric values
        ddof: Delta degrees of freedom (0 for population, 1 for sample)
        
    Returns:
        The variance
        
    Example:
        >>> variance([1, 2, 3, 4, 5])
        2.0
    """
    if not values:
        raise ValueError("Cannot calculate variance of empty list")
    if len(values) - ddof <= 0:
        raise ValueError("Not enough data points for the specified ddof")
    
    m = mean(values)
    squared_diffs = [(x - m) ** 2 for x in values]
    return sum(squared_diffs) / (len(values) - ddof)


def std(values: List[Number], ddof: int = 0) -> float:
    """
    Calculate the standard deviation of a list of values.
    
    Args:
        values: List of numeric values
        ddof: Delta degrees of freedom (0 for population, 1 for sample)
        
    Returns:
        The standard deviation
        
    Example:
        >>> std([1, 2, 3, 4, 5])
        1.4142135623730951
    """
    return math.sqrt(variance(values, ddof))


def median(values: List[Number]) -> float:
    """
    Calculate the median of a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        The median value
        
    Example:
        >>> median([1, 2, 3, 4, 5])
        3
    """
    if not values:
        raise ValueError("Cannot calculate median of empty list")
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n % 2 == 1:
        return sorted_values[n // 2]
    else:
        mid = n // 2
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def percentile(values: List[Number], p: float) -> float:
    """
    Calculate the p-th percentile of a list of values.
    
    Args:
        values: List of numeric values
        p: Percentile to compute (0-100)
        
    Returns:
        The p-th percentile value
        
    Example:
        >>> percentile([1, 2, 3, 4, 5], 50)
        3.0
    """
    if not values:
        raise ValueError("Cannot calculate percentile of empty list")
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Linear interpolation
    k = (p / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_values[int(k)]
    
    return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


def normalize(values: List[Number], method: str = "minmax") -> List[float]:
    """
    Normalize a list of values.
    
    Args:
        values: List of numeric values
        method: Normalization method ("minmax" or "zscore")
        
    Returns:
        Normalized values
        
    Example:
        >>> normalize([1, 2, 3, 4, 5])
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    if not values:
        return []
    
    if method == "minmax":
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return [0.0] * len(values)
        return [(x - min_val) / (max_val - min_val) for x in values]
    
    elif method == "zscore":
        m = mean(values)
        s = std(values)
        if s == 0:
            return [0.0] * len(values)
        return [(x - m) / s for x in values]
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def softmax(values: List[Number]) -> List[float]:
    """
    Apply softmax function to a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Softmax probabilities
        
    Example:
        >>> softmax([1, 2, 3])
        [0.09003057317038046, 0.24472847105479767, 0.6652409557748219]
    """
    if not values:
        return []
    
    # Subtract max for numerical stability
    max_val = max(values)
    exp_values = [math.exp(x - max_val) for x in values]
    sum_exp = sum(exp_values)
    
    return [x / sum_exp for x in exp_values]


def sigmoid(x: Number) -> float:
    """
    Apply sigmoid function.
    
    Args:
        x: Input value
        
    Returns:
        Sigmoid output
        
    Example:
        >>> sigmoid(0)
        0.5
    """
    # Clip to prevent overflow
    if x < -709:
        return 0.0
    if x > 709:
        return 1.0
    return 1 / (1 + math.exp(-x))


def cosine_similarity(vec1: List[Number], vec2: List[Number]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (-1 to 1)
        
    Example:
        >>> cosine_similarity([1, 0, 1], [0, 1, 1])
        0.5
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    
    if not vec1:
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: List[Number], vec2: List[Number]) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance
        
    Example:
        >>> euclidean_distance([0, 0], [3, 4])
        5.0
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Jaccard similarity score (0 to 1)
        
    Example:
        >>> jaccard_similarity({1, 2, 3}, {2, 3, 4})
        0.5
    """
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance (number of insertions, deletions, substitutions)
        
    Example:
        >>> levenshtein_distance("kitten", "sitting")
        3
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    
    if not s2:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            cost = 0 if c1 == c2 else 1
            
            current_row.append(min(
                previous_row[j + 1] + 1,      # Deletion
                current_row[j] + 1,            # Insertion
                previous_row[j] + cost         # Substitution
            ))
        
        previous_row = current_row
    
    return previous_row[-1]


def dot_product(vec1: List[Number], vec2: List[Number]) -> float:
    """
    Calculate dot product of two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Dot product value
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    
    return sum(a * b for a, b in zip(vec1, vec2))


def l2_norm(vec: List[Number]) -> float:
    """
    Calculate L2 norm (Euclidean norm) of a vector.
    
    Args:
        vec: Input vector
        
    Returns:
        L2 norm value
    """
    return math.sqrt(sum(x * x for x in vec))


def l1_norm(vec: List[Number]) -> float:
    """
    Calculate L1 norm (Manhattan norm) of a vector.
    
    Args:
        vec: Input vector
        
    Returns:
        L1 norm value
    """
    return sum(abs(x) for x in vec)


def manhattan_distance(vec1: List[Number], vec2: List[Number]) -> float:
    """
    Calculate Manhattan distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Manhattan distance
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    
    return sum(abs(a - b) for a, b in zip(vec1, vec2))


def hamming_distance(s1: str, s2: str) -> int:
    """
    Calculate Hamming distance between two strings of equal length.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Hamming distance
    """
    if len(s1) != len(s2):
        raise ValueError("Strings must have the same length")
    
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def log_loss(y_true: List[int], y_pred: List[float], eps: float = 1e-15) -> float:
    """
    Calculate log loss (cross-entropy loss).
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities
        eps: Small value to avoid log(0)
        
    Returns:
        Log loss value
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")
    
    # Clip predictions to avoid log(0)
    y_pred_clipped = [max(eps, min(1 - eps, p)) for p in y_pred]
    
    loss = 0.0
    for true, pred in zip(y_true, y_pred_clipped):
        loss += -(true * math.log(pred) + (1 - true) * math.log(1 - pred))
    
    return loss / len(y_true)


def entropy(probabilities: List[float], base: float = math.e) -> float:
    """
    Calculate entropy of a probability distribution.
    
    Args:
        probabilities: Probability distribution (must sum to 1)
        base: Logarithm base (default: e for natural log)
        
    Returns:
        Entropy value
    """
    log_func = math.log if base == math.e else lambda x: math.log(x) / math.log(base)
    
    h = 0.0
    for p in probabilities:
        if p > 0:
            h -= p * log_func(p)
    
    return h


def cross_entropy(p: List[float], q: List[float], eps: float = 1e-15) -> float:
    """
    Calculate cross-entropy between two distributions.
    
    Args:
        p: True distribution
        q: Predicted distribution
        eps: Small value to avoid log(0)
        
    Returns:
        Cross-entropy value
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have the same length")
    
    ce = 0.0
    for pi, qi in zip(p, q):
        qi_clipped = max(eps, qi)
        if pi > 0:
            ce -= pi * math.log(qi_clipped)
    
    return ce


def kl_divergence(p: List[float], q: List[float], eps: float = 1e-15) -> float:
    """
    Calculate KL divergence from distribution p to q.
    
    Args:
        p: True distribution
        q: Approximate distribution
        eps: Small value to avoid division by zero
        
    Returns:
        KL divergence value
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have the same length")
    
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            qi_clipped = max(eps, qi)
            kl += pi * math.log(pi / qi_clipped)
    
    return kl
