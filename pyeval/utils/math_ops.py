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


# =============================================================================
# Statistical Testing Functions
# =============================================================================

import random
from typing import Dict, Any


def bootstrap_confidence_interval(data: List[Number], 
                                  statistic: str = "mean",
                                  confidence: float = 0.95,
                                  n_bootstrap: int = 1000,
                                  seed: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: Sample data
        statistic: Statistic to compute ("mean", "median", "std")
        confidence: Confidence level (0-1)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with point estimate and confidence interval
        
    Example:
        >>> result = bootstrap_confidence_interval([1, 2, 3, 4, 5], "mean", 0.95)
        >>> 'ci_lower' in result and 'ci_upper' in result
        True
    """
    if seed is not None:
        random.seed(seed)
    
    stat_funcs = {
        "mean": mean,
        "median": median,
        "std": lambda x: std(x, ddof=1)
    }
    
    if statistic not in stat_funcs:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    stat_func = stat_funcs[statistic]
    n = len(data)
    
    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n-1)] for _ in range(n)]
        bootstrap_stats.append(stat_func(sample))
    
    # Sort bootstrap statistics
    bootstrap_stats.sort()
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap) - 1
    
    return {
        'point_estimate': stat_func(data),
        'ci_lower': bootstrap_stats[lower_idx],
        'ci_upper': bootstrap_stats[upper_idx],
        'confidence': confidence,
        'n_bootstrap': n_bootstrap,
        'se': std(bootstrap_stats, ddof=1)  # Bootstrap standard error
    }


def paired_t_test(sample1: List[Number], sample2: List[Number]) -> Dict[str, float]:
    """
    Perform paired t-test (two-tailed).
    
    Tests whether the mean difference between paired observations is zero.
    
    Args:
        sample1: First sample
        sample2: Second sample (paired with sample1)
        
    Returns:
        Dictionary with t-statistic, p-value approximation, and effect size
        
    Example:
        >>> result = paired_t_test([1, 2, 3, 4], [1.5, 2.5, 3.5, 4.5])
        >>> 't_statistic' in result
        True
    """
    if len(sample1) != len(sample2):
        raise ValueError("Samples must have the same length")
    
    n = len(sample1)
    if n < 2:
        raise ValueError("Need at least 2 paired observations")
    
    # Calculate differences
    differences = [a - b for a, b in zip(sample1, sample2)]
    
    # Mean and std of differences
    mean_diff = mean(differences)
    se = std(differences, ddof=1) / math.sqrt(n)
    
    if se == 0:
        return {
            't_statistic': 0.0 if mean_diff == 0 else float('inf'),
            'p_value': 1.0 if mean_diff == 0 else 0.0,
            'mean_difference': mean_diff,
            'cohens_d': 0.0,
            'df': n - 1
        }
    
    t_stat = mean_diff / se
    df = n - 1
    
    # Approximate p-value using normal distribution for large df
    # For small df, this is an approximation
    z = abs(t_stat)
    p_value = 2 * (1 - _normal_cdf(z))
    
    # Cohen's d effect size
    cohens_d = mean_diff / std(differences, ddof=1)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_difference': mean_diff,
        'se': se,
        'cohens_d': cohens_d,
        'df': df
    }


def _normal_cdf(z: float) -> float:
    """Approximate standard normal CDF using error function approximation."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def independent_t_test(sample1: List[Number], sample2: List[Number],
                       equal_var: bool = True) -> Dict[str, float]:
    """
    Perform independent samples t-test (two-tailed).
    
    Args:
        sample1: First sample
        sample2: Second sample
        equal_var: Assume equal variances (True) or use Welch's t-test (False)
        
    Returns:
        Dictionary with t-statistic, p-value approximation, and effect size
    """
    n1, n2 = len(sample1), len(sample2)
    
    if n1 < 2 or n2 < 2:
        raise ValueError("Each sample needs at least 2 observations")
    
    mean1, mean2 = mean(sample1), mean(sample2)
    var1 = variance(sample1, ddof=1)
    var2 = variance(sample2, ddof=1)
    
    mean_diff = mean1 - mean2
    
    if equal_var:
        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        se = math.sqrt(pooled_var * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        # Welch's t-test
        se = math.sqrt(var1/n1 + var2/n2)
        # Welch-Satterthwaite df approximation
        num = (var1/n1 + var2/n2) ** 2
        denom = (var1/n1)**2 / (n1-1) + (var2/n2)**2 / (n2-1)
        df = num / denom if denom > 0 else n1 + n2 - 2
    
    if se == 0:
        return {
            't_statistic': 0.0 if mean_diff == 0 else float('inf'),
            'p_value': 1.0 if mean_diff == 0 else 0.0,
            'mean_difference': mean_diff,
            'cohens_d': 0.0,
            'df': df
        }
    
    t_stat = mean_diff / se
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    
    # Cohen's d
    pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_difference': mean_diff,
        'se': se,
        'cohens_d': cohens_d,
        'df': df
    }


def wilcoxon_signed_rank_test(sample1: List[Number], 
                               sample2: List[Number]) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric paired test).
    
    Args:
        sample1: First sample
        sample2: Second sample (paired)
        
    Returns:
        Dictionary with test statistic and approximate p-value
    """
    if len(sample1) != len(sample2):
        raise ValueError("Samples must have the same length")
    
    n = len(sample1)
    
    # Calculate differences (excluding zeros)
    differences = []
    for a, b in zip(sample1, sample2):
        diff = a - b
        if diff != 0:
            differences.append(diff)
    
    if not differences:
        return {'w_statistic': 0.0, 'p_value': 1.0, 'n_nonzero': 0}
    
    n_nonzero = len(differences)
    
    # Rank by absolute value
    abs_diffs = [(abs(d), d) for d in differences]
    abs_diffs.sort(key=lambda x: x[0])
    
    # Assign ranks (average for ties)
    ranks = []
    i = 0
    while i < len(abs_diffs):
        j = i
        while j < len(abs_diffs) and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # Average rank (1-indexed)
        for k in range(i, j):
            ranks.append((avg_rank, abs_diffs[k][1]))
        i = j
    
    # Calculate W+ and W-
    w_plus = sum(rank for rank, diff in ranks if diff > 0)
    w_minus = sum(rank for rank, diff in ranks if diff < 0)
    
    w_stat = min(w_plus, w_minus)
    
    # Normal approximation for p-value (n >= 10)
    expected = n_nonzero * (n_nonzero + 1) / 4
    var_w = n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24
    
    if var_w > 0:
        z = (w_stat - expected) / math.sqrt(var_w)
        p_value = 2 * (1 - _normal_cdf(abs(z)))
    else:
        p_value = 1.0
    
    return {
        'w_statistic': w_stat,
        'w_plus': w_plus,
        'w_minus': w_minus,
        'p_value': p_value,
        'n_nonzero': n_nonzero
    }


def mann_whitney_u_test(sample1: List[Number], 
                        sample2: List[Number]) -> Dict[str, float]:
    """
    Perform Mann-Whitney U test (non-parametric independent test).
    
    Args:
        sample1: First sample
        sample2: Second sample
        
    Returns:
        Dictionary with U statistic and approximate p-value
    """
    n1, n2 = len(sample1), len(sample2)
    
    if n1 == 0 or n2 == 0:
        raise ValueError("Both samples must have at least one observation")
    
    # Combine and rank all observations
    combined = [(x, 0) for x in sample1] + [(x, 1) for x in sample2]
    combined.sort(key=lambda x: x[0])
    
    # Assign ranks (average for ties)
    ranks = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks.append((avg_rank, combined[k][1]))
        i = j
    
    # Sum of ranks for each group
    r1 = sum(rank for rank, group in ranks if group == 0)
    
    # Calculate U
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u_stat = min(u1, u2)
    
    # Normal approximation for p-value
    expected = n1 * n2 / 2
    var_u = n1 * n2 * (n1 + n2 + 1) / 12
    
    if var_u > 0:
        z = (u_stat - expected) / math.sqrt(var_u)
        p_value = 2 * (1 - _normal_cdf(abs(z)))
    else:
        p_value = 1.0
    
    return {
        'u_statistic': u_stat,
        'u1': u1,
        'u2': u2,
        'p_value': p_value,
        'effect_size': u_stat / (n1 * n2)  # Common language effect size
    }


def mcnemar_test(table: List[List[int]]) -> Dict[str, float]:
    """
    Perform McNemar's test for paired nominal data.
    
    Used to compare two classifiers on the same test set.
    
    Args:
        table: 2x2 contingency table [[a, b], [c, d]]
               where b = classifier1 correct, classifier2 wrong
               and c = classifier1 wrong, classifier2 correct
        
    Returns:
        Dictionary with chi-square statistic and p-value
    """
    if len(table) != 2 or len(table[0]) != 2 or len(table[1]) != 2:
        raise ValueError("Table must be 2x2")
    
    b = table[0][1]  # Classifier 1 correct, Classifier 2 wrong
    c = table[1][0]  # Classifier 1 wrong, Classifier 2 correct
    
    if b + c == 0:
        return {'chi_square': 0.0, 'p_value': 1.0, 'b': b, 'c': c}
    
    # McNemar's chi-square statistic
    chi_sq = (abs(b - c) - 1) ** 2 / (b + c)  # With continuity correction
    
    # P-value from chi-square distribution with df=1
    # Using Wilson-Hilferty approximation
    p_value = 1 - _chi_square_cdf(chi_sq, df=1)
    
    return {
        'chi_square': chi_sq,
        'p_value': p_value,
        'b': b,
        'c': c
    }


def _chi_square_cdf(x: float, df: int) -> float:
    """Approximate chi-square CDF using gamma function relation."""
    if x <= 0:
        return 0.0
    
    # For df=1, chi-square CDF = 2 * Phi(sqrt(x)) - 1
    if df == 1:
        return 2 * _normal_cdf(math.sqrt(x)) - 1
    
    # For other df, use approximation
    k = df / 2
    z = x / 2
    
    # Incomplete gamma approximation
    # Using series expansion for small z
    if z < k + 1:
        sum_val = 1.0
        term = 1.0
        for n in range(1, 100):
            term *= z / (k + n)
            sum_val += term
            if abs(term) < 1e-10:
                break
        return (z ** k * math.exp(-z) * sum_val) / math.gamma(k + 1)
    else:
        # Use continued fraction for large z
        return 1.0 - 0.5 * math.erfc(math.sqrt(z - k))


def cohens_d(sample1: List[Number], sample2: List[Number]) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        sample1: First sample
        sample2: Second sample
        
    Returns:
        Cohen's d value
        
    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = mean(sample1), mean(sample2)
    var1 = variance(sample1, ddof=1)
    var2 = variance(sample2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def hedges_g(sample1: List[Number], sample2: List[Number]) -> float:
    """
    Calculate Hedges' g effect size (bias-corrected Cohen's d).
    
    Args:
        sample1: First sample
        sample2: Second sample
        
    Returns:
        Hedges' g value
    """
    d = cohens_d(sample1, sample2)
    n = len(sample1) + len(sample2)
    
    # Correction factor
    correction = 1 - 3 / (4 * n - 9)
    
    return d * correction


def glass_delta(control: List[Number], treatment: List[Number]) -> float:
    """
    Calculate Glass's delta effect size.
    
    Uses only the control group's standard deviation.
    
    Args:
        control: Control group sample
        treatment: Treatment group sample
        
    Returns:
        Glass's delta value
    """
    control_std = std(control, ddof=1)
    
    if control_std == 0:
        return 0.0
    
    return (mean(treatment) - mean(control)) / control_std


def correlation_coefficient(x: List[Number], y: List[Number]) -> Dict[str, float]:
    """
    Calculate Pearson correlation coefficient and related statistics.
    
    Args:
        x: First variable
        y: Second variable
        
    Returns:
        Dictionary with correlation, r-squared, and significance
    """
    if len(x) != len(y):
        raise ValueError("Variables must have the same length")
    
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 observations")
    
    mean_x, mean_y = mean(x), mean(y)
    
    # Calculate covariance and standard deviations
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (n - 1)
    std_x = std(x, ddof=1)
    std_y = std(y, ddof=1)
    
    if std_x == 0 or std_y == 0:
        return {'r': 0.0, 'r_squared': 0.0, 'p_value': 1.0}
    
    r = cov / (std_x * std_y)
    r_squared = r ** 2
    
    # T-test for significance
    if abs(r) < 1:
        t_stat = r * math.sqrt((n - 2) / (1 - r ** 2))
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    else:
        p_value = 0.0
    
    return {
        'r': r,
        'r_squared': r_squared,
        'p_value': p_value,
        'n': n
    }


def spearman_correlation(x: List[Number], y: List[Number]) -> Dict[str, float]:
    """
    Calculate Spearman rank correlation coefficient.
    
    Args:
        x: First variable
        y: Second variable
        
    Returns:
        Dictionary with correlation coefficient
    """
    if len(x) != len(y):
        raise ValueError("Variables must have the same length")
    
    n = len(x)
    
    # Convert to ranks
    def to_ranks(data: List[Number]) -> List[float]:
        indexed = [(val, i) for i, val in enumerate(data)]
        indexed.sort(key=lambda x: x[0])
        
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and indexed[j][0] == indexed[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2
            for k in range(i, j):
                ranks[indexed[k][1]] = avg_rank
            i = j
        return ranks
    
    ranks_x = to_ranks(x)
    ranks_y = to_ranks(y)
    
    # Calculate Pearson correlation on ranks
    return correlation_coefficient(ranks_x, ranks_y)

