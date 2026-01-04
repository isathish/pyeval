"""
Fairness Metrics - Pure Python Implementation
=============================================

Evaluation metrics for AI fairness including demographic parity,
equalized odds, disparate impact, and statistical parity.
"""

from typing import List, Dict, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass
from collections import Counter
import math

from pyeval.utils.data_ops import check_consistent_length, unique_labels
from pyeval.utils.math_ops import mean


# =============================================================================
# Core Fairness Metrics
# =============================================================================

def demographic_parity(y_pred: List[int], sensitive_features: List[Any],
                       favorable_outcome: int = 1) -> Dict[str, Any]:
    """
    Calculate demographic parity (statistical parity).
    
    Demographic parity requires that the probability of positive outcome
    is equal across all groups defined by sensitive features.
    
    DP Difference = max(P(Y=1|G=g)) - min(P(Y=1|G=g))
    
    Args:
        y_pred: Predicted labels
        sensitive_features: Sensitive attribute values (e.g., gender, race)
        favorable_outcome: Value representing favorable outcome
        
    Returns:
        Dictionary with demographic parity metrics
        
    Example:
        >>> y_pred = [1, 0, 1, 1, 0, 0, 1, 0]
        >>> groups = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        >>> result = demographic_parity(y_pred, groups)
        >>> result['dp_difference']  # Closer to 0 is more fair
    """
    check_consistent_length(y_pred, sensitive_features)
    
    # Group by sensitive feature
    groups = unique_labels(sensitive_features)
    
    group_rates = {}
    group_sizes = {}
    
    for group in groups:
        group_preds = [y for y, g in zip(y_pred, sensitive_features) if g == group]
        group_sizes[group] = len(group_preds)
        
        if group_preds:
            favorable_count = sum(1 for y in group_preds if y == favorable_outcome)
            group_rates[group] = favorable_count / len(group_preds)
        else:
            group_rates[group] = 0.0
    
    # Calculate disparities
    rates = list(group_rates.values())
    
    if not rates:
        return {
            'dp_difference': 0.0,
            'dp_ratio': 1.0,
            'is_fair': True,
            'group_rates': {},
            'group_sizes': {}
        }
    
    max_rate = max(rates)
    min_rate = min(rates)
    
    dp_difference = max_rate - min_rate
    dp_ratio = min_rate / max_rate if max_rate > 0 else 1.0
    
    # Typically, DP difference < 0.1 or ratio > 0.8 is considered fair
    is_fair = dp_difference < 0.1 or dp_ratio > 0.8
    
    return {
        'dp_difference': dp_difference,
        'dp_ratio': dp_ratio,
        'is_fair': is_fair,
        'group_rates': group_rates,
        'group_sizes': group_sizes,
        'max_rate_group': max(group_rates, key=group_rates.get),
        'min_rate_group': min(group_rates, key=group_rates.get)
    }


def equalized_odds(y_true: List[int], y_pred: List[int],
                   sensitive_features: List[Any],
                   favorable_outcome: int = 1) -> Dict[str, Any]:
    """
    Calculate equalized odds.
    
    Equalized odds requires equal true positive rates and false positive rates
    across all groups.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Sensitive attribute values
        favorable_outcome: Value representing favorable outcome
        
    Returns:
        Dictionary with equalized odds metrics
        
    Example:
        >>> y_true = [1, 0, 1, 1, 0, 0, 1, 0]
        >>> y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
        >>> groups = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        >>> result = equalized_odds(y_true, y_pred, groups)
        >>> result['tpr_difference']  # Closer to 0 is more fair
    """
    check_consistent_length(y_true, y_pred, sensitive_features)
    
    groups = unique_labels(sensitive_features)
    
    group_tpr = {}  # True Positive Rate
    group_fpr = {}  # False Positive Rate
    group_tnr = {}  # True Negative Rate
    group_fnr = {}  # False Negative Rate
    
    for group in groups:
        # Get predictions and labels for this group
        indices = [i for i, g in enumerate(sensitive_features) if g == group]
        
        group_true = [y_true[i] for i in indices]
        group_pred = [y_pred[i] for i in indices]
        
        # Calculate confusion matrix components
        tp = sum(1 for t, p in zip(group_true, group_pred) 
                 if t == favorable_outcome and p == favorable_outcome)
        fn = sum(1 for t, p in zip(group_true, group_pred) 
                 if t == favorable_outcome and p != favorable_outcome)
        fp = sum(1 for t, p in zip(group_true, group_pred) 
                 if t != favorable_outcome and p == favorable_outcome)
        tn = sum(1 for t, p in zip(group_true, group_pred) 
                 if t != favorable_outcome and p != favorable_outcome)
        
        # Calculate rates
        group_tpr[group] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        group_fpr[group] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        group_tnr[group] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        group_fnr[group] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Calculate disparities
    tpr_values = list(group_tpr.values())
    fpr_values = list(group_fpr.values())
    
    tpr_difference = max(tpr_values) - min(tpr_values) if tpr_values else 0.0
    fpr_difference = max(fpr_values) - min(fpr_values) if fpr_values else 0.0
    
    # Equalized odds is satisfied when both TPR and FPR are equal across groups
    eo_difference = max(tpr_difference, fpr_difference)
    
    # Average equalized odds difference
    average_odds_difference = (tpr_difference + fpr_difference) / 2
    
    is_fair = eo_difference < 0.1
    
    return {
        'tpr_difference': tpr_difference,
        'fpr_difference': fpr_difference,
        'eo_difference': eo_difference,
        'average_odds_difference': average_odds_difference,
        'is_fair': is_fair,
        'group_tpr': group_tpr,
        'group_fpr': group_fpr,
        'group_tnr': group_tnr,
        'group_fnr': group_fnr
    }


def equal_opportunity(y_true: List[int], y_pred: List[int],
                      sensitive_features: List[Any],
                      favorable_outcome: int = 1) -> Dict[str, Any]:
    """
    Calculate equal opportunity.
    
    Equal opportunity requires equal true positive rates across groups.
    This is a relaxation of equalized odds.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Sensitive attribute values
        favorable_outcome: Value representing favorable outcome
        
    Returns:
        Dictionary with equal opportunity metrics
    """
    eo_result = equalized_odds(y_true, y_pred, sensitive_features, favorable_outcome)
    
    return {
        'tpr_difference': eo_result['tpr_difference'],
        'is_fair': eo_result['tpr_difference'] < 0.1,
        'group_tpr': eo_result['group_tpr']
    }


def disparate_impact(y_pred: List[int], sensitive_features: List[Any],
                     favorable_outcome: int = 1,
                     reference_group: Optional[Any] = None) -> Dict[str, Any]:
    """
    Calculate disparate impact ratio.
    
    DI Ratio = P(Y=1|G=unprivileged) / P(Y=1|G=privileged)
    
    A ratio < 0.8 is often considered evidence of disparate impact (80% rule).
    
    Args:
        y_pred: Predicted labels
        sensitive_features: Sensitive attribute values
        favorable_outcome: Value representing favorable outcome
        reference_group: Reference (privileged) group for comparison
        
    Returns:
        Dictionary with disparate impact metrics
        
    Example:
        >>> y_pred = [1, 0, 1, 1, 0, 0, 0, 0]
        >>> groups = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        >>> result = disparate_impact(y_pred, groups, reference_group='A')
        >>> result['di_ratio']  # Should be >= 0.8 to avoid disparate impact
    """
    dp_result = demographic_parity(y_pred, sensitive_features, favorable_outcome)
    
    group_rates = dp_result['group_rates']
    
    if not group_rates:
        return {
            'di_ratio': 1.0,
            'has_disparate_impact': False,
            'group_ratios': {}
        }
    
    # Determine reference group
    if reference_group is None:
        # Use group with highest rate as reference
        reference_group = max(group_rates, key=group_rates.get)
    
    reference_rate = group_rates.get(reference_group, 0.0)
    
    if reference_rate == 0:
        return {
            'di_ratio': 1.0 if all(r == 0 for r in group_rates.values()) else 0.0,
            'has_disparate_impact': any(r > 0 for r in group_rates.values()),
            'group_ratios': {g: 0.0 for g in group_rates if g != reference_group}
        }
    
    # Calculate DI ratio for each group vs reference
    group_ratios = {}
    for group, rate in group_rates.items():
        if group != reference_group:
            group_ratios[group] = rate / reference_rate
    
    # Overall DI ratio (minimum ratio)
    di_ratio = min(group_ratios.values()) if group_ratios else 1.0
    
    # 80% rule
    has_disparate_impact = di_ratio < 0.8
    
    return {
        'di_ratio': di_ratio,
        'has_disparate_impact': has_disparate_impact,
        'reference_group': reference_group,
        'reference_rate': reference_rate,
        'group_ratios': group_ratios,
        'passes_80_percent_rule': not has_disparate_impact
    }


def statistical_parity_difference(y_pred: List[int], sensitive_features: List[Any],
                                  favorable_outcome: int = 1) -> float:
    """
    Calculate statistical parity difference.
    
    SPD = P(Y=1|G=unprivileged) - P(Y=1|G=privileged)
    
    Args:
        y_pred: Predicted labels
        sensitive_features: Sensitive attribute values
        favorable_outcome: Value representing favorable outcome
        
    Returns:
        Statistical parity difference (closer to 0 is more fair)
    """
    dp_result = demographic_parity(y_pred, sensitive_features, favorable_outcome)
    return dp_result['dp_difference']


# =============================================================================
# Calibration Metrics
# =============================================================================

def calibration_by_group(y_true: List[int], y_scores: List[float],
                         sensitive_features: List[Any],
                         n_bins: int = 10) -> Dict[str, Any]:
    """
    Calculate calibration metrics by group.
    
    Calibration measures whether predicted probabilities match actual outcomes.
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        sensitive_features: Sensitive attribute values
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with calibration metrics per group
    """
    check_consistent_length(y_true, y_scores, sensitive_features)
    
    groups = unique_labels(sensitive_features)
    
    group_calibration = {}
    
    for group in groups:
        indices = [i for i, g in enumerate(sensitive_features) if g == group]
        
        group_true = [y_true[i] for i in indices]
        group_scores = [y_scores[i] for i in indices]
        
        if not group_true:
            group_calibration[group] = {'ece': 0.0, 'mce': 0.0}
            continue
        
        # Bin predictions
        bins = [[] for _ in range(n_bins)]
        
        for true_val, score in zip(group_true, group_scores):
            bin_idx = min(int(score * n_bins), n_bins - 1)
            bins[bin_idx].append((true_val, score))
        
        # Calculate ECE (Expected Calibration Error)
        ece = 0.0
        mce = 0.0
        total = len(group_true)
        
        for bin_data in bins:
            if bin_data:
                bin_size = len(bin_data)
                avg_confidence = mean([s for _, s in bin_data])
                avg_accuracy = mean([t for t, _ in bin_data])
                
                bin_error = abs(avg_accuracy - avg_confidence)
                ece += (bin_size / total) * bin_error
                mce = max(mce, bin_error)
        
        group_calibration[group] = {
            'ece': ece,
            'mce': mce
        }
    
    # Check calibration fairness
    eces = [v['ece'] for v in group_calibration.values()]
    calibration_difference = max(eces) - min(eces) if eces else 0.0
    
    return {
        'group_calibration': group_calibration,
        'calibration_difference': calibration_difference,
        'is_calibration_fair': calibration_difference < 0.05
    }


# =============================================================================
# Individual Fairness
# =============================================================================

def individual_fairness(X: List[List[float]], y_pred: List[int],
                        distance_threshold: float = 0.1) -> Dict[str, float]:
    """
    Calculate individual fairness metric.
    
    Individual fairness: similar individuals should receive similar predictions.
    
    Args:
        X: Feature vectors for each sample
        y_pred: Predicted labels
        distance_threshold: Threshold for considering samples similar
        
    Returns:
        Dictionary with individual fairness metrics
    """
    from pyeval.utils.math_ops import euclidean_distance
    
    check_consistent_length(X, y_pred)
    
    n = len(X)
    if n < 2:
        return {'individual_fairness': 1.0, 'consistency': 1.0}
    
    violations = 0
    similar_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate feature distance
            dist = euclidean_distance(X[i], X[j])
            
            # Normalize by feature dimension
            max_dist = math.sqrt(len(X[i]))
            normalized_dist = dist / max_dist if max_dist > 0 else dist
            
            if normalized_dist <= distance_threshold:
                similar_pairs += 1
                # Similar individuals should have same prediction
                if y_pred[i] != y_pred[j]:
                    violations += 1
    
    if similar_pairs == 0:
        consistency = 1.0
    else:
        consistency = 1 - (violations / similar_pairs)
    
    return {
        'individual_fairness': consistency,
        'consistency': consistency,
        'violations': violations,
        'similar_pairs': similar_pairs
    }


# =============================================================================
# Counterfactual Fairness
# =============================================================================

def counterfactual_fairness_check(predictions_original: List[int],
                                   predictions_counterfactual: List[int]) -> Dict[str, Any]:
    """
    Check counterfactual fairness.
    
    Counterfactual fairness: prediction should be same if sensitive attribute
    were different (in a counterfactual world).
    
    Args:
        predictions_original: Predictions with original sensitive attributes
        predictions_counterfactual: Predictions with flipped sensitive attributes
        
    Returns:
        Dictionary with counterfactual fairness metrics
    """
    check_consistent_length(predictions_original, predictions_counterfactual)
    
    n = len(predictions_original)
    if n == 0:
        return {'counterfactual_fairness': 1.0, 'flip_rate': 0.0}
    
    # Count prediction flips
    flips = sum(1 for o, c in zip(predictions_original, predictions_counterfactual) 
                if o != c)
    
    flip_rate = flips / n
    
    return {
        'counterfactual_fairness': 1 - flip_rate,
        'flip_rate': flip_rate,
        'total_flips': flips,
        'total_samples': n,
        'is_counterfactually_fair': flip_rate < 0.05
    }


# =============================================================================
# Comprehensive Fairness Analysis
# =============================================================================

def comprehensive_fairness_analysis(y_true: List[int], y_pred: List[int],
                                    sensitive_features: List[Any],
                                    y_scores: Optional[List[float]] = None,
                                    favorable_outcome: int = 1) -> Dict[str, Any]:
    """
    Perform comprehensive fairness analysis.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Sensitive attribute values
        y_scores: Predicted probabilities (optional)
        favorable_outcome: Value representing favorable outcome
        
    Returns:
        Dictionary with all fairness metrics
    """
    results = {
        'demographic_parity': demographic_parity(y_pred, sensitive_features, favorable_outcome),
        'equalized_odds': equalized_odds(y_true, y_pred, sensitive_features, favorable_outcome),
        'equal_opportunity': equal_opportunity(y_true, y_pred, sensitive_features, favorable_outcome),
        'disparate_impact': disparate_impact(y_pred, sensitive_features, favorable_outcome),
    }
    
    if y_scores:
        results['calibration'] = calibration_by_group(y_true, y_scores, sensitive_features)
    
    # Overall fairness assessment
    dp_fair = results['demographic_parity']['is_fair']
    eo_fair = results['equalized_odds']['is_fair']
    di_fair = not results['disparate_impact']['has_disparate_impact']
    
    results['overall_assessment'] = {
        'passes_demographic_parity': dp_fair,
        'passes_equalized_odds': eo_fair,
        'passes_disparate_impact': di_fair,
        'overall_fair': dp_fair and eo_fair and di_fair,
        'fairness_score': (int(dp_fair) + int(eo_fair) + int(di_fair)) / 3
    }
    
    return results


# =============================================================================
# Fairness Metrics Class
# =============================================================================

@dataclass
class FairnessMetrics:
    """Container for fairness evaluation metrics."""
    
    dp_difference: float = 0.0
    dp_ratio: float = 1.0
    tpr_difference: float = 0.0
    fpr_difference: float = 0.0
    di_ratio: float = 1.0
    overall_fair: bool = True
    
    @classmethod
    def compute(cls, y_true: List[int], y_pred: List[int],
                sensitive_features: List[Any],
                favorable_outcome: int = 1) -> 'FairnessMetrics':
        """
        Compute all fairness metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Sensitive attribute values
            favorable_outcome: Value representing favorable outcome
            
        Returns:
            FairnessMetrics object
        """
        dp = demographic_parity(y_pred, sensitive_features, favorable_outcome)
        eo = equalized_odds(y_true, y_pred, sensitive_features, favorable_outcome)
        di = disparate_impact(y_pred, sensitive_features, favorable_outcome)
        
        overall = dp['is_fair'] and eo['is_fair'] and not di['has_disparate_impact']
        
        return cls(
            dp_difference=dp['dp_difference'],
            dp_ratio=dp['dp_ratio'],
            tpr_difference=eo['tpr_difference'],
            fpr_difference=eo['fpr_difference'],
            di_ratio=di['di_ratio'],
            overall_fair=overall
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'dp_difference': self.dp_difference,
            'dp_ratio': self.dp_ratio,
            'tpr_difference': self.tpr_difference,
            'fpr_difference': self.fpr_difference,
            'di_ratio': self.di_ratio,
            'overall_fair': self.overall_fair
        }
    
    def fairness_score(self) -> float:
        """
        Calculate overall fairness score (0 to 1, higher is more fair).
        """
        # Each metric contributes to the score
        dp_score = 1 - self.dp_difference
        eo_score = 1 - max(self.tpr_difference, self.fpr_difference)
        di_score = min(1.0, self.di_ratio / 0.8)  # Normalized to 80% rule
        
        return (dp_score + eo_score + di_score) / 3


# =============================================================================
# True Positive Rate Difference
# =============================================================================

def true_positive_rate_difference(y_true: List[int], y_pred: List[int],
                                   sensitive_features: List[Any],
                                   favorable_outcome: int = 1) -> float:
    """
    Calculate the difference in True Positive Rates between groups.
    
    TPR Difference = max(TPR) - min(TPR) across groups
    
    A value closer to 0 indicates more fairness.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        sensitive_features: Group membership for each sample
        favorable_outcome: Value representing favorable outcome
        
    Returns:
        TPR difference (0 = perfectly fair)
        
    Example:
        >>> y_true = [1, 0, 1, 1, 0, 0, 1, 0]
        >>> y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
        >>> groups = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        >>> true_positive_rate_difference(y_true, y_pred, groups)
    """
    eo_result = equalized_odds(y_true, y_pred, sensitive_features, favorable_outcome)
    return eo_result['tpr_difference']


# =============================================================================
# False Positive Rate Difference
# =============================================================================

def false_positive_rate_difference(y_true: List[int], y_pred: List[int],
                                    sensitive_features: List[Any],
                                    favorable_outcome: int = 1) -> float:
    """
    Calculate the difference in False Positive Rates between groups.
    
    FPR Difference = max(FPR) - min(FPR) across groups
    
    A value closer to 0 indicates more fairness.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        sensitive_features: Group membership for each sample
        favorable_outcome: Value representing favorable outcome
        
    Returns:
        FPR difference (0 = perfectly fair)
        
    Example:
        >>> y_true = [1, 0, 1, 1, 0, 0, 1, 0]
        >>> y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
        >>> groups = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        >>> false_positive_rate_difference(y_true, y_pred, groups)
    """
    eo_result = equalized_odds(y_true, y_pred, sensitive_features, favorable_outcome)
    return eo_result['fpr_difference']
