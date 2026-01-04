"""
Machine Learning Metrics - Pure Python Implementation
=====================================================

Classification, Regression, and Clustering evaluation metrics.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import math

from pyeval.utils.data_ops import (
    binary_confusion_matrix, 
    multiclass_confusion_matrix,
    unique_labels,
    check_consistent_length,
    type_of_target
)
from pyeval.utils.math_ops import mean, euclidean_distance


# =============================================================================
# Classification Metrics
# =============================================================================

def accuracy_score(y_true: List[Any], y_pred: List[Any], 
                   normalize: bool = True) -> float:
    """
    Calculate accuracy classification score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        normalize: If True, return fraction; if False, return count
        
    Returns:
        Accuracy score
        
    Example:
        >>> accuracy_score([1, 0, 1, 1], [1, 0, 0, 1])
        0.75
    """
    check_consistent_length(y_true, y_pred)
    
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    
    if normalize:
        return correct / len(y_true) if y_true else 0.0
    return correct


def precision_score(y_true: List[int], y_pred: List[int], 
                    average: str = 'binary', pos_label: int = 1,
                    zero_division: float = 0.0) -> Union[float, List[float]]:
    """
    Calculate precision score.
    
    Precision = TP / (TP + FP)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: 'binary', 'micro', 'macro', 'weighted', or None
        pos_label: Positive class for binary classification
        zero_division: Value to return when there's no positive prediction
        
    Returns:
        Precision score(s)
        
    Example:
        >>> precision_score([1, 0, 1, 1], [1, 1, 0, 1])
        0.6666666666666666
    """
    check_consistent_length(y_true, y_pred)
    
    if average == 'binary':
        cm = binary_confusion_matrix(y_true, y_pred)
        tp, fp = cm['tp'], cm['fp']
        if tp + fp == 0:
            return zero_division
        return tp / (tp + fp)
    
    labels = sorted(unique_labels(y_true, y_pred))
    matrix = multiclass_confusion_matrix(y_true, y_pred, labels)
    
    precisions = []
    supports = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(len(labels))) - tp
        supports.append(sum(matrix[i]))
        
        if tp + fp == 0:
            precisions.append(zero_division)
        else:
            precisions.append(tp / (tp + fp))
    
    if average is None:
        return precisions
    elif average == 'micro':
        tp_sum = sum(matrix[i][i] for i in range(len(labels)))
        fp_sum = sum(sum(matrix[j][i] for j in range(len(labels))) - matrix[i][i] 
                     for i in range(len(labels)))
        if tp_sum + fp_sum == 0:
            return zero_division
        return tp_sum / (tp_sum + fp_sum)
    elif average == 'macro':
        return mean(precisions)
    elif average == 'weighted':
        total_support = sum(supports)
        if total_support == 0:
            return zero_division
        return sum(p * s for p, s in zip(precisions, supports)) / total_support
    
    raise ValueError(f"Unknown average: {average}")


def recall_score(y_true: List[int], y_pred: List[int], 
                 average: str = 'binary', pos_label: int = 1,
                 zero_division: float = 0.0) -> Union[float, List[float]]:
    """
    Calculate recall (sensitivity, true positive rate) score.
    
    Recall = TP / (TP + FN)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: 'binary', 'micro', 'macro', 'weighted', or None
        pos_label: Positive class for binary classification
        zero_division: Value to return when there's no positive samples
        
    Returns:
        Recall score(s)
        
    Example:
        >>> recall_score([1, 0, 1, 1], [1, 1, 0, 1])
        0.6666666666666666
    """
    check_consistent_length(y_true, y_pred)
    
    if average == 'binary':
        cm = binary_confusion_matrix(y_true, y_pred)
        tp, fn = cm['tp'], cm['fn']
        if tp + fn == 0:
            return zero_division
        return tp / (tp + fn)
    
    labels = sorted(unique_labels(y_true, y_pred))
    matrix = multiclass_confusion_matrix(y_true, y_pred, labels)
    
    recalls = []
    supports = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fn = sum(matrix[i]) - tp
        supports.append(sum(matrix[i]))
        
        if tp + fn == 0:
            recalls.append(zero_division)
        else:
            recalls.append(tp / (tp + fn))
    
    if average is None:
        return recalls
    elif average == 'micro':
        tp_sum = sum(matrix[i][i] for i in range(len(labels)))
        fn_sum = sum(sum(matrix[i]) - matrix[i][i] for i in range(len(labels)))
        if tp_sum + fn_sum == 0:
            return zero_division
        return tp_sum / (tp_sum + fn_sum)
    elif average == 'macro':
        return mean(recalls)
    elif average == 'weighted':
        total_support = sum(supports)
        if total_support == 0:
            return zero_division
        return sum(r * s for r, s in zip(recalls, supports)) / total_support
    
    raise ValueError(f"Unknown average: {average}")


def f1_score(y_true: List[int], y_pred: List[int], 
             average: str = 'binary', pos_label: int = 1,
             zero_division: float = 0.0) -> Union[float, List[float]]:
    """
    Calculate F1 score (harmonic mean of precision and recall).
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: 'binary', 'micro', 'macro', 'weighted', or None
        pos_label: Positive class for binary classification
        zero_division: Value to return on zero division
        
    Returns:
        F1 score(s)
        
    Example:
        >>> f1_score([1, 0, 1, 1], [1, 1, 0, 1])
        0.6666666666666666
    """
    prec = precision_score(y_true, y_pred, average=average, 
                           pos_label=pos_label, zero_division=zero_division)
    rec = recall_score(y_true, y_pred, average=average, 
                       pos_label=pos_label, zero_division=zero_division)
    
    if average is None:
        f1s = []
        for p, r in zip(prec, rec):
            if p + r == 0:
                f1s.append(zero_division)
            else:
                f1s.append(2 * p * r / (p + r))
        return f1s
    
    if prec + rec == 0:
        return zero_division
    return 2 * prec * rec / (prec + rec)


def fbeta_score(y_true: List[int], y_pred: List[int], beta: float,
                average: str = 'binary', pos_label: int = 1,
                zero_division: float = 0.0) -> Union[float, List[float]]:
    """
    Calculate F-beta score.
    
    F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        beta: Weight of recall vs precision
        average: 'binary', 'micro', 'macro', 'weighted', or None
        pos_label: Positive class for binary classification
        zero_division: Value to return on zero division
        
    Returns:
        F-beta score(s)
    """
    prec = precision_score(y_true, y_pred, average=average, 
                           pos_label=pos_label, zero_division=zero_division)
    rec = recall_score(y_true, y_pred, average=average, 
                       pos_label=pos_label, zero_division=zero_division)
    
    beta_sq = beta ** 2
    
    if average is None:
        scores = []
        for p, r in zip(prec, rec):
            denom = beta_sq * p + r
            if denom == 0:
                scores.append(zero_division)
            else:
                scores.append((1 + beta_sq) * p * r / denom)
        return scores
    
    denom = beta_sq * prec + rec
    if denom == 0:
        return zero_division
    return (1 + beta_sq) * prec * rec / denom


def specificity_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate specificity (true negative rate).
    
    Specificity = TN / (TN + FP)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Specificity score
    """
    cm = binary_confusion_matrix(y_true, y_pred)
    tn, fp = cm['tn'], cm['fp']
    
    if tn + fp == 0:
        return 0.0
    return tn / (tn + fp)


def matthews_corrcoef(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate Matthews Correlation Coefficient.
    
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        MCC score (-1 to 1)
        
    Example:
        >>> matthews_corrcoef([1, 0, 1, 1], [1, 0, 0, 1])
        0.5773502691896258
    """
    cm = binary_confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']
    
    numerator = tp * tn - fp * fn
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    return numerator / denominator


def cohen_kappa_score(y_true: List[Any], y_pred: List[Any]) -> float:
    """
    Calculate Cohen's Kappa coefficient.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Kappa score (-1 to 1, where 1 = perfect agreement)
    """
    check_consistent_length(y_true, y_pred)
    
    labels = sorted(unique_labels(y_true, y_pred))
    matrix = multiclass_confusion_matrix(y_true, y_pred, labels)
    n = len(y_true)
    
    if n == 0:
        return 0.0
    
    # Observed agreement (accuracy)
    po = sum(matrix[i][i] for i in range(len(labels))) / n
    
    # Expected agreement by chance
    pe = 0.0
    for i in range(len(labels)):
        row_sum = sum(matrix[i]) / n
        col_sum = sum(matrix[j][i] for j in range(len(labels))) / n
        pe += row_sum * col_sum
    
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def roc_auc_score(y_true: List[int], y_scores: List[float]) -> float:
    """
    Calculate Area Under the ROC Curve (AUC).
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        
    Returns:
        AUC score (0 to 1)
        
    Example:
        >>> roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        0.75
    """
    check_consistent_length(y_true, y_scores)
    
    # Get sorted indices by score (descending)
    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined, return 0.5
    
    # Calculate AUC using the trapezoidal rule
    tpr_prev = 0.0
    fpr_prev = 0.0
    auc = 0.0
    tp = 0
    fp = 0
    
    prev_score = None
    for i in sorted_indices:
        score = y_scores[i]
        
        if prev_score is not None and score != prev_score:
            # Add trapezoidal area
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev = tpr
            fpr_prev = fpr
        
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        
        prev_score = score
    
    # Add final trapezoid
    tpr = tp / n_pos
    fpr = fp / n_neg
    auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
    
    return auc


def roc_curve(y_true: List[int], y_scores: List[float]) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate ROC curve.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    check_consistent_length(y_true, y_scores)
    
    # Sort by score descending
    sorted_pairs = sorted(zip(y_scores, y_true), reverse=True)
    
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    
    fprs = [0.0]
    tprs = [0.0]
    thresholds = [float('inf')]
    
    tp = 0
    fp = 0
    prev_score = None
    
    for score, label in sorted_pairs:
        if prev_score is not None and score != prev_score:
            fprs.append(fp / n_neg if n_neg > 0 else 0)
            tprs.append(tp / n_pos if n_pos > 0 else 0)
            thresholds.append(score)
        
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    
    fprs.append(fp / n_neg if n_neg > 0 else 0)
    tprs.append(tp / n_pos if n_pos > 0 else 0)
    thresholds.append(prev_score)
    
    return fprs, tprs, thresholds


def precision_recall_curve(y_true: List[int], y_scores: List[float]) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    check_consistent_length(y_true, y_scores)
    
    sorted_pairs = sorted(zip(y_scores, y_true), reverse=True)
    
    n_pos = sum(y_true)
    
    precisions = []
    recalls = []
    thresholds = []
    
    tp = 0
    fp = 0
    
    for score, label in sorted_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / n_pos if n_pos > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(score)
    
    return precisions, recalls, thresholds


def average_precision_score(y_true: List[int], y_scores: List[float]) -> float:
    """
    Calculate average precision (AP) from prediction scores.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        
    Returns:
        Average precision score
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    
    # Calculate AP as sum of (R_n - R_{n-1}) * P_n
    ap = 0.0
    prev_recall = 0.0
    
    for precision, recall in zip(precisions, recalls):
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    
    return ap


def confusion_matrix(y_true: List[Any], y_pred: List[Any], 
                     labels: Optional[List[Any]] = None) -> List[List[int]]:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels (uses unique labels if None)
        
    Returns:
        Confusion matrix as 2D list
        
    Example:
        >>> confusion_matrix([0, 1, 2, 0], [0, 2, 2, 0])
        [[2, 0, 0], [0, 0, 1], [0, 0, 1]]
    """
    return multiclass_confusion_matrix(y_true, y_pred, labels)


def classification_report(y_true: List[Any], y_pred: List[Any],
                          labels: Optional[List[Any]] = None,
                          output_dict: bool = False) -> Union[str, Dict]:
    """
    Build a classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels
        output_dict: If True, return as dictionary
        
    Returns:
        Classification report as string or dict
    """
    if labels is None:
        labels = sorted(unique_labels(y_true, y_pred))
    
    report_dict = {}
    
    for label in labels:
        # Binary approach for each label
        y_true_bin = [1 if t == label else 0 for t in y_true]
        y_pred_bin = [1 if p == label else 0 for p in y_pred]
        
        prec = precision_score(y_true_bin, y_pred_bin)
        rec = recall_score(y_true_bin, y_pred_bin)
        f1 = f1_score(y_true_bin, y_pred_bin)
        support = sum(y_true_bin)
        
        report_dict[str(label)] = {
            'precision': prec,
            'recall': rec,
            'f1-score': f1,
            'support': support
        }
    
    # Add averages
    macro_prec = mean([report_dict[str(l)]['precision'] for l in labels])
    macro_rec = mean([report_dict[str(l)]['recall'] for l in labels])
    macro_f1 = mean([report_dict[str(l)]['f1-score'] for l in labels])
    
    report_dict['macro avg'] = {
        'precision': macro_prec,
        'recall': macro_rec,
        'f1-score': macro_f1,
        'support': len(y_true)
    }
    
    total_support = sum(report_dict[str(l)]['support'] for l in labels)
    weighted_prec = sum(report_dict[str(l)]['precision'] * report_dict[str(l)]['support'] 
                        for l in labels) / total_support if total_support > 0 else 0
    weighted_rec = sum(report_dict[str(l)]['recall'] * report_dict[str(l)]['support'] 
                       for l in labels) / total_support if total_support > 0 else 0
    weighted_f1 = sum(report_dict[str(l)]['f1-score'] * report_dict[str(l)]['support'] 
                      for l in labels) / total_support if total_support > 0 else 0
    
    report_dict['weighted avg'] = {
        'precision': weighted_prec,
        'recall': weighted_rec,
        'f1-score': weighted_f1,
        'support': len(y_true)
    }
    
    report_dict['accuracy'] = accuracy_score(y_true, y_pred)
    
    if output_dict:
        return report_dict
    
    # Format as string
    headers = ['', 'precision', 'recall', 'f1-score', 'support']
    rows = []
    
    for label in labels:
        d = report_dict[str(label)]
        rows.append([str(label), f"{d['precision']:.2f}", f"{d['recall']:.2f}", 
                     f"{d['f1-score']:.2f}", str(d['support'])])
    
    rows.append([''] * 5)  # Empty row
    rows.append(['accuracy', '', '', f"{report_dict['accuracy']:.2f}", str(len(y_true))])
    
    for avg_type in ['macro avg', 'weighted avg']:
        d = report_dict[avg_type]
        rows.append([avg_type, f"{d['precision']:.2f}", f"{d['recall']:.2f}", 
                     f"{d['f1-score']:.2f}", str(d['support'])])
    
    # Build formatted output
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(5)]
    
    lines = []
    header_line = ''.join(h.rjust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append('')
    
    for row in rows:
        lines.append(''.join(str(c).rjust(w) for c, w in zip(row, col_widths)))
    
    return '\n'.join(lines)


# =============================================================================
# Regression Metrics
# =============================================================================

def mean_squared_error(y_true: List[float], y_pred: List[float], 
                       squared: bool = True) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        squared: If True, return MSE; if False, return RMSE
        
    Returns:
        MSE or RMSE
        
    Example:
        >>> mean_squared_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
        0.375
    """
    check_consistent_length(y_true, y_pred)
    
    mse = mean([(t - p) ** 2 for t, p in zip(y_true, y_pred)])
    
    if squared:
        return mse
    return math.sqrt(mse)


def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE
    """
    return mean_squared_error(y_true, y_pred, squared=False)


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAE
        
    Example:
        >>> mean_absolute_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
        0.5
    """
    check_consistent_length(y_true, y_pred)
    return mean([abs(t - p) for t, p in zip(y_true, y_pred)])


def mean_absolute_percentage_error(y_true: List[float], y_pred: List[float],
                                    epsilon: float = 1e-10) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE (as a fraction, not percentage)
    """
    check_consistent_length(y_true, y_pred)
    return mean([abs((t - p) / (abs(t) + epsilon)) for t, p in zip(y_true, y_pred)])


def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    R² = 1 - SS_res / SS_tot
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R² score (best is 1.0, can be negative)
        
    Example:
        >>> r2_score([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
        0.9486081370449679
    """
    check_consistent_length(y_true, y_pred)
    
    y_mean = mean(y_true)
    ss_tot = sum((t - y_mean) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    
    if ss_tot == 0:
        return 0.0 if ss_res > 0 else 1.0
    
    return 1 - (ss_res / ss_tot)


def adjusted_r2_score(y_true: List[float], y_pred: List[float], 
                      n_features: int) -> float:
    """
    Calculate Adjusted R-squared.
    
    Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        n_features: Number of features in the model
        
    Returns:
        Adjusted R² score
    """
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    
    if n - n_features - 1 <= 0:
        return r2
    
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


def explained_variance_score(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Explained Variance score.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Explained variance score
    """
    check_consistent_length(y_true, y_pred)
    
    residuals = [t - p for t, p in zip(y_true, y_pred)]
    var_residuals = mean([r ** 2 for r in residuals]) - mean(residuals) ** 2
    var_true = mean([t ** 2 for t in y_true]) - mean(y_true) ** 2
    
    if var_true == 0:
        return 0.0 if var_residuals > 0 else 1.0
    
    return 1 - (var_residuals / var_true)


def max_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate maximum residual error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Maximum error
    """
    check_consistent_length(y_true, y_pred)
    return max(abs(t - p) for t, p in zip(y_true, y_pred))


def median_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Median Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Median absolute error
    """
    from pyeval.utils.math_ops import median as calc_median
    
    check_consistent_length(y_true, y_pred)
    errors = [abs(t - p) for t, p in zip(y_true, y_pred)]
    return calc_median(errors)


# =============================================================================
# Clustering Metrics
# =============================================================================

def silhouette_score(X: List[List[float]], labels: List[int]) -> float:
    """
    Calculate Silhouette Score for clustering.
    
    The silhouette value measures how similar a point is to its own cluster
    compared to other clusters.
    
    Args:
        X: Data points as list of feature vectors
        labels: Cluster labels for each point
        
    Returns:
        Mean silhouette score (-1 to 1)
        
    Example:
        >>> X = [[1, 2], [1.5, 1.8], [5, 8], [5.5, 8.2]]
        >>> labels = [0, 0, 1, 1]
        >>> silhouette_score(X, labels)  # Returns value between -1 and 1
    """
    check_consistent_length(X, labels)
    
    n_samples = len(X)
    if n_samples < 2:
        return 0.0
    
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0
    
    # Compute pairwise distances
    distances = [[0.0] * n_samples for _ in range(n_samples)]
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = euclidean_distance(X[i], X[j])
            distances[i][j] = d
            distances[j][i] = d
    
    silhouette_values = []
    
    for i in range(n_samples):
        cluster_i = labels[i]
        
        # Calculate a(i): mean distance to points in same cluster
        same_cluster = [j for j in range(n_samples) if labels[j] == cluster_i and j != i]
        if not same_cluster:
            a_i = 0.0
        else:
            a_i = mean([distances[i][j] for j in same_cluster])
        
        # Calculate b(i): min mean distance to points in other clusters
        b_i = float('inf')
        for cluster in unique_labels:
            if cluster == cluster_i:
                continue
            other_cluster = [j for j in range(n_samples) if labels[j] == cluster]
            if other_cluster:
                mean_dist = mean([distances[i][j] for j in other_cluster])
                b_i = min(b_i, mean_dist)
        
        if b_i == float('inf'):
            b_i = 0.0
        
        # Calculate silhouette
        max_ab = max(a_i, b_i)
        if max_ab == 0:
            s_i = 0.0
        else:
            s_i = (b_i - a_i) / max_ab
        
        silhouette_values.append(s_i)
    
    return mean(silhouette_values)


def davies_bouldin_score(X: List[List[float]], labels: List[int]) -> float:
    """
    Calculate Davies-Bouldin Score for clustering.
    
    Lower values indicate better clustering.
    
    Args:
        X: Data points as list of feature vectors
        labels: Cluster labels for each point
        
    Returns:
        Davies-Bouldin score (lower is better)
    """
    check_consistent_length(X, labels)
    
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
    
    # Calculate cluster centroids and scatter
    centroids = {}
    scatters = {}
    
    for cluster in unique_labels:
        cluster_points = [X[i] for i in range(len(X)) if labels[i] == cluster]
        n_features = len(X[0])
        
        # Centroid
        centroid = [mean([p[f] for p in cluster_points]) for f in range(n_features)]
        centroids[cluster] = centroid
        
        # Scatter (average distance to centroid)
        scatter = mean([euclidean_distance(p, centroid) for p in cluster_points])
        scatters[cluster] = scatter
    
    # Calculate Davies-Bouldin index
    db_values = []
    
    for i, cluster_i in enumerate(unique_labels):
        max_ratio = 0.0
        
        for j, cluster_j in enumerate(unique_labels):
            if i == j:
                continue
            
            # R_ij = (S_i + S_j) / M_ij
            centroid_dist = euclidean_distance(centroids[cluster_i], centroids[cluster_j])
            
            if centroid_dist == 0:
                continue
            
            ratio = (scatters[cluster_i] + scatters[cluster_j]) / centroid_dist
            max_ratio = max(max_ratio, ratio)
        
        db_values.append(max_ratio)
    
    return mean(db_values)


def calinski_harabasz_score(X: List[List[float]], labels: List[int]) -> float:
    """
    Calculate Calinski-Harabasz Score (Variance Ratio Criterion).
    
    Higher values indicate better clustering.
    
    Args:
        X: Data points as list of feature vectors
        labels: Cluster labels for each point
        
    Returns:
        Calinski-Harabasz score (higher is better)
    """
    check_consistent_length(X, labels)
    
    n_samples = len(X)
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)
    n_features = len(X[0])
    
    if n_clusters < 2 or n_clusters >= n_samples:
        return 0.0
    
    # Overall centroid
    overall_centroid = [mean([X[i][f] for i in range(n_samples)]) for f in range(n_features)]
    
    # Between-cluster dispersion (B_k)
    bgss = 0.0  # Between-group sum of squares
    
    # Within-cluster dispersion (W_k)
    wgss = 0.0  # Within-group sum of squares
    
    for cluster in unique_labels:
        cluster_indices = [i for i in range(n_samples) if labels[i] == cluster]
        n_cluster = len(cluster_indices)
        
        # Cluster centroid
        cluster_centroid = [mean([X[i][f] for i in cluster_indices]) for f in range(n_features)]
        
        # Between-group contribution
        bgss += n_cluster * euclidean_distance(cluster_centroid, overall_centroid) ** 2
        
        # Within-group contribution
        for i in cluster_indices:
            wgss += euclidean_distance(X[i], cluster_centroid) ** 2
    
    if wgss == 0:
        return 0.0
    
    return (bgss / (n_clusters - 1)) / (wgss / (n_samples - n_clusters))


# =============================================================================
# Metric Classes
# =============================================================================

@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    
    @classmethod
    def compute(cls, y_true: List[Any], y_pred: List[Any], 
                y_scores: Optional[List[float]] = None) -> 'ClassificationMetrics':
        """
        Compute all classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_scores: Predicted scores/probabilities (for ROC-AUC)
            
        Returns:
            ClassificationMetrics object
        """
        target_type = type_of_target(y_true)
        average = 'binary' if target_type == 'binary' else 'weighted'
        
        cm = confusion_matrix(y_true, y_pred)
        
        roc = None
        if y_scores is not None and target_type == 'binary':
            roc = roc_auc_score(y_true, y_scores)
        
        return cls(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average=average),
            recall=recall_score(y_true, y_pred, average=average),
            f1=f1_score(y_true, y_pred, average=average),
            roc_auc=roc,
            confusion_matrix=cm
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'roc_auc': self.roc_auc,
            'confusion_matrix': self.confusion_matrix
        }


@dataclass
class RegressionMetrics:
    """Container for regression metrics."""
    
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    explained_variance: float = 0.0
    
    @classmethod
    def compute(cls, y_true: List[float], y_pred: List[float]) -> 'RegressionMetrics':
        """
        Compute all regression metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            RegressionMetrics object
        """
        return cls(
            mse=mean_squared_error(y_true, y_pred),
            rmse=root_mean_squared_error(y_true, y_pred),
            mae=mean_absolute_error(y_true, y_pred),
            r2=r2_score(y_true, y_pred),
            explained_variance=explained_variance_score(y_true, y_pred)
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'explained_variance': self.explained_variance
        }


@dataclass
class ClusteringMetrics:
    """Container for clustering metrics."""
    
    silhouette: float = 0.0
    davies_bouldin: float = 0.0
    calinski_harabasz: float = 0.0
    
    @classmethod
    def compute(cls, X: List[List[float]], labels: List[int]) -> 'ClusteringMetrics':
        """
        Compute all clustering metrics.
        
        Args:
            X: Data points
            labels: Cluster labels
            
        Returns:
            ClusteringMetrics object
        """
        return cls(
            silhouette=silhouette_score(X, labels),
            davies_bouldin=davies_bouldin_score(X, labels),
            calinski_harabasz=calinski_harabasz_score(X, labels)
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'silhouette': self.silhouette,
            'davies_bouldin': self.davies_bouldin,
            'calinski_harabasz': self.calinski_harabasz
        }
