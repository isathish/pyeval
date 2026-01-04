"""
Visualization Operations - Pure Python ASCII-based Visualizations
==================================================================

Text-based visualizations for evaluation metrics without external dependencies.
"""

from typing import List, Dict, Optional, Any, Union

Number = Union[int, float]


# =============================================================================
# Confusion Matrix Visualization
# =============================================================================

def confusion_matrix_display(matrix: List[List[int]], 
                             labels: Optional[List[str]] = None,
                             width: int = 80) -> str:
    """
    Display a confusion matrix in ASCII format.
    
    Args:
        matrix: 2D confusion matrix
        labels: Class labels
        width: Maximum width of display
        
    Returns:
        ASCII string representation
        
    Example:
        >>> matrix = [[50, 10], [5, 35]]
        >>> print(confusion_matrix_display(matrix, ['Cat', 'Dog']))
    """
    n_classes = len(matrix)
    
    if labels is None:
        labels = [str(i) for i in range(n_classes)]
    
    # Calculate cell width
    max_val = max(max(row) for row in matrix)
    cell_width = max(len(str(max_val)), max(len(label) for label in labels)) + 2
    
    lines = []
    
    # Title
    lines.append("=" * width)
    lines.append("CONFUSION MATRIX".center(width))
    lines.append("=" * width)
    lines.append("")
    
    # Header row
    header = " " * (cell_width + 2) + "Predicted".center(cell_width * n_classes)
    lines.append(header)
    
    # Column labels
    col_labels = " " * (cell_width + 2)
    for label in labels:
        col_labels += label.center(cell_width)
    lines.append(col_labels)
    lines.append(" " * (cell_width + 2) + "-" * (cell_width * n_classes))
    
    # Rows
    for i, row in enumerate(matrix):
        if i == n_classes // 2:
            prefix = "Actual "
        else:
            prefix = "       "
        
        line = prefix[:7] + labels[i].rjust(cell_width - 2) + " |"
        for val in row:
            line += str(val).center(cell_width)
        lines.append(line)
    
    lines.append("")
    
    # Statistics
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(n_classes))
    accuracy = correct / total if total > 0 else 0.0
    
    lines.append(f"Total samples: {total}")
    lines.append(f"Correctly classified: {correct}")
    lines.append(f"Accuracy: {accuracy:.2%}")
    lines.append("=" * width)
    
    return "\n".join(lines)


def classification_report_display(y_true: List[Any], y_pred: List[Any],
                                  labels: Optional[List[Any]] = None,
                                  width: int = 80) -> str:
    """
    Generate a classification report in ASCII format.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of labels to include
        width: Maximum width
        
    Returns:
        ASCII classification report
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    # Calculate metrics per class
    metrics = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for t in y_true if t == label)
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    lines = []
    lines.append("=" * width)
    lines.append("CLASSIFICATION REPORT".center(width))
    lines.append("=" * width)
    lines.append("")
    
    # Header
    header = f"{'Class':15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Per-class metrics
    total_support = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    
    for label in labels:
        m = metrics[label]
        line = f"{str(label):15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}"
        lines.append(line)
        
        total_support += m['support']
        weighted_precision += m['precision'] * m['support']
        weighted_recall += m['recall'] * m['support']
        weighted_f1 += m['f1'] * m['support']
    
    lines.append("-" * len(header))
    
    # Averages
    if total_support > 0:
        macro_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
        macro_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
        macro_f1 = sum(m['f1'] for m in metrics.values()) / len(metrics)
        
        weighted_precision /= total_support
        weighted_recall /= total_support
        weighted_f1 /= total_support
        
        lines.append(f"{'Macro avg':15} {macro_precision:>10.4f} {macro_recall:>10.4f} {macro_f1:>10.4f} {total_support:>10}")
        lines.append(f"{'Weighted avg':15} {weighted_precision:>10.4f} {weighted_recall:>10.4f} {weighted_f1:>10.4f} {total_support:>10}")
    
    lines.append("")
    lines.append("=" * width)
    
    return "\n".join(lines)


# =============================================================================
# Bar Charts
# =============================================================================

def horizontal_bar_chart(values: Dict[str, float], 
                         title: str = "Bar Chart",
                         width: int = 50,
                         char: str = "█") -> str:
    """
    Create a horizontal bar chart in ASCII.
    
    Args:
        values: Dictionary mapping labels to values
        title: Chart title
        width: Width of bar area
        char: Character to use for bars
        
    Returns:
        ASCII bar chart string
    """
    if not values:
        return "No data to display"
    
    max_val = max(values.values())
    max_label_len = max(len(str(k)) for k in values.keys())
    
    lines = []
    lines.append("=" * (max_label_len + width + 15))
    lines.append(title.center(max_label_len + width + 15))
    lines.append("=" * (max_label_len + width + 15))
    lines.append("")
    
    for label, value in values.items():
        bar_len = int(value / max_val * width) if max_val > 0 else 0
        bar = char * bar_len
        line = f"{str(label):>{max_label_len}} | {bar} {value:.4f}"
        lines.append(line)
    
    lines.append("")
    lines.append("=" * (max_label_len + width + 15))
    
    return "\n".join(lines)


def histogram_display(values: List[Number], 
                      bins: int = 10,
                      title: str = "Histogram",
                      width: int = 50,
                      char: str = "█") -> str:
    """
    Create a histogram in ASCII.
    
    Args:
        values: List of numeric values
        bins: Number of bins
        title: Chart title
        width: Width of bar area
        char: Character to use for bars
        
    Returns:
        ASCII histogram string
    """
    if not values:
        return "No data to display"
    
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        bin_edges = [min_val - 0.5, max_val + 0.5]
        bins = 1
    else:
        bin_width = (max_val - min_val) / bins
        bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
    
    # Count values in each bin
    counts = [0] * bins
    for v in values:
        for i in range(bins):
            if i == bins - 1:
                if bin_edges[i] <= v <= bin_edges[i + 1]:
                    counts[i] += 1
                    break
            else:
                if bin_edges[i] <= v < bin_edges[i + 1]:
                    counts[i] += 1
                    break
    
    max_count = max(counts) if counts else 1
    
    lines = []
    lines.append("=" * (width + 30))
    lines.append(title.center(width + 30))
    lines.append("=" * (width + 30))
    lines.append("")
    
    for i, count in enumerate(counts):
        bar_len = int(count / max_count * width) if max_count > 0 else 0
        bar = char * bar_len
        bin_range = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
        line = f"{bin_range:>20} | {bar} {count}"
        lines.append(line)
    
    lines.append("")
    lines.append(f"Total: {len(values)} | Min: {min_val:.4f} | Max: {max_val:.4f}")
    lines.append("=" * (width + 30))
    
    return "\n".join(lines)


# =============================================================================
# ROC and PR Curve Approximations
# =============================================================================

def roc_curve_display(y_true: List[int], y_scores: List[float],
                      thresholds: int = 10,
                      width: int = 50,
                      height: int = 20) -> str:
    """
    Display an ASCII approximation of an ROC curve.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Predicted scores/probabilities
        thresholds: Number of threshold points to evaluate
        width: Width of the plot
        height: Height of the plot
        
    Returns:
        ASCII ROC curve string
    """
    # Calculate TPR and FPR at different thresholds
    threshold_values = [i / thresholds for i in range(thresholds + 1)]
    
    points = []
    for thresh in threshold_values:
        tp = sum(1 for t, s in zip(y_true, y_scores) if t == 1 and s >= thresh)
        fp = sum(1 for t, s in zip(y_true, y_scores) if t == 0 and s >= thresh)
        fn = sum(1 for t, s in zip(y_true, y_scores) if t == 1 and s < thresh)
        tn = sum(1 for t, s in zip(y_true, y_scores) if t == 0 and s < thresh)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        points.append((fpr, tpr))
    
    # Calculate AUC using trapezoidal rule
    points.sort()
    auc = 0
    for i in range(1, len(points)):
        auc += (points[i][0] - points[i-1][0]) * (points[i][1] + points[i-1][1]) / 2
    
    # Create ASCII plot
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Draw axes
    for i in range(height):
        grid[i][0] = '|'
    for j in range(width):
        grid[height-1][j] = '-'
    grid[height-1][0] = '+'
    
    # Draw diagonal (random classifier)
    for i in range(min(width, height)):
        x = int(i * width / height)
        y = height - 1 - i
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = '.'
    
    # Plot ROC points
    for fpr, tpr in points:
        x = int(fpr * (width - 1))
        y = height - 1 - int(tpr * (height - 1))
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = '*'
    
    lines = []
    lines.append("=" * (width + 10))
    lines.append("ROC CURVE".center(width + 10))
    lines.append("=" * (width + 10))
    lines.append("")
    
    # Y-axis label
    lines.append("TPR")
    lines.append("1.0 " + "".join(grid[0]))
    for i, row in enumerate(grid[1:], 1):
        if i == height // 2:
            lines.append("    " + "".join(row))
        else:
            lines.append("    " + "".join(row))
    lines.append("0.0 " + "    " + "FPR".center(width - 10) + "1.0")
    
    lines.append("")
    lines.append(f"AUC = {auc:.4f}")
    lines.append("* = ROC points, . = random classifier baseline")
    lines.append("=" * (width + 10))
    
    return "\n".join(lines)


def pr_curve_display(y_true: List[int], y_scores: List[float],
                     thresholds: int = 10,
                     width: int = 50,
                     height: int = 20) -> str:
    """
    Display an ASCII approximation of a Precision-Recall curve.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Predicted scores/probabilities  
        thresholds: Number of threshold points to evaluate
        width: Width of the plot
        height: Height of the plot
        
    Returns:
        ASCII PR curve string
    """
    threshold_values = [i / thresholds for i in range(thresholds + 1)]
    
    points = []
    for thresh in threshold_values:
        tp = sum(1 for t, s in zip(y_true, y_scores) if t == 1 and s >= thresh)
        fp = sum(1 for t, s in zip(y_true, y_scores) if t == 0 and s >= thresh)
        fn = sum(1 for t, s in zip(y_true, y_scores) if t == 1 and s < thresh)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        points.append((recall, precision))
    
    # Create ASCII plot
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Draw axes
    for i in range(height):
        grid[i][0] = '|'
    for j in range(width):
        grid[height-1][j] = '-'
    grid[height-1][0] = '+'
    
    # Plot PR points
    for recall, precision in points:
        x = int(recall * (width - 1))
        y = height - 1 - int(precision * (height - 1))
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = '*'
    
    lines = []
    lines.append("=" * (width + 10))
    lines.append("PRECISION-RECALL CURVE".center(width + 10))
    lines.append("=" * (width + 10))
    lines.append("")
    
    lines.append("Precision")
    lines.append("1.0 " + "".join(grid[0]))
    for row in grid[1:]:
        lines.append("    " + "".join(row))
    lines.append("0.0 " + "    " + "Recall".center(width - 10) + "1.0")
    
    lines.append("")
    lines.append("* = PR curve points")
    lines.append("=" * (width + 10))
    
    return "\n".join(lines)


# =============================================================================
# Score Distribution Visualization
# =============================================================================

def score_distribution_display(scores: Dict[str, List[float]],
                               title: str = "Score Distribution",
                               width: int = 60) -> str:
    """
    Display score distributions for multiple groups.
    
    Args:
        scores: Dictionary mapping group names to score lists
        title: Chart title
        width: Display width
        
    Returns:
        ASCII score distribution display
    """
    lines = []
    lines.append("=" * width)
    lines.append(title.center(width))
    lines.append("=" * width)
    lines.append("")
    
    for group, values in scores.items():
        if not values:
            continue
            
        mean_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
        
        # Mini histogram
        bins = 10
        if max_val > min_val:
            bin_counts = [0] * bins
            for v in values:
                bin_idx = min(int((v - min_val) / (max_val - min_val) * bins), bins - 1)
                bin_counts[bin_idx] += 1
            
            max_count = max(bin_counts)
            hist = ""
            for count in bin_counts:
                bar_len = int(count / max_count * 5) if max_count > 0 else 0
                hist += "▁▂▃▄▅▆▇█"[min(bar_len, 7)]
        else:
            hist = "█" * 10
        
        lines.append(f"{group}:")
        lines.append(f"  {hist}")
        lines.append(f"  n={len(values):5} | mean={mean_val:7.4f} | std={std_val:7.4f}")
        lines.append(f"  min={min_val:7.4f} | max={max_val:7.4f}")
        lines.append("")
    
    lines.append("=" * width)
    
    return "\n".join(lines)


# =============================================================================
# Comparison Tables
# =============================================================================

def metric_comparison_table(metrics: Dict[str, Dict[str, float]],
                            title: str = "Metric Comparison",
                            width: int = 80) -> str:
    """
    Display a comparison table of metrics across models/experiments.
    
    Args:
        metrics: Dict mapping model names to dicts of metric values
        title: Table title
        width: Display width
        
    Returns:
        ASCII comparison table
    """
    if not metrics:
        return "No metrics to display"
    
    # Get all metric names
    all_metrics = set()
    for m in metrics.values():
        all_metrics.update(m.keys())
    all_metrics = sorted(all_metrics)
    
    model_names = list(metrics.keys())
    
    lines = []
    lines.append("=" * width)
    lines.append(title.center(width))
    lines.append("=" * width)
    lines.append("")
    
    # Header
    col_width = max(10, (width - 15) // len(model_names))
    header = f"{'Metric':15}"
    for name in model_names:
        header += f"{name[:col_width]:>{col_width}}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Rows
    for metric_name in all_metrics:
        row = f"{metric_name:15}"
        values = []
        for model_name in model_names:
            val = metrics[model_name].get(metric_name, float('nan'))
            values.append(val)
            row += f"{val:>{col_width}.4f}"
        
        # (Can't color in ASCII, but could add marker)
        lines.append(row)
    
    lines.append("")
    lines.append("=" * width)
    
    return "\n".join(lines)


def progress_bar(current: int, total: int, 
                 width: int = 40,
                 prefix: str = "Progress",
                 suffix: str = "") -> str:
    """
    Create a progress bar string.
    
    Args:
        current: Current progress value
        total: Total value
        width: Bar width
        prefix: Text before bar
        suffix: Text after bar
        
    Returns:
        Progress bar string
    """
    if total <= 0:
        return f"{prefix}: [{'?' * width}] ?% {suffix}"
    
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    
    return f"{prefix}: [{bar}] {percent:6.1%} ({current}/{total}) {suffix}"


def sparkline(values: List[Number], width: int = 20) -> str:
    """
    Create a sparkline string representation of values.
    
    Args:
        values: List of numeric values
        width: Maximum width
        
    Returns:
        Sparkline string
    """
    if not values:
        return ""
    
    # Resample if needed
    if len(values) > width:
        step = len(values) / width
        resampled = []
        for i in range(width):
            start = int(i * step)
            end = int((i + 1) * step)
            chunk = values[start:end]
            resampled.append(sum(chunk) / len(chunk) if chunk else 0)
        values = resampled
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return "▄" * len(values)
    
    blocks = "▁▂▃▄▅▆▇█"
    result = ""
    
    for v in values:
        normalized = (v - min_val) / (max_val - min_val)
        idx = min(int(normalized * 7), 7)
        result += blocks[idx]
    
    return result


def error_analysis_display(errors: List[Dict[str, Any]],
                           max_show: int = 10,
                           width: int = 80) -> str:
    """
    Display error analysis in ASCII format.
    
    Args:
        errors: List of error dictionaries with 'input', 'expected', 'actual', etc.
        max_show: Maximum number of errors to display
        width: Display width
        
    Returns:
        ASCII error analysis display
    """
    lines = []
    lines.append("=" * width)
    lines.append("ERROR ANALYSIS".center(width))
    lines.append("=" * width)
    lines.append("")
    lines.append(f"Total errors: {len(errors)}")
    lines.append(f"Showing first {min(len(errors), max_show)}:")
    lines.append("")
    
    for i, error in enumerate(errors[:max_show]):
        lines.append(f"Error #{i + 1}:")
        lines.append("-" * 40)
        
        for key, value in error.items():
            val_str = str(value)
            if len(val_str) > width - 15:
                val_str = val_str[:width - 18] + "..."
            lines.append(f"  {key:12}: {val_str}")
        
        lines.append("")
    
    if len(errors) > max_show:
        lines.append(f"... and {len(errors) - max_show} more errors")
    
    lines.append("=" * width)
    
    return "\n".join(lines)
