"""
Data Operations - Pure Python Implementation
============================================

Data manipulation and validation functions for evaluation metrics.
"""

from typing import List, Dict, Set, Tuple, Any, Optional
from collections import Counter


def binary_confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """
    Calculate binary confusion matrix components.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        
    Returns:
        Dictionary with tp, tn, fp, fn counts
        
    Example:
        >>> binary_confusion_matrix([1, 0, 1, 1], [1, 0, 0, 1])
        {'tp': 2, 'tn': 1, 'fp': 0, 'fn': 1}
    """
    check_consistent_length(y_true, y_pred)
    
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def multiclass_confusion_matrix(y_true: List[Any], y_pred: List[Any], 
                                 labels: Optional[List[Any]] = None) -> List[List[int]]:
    """
    Calculate multiclass confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of labels to include (uses unique labels if None)
        
    Returns:
        2D confusion matrix (rows=true, cols=predicted)
        
    Example:
        >>> multiclass_confusion_matrix([0, 1, 2, 0], [0, 2, 2, 0])
        [[2, 0, 0], [0, 0, 1], [0, 0, 1]]
    """
    check_consistent_length(y_true, y_pred)
    
    if labels is None:
        labels = sorted(unique_labels(y_true, y_pred))
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)
    
    matrix = [[0] * n_labels for _ in range(n_labels)]
    
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            matrix[label_to_idx[true]][label_to_idx[pred]] += 1
    
    return matrix


def unique_labels(*arrays) -> Set[Any]:
    """
    Extract unique labels from arrays.
    
    Args:
        *arrays: Variable number of label arrays
        
    Returns:
        Set of unique labels
        
    Example:
        >>> unique_labels([1, 2, 3], [2, 3, 4])
        {1, 2, 3, 4}
    """
    labels = set()
    for arr in arrays:
        labels.update(arr)
    return labels


def check_consistent_length(*arrays) -> None:
    """
    Check that all arrays have consistent length.
    
    Args:
        *arrays: Variable number of arrays
        
    Raises:
        ValueError: If arrays have inconsistent lengths
    """
    lengths = [len(arr) for arr in arrays if arr is not None]
    
    if len(set(lengths)) > 1:
        raise ValueError(f"Found arrays with inconsistent lengths: {lengths}")


def type_of_target(y: List[Any]) -> str:
    """
    Determine the type of target array.
    
    Args:
        y: Target array
        
    Returns:
        Target type: 'binary', 'multiclass', 'multilabel', 'continuous', 'unknown'
        
    Example:
        >>> type_of_target([0, 1, 1, 0])
        'binary'
        >>> type_of_target([0, 1, 2, 3])
        'multiclass'
    """
    if not y:
        return 'unknown'
    
    unique = set(y)
    
    # Check for binary
    if unique == {0, 1} or unique == {0} or unique == {1}:
        return 'binary'
    
    # Check if all integers
    if all(isinstance(val, int) for val in y):
        if len(unique) <= 2:
            return 'binary'
        return 'multiclass'
    
    # Check for continuous
    if all(isinstance(val, (int, float)) for val in y):
        return 'continuous'
    
    # Check for multilabel (list of lists)
    if all(isinstance(val, (list, tuple, set)) for val in y):
        return 'multilabel'
    
    return 'unknown'


def stratified_split(data: List[Any], labels: List[Any], 
                     test_size: float = 0.2, random_seed: int = 42) -> Tuple[List, List, List, List]:
    """
    Perform stratified train-test split.
    
    Args:
        data: Data points
        labels: Corresponding labels
        test_size: Fraction of data for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, test_data, train_labels, test_labels)
    """
    check_consistent_length(data, labels)
    
    # Group indices by label
    label_indices: Dict[Any, List[int]] = {}
    for i, label in enumerate(labels):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(i)
    
    # Simple deterministic shuffle based on seed
    def shuffle_indices(indices: List[int], seed: int) -> List[int]:
        # Simple LCG-based shuffle
        n = len(indices)
        shuffled = indices.copy()
        for i in range(n - 1, 0, -1):
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            j = seed % (i + 1)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        return shuffled
    
    train_indices = []
    test_indices = []
    
    seed = random_seed
    for label, indices in label_indices.items():
        shuffled = shuffle_indices(indices, seed)
        n_test = max(1, int(len(indices) * test_size))
        test_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])
        seed += 1
    
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    train_labels = [labels[i] for i in train_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_data, test_data, train_labels, test_labels


def k_fold_indices(n_samples: int, k: int = 5, shuffle: bool = True, 
                   random_seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """
    Generate k-fold cross-validation indices.
    
    Args:
        n_samples: Number of samples
        k: Number of folds
        shuffle: Whether to shuffle indices
        random_seed: Random seed for shuffling
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    indices = list(range(n_samples))
    
    if shuffle:
        # Simple LCG-based shuffle
        seed = random_seed
        for i in range(n_samples - 1, 0, -1):
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            j = seed % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
    
    fold_sizes = [n_samples // k] * k
    for i in range(n_samples % k):
        fold_sizes[i] += 1
    
    folds = []
    current = 0
    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size])
        current += fold_size
    
    result = []
    for i in range(k):
        test_indices = folds[i]
        train_indices = []
        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])
        result.append((train_indices, test_indices))
    
    return result


def one_hot_encode(labels: List[int], n_classes: Optional[int] = None) -> List[List[int]]:
    """
    One-hot encode integer labels.
    
    Args:
        labels: List of integer labels
        n_classes: Number of classes (inferred if None)
        
    Returns:
        One-hot encoded matrix
        
    Example:
        >>> one_hot_encode([0, 1, 2])
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """
    if n_classes is None:
        n_classes = max(labels) + 1
    
    encoded = []
    for label in labels:
        row = [0] * n_classes
        if 0 <= label < n_classes:
            row[label] = 1
        encoded.append(row)
    
    return encoded


def label_binarize(labels: List[Any], classes: List[Any]) -> List[List[int]]:
    """
    Binarize labels in a one-vs-all fashion.
    
    Args:
        labels: List of labels
        classes: List of classes
        
    Returns:
        Binary indicator matrix
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    
    binarized = []
    for label in labels:
        row = [0] * n_classes
        if label in class_to_idx:
            row[class_to_idx[label]] = 1
        binarized.append(row)
    
    return binarized


def class_distribution(labels: List[Any]) -> Dict[Any, float]:
    """
    Calculate class distribution (proportions).
    
    Args:
        labels: List of labels
        
    Returns:
        Dictionary mapping labels to their proportions
        
    Example:
        >>> class_distribution([0, 0, 1, 1, 1])
        {0: 0.4, 1: 0.6}
    """
    counts = Counter(labels)
    total = len(labels)
    
    return {label: count / total for label, count in counts.items()}


def sample_weights_from_class_weights(labels: List[Any], 
                                       class_weights: Dict[Any, float]) -> List[float]:
    """
    Generate sample weights from class weights.
    
    Args:
        labels: List of labels
        class_weights: Dictionary mapping labels to weights
        
    Returns:
        List of sample weights
    """
    return [class_weights.get(label, 1.0) for label in labels]


def balanced_class_weights(labels: List[Any]) -> Dict[Any, float]:
    """
    Calculate balanced class weights for imbalanced datasets.
    
    Args:
        labels: List of labels
        
    Returns:
        Dictionary mapping labels to balanced weights
        
    Example:
        >>> balanced_class_weights([0, 0, 0, 0, 1])
        {0: 0.625, 1: 2.5}
    """
    counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(counts)
    
    weights = {}
    for label, count in counts.items():
        weights[label] = n_samples / (n_classes * count)
    
    return weights


def bootstrap_sample(data: List[Any], n_samples: Optional[int] = None,
                     random_seed: int = 42) -> Tuple[List[Any], List[int]]:
    """
    Generate a bootstrap sample.
    
    Args:
        data: Original data
        n_samples: Number of samples (defaults to len(data))
        random_seed: Random seed
        
    Returns:
        Tuple of (bootstrap_sample, indices)
    """
    n = len(data)
    if n_samples is None:
        n_samples = n
    
    # Simple LCG for random indices
    seed = random_seed
    indices = []
    for _ in range(n_samples):
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        indices.append(seed % n)
    
    sample = [data[i] for i in indices]
    return sample, indices


def group_by(data: List[Any], keys: List[Any]) -> Dict[Any, List[Any]]:
    """
    Group data by keys.
    
    Args:
        data: Data items
        keys: Grouping keys
        
    Returns:
        Dictionary mapping keys to grouped data
    """
    check_consistent_length(data, keys)
    
    groups: Dict[Any, List[Any]] = {}
    for item, key in zip(data, keys):
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    
    return groups


def flatten(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested list.
    
    Args:
        nested_list: List of lists
        
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def batch_iterator(data: List[Any], batch_size: int):
    """
    Iterate over data in batches.
    
    Args:
        data: Data to batch
        batch_size: Size of each batch
        
    Yields:
        Batches of data
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
