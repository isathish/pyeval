"""
Utility functions for PyEval
"""

from pyeval.utils.math_ops import (
    mean,
    variance,
    std,
    median,
    percentile,
    normalize,
    softmax,
    sigmoid,
    cosine_similarity,
    euclidean_distance,
    jaccard_similarity,
    levenshtein_distance,
)

from pyeval.utils.text_ops import (
    tokenize,
    ngrams,
    word_ngrams,
    char_ngrams,
    stem,
    lemmatize,
    remove_punctuation,
    remove_stopwords,
    normalize_text,
)

from pyeval.utils.data_ops import (
    binary_confusion_matrix,
    multiclass_confusion_matrix,
    unique_labels,
    check_consistent_length,
    type_of_target,
)

__all__ = [
    # Math operations
    "mean",
    "variance", 
    "std",
    "median",
    "percentile",
    "normalize",
    "softmax",
    "sigmoid",
    "cosine_similarity",
    "euclidean_distance",
    "jaccard_similarity",
    "levenshtein_distance",
    # Text operations
    "tokenize",
    "ngrams",
    "word_ngrams",
    "char_ngrams",
    "stem",
    "lemmatize",
    "remove_punctuation",
    "remove_stopwords",
    "normalize_text",
    # Data operations
    "binary_confusion_matrix",
    "multiclass_confusion_matrix",
    "unique_labels",
    "check_consistent_length",
    "type_of_target",
]
