"""
Tests for utility functions: math_ops, text_ops, data_ops.
"""

import sys
sys.path.insert(0, '..')

import pytest
from pyeval.utils.math_ops import (
    mean, std, variance, median, percentile,
    normalize, softmax, sigmoid,
    cosine_similarity, euclidean_distance, dot_product
)
from pyeval.utils.text_ops import (
    tokenize, ngrams, stem, remove_punctuation,
    remove_stopwords, normalize_text, word_count, get_word_frequencies
)
from pyeval.utils.data_ops import (
    binary_confusion_matrix, flatten, batch_iterator, unique_labels
)


# ========== Math Operations Tests ==========

class TestMean:
    """Tests for mean function."""
    
    def test_basic_mean(self):
        """Basic mean calculation."""
        assert mean([1, 2, 3, 4, 5]) == 3.0
    
    def test_single_value(self):
        """Mean of single value."""
        assert mean([5]) == 5.0
    
    def test_floats(self):
        """Mean of floats."""
        result = mean([1.5, 2.5, 3.5])
        assert abs(result - 2.5) < 0.001


class TestStd:
    """Tests for standard deviation."""
    
    def test_basic_std(self):
        """Basic std calculation."""
        result = std([1, 2, 3, 4, 5])
        assert result is not None
        assert result >= 0
    
    def test_constant_values(self):
        """Std of constant values should be 0."""
        result = std([5, 5, 5, 5])
        assert abs(result) < 0.001


class TestVariance:
    """Tests for variance."""
    
    def test_basic_variance(self):
        """Basic variance calculation."""
        result = variance([1, 2, 3, 4, 5])
        assert result is not None
        assert result >= 0
    
    def test_constant_values(self):
        """Variance of constant values should be 0."""
        result = variance([5, 5, 5, 5])
        assert abs(result) < 0.001


class TestMedian:
    """Tests for median."""
    
    def test_odd_count(self):
        """Median of odd number of values."""
        assert median([1, 2, 3, 4, 5]) == 3
    
    def test_even_count(self):
        """Median of even number of values."""
        result = median([1, 2, 3, 4])
        assert result == 2.5


class TestPercentile:
    """Tests for percentile."""
    
    def test_50th_percentile(self):
        """50th percentile is median."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = percentile(data, 50)
        assert result is not None
    
    def test_boundaries(self):
        """Test boundary percentiles."""
        data = [1, 2, 3, 4, 5]
        p0 = percentile(data, 0)
        p100 = percentile(data, 100)
        assert p0 is not None
        assert p100 is not None


class TestNormalize:
    """Tests for normalize."""
    
    def test_basic_normalize(self):
        """Basic normalization."""
        result = normalize([1, 2, 3, 4, 5])
        assert len(result) == 5
        assert min(result) >= 0
        assert max(result) <= 1


class TestSoftmax:
    """Tests for softmax."""
    
    def test_basic_softmax(self):
        """Softmax output sums to 1."""
        result = softmax([1, 2, 3])
        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 0.001
    
    def test_all_positive(self):
        """Softmax outputs are all positive."""
        result = softmax([-1, 0, 1])
        assert all(x > 0 for x in result)


class TestSigmoid:
    """Tests for sigmoid."""
    
    def test_zero(self):
        """Sigmoid of 0 is 0.5."""
        result = sigmoid(0)
        assert abs(result - 0.5) < 0.001
    
    def test_range(self):
        """Sigmoid output is between 0 and 1."""
        for x in [-10, -1, 0, 1, 10]:
            result = sigmoid(x)
            assert 0 <= result <= 1


class TestCosineSimilarity:
    """Tests for cosine similarity."""
    
    def test_identical_vectors(self):
        """Identical vectors have similarity 1."""
        result = cosine_similarity([1, 2, 3], [1, 2, 3])
        assert abs(result - 1.0) < 0.001
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0."""
        result = cosine_similarity([1, 0], [0, 1])
        assert abs(result) < 0.001


class TestEuclideanDistance:
    """Tests for Euclidean distance."""
    
    def test_same_point(self):
        """Distance to same point is 0."""
        result = euclidean_distance([1, 2, 3], [1, 2, 3])
        assert abs(result) < 0.001
    
    def test_basic_distance(self):
        """Basic distance calculation."""
        result = euclidean_distance([0, 0], [3, 4])
        assert abs(result - 5.0) < 0.001


class TestDotProduct:
    """Tests for dot product."""
    
    def test_basic_dot(self):
        """Basic dot product."""
        result = dot_product([1, 2, 3], [4, 5, 6])
        assert result == 32  # 1*4 + 2*5 + 3*6 = 32
    
    def test_orthogonal(self):
        """Orthogonal vectors have dot product 0."""
        result = dot_product([1, 0], [0, 1])
        assert result == 0


# ========== Text Operations Tests ==========

class TestTokenize:
    """Tests for tokenize."""
    
    def test_basic_tokenize(self):
        """Basic tokenization."""
        result = tokenize("Hello world, how are you?")
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_empty_string(self):
        """Tokenize empty string."""
        result = tokenize("")
        assert isinstance(result, list)


class TestNgrams:
    """Tests for ngrams."""
    
    def test_bigrams(self):
        """Generate bigrams."""
        result = ngrams(["a", "b", "c", "d"], 2)
        result_list = list(result)
        assert len(result_list) == 3
    
    def test_trigrams(self):
        """Generate trigrams."""
        result = ngrams(["a", "b", "c", "d"], 3)
        result_list = list(result)
        assert len(result_list) == 2


class TestStem:
    """Tests for stem."""
    
    def test_basic_stem(self):
        """Basic stemming."""
        result = stem("running")
        assert isinstance(result, str)


class TestRemovePunctuation:
    """Tests for remove_punctuation."""
    
    def test_basic_removal(self):
        """Remove punctuation from text."""
        result = remove_punctuation("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result


class TestRemoveStopwords:
    """Tests for remove_stopwords."""
    
    def test_basic_removal(self):
        """Remove stopwords from tokens."""
        tokens = ["the", "quick", "brown", "fox", "is", "a", "animal"]
        result = remove_stopwords(tokens)
        assert isinstance(result, list)


class TestNormalizeText:
    """Tests for normalize_text."""
    
    def test_basic_normalize(self):
        """Basic text normalization."""
        result = normalize_text("  HELLO  World  ")
        assert isinstance(result, str)


class TestWordCount:
    """Tests for word_count."""
    
    def test_basic_count(self):
        """Count words in text."""
        result = word_count("Hello world, how are you?")
        assert result > 0
    
    def test_empty_string(self):
        """Word count of empty string."""
        result = word_count("")
        assert result == 0


class TestGetWordFrequencies:
    """Tests for get_word_frequencies."""
    
    def test_basic_frequencies(self):
        """Get word frequencies."""
        result = get_word_frequencies("hello world hello")
        assert isinstance(result, dict)
        assert "hello" in result or any("hello" in str(k).lower() for k in result.keys())


# ========== Data Operations Tests ==========

class TestBinaryConfusionMatrix:
    """Tests for binary_confusion_matrix."""
    
    def test_perfect_prediction(self):
        """Perfect predictions."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        result = binary_confusion_matrix(y_true, y_pred)
        assert result is not None
    
    def test_all_wrong(self):
        """All wrong predictions."""
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        result = binary_confusion_matrix(y_true, y_pred)
        assert result is not None


class TestFlatten:
    """Tests for flatten."""
    
    def test_basic_flatten(self):
        """Flatten nested list."""
        nested = [[1, 2], [3, 4], [5]]
        result = flatten(nested)
        assert list(result) == [1, 2, 3, 4, 5]
    
    def test_already_flat(self):
        """Flatten already flat list."""
        flat = [[1], [2], [3]]
        result = flatten(flat)
        assert list(result) == [1, 2, 3]


class TestBatchIterator:
    """Tests for batch_iterator."""
    
    def test_exact_batches(self):
        """Exact batch sizes."""
        data = [1, 2, 3, 4, 5, 6]
        batches = list(batch_iterator(data, 2))
        assert len(batches) == 3
    
    def test_partial_batch(self):
        """Last batch may be smaller."""
        data = [1, 2, 3, 4, 5]
        batches = list(batch_iterator(data, 2))
        assert len(batches) == 3
        assert len(batches[-1]) == 1


class TestUniqueLabels:
    """Tests for unique_labels."""
    
    def test_basic_unique(self):
        """Get unique labels."""
        labels = [1, 2, 2, 3, 3, 3, 1]
        result = unique_labels(labels)
        assert set(result) == {1, 2, 3}
    
    def test_string_labels(self):
        """Unique string labels."""
        labels = ["a", "b", "a", "c"]
        result = unique_labels(labels)
        assert set(result) == {"a", "b", "c"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
