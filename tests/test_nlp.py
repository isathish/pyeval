"""
Unit Tests for PyEval NLP Metrics
"""

import sys
sys.path.insert(0, '..')

from pyeval.nlp import (
    bleu_score, sentence_bleu, rouge_score, rouge_n, rouge_l,
    meteor_score, ter_score, distinct_n,
    NLPMetrics
)


class TestBLEUScore:
    """Tests for BLEU score."""
    
    def test_bleu_perfect_match(self):
        reference = "the cat sat on the mat"
        hypothesis = "the cat sat on the mat"
        result = bleu_score([reference], hypothesis)
        assert result['bleu'] > 0.9
    
    def test_bleu_partial_match(self):
        reference = "the cat sat on the mat"
        hypothesis = "a cat is on the mat"
        result = bleu_score([reference], hypothesis)
        assert 0 <= result['bleu'] < 1.0  # Can be 0 if no 4-gram matches
    
    def test_bleu_no_match(self):
        reference = "the cat sat on the mat"
        hypothesis = "completely different sentence here"
        result = bleu_score([reference], hypothesis)
        assert result['bleu'] < 0.5
    
    def test_sentence_bleu(self):
        reference = "the quick brown fox"
        hypothesis = "the fast brown fox"
        score = sentence_bleu(reference, hypothesis)
        assert 0 <= score <= 1.0


class TestROUGEScore:
    """Tests for ROUGE score."""
    
    def test_rouge_perfect_match(self):
        reference = "the cat sat on the mat"
        hypothesis = "the cat sat on the mat"
        result = rouge_score(reference, hypothesis)
        assert result['rouge1']['f1'] == 1.0
        assert result['rougeL']['f1'] == 1.0
    
    def test_rouge_partial_match(self):
        reference = "the cat sat on the mat"
        hypothesis = "a cat is on the mat"
        result = rouge_score(reference, hypothesis)
        assert 0 < result['rouge1']['f1'] < 1.0
    
    def test_rouge_n_unigram(self):
        reference = "the cat sat"
        hypothesis = "the cat"
        result = rouge_n(reference, hypothesis, n=1)
        # 2 out of 3 words match, recall = 2/3
        assert result['recall'] == 2/3
    
    def test_rouge_l(self):
        reference = "the cat sat on the mat"
        hypothesis = "the cat on mat"
        result = rouge_l(reference, hypothesis)
        assert 0 < result['f1'] < 1.0


class TestMETEOR:
    """Tests for METEOR score."""
    
    def test_meteor_perfect_match(self):
        reference = "the cat sat on the mat"
        hypothesis = "the cat sat on the mat"
        result = meteor_score(reference, hypothesis)
        assert result['meteor'] > 0.9
    
    def test_meteor_partial_match(self):
        reference = "the cat sat on the mat"
        hypothesis = "a cat is on the mat"
        result = meteor_score(reference, hypothesis)
        assert 0 < result['meteor'] < 1.0


class TestTER:
    """Tests for TER score."""
    
    def test_ter_perfect(self):
        reference = "the cat sat on the mat"
        hypothesis = "the cat sat on the mat"
        result = ter_score(reference, hypothesis)
        assert result['ter'] == 0.0
    
    def test_ter_basic(self):
        reference = "the cat sat on the mat"
        hypothesis = "a cat is on a mat"
        result = ter_score(reference, hypothesis)
        assert result['ter'] > 0


class TestDistinctN:
    """Tests for Distinct-N metric."""
    
    def test_distinct_1(self):
        texts = ["hello world", "hello there"]
        score = distinct_n(texts, n=1)
        # unique unigrams / total unigrams
        assert 0 < score <= 1.0
    
    def test_distinct_2(self):
        texts = ["the cat sat", "the cat ate"]
        score = distinct_n(texts, n=2)
        assert 0 < score <= 1.0


class TestNLPMetrics:
    """Tests for NLPMetrics class."""
    
    def test_compute(self):
        reference = "the quick brown fox"
        hypothesis = "a fast brown fox"
        metrics = NLPMetrics.compute(reference, hypothesis)
        assert isinstance(metrics.bleu, float)
        assert isinstance(metrics.rouge1_f1, float)
        assert isinstance(metrics.meteor, float)


def run_tests():
    """Run all tests."""
    import traceback
    
    test_classes = [
        TestBLEUScore,
        TestROUGEScore,
        TestMETEOR,
        TestTER,
        TestDistinctN,
        TestNLPMetrics,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {test_class.__name__}.{method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {test_class.__name__}.{method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {test_class.__name__}.{method_name}: {e}")
                    traceback.print_exc()
                    failed += 1
    
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*40}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
