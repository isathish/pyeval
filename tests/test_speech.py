"""
Unit Tests for PyEval Speech Metrics
"""

import sys

sys.path.insert(0, "..")

from pyeval.speech import (
    word_error_rate,
    character_error_rate,
    match_error_rate,
    word_information_lost,
    sentence_error_rate,
    SpeechMetrics,
)


class TestWER:
    """Tests for Word Error Rate."""

    def test_wer_perfect(self):
        reference = "the cat sat on the mat"
        hypothesis = "the cat sat on the mat"
        result = word_error_rate(reference, hypothesis)
        assert result["wer"] == 0.0

    def test_wer_insertion(self):
        reference = "hello world"
        hypothesis = "hello there world"
        result = word_error_rate(reference, hypothesis)
        # 1 insertion / 2 words = 0.5
        assert result["wer"] == 0.5
        assert result["insertions"] == 1

    def test_wer_deletion(self):
        reference = "hello there world"
        hypothesis = "hello world"
        result = word_error_rate(reference, hypothesis)
        # 1 deletion / 3 words = 1/3
        assert abs(result["wer"] - 1 / 3) < 0.01
        assert result["deletions"] == 1

    def test_wer_substitution(self):
        reference = "the cat sat"
        hypothesis = "the dog sat"
        result = word_error_rate(reference, hypothesis)
        # 1 substitution / 3 words = 1/3
        assert abs(result["wer"] - 1 / 3) < 0.01
        assert result["substitutions"] == 1


class TestCER:
    """Tests for Character Error Rate."""

    def test_cer_perfect(self):
        reference = "hello"
        hypothesis = "hello"
        result = character_error_rate(reference, hypothesis)
        assert result["cer"] == 0.0

    def test_cer_basic(self):
        reference = "hello"
        hypothesis = "hallo"
        result = character_error_rate(reference, hypothesis)
        # 1 substitution / 5 characters = 0.2
        assert result["cer"] == 0.2


class TestOtherSpeechMetrics:
    """Tests for other speech metrics."""

    def test_mer(self):
        reference = "the cat sat"
        hypothesis = "the dog sat"
        result = match_error_rate(reference, hypothesis)
        assert 0 <= result["mer"] <= 1.0

    def test_wil(self):
        reference = "hello world"
        hypothesis = "hello there"
        result = word_information_lost(reference, hypothesis)
        assert 0 <= result["wil"] <= 1.0

    def test_ser_perfect(self):
        references = ["hello world", "foo bar"]
        hypotheses = ["hello world", "foo bar"]
        result = sentence_error_rate(references, hypotheses)
        assert result["ser"] == 0.0

    def test_ser_partial(self):
        references = ["hello world", "foo bar"]
        hypotheses = ["hello world", "foo baz"]
        result = sentence_error_rate(references, hypotheses)
        assert result["ser"] == 0.5


class TestSpeechMetrics:
    """Tests for SpeechMetrics class."""

    def test_compute(self):
        reference = "the quick brown fox"
        hypothesis = "the fast brown fox"
        metrics = SpeechMetrics.compute(reference, hypothesis)
        assert isinstance(metrics.wer, float)
        assert isinstance(metrics.cer, float)
        assert 0 <= metrics.wer <= 1.0


def run_tests():
    """Run all tests."""
    import traceback

    test_classes = [
        TestWER,
        TestCER,
        TestOtherSpeechMetrics,
        TestSpeechMetrics,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
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
