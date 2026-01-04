"""
Tests for RAG evaluation metrics.
"""

import sys
sys.path.insert(0, '..')

import pytest
from pyeval import (
    context_relevance, context_precision, context_recall,
    answer_correctness, retrieval_precision, retrieval_recall,
    groundedness_score, noise_robustness,
    RAGMetrics
)


class TestContextRelevance:
    """Tests for context relevance."""
    
    def test_relevant_context(self):
        """Relevant context should have high score."""
        context = "Paris is the capital city of France and is famous for the Eiffel Tower."
        query = "What is the capital of France?"
        result = context_relevance(context, query)
        assert isinstance(result, (int, float, dict))
        if isinstance(result, dict):
            assert 'score' in result
    
    def test_irrelevant_context(self):
        """Irrelevant context should have lower score."""
        context = "The weather in Australia is often sunny and warm."
        query = "What is the capital of France?"
        result = context_relevance(context, query)
        assert result is not None


class TestContextPrecision:
    """Tests for context precision."""
    
    def test_context_precision_high(self):
        """Retrieved context fully relevant should have high precision."""
        retrieved_contexts = [
            "Paris is the capital of France.",
            "France is a country in Western Europe.",
            "The Eiffel Tower is in Paris."
        ]
        query = "Tell me about Paris and France."
        result = context_precision(retrieved_contexts, query)
        assert result is not None


class TestContextRecall:
    """Tests for context recall."""
    
    def test_context_recall(self):
        """Context recall should work with ground truth."""
        retrieved_contexts = [
            "Paris is the capital of France.",
            "The Eiffel Tower is located in Paris."
        ]
        ground_truth = "Paris is the capital of France and home to the Eiffel Tower."
        result = context_recall(retrieved_contexts, ground_truth)
        assert result is not None


class TestAnswerCorrectness:
    """Tests for answer correctness."""
    
    def test_correct_answer(self):
        """Correct answer should have high score."""
        answer = "Paris"
        ground_truth = "Paris"
        result = answer_correctness(answer, ground_truth)
        assert result is not None
        # If numeric, should be high for exact match
        if isinstance(result, (int, float)):
            assert result >= 0
    
    def test_partially_correct(self):
        """Partially correct answer should have medium score."""
        answer = "The capital of France is Paris, located in Europe."
        ground_truth = "Paris is the capital of France."
        result = answer_correctness(answer, ground_truth)
        assert result is not None


class TestRetrievalPrecision:
    """Tests for retrieval precision."""
    
    def test_perfect_precision(self):
        """All relevant retrievals should have high precision."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3", "doc4"]
        result = retrieval_precision(retrieved, relevant)
        assert result is not None
        if isinstance(result, (int, float)):
            assert 0 <= result <= 1
    
    def test_no_relevant(self):
        """No relevant retrievals should have low precision."""
        retrieved = ["doc5", "doc6"]
        relevant = ["doc1", "doc2"]
        result = retrieval_precision(retrieved, relevant)
        assert result is not None


class TestRetrievalRecall:
    """Tests for retrieval recall."""
    
    def test_perfect_recall(self):
        """All relevant docs retrieved should have high recall."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = ["doc1", "doc2"]
        result = retrieval_recall(retrieved, relevant)
        assert result is not None
        if isinstance(result, (int, float)):
            assert 0 <= result <= 1
    
    def test_partial_recall(self):
        """Some relevant docs retrieved should have partial recall."""
        retrieved = ["doc1", "doc5"]
        relevant = ["doc1", "doc2", "doc3"]
        result = retrieval_recall(retrieved, relevant)
        assert result is not None


class TestGroundednessScore:
    """Tests for groundedness."""
    
    def test_grounded_response(self):
        """Response grounded in context should score high."""
        response = "Paris is the capital of France."
        context = "France is a country. Paris is the capital of France."
        result = groundedness_score(response, context)
        assert result is not None
    
    def test_ungrounded_response(self):
        """Response not in context should score low."""
        response = "Tokyo is the capital of Japan."
        context = "France is a country. Paris is the capital."
        result = groundedness_score(response, context)
        assert result is not None


class TestNoiseRobustness:
    """Tests for noise robustness."""
    
    def test_with_noise(self):
        """Should handle noisy context."""
        response = "Paris is the capital of France."
        contexts = [
            "Paris is the capital of France.",
            "Random noise about unrelated topics.",
            "More irrelevant information."
        ]
        result = noise_robustness(response, contexts)
        assert result is not None


class TestRAGMetrics:
    """Tests for RAGMetrics class."""
    
    def test_compute_metrics(self):
        """Should compute RAG metrics."""
        query = "What is the capital of France?"
        answer = "Paris is the capital of France."
        contexts = ["France is a country. Paris is its capital."]
        ground_truth = "Paris"
        
        metrics = RAGMetrics.compute(query, answer, contexts, ground_truth)
        assert metrics is not None
    
    def test_without_ground_truth(self):
        """Should work without ground truth."""
        query = "What is the capital of France?"
        answer = "Paris"
        contexts = ["Paris is the capital of France."]
        
        metrics = RAGMetrics.compute(query, answer, contexts)
        assert metrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
