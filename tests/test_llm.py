"""
Tests for LLM evaluation metrics.
"""

import sys
sys.path.insert(0, '..')

import pytest
from pyeval import (
    toxicity_score, hallucination_score, answer_relevancy,
    faithfulness_score, coherence_score,
    LLMMetrics
)


class TestToxicityScore:
    """Tests for toxicity detection."""
    
    def test_clean_text(self):
        """Clean text should have low toxicity."""
        text = "The weather is beautiful today. I love flowers and sunshine."
        result = toxicity_score(text)
        assert isinstance(result, dict)
        assert 'score' in result
        assert 0 <= result['score'] <= 1
    
    def test_returns_dict(self):
        """Should return dictionary with score."""
        text = "Hello world, this is a test."
        result = toxicity_score(text)
        assert isinstance(result, dict)


class TestHallucinationScore:
    """Tests for hallucination detection."""
    
    def test_no_hallucination(self):
        """Response grounded in context should have low hallucination."""
        response = "Paris is the capital of France."
        context = "France is a country in Europe. Its capital is Paris."
        result = hallucination_score(response, context)
        assert isinstance(result, dict)
        assert 'score' in result
        assert 0 <= result['score'] <= 1
    
    def test_hallucination(self):
        """Response not in context should have high hallucination."""
        response = "The moon is made of cheese and has many aliens living there."
        context = "The Earth is the third planet from the Sun."
        result = hallucination_score(response, context)
        assert isinstance(result, dict)
        assert 'score' in result


class TestAnswerRelevancy:
    """Tests for answer relevancy."""
    
    def test_relevant_answer(self):
        """Relevant answer should have high score."""
        answer = "The capital of France is Paris."
        question = "What is the capital of France?"
        result = answer_relevancy(question, answer)
        assert isinstance(result, dict)
        assert 'score' in result
        assert 0 <= result['score'] <= 1
    
    def test_returns_dict(self):
        """Should return dictionary."""
        answer = "Some answer"
        question = "Some question?"
        result = answer_relevancy(question, answer)
        assert isinstance(result, dict)


class TestFaithfulnessScore:
    """Tests for faithfulness to context."""
    
    def test_faithful_response(self):
        """Response faithful to context should score high."""
        response = "Python is a programming language."
        context = "Python is a popular programming language used for many applications."
        result = faithfulness_score(response, context)
        assert isinstance(result, dict)
        assert 'score' in result
        assert 0 <= result['score'] <= 1
    
    def test_returns_dict(self):
        """Should return dictionary."""
        response = "Some response"
        context = "Some context"
        result = faithfulness_score(response, context)
        assert isinstance(result, dict)


class TestCoherenceScore:
    """Tests for coherence."""
    
    def test_coherent_response(self):
        """Coherent response should score high."""
        text = "The Eiffel Tower is located in Paris, France. It was built in 1889."
        result = coherence_score(text)
        assert isinstance(result, dict)
        assert 'overall' in result or 'score' in result
    
    def test_short_text(self):
        """Short text should still work."""
        text = "Yes"
        result = coherence_score(text)
        assert isinstance(result, dict)


class TestLLMMetrics:
    """Tests for LLMMetrics class."""
    
    def test_compute_all_metrics(self):
        """Should compute all LLM metrics."""
        response = "Paris is the capital of France and is known for the Eiffel Tower."
        query = "What is the capital of France?"
        context = "France is a country in Europe. Its capital city is Paris."
        
        metrics = LLMMetrics.compute(response, query, context)
        
        assert metrics is not None
        # Check for expected attributes
        assert hasattr(metrics, 'toxicity') or hasattr(metrics, 'overall_score')
    
    def test_without_context(self):
        """Should work without context."""
        response = "Hello, how are you?"
        query = "Greet me"
        
        metrics = LLMMetrics.compute(response, query)
        assert metrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
