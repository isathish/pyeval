"""
Tests for new advanced metrics added to pyeval.
"""

import pytest
from pyeval.ml import (
    balanced_accuracy_score, log_loss, brier_score, hamming_loss,
    jaccard_score, top_k_accuracy, expected_calibration_error,
    mean_squared_log_error, symmetric_mape, huber_loss, quantile_loss,
    normalized_rmse, adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score
)
from pyeval.nlp import (
    chrf_score, text_entropy, repetition_ratio, compression_ratio,
    coverage_score, density_score, lexical_diversity, sentence_bleu
)
from pyeval.llm import (
    bias_detection, instruction_following_score, multi_turn_coherence,
    summarization_quality, response_diversity
)
from pyeval.rag import (
    context_entity_recall, answer_attribution, context_utilization,
    question_answer_relevance, rag_pipeline_score
)
from pyeval.speech import (
    slot_error_rate, intent_accuracy, phoneme_error_rate,
    keyword_spotting_metrics, mean_opinion_score, speech_intelligibility_index,
    fluency_score as speech_fluency_score
)
from pyeval.recommender import (
    serendipity, gini_index, expected_percentile_ranking, auc_score,
    inter_list_diversity, entropy_diversity, surprisal, ranking_correlation,
    beyond_accuracy_metrics
)
from pyeval.utils.math_ops import (
    bootstrap_confidence_interval, paired_t_test, independent_t_test,
    wilcoxon_signed_rank_test, mann_whitney_u_test, mcnemar_test,
    cohens_d, hedges_g, glass_delta, correlation_coefficient, spearman_correlation
)
from pyeval.utils.viz_ops import (
    confusion_matrix_display, classification_report_display,
    horizontal_bar_chart, histogram_display, sparkline, progress_bar
)


class TestNewMLMetrics:
    """Tests for new ML classification, regression and clustering metrics."""
    
    def test_balanced_accuracy(self):
        y_true = [0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 0, 0]
        # Class 0: 2/2 correct = 1.0, Class 1: 2/4 correct = 0.5
        # Balanced = (1.0 + 0.5) / 2 = 0.75
        result = balanced_accuracy_score(y_true, y_pred)
        assert 0.7 < result < 0.8
    
    def test_log_loss(self):
        y_true = [0, 1, 1, 0]
        y_prob = [0.1, 0.9, 0.8, 0.3]
        result = log_loss(y_true, y_prob)
        assert isinstance(result, float)
        assert result > 0  # Log loss is always positive
    
    def test_brier_score(self):
        y_true = [1, 0, 1, 1]
        y_prob = [0.9, 0.1, 0.8, 0.6]
        result = brier_score(y_true, y_prob)
        assert 0 <= result <= 1
    
    def test_hamming_loss(self):
        y_true = [[1, 0, 1], [0, 1, 1]]
        y_pred = [[1, 0, 0], [0, 1, 1]]
        result = hamming_loss(y_true, y_pred)
        assert result == pytest.approx(1/6, abs=0.01)
    
    def test_jaccard_score(self):
        y_true = [1, 1, 1, 0, 0]
        y_pred = [1, 1, 0, 0, 1]
        result = jaccard_score(y_true, y_pred)
        assert 0 <= result <= 1
    
    def test_top_k_accuracy(self):
        y_true = [0, 1, 2]
        y_prob = [[0.8, 0.1, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]]
        result = top_k_accuracy(y_true, y_prob, k=2)
        assert result == 1.0  # All correct within top 2
    
    def test_mean_squared_log_error(self):
        y_true = [1, 2, 3, 4]
        y_pred = [1.1, 2.2, 2.8, 4.1]
        result = mean_squared_log_error(y_true, y_pred)
        assert result >= 0
    
    def test_symmetric_mape(self):
        y_true = [100, 200, 300]
        y_pred = [110, 190, 310]
        result = symmetric_mape(y_true, y_pred)
        assert 0 <= result  # sMAPE is non-negative
    
    def test_adjusted_rand_score(self):
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 0, 1, 1, 1, 2]
        result = adjusted_rand_score(labels_true, labels_pred)
        assert -1 <= result <= 1
    
    def test_v_measure_score(self):
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 0, 1, 1, 2, 2]
        result = v_measure_score(labels_true, labels_pred)
        assert result == pytest.approx(1.0, abs=0.01)


class TestNewNLPMetrics:
    """Tests for new NLP metrics."""
    
    def test_chrf_score(self):
        reference = "Hello world test"
        candidate = "Hello world test"
        result = chrf_score(reference, candidate)
        assert 'chrf' in result
        assert 0.9 <= result['chrf'] <= 1.0
    
    def test_text_entropy(self):
        text = "hello hello world world test test"
        result = text_entropy(text)
        assert result > 0  # Entropy should be positive
    
    def test_repetition_ratio(self):
        text = "hello hello world"
        result = repetition_ratio(text)
        assert 0 <= result <= 1
    
    def test_lexical_diversity(self):
        text = "the cat sat on the mat the dog ran"
        result = lexical_diversity(text)
        assert 'ttr' in result
        assert 0 < result['ttr'] <= 1
    
    def test_sentence_bleu(self):
        reference = "the cat is on the mat"
        hypothesis = "the cat is on mat"
        result = sentence_bleu(reference, hypothesis)
        assert 0 <= result <= 1


class TestNewLLMMetrics:
    """Tests for new LLM evaluation metrics."""
    
    def test_bias_detection(self):
        text = "The doctor performed surgery"  # Neutral text
        result = bias_detection(text)
        assert 'bias_score' in result
        assert 0 <= result['bias_score'] <= 1
    
    def test_instruction_following(self):
        instruction = "Write a short sentence about cats"
        response = "Cats are wonderful pets that bring joy to many homes"
        result = instruction_following_score(instruction, response)
        assert 'instruction_following' in result
    
    def test_multi_turn_coherence(self):
        turns = [
            {'role': 'user', 'content': "What is Python?"},
            {'role': 'assistant', 'content': "Python is a programming language"},
            {'role': 'user', 'content': "What can I do with it?"},
            {'role': 'assistant', 'content': "You can build applications"}
        ]
        result = multi_turn_coherence(turns)
        assert 'coherence' in result
    
    def test_summarization_quality(self):
        source = "Python is a programming language. It is popular for data science."
        summary = "Python is popular for data science."
        result = summarization_quality(source, summary)
        assert 'quality' in result
    
    def test_response_diversity(self):
        responses = ["Hello world", "Hello there", "Hi everyone"]
        result = response_diversity(responses)
        assert 'diversity' in result


class TestNewRAGMetrics:
    """Tests for new RAG metrics."""
    
    def test_context_entity_recall(self):
        contexts = ["John Smith is the CEO of TechCorp"]
        ground_truth = "John Smith leads TechCorp"
        result = context_entity_recall(contexts, ground_truth)
        assert 'entity_recall' in result
    
    def test_answer_attribution(self):
        answer = "Python is great for data science"
        contexts = ["Python is a popular programming language", 
                   "Data science uses Python extensively"]
        result = answer_attribution(answer, contexts)
        assert 'attribution_rate' in result
    
    def test_context_utilization(self):
        answer = "Python is useful for machine learning"
        contexts = ["Python is great for ML", "Java is also popular"]
        result = context_utilization(answer, contexts)
        assert 'average_utilization' in result
    
    def test_question_answer_relevance(self):
        question = "What is Python?"
        answer = "Python is a programming language"
        result = question_answer_relevance(question, answer)
        assert 'relevance' in result
    
    def test_rag_pipeline_score(self):
        question = "What is machine learning?"
        contexts = ["Machine learning is a subset of AI"]
        answer = "Machine learning is a type of artificial intelligence"
        result = rag_pipeline_score(question, contexts, answer)
        assert 'overall_score' in result


class TestNewSpeechMetrics:
    """Tests for new speech metrics."""
    
    def test_slot_error_rate(self):
        ref_slots = {'city': 'Paris', 'date': 'tomorrow'}
        hyp_slots = {'city': 'Paris', 'date': 'today'}
        result = slot_error_rate(ref_slots, hyp_slots)
        assert 'slot_error_rate' in result
    
    def test_intent_accuracy(self):
        ref_intents = ['booking', 'query', 'booking']
        hyp_intents = ['booking', 'query', 'cancel']
        result = intent_accuracy(ref_intents, hyp_intents)
        assert 'intent_accuracy' in result
        assert result['intent_accuracy'] == pytest.approx(2/3, abs=0.01)
    
    def test_phoneme_error_rate(self):
        ref_phonemes = ['AH', 'B', 'AW', 'T']
        hyp_phonemes = ['AH', 'B', 'AH', 'T']
        result = phoneme_error_rate(ref_phonemes, hyp_phonemes)
        assert 'per' in result
    
    def test_mean_opinion_score(self):
        scores = [4.0, 4.5, 3.5, 4.0, 5.0]
        result = mean_opinion_score(scores)
        assert 'mos' in result
        assert 3.5 < result['mos'] < 4.5
    
    def test_speech_intelligibility_index(self):
        result = speech_intelligibility_index(15.0)  # 15 dB SNR
        assert 0 <= result <= 1


class TestNewRecommenderMetrics:
    """Tests for new recommender metrics."""
    
    def test_serendipity(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = [3, 5, 7]
        expected = [1, 2]  # Popular items
        result = serendipity(recommended, relevant, expected)
        assert 0 <= result <= 1
    
    def test_gini_index(self):
        recommendations = [[1, 2, 3], [1, 2, 4], [1, 3, 5]]
        catalog = [1, 2, 3, 4, 5]
        result = gini_index(recommendations, catalog)
        assert 0 <= result <= 1
    
    def test_inter_list_diversity(self):
        recommendations = [[1, 2, 3], [4, 5, 6], [1, 4, 7]]
        result = inter_list_diversity(recommendations)
        assert 0 <= result <= 1
    
    def test_ranking_correlation(self):
        recommended = [1, 2, 3, 4, 5]
        truth = [1, 3, 2, 5, 4]
        result = ranking_correlation(recommended, truth)
        assert 'kendall_tau' in result
        assert 'spearman_rho' in result


class TestStatisticalUtilities:
    """Tests for statistical testing utilities."""
    
    def test_bootstrap_ci(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = bootstrap_confidence_interval(data, 'mean', n_bootstrap=100, seed=42)
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert result['ci_lower'] < result['point_estimate'] < result['ci_upper']
    
    def test_paired_t_test(self):
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [1.5, 2.5, 3.5, 4.5, 5.5]
        result = paired_t_test(sample1, sample2)
        assert 't_statistic' in result
        assert 'p_value' in result
    
    def test_independent_t_test(self):
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [2, 3, 4, 5, 6]
        result = independent_t_test(sample1, sample2)
        assert 't_statistic' in result
        assert 'cohens_d' in result
    
    def test_wilcoxon_test(self):
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [1.2, 2.1, 2.9, 4.2, 4.8]
        result = wilcoxon_signed_rank_test(sample1, sample2)
        assert 'w_statistic' in result
    
    def test_mann_whitney_test(self):
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [3, 4, 5, 6, 7]
        result = mann_whitney_u_test(sample1, sample2)
        assert 'u_statistic' in result
    
    def test_mcnemar_test(self):
        table = [[20, 5], [10, 15]]
        result = mcnemar_test(table)
        assert 'chi_square' in result
        assert 'p_value' in result
    
    def test_cohens_d(self):
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [2, 3, 4, 5, 6]
        result = cohens_d(sample1, sample2)
        assert isinstance(result, float)
    
    def test_correlation_coefficient(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]
        result = correlation_coefficient(x, y)
        assert 'r' in result
        assert 'r_squared' in result
        assert -1 <= result['r'] <= 1


class TestVisualizationUtilities:
    """Tests for ASCII visualization utilities."""
    
    def test_confusion_matrix_display(self):
        matrix = [[50, 10], [5, 35]]
        result = confusion_matrix_display(matrix, ['Cat', 'Dog'])
        assert 'CONFUSION MATRIX' in result
        assert 'Cat' in result
        assert 'Dog' in result
    
    def test_classification_report_display(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 2, 2, 2]
        result = classification_report_display(y_true, y_pred)
        assert 'CLASSIFICATION REPORT' in result
        assert 'Precision' in result
    
    def test_horizontal_bar_chart(self):
        values = {'A': 0.8, 'B': 0.6, 'C': 0.9}
        result = horizontal_bar_chart(values, title="Test Chart")
        assert 'Test Chart' in result
        assert 'A' in result
    
    def test_histogram_display(self):
        values = [1, 2, 2, 3, 3, 3, 4, 4, 5]
        result = histogram_display(values, bins=5, title="Test Histogram")
        assert 'Test Histogram' in result
    
    def test_sparkline(self):
        values = [1, 2, 5, 3, 8, 4, 2]
        result = sparkline(values)
        assert len(result) > 0
        assert all(c in "▁▂▃▄▅▆▇█" for c in result)
    
    def test_progress_bar(self):
        result = progress_bar(50, 100)
        assert '50.0%' in result
        assert '█' in result
