#!/usr/bin/env python3
"""
PyEval - Comprehensive Examples
================================

This file demonstrates all the features of the pyeval package.
Run this file to see example outputs for all metric categories.

No external dependencies required!
"""

import sys
sys.path.insert(0, '..')

from pyeval import (
    # ML Metrics
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score,
    ClassificationMetrics, RegressionMetrics, ClusteringMetrics,
    
    # NLP Metrics
    bleu_score, rouge_score, meteor_score, NLPMetrics,
    
    # LLM Metrics
    hallucination_score, answer_relevancy, faithfulness_score,
    toxicity_score, coherence_score, LLMMetrics,
    
    # RAG Metrics
    context_relevance, answer_correctness, retrieval_precision,
    retrieval_recall, RAGMetrics,
    
    # Fairness Metrics
    demographic_parity, equalized_odds, disparate_impact,
    FairnessMetrics,
    
    # Speech Metrics
    word_error_rate, character_error_rate, SpeechMetrics,
    
    # Recommender Metrics
    precision_at_k, recall_at_k, ndcg_at_k, mean_average_precision,
    hit_rate, RecommenderMetrics,
    
    # Evaluator
    Evaluator, EvaluationReport,
    
    # Tracking
    ExperimentTracker,
)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# =============================================================================
# 1. ML Classification Metrics
# =============================================================================

def demo_classification():
    print_section("ML CLASSIFICATION METRICS")
    
    # Binary classification
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    y_prob = [0.9, 0.1, 0.4, 0.8, 0.2, 0.95, 0.6, 0.3, 0.85, 0.45]
    
    print("Binary Classification Example:")
    print(f"  y_true: {y_true}")
    print(f"  y_pred: {y_pred}")
    print()
    
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}")
    
    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    for row in cm:
        print(f"    {row}")
    
    # Using ClassificationMetrics class
    print("\n  Using ClassificationMetrics.compute():")
    metrics = ClassificationMetrics.compute(y_true, y_pred)
    print(f"    {metrics}")
    
    # Multiclass classification
    print("\n\nMulticlass Classification Example:")
    y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    y_pred_multi = [0, 1, 1, 0, 2, 2, 0, 0, 2, 1]
    
    print(f"  Accuracy (macro):  {accuracy_score(y_true_multi, y_pred_multi):.4f}")
    print(f"  Precision (macro): {precision_score(y_true_multi, y_pred_multi, average='macro'):.4f}")
    print(f"  Recall (macro):    {recall_score(y_true_multi, y_pred_multi, average='macro'):.4f}")
    print(f"  F1 (macro):        {f1_score(y_true_multi, y_pred_multi, average='macro'):.4f}")


# =============================================================================
# 2. ML Regression Metrics
# =============================================================================

def demo_regression():
    print_section("ML REGRESSION METRICS")
    
    y_true = [3.0, 5.5, 2.5, 7.0, 4.5, 6.0, 8.5, 1.5]
    y_pred = [2.8, 5.2, 2.9, 7.3, 4.2, 5.8, 8.2, 1.8]
    
    print("Regression Example:")
    print(f"  y_true: {y_true}")
    print(f"  y_pred: {y_pred}")
    print()
    
    print(f"  MSE:  {mean_squared_error(y_true, y_pred):.4f}")
    print(f"  RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
    print(f"  MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  R²:   {r2_score(y_true, y_pred):.4f}")
    
    # Using RegressionMetrics class
    print("\n  Using RegressionMetrics.compute():")
    metrics = RegressionMetrics.compute(y_true, y_pred)
    print(f"    MSE={metrics.mse:.4f}, RMSE={metrics.rmse:.4f}, "
          f"MAE={metrics.mae:.4f}, R²={metrics.r2:.4f}")


# =============================================================================
# 3. Clustering Metrics
# =============================================================================

def demo_clustering():
    print_section("CLUSTERING METRICS")
    
    # Sample 2D data points with cluster assignments
    X = [
        [1.0, 1.0], [1.5, 1.5], [1.2, 1.3],  # Cluster 0
        [5.0, 5.0], [5.5, 5.5], [5.2, 5.3],  # Cluster 1
        [9.0, 1.0], [9.5, 1.5], [9.2, 1.3],  # Cluster 2
    ]
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    
    print("Clustering Example:")
    print(f"  Number of samples: {len(X)}")
    print(f"  Number of clusters: {len(set(labels))}")
    print()
    
    print(f"  Silhouette Score: {silhouette_score(X, labels):.4f}")
    
    # Using ClusteringMetrics class
    metrics = ClusteringMetrics.compute(X, labels)
    print(f"\n  Using ClusteringMetrics.compute():")
    print(f"    Silhouette: {metrics.silhouette:.4f}")


# =============================================================================
# 4. NLP Metrics
# =============================================================================

def demo_nlp():
    print_section("NLP METRICS (BLEU, ROUGE, METEOR)")
    
    reference = "The quick brown fox jumps over the lazy dog"
    hypothesis = "A fast brown fox leaps over a lazy dog"
    
    print("Text Generation Example:")
    print(f"  Reference:  '{reference}'")
    print(f"  Hypothesis: '{hypothesis}'")
    print()
    
    # BLEU Score (pass strings - function tokenizes internally)
    bleu = bleu_score([reference], hypothesis)
    print(f"  BLEU Score: {bleu['bleu']:.4f}")
    
    # ROUGE Scores
    rouge = rouge_score(reference, hypothesis)
    print(f"\n  ROUGE Scores:")
    print(f"    ROUGE-1 F1: {rouge['rouge1']['f1']:.4f}")
    print(f"    ROUGE-2 F1: {rouge['rouge2']['f1']:.4f}")
    print(f"    ROUGE-L F1: {rouge['rougeL']['f1']:.4f}")
    
    # METEOR Score (single reference string, candidate string)
    meteor = meteor_score(reference, hypothesis)
    print(f"\n  METEOR Score: {meteor['meteor']:.4f}")
    
    # Using NLPMetrics class (single reference, single candidate)
    print("\n  Using NLPMetrics.compute():")
    metrics = NLPMetrics.compute(reference, hypothesis)
    print(f"    BLEU={metrics.bleu:.4f}, ROUGE-1={metrics.rouge1_f1:.4f}, "
          f"METEOR={metrics.meteor:.4f}")


# =============================================================================
# 5. LLM Evaluation Metrics
# =============================================================================

def demo_llm():
    print_section("LLM EVALUATION METRICS")
    
    query = "What is the capital of France?"
    response = "The capital of France is Paris, which is known for the Eiffel Tower."
    context = "France is a country in Western Europe. Its capital city is Paris, home to landmarks like the Eiffel Tower and the Louvre Museum."
    
    print("LLM Response Evaluation:")
    print(f"  Query:    '{query}'")
    print(f"  Response: '{response}'")
    print(f"  Context:  '{context[:50]}...'")
    print()
    
    # All LLM functions return dictionaries with scores
    toxicity = toxicity_score(response)
    coherence = coherence_score(response)
    relevancy = answer_relevancy(query, response)
    faithful = faithfulness_score(response, context)
    hallucination = hallucination_score(response, context)
    
    print(f"  Toxicity Score:     {toxicity['toxicity']:.4f} (lower is better)")
    print(f"  Coherence Score:    {coherence['coherence']:.4f}")
    print(f"  Relevancy Score:    {relevancy['relevancy']:.4f}")
    print(f"  Faithfulness Score: {faithful['faithfulness']:.4f}")
    print(f"  Hallucination Score: {hallucination['hallucination_score']:.4f} (lower is better)")
    
    # Using LLMMetrics class
    print("\n  Using LLMMetrics.compute():")
    metrics = LLMMetrics.compute(response, query, context)
    print(f"    Overall Score: {metrics.overall_score():.4f}")


# =============================================================================
# 6. RAG Evaluation Metrics
# =============================================================================

def demo_rag():
    print_section("RAG EVALUATION METRICS")
    
    query = "What are the benefits of exercise?"
    contexts = [
        "Regular exercise improves cardiovascular health and strengthens muscles.",
        "Exercise releases endorphins which improve mood and reduce stress.",
        "Physical activity helps maintain a healthy weight and boosts energy."
    ]
    response = "Exercise has many benefits including improved heart health, better mood, and increased energy levels."
    ground_truth = "Exercise improves cardiovascular health, mental well-being, and helps maintain healthy weight."
    
    print("RAG Pipeline Evaluation:")
    print(f"  Query: '{query}'")
    print(f"  Response: '{response}'")
    print(f"  Contexts: {len(contexts)} retrieved")
    print()
    
    # context_relevance and answer_correctness return dicts
    ctx_rel = context_relevance(query, contexts)
    ans_corr = answer_correctness(response, ground_truth)
    
    print(f"  Context Relevance:    {ctx_rel['overall_relevance']:.4f}")
    print(f"  Answer Correctness:   {ans_corr['correctness']:.4f}")
    print(f"  Retrieval Precision:  {retrieval_precision(contexts, [ground_truth]):.4f}")
    print(f"  Retrieval Recall:     {retrieval_recall(contexts, [ground_truth]):.4f}")
    
    # Using RAGMetrics class
    # Signature: compute(question, answer, contexts, ground_truth_answer, ground_truth_contexts)
    print("\n  Using RAGMetrics.compute():")
    metrics = RAGMetrics.compute(query, response, contexts, ground_truth)
    print(f"    Context Relevance: {metrics.context_relevance:.4f}")
    print(f"    Answer Correctness: {metrics.answer_correctness:.4f}")


# =============================================================================
# 7. Fairness Metrics
# =============================================================================

def demo_fairness():
    print_section("FAIRNESS METRICS")
    
    # Model predictions and sensitive attributes
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    sensitive = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 
                 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
    
    print("Fairness Evaluation:")
    print(f"  Samples: {len(y_true)}")
    print(f"  Groups: A ({sensitive.count('A')}), B ({sensitive.count('B')})")
    print()
    
    # All fairness functions return dictionaries
    dp = demographic_parity(y_pred, sensitive)
    eo = equalized_odds(y_true, y_pred, sensitive)
    di = disparate_impact(y_pred, sensitive)
    
    print(f"  Demographic Parity Diff: {dp['dp_difference']:.4f}")
    print(f"  Equalized Odds Diff:     {eo['eo_difference']:.4f}")
    print(f"  Disparate Impact Ratio:  {di['di_ratio']:.4f}")
    
    # Using FairnessMetrics class
    print("\n  Using FairnessMetrics.compute():")
    metrics = FairnessMetrics.compute(y_true, y_pred, sensitive)
    print(f"    Demographic Parity: {metrics.dp_difference:.4f}")
    print(f"    Equalized Odds (TPR diff): {metrics.tpr_difference:.4f}")


# =============================================================================
# 8. Speech Recognition Metrics
# =============================================================================

def demo_speech():
    print_section("SPEECH RECOGNITION METRICS")
    
    reference = "the quick brown fox jumps over the lazy dog"
    hypothesis = "the quick brown fox jumps over lazy dog"
    
    print("Speech Recognition Evaluation:")
    print(f"  Reference:  '{reference}'")
    print(f"  Hypothesis: '{hypothesis}'")
    print()
    
    # Speech metrics return dicts
    wer = word_error_rate(reference, hypothesis)
    cer = character_error_rate(reference, hypothesis)
    
    print(f"  Word Error Rate (WER):      {wer['wer']:.4f}")
    print(f"  Character Error Rate (CER): {cer['cer']:.4f}")
    
    # Using SpeechMetrics class
    print("\n  Using SpeechMetrics.compute():")
    metrics = SpeechMetrics.compute(reference, hypothesis)
    print(f"    WER: {metrics.wer:.4f}, CER: {metrics.cer:.4f}")


# =============================================================================
# 9. Recommender System Metrics
# =============================================================================

def demo_recommender():
    print_section("RECOMMENDER SYSTEM METRICS")
    
    # User's recommendations and actual relevant items
    recommended = [101, 203, 45, 67, 89, 12, 34, 56, 78, 90]
    relevant = [45, 89, 78, 123, 456]  # Ground truth relevant items
    
    print("Recommender System Evaluation:")
    print(f"  Recommended items: {recommended[:5]}...")
    print(f"  Relevant items:    {relevant}")
    print()
    
    for k in [5, 10]:
        print(f"  @K={k}:")
        print(f"    Precision@{k}: {precision_at_k(recommended, relevant, k):.4f}")
        print(f"    Recall@{k}:    {recall_at_k(recommended, relevant, k):.4f}")
        print(f"    NDCG@{k}:      {ndcg_at_k(recommended, relevant, k):.4f}")
        print(f"    Hit Rate@{k}:  {hit_rate(recommended, relevant, k):.4f}")
    
    # Multiple users
    print("\n  Multi-User Metrics (MAP):")
    all_recommendations = [
        [101, 203, 45, 67, 89],
        [12, 45, 78, 90, 100],
        [45, 67, 89, 101, 203]
    ]
    all_relevants = [
        [45, 89, 123],
        [45, 78, 200],
        [67, 101, 500]
    ]
    
    map_score = mean_average_precision(all_recommendations, all_relevants)
    print(f"    MAP: {map_score:.4f}")
    
    # Using RecommenderMetrics class
    print("\n  Using RecommenderMetrics.compute():")
    metrics = RecommenderMetrics.compute(recommended, relevant, k=5)
    print(f"    P@5={metrics.precision_at_k:.4f}, R@5={metrics.recall_at_k:.4f}, "
          f"NDCG@5={metrics.ndcg_at_k:.4f}")


# =============================================================================
# 10. Unified Evaluator
# =============================================================================

def demo_evaluator():
    print_section("UNIFIED EVALUATOR")
    
    evaluator = Evaluator("Demo Evaluation")
    
    # Classification evaluation
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    report = evaluator.evaluate_classification(y_true, y_pred, name="Binary Classifier")
    print(report.summary())
    
    # NLP evaluation
    references = ["The cat sat on the mat", "Hello world"]
    hypotheses = ["A cat is on the mat", "Hello there world"]
    report = evaluator.evaluate_generation(references, hypotheses, name="Text Generator")
    print(report.summary())
    
    # Compare all reports
    print(evaluator.compare_reports())


# =============================================================================
# 11. Experiment Tracking
# =============================================================================

def demo_tracking():
    print_section("EXPERIMENT TRACKING")
    
    tracker = ExperimentTracker("ml_experiments")
    
    # Simulate multiple experiment runs
    experiments = [
        {"name": "model_v1", "lr": 0.01, "epochs": 10, "acc": 0.85, "f1": 0.82},
        {"name": "model_v2", "lr": 0.001, "epochs": 20, "acc": 0.88, "f1": 0.85},
        {"name": "model_v3", "lr": 0.005, "epochs": 15, "acc": 0.92, "f1": 0.90},
    ]
    
    for exp in experiments:
        run = tracker.create_run(exp["name"])
        run.log_params({"learning_rate": exp["lr"], "epochs": exp["epochs"]})
        run.log_metrics({"accuracy": exp["acc"], "f1_score": exp["f1"]})
        run.set_status("completed")
        tracker.save_run(run)
    
    # Print comparison
    print(tracker.compare_runs())
    
    # Get best run
    best = tracker.get_best_run("accuracy")
    if best:
        print(f"Best run by accuracy: {best.experiment_name}")
        print(f"  Accuracy: {best.metrics['accuracy']:.4f}")
        print(f"  Parameters: {best.parameters}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  PYEVAL - COMPREHENSIVE EVALUATION FRAMEWORK DEMO")
    print("  No external dependencies required!")
    print("="*60)
    
    demo_classification()
    demo_regression()
    demo_clustering()
    demo_nlp()
    demo_llm()
    demo_rag()
    demo_fairness()
    demo_speech()
    demo_recommender()
    demo_evaluator()
    demo_tracking()
    
    print("\n" + "="*60)
    print("  DEMO COMPLETE!")
    print("  PyEval provides 100+ metrics with zero dependencies.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
