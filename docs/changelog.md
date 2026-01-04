# Changelog

All notable changes to PyEval will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Nothing yet

### Security
- Nothing yet

---

## [1.0.0] - 2024-01-15

### Added

#### Core Metrics
- **ML Classification Metrics**
  - `accuracy_score` - Overall accuracy measurement
  - `precision_score` - Precision with multi-class support
  - `recall_score` - Recall with multi-class support
  - `f1_score` - F1 score with averaging options
  - `fbeta_score` - Fβ score with configurable beta
  - `confusion_matrix` - Full confusion matrix computation
  - `roc_auc_score` - ROC AUC for binary and multi-class
  - `average_precision_score` - Average precision (PR-AUC)
  - `matthews_corrcoef` - Matthews Correlation Coefficient
  - `cohen_kappa_score` - Cohen's Kappa
  - `hamming_loss` - Hamming loss
  - `hinge_loss` - Hinge loss
  - `log_loss` - Log loss (cross-entropy)
  - `jaccard_score` - Jaccard similarity
  - `balanced_accuracy_score` - Balanced accuracy
  - `top_k_accuracy_score` - Top-K accuracy

- **ML Regression Metrics**
  - `mean_squared_error` (MSE)
  - `root_mean_squared_error` (RMSE)
  - `mean_absolute_error` (MAE)
  - `mean_absolute_percentage_error` (MAPE)
  - `r2_score` - R² coefficient of determination
  - `explained_variance_score`
  - `max_error`
  - `median_absolute_error`
  - `mean_squared_log_error` (MSLE)
  - `mean_poisson_deviance`
  - `mean_gamma_deviance`
  - `mean_tweedie_deviance`
  - `d2_tweedie_score`

- **NLP Metrics**
  - `bleu_score` - BLEU (1-4 grams)
  - `rouge_score` - ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum
  - `meteor_score` - METEOR
  - `ter_score` - Translation Edit Rate
  - `wer` - Word Error Rate
  - `cer` - Character Error Rate
  - `chrF_score` - chrF and chrF++
  - `bert_score` - BERTScore (precision, recall, F1)
  - `perplexity` - Language model perplexity
  - `sentence_bleu` - Sentence-level BLEU
  - `corpus_bleu` - Corpus-level BLEU

- **LLM Metrics**
  - `faithfulness` - Response faithfulness to context
  - `relevance` - Response relevance to query
  - `coherence` - Response coherence
  - `fluency` - Language fluency
  - `toxicity` - Toxicity detection
  - `bias_detection` - Bias detection across categories
  - `hallucination_score` - Hallucination detection
  - `consistency` - Response consistency
  - `groundedness` - Response groundedness to sources
  - `instruction_following` - Instruction adherence

- **RAG Metrics**
  - `context_precision` - Precision of retrieved context
  - `context_recall` - Recall of retrieved context
  - `answer_relevance` - Answer relevance to query
  - `answer_faithfulness` - Answer faithfulness to context
  - `answer_correctness` - Answer correctness
  - `context_utilization` - Context utilization efficiency
  - `retrieval_precision` - Retrieval precision
  - `retrieval_recall` - Retrieval recall
  - `retrieval_f1` - Retrieval F1 score
  - `answer_similarity` - Answer semantic similarity
  - `noise_robustness` - Robustness to noise in context

- **Fairness Metrics**
  - `demographic_parity` - Demographic parity ratio
  - `equalized_odds` - Equalized odds (TPR/FPR)
  - `equal_opportunity` - Equal opportunity (TPR)
  - `predictive_parity` - Predictive parity (PPV)
  - `calibration` - Calibration fairness
  - `disparate_impact` - Disparate impact ratio
  - `statistical_parity_difference`
  - `average_odds_difference`
  - `individual_fairness` - Individual fairness
  - `counterfactual_fairness` - Counterfactual fairness

- **Speech Metrics**
  - `wer` - Word Error Rate
  - `cer` - Character Error Rate
  - `mer` - Match Error Rate
  - `wil` - Word Information Lost
  - `wip` - Word Information Preserved
  - `phoneme_error_rate` (PER)
  - `pesq` - PESQ score
  - `stoi` - STOI score
  - `si_sdr` - SI-SDR
  - `snr` - Signal-to-Noise Ratio
  - `real_time_factor` (RTF)

- **Recommender Metrics**
  - `precision_at_k` - Precision@K
  - `recall_at_k` - Recall@K
  - `ndcg_at_k` - NDCG@K
  - `map_at_k` - MAP@K
  - `mrr` - Mean Reciprocal Rank
  - `hit_rate` - Hit Rate
  - `coverage` - Catalog coverage
  - `diversity` - Recommendation diversity
  - `novelty` - Recommendation novelty
  - `serendipity` - Recommendation serendipity
  - `personalization` - Personalization score

#### Framework Features
- **Decorators**
  - `@timed` - Execution time measurement
  - `@memoize` - Result caching
  - `@retry` - Automatic retry with backoff
  - `@logged` - Automatic logging
  - `@require_same_length` - Input validation
  - `@require_non_empty` - Non-empty validation
  - `@deprecated` - Deprecation warnings
  - `@experimental` - Experimental feature marking

- **Validators**
  - `TypeValidator` - Type checking
  - `ListValidator` - List/array validation
  - `NumericValidator` - Numeric range/type validation
  - `StringValidator` - String format validation
  - `SchemaValidator` - JSON schema validation

- **Patterns**
  - Strategy Pattern - Swappable metric implementations
  - Factory Pattern - Metric creation
  - Composite Pattern - Combined metrics
  - Builder Pattern - Fluent configuration
  - Observer Pattern - Metric event handling

- **Pipelines**
  - `Pipeline` - Fluent evaluation pipeline
  - `ConditionalPipeline` - Conditional branching
  - `CompositePipeline` - Pipeline composition
  - `ParallelPipeline` - Parallel execution

- **Functional Utilities**
  - `compose` / `pipe` - Function composition
  - `curry` / `partial` - Partial application
  - `memoize` / `memoize_with_ttl` - Caching
  - `Maybe` / `Either` / `Result` - Monads

- **Statistical Utilities**
  - Confidence intervals (bootstrap, normal, Wilson)
  - Significance tests (paired t-test, McNemar, bootstrap)
  - Multiple comparison corrections (Bonferroni, Holm)
  - Effect size calculations (Cohen's d, Cliff's delta)

- **Visualization**
  - Confusion matrix plots
  - ROC/PR curves
  - Calibration plots
  - Metric comparison charts
  - Dashboard generation

### Notes
- Zero external dependencies (pure Python)
- Python 3.8+ compatible
- 302 comprehensive tests
- 327+ public exports

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2024-01-15 | Initial release with 327+ metrics and utilities |

---

[Unreleased]: https://github.com/yourusername/pyeval/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/pyeval/releases/tag/v1.0.0
