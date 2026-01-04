"""
Recommender System Metrics - Pure Python Implementation
========================================================

Evaluation metrics for recommendation systems including
Precision@K, Recall@K, NDCG, MAP, and Hit Rate.
"""

from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import math

from pyeval.utils.math_ops import mean


# =============================================================================
# Precision@K
# =============================================================================


def precision_at_k(
    recommended: List[Any], relevant: List[Any], k: Optional[int] = None
) -> float:
    """
    Calculate Precision at K.

    Precision@K = (relevant items in top-K) / K

    Args:
        recommended: List of recommended items (ordered by relevance)
        relevant: List of relevant items (ground truth)
        k: Number of top recommendations to consider

    Returns:
        Precision@K score (0 to 1)

    Example:
        >>> recommended = [1, 2, 3, 4, 5]
        >>> relevant = [1, 3, 5, 7]
        >>> precision_at_k(recommended, relevant, k=3)
        0.6666666666666666
    """
    if k is not None:
        recommended = recommended[:k]
    else:
        k = len(recommended)

    if k == 0:
        return 0.0

    relevant_set = set(relevant)
    hits = sum(1 for item in recommended if item in relevant_set)

    return hits / k


def mean_precision_at_k(
    recommendations: List[List[Any]], relevants: List[List[Any]], k: int
) -> float:
    """
    Calculate Mean Precision at K across multiple users.

    Args:
        recommendations: List of recommendation lists (one per user)
        relevants: List of relevant item lists (one per user)
        k: Number of top recommendations to consider

    Returns:
        Mean Precision@K score
    """
    if len(recommendations) != len(relevants):
        raise ValueError("Number of users must match")

    precisions = [
        precision_at_k(rec, rel, k) for rec, rel in zip(recommendations, relevants)
    ]

    return mean(precisions) if precisions else 0.0


# =============================================================================
# Recall@K
# =============================================================================


def recall_at_k(
    recommended: List[Any], relevant: List[Any], k: Optional[int] = None
) -> float:
    """
    Calculate Recall at K.

    Recall@K = (relevant items in top-K) / (total relevant items)

    Args:
        recommended: List of recommended items (ordered by relevance)
        relevant: List of relevant items (ground truth)
        k: Number of top recommendations to consider

    Returns:
        Recall@K score (0 to 1)

    Example:
        >>> recommended = [1, 2, 3, 4, 5]
        >>> relevant = [1, 3, 5, 7]
        >>> recall_at_k(recommended, relevant, k=3)
        0.5
    """
    if k is not None:
        recommended = recommended[:k]

    if not relevant:
        return 1.0 if not recommended else 0.0

    relevant_set = set(relevant)
    hits = sum(1 for item in recommended if item in relevant_set)

    return hits / len(relevant)


def mean_recall_at_k(
    recommendations: List[List[Any]], relevants: List[List[Any]], k: int
) -> float:
    """
    Calculate Mean Recall at K across multiple users.

    Args:
        recommendations: List of recommendation lists
        relevants: List of relevant item lists
        k: Number of top recommendations to consider

    Returns:
        Mean Recall@K score
    """
    if len(recommendations) != len(relevants):
        raise ValueError("Number of users must match")

    recalls = [recall_at_k(rec, rel, k) for rec, rel in zip(recommendations, relevants)]

    return mean(recalls) if recalls else 0.0


# =============================================================================
# F1@K
# =============================================================================


def f1_at_k(
    recommended: List[Any], relevant: List[Any], k: Optional[int] = None
) -> float:
    """
    Calculate F1 score at K.

    F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

    Args:
        recommended: List of recommended items
        relevant: List of relevant items
        k: Number of top recommendations to consider

    Returns:
        F1@K score (0 to 1)
    """
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


# =============================================================================
# NDCG (Normalized Discounted Cumulative Gain)
# =============================================================================


def dcg_at_k(relevances: List[float], k: Optional[int] = None) -> float:
    """
    Calculate Discounted Cumulative Gain at K.

    DCG@K = sum(rel_i / log2(i + 1)) for i in 1..K

    Args:
        relevances: List of relevance scores (ordered by ranking)
        k: Number of top positions to consider

    Returns:
        DCG@K score
    """
    if k is not None:
        relevances = relevances[:k]

    dcg = 0.0
    for i, rel in enumerate(relevances, 1):
        dcg += rel / math.log2(i + 1)

    return dcg


def idcg_at_k(relevances: List[float], k: Optional[int] = None) -> float:
    """
    Calculate Ideal DCG at K (best possible DCG).

    Args:
        relevances: List of relevance scores
        k: Number of top positions to consider

    Returns:
        Ideal DCG@K score
    """
    # Sort relevances in descending order
    sorted_relevances = sorted(relevances, reverse=True)
    return dcg_at_k(sorted_relevances, k)


def ndcg_at_k(
    recommended: List[Any],
    relevant: Union[List[Any], Dict[Any, float]],
    k: Optional[int] = None,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    NDCG@K = DCG@K / IDCG@K

    Args:
        recommended: List of recommended items (ordered by ranking)
        relevant: List of relevant items OR dict mapping items to relevance scores
        k: Number of top positions to consider

    Returns:
        NDCG@K score (0 to 1)

    Example:
        >>> recommended = [1, 2, 3, 4, 5]
        >>> relevant = {1: 1.0, 3: 0.8, 5: 0.5, 7: 1.0}  # with graded relevance
        >>> ndcg_at_k(recommended, relevant, k=5)
    """
    if k is not None:
        recommended = recommended[:k]

    # Convert to relevance dict if list
    if isinstance(relevant, list):
        relevant_dict = {item: 1.0 for item in relevant}
    else:
        relevant_dict = relevant

    # Get relevance scores for recommended items
    relevances = [relevant_dict.get(item, 0.0) for item in recommended]

    # Calculate DCG
    dcg = dcg_at_k(relevances, k)

    # Calculate IDCG (using all relevance scores)
    all_relevances = list(relevant_dict.values())
    idcg = idcg_at_k(all_relevances, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mean_ndcg_at_k(
    recommendations: List[List[Any]],
    relevants: List[Union[List[Any], Dict[Any, float]]],
    k: int,
) -> float:
    """
    Calculate Mean NDCG at K across multiple users.

    Args:
        recommendations: List of recommendation lists
        relevants: List of relevant items (lists or dicts)
        k: Number of top positions to consider

    Returns:
        Mean NDCG@K score
    """
    if len(recommendations) != len(relevants):
        raise ValueError("Number of users must match")

    ndcgs = [ndcg_at_k(rec, rel, k) for rec, rel in zip(recommendations, relevants)]

    return mean(ndcgs) if ndcgs else 0.0


# =============================================================================
# Mean Average Precision (MAP)
# =============================================================================


def average_precision(recommended: List[Any], relevant: List[Any]) -> float:
    """
    Calculate Average Precision.

    AP = (1/R) * sum(P@k * rel(k)) for k in 1..N

    Where R = number of relevant items, P@k = precision at k,
    rel(k) = 1 if item at k is relevant, 0 otherwise.

    Args:
        recommended: List of recommended items (ordered by ranking)
        relevant: List of relevant items

    Returns:
        Average Precision score (0 to 1)

    Example:
        >>> recommended = [1, 4, 2, 5, 3]
        >>> relevant = [1, 2, 3]
        >>> average_precision(recommended, relevant)
    """
    if not relevant:
        return 1.0 if not recommended else 0.0

    relevant_set = set(relevant)

    hits = 0
    precision_sum = 0.0

    for i, item in enumerate(recommended, 1):
        if item in relevant_set:
            hits += 1
            precision_sum += hits / i

    return precision_sum / len(relevant)


def mean_average_precision(
    recommendations: List[List[Any]], relevants: List[List[Any]]
) -> float:
    """
    Calculate Mean Average Precision (MAP).

    MAP = mean(AP for each user)

    Args:
        recommendations: List of recommendation lists
        relevants: List of relevant item lists

    Returns:
        MAP score (0 to 1)

    Example:
        >>> recommendations = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        >>> relevants = [[1, 3, 5], [2, 4]]
        >>> mean_average_precision(recommendations, relevants)
    """
    if len(recommendations) != len(relevants):
        raise ValueError("Number of users must match")

    aps = [average_precision(rec, rel) for rec, rel in zip(recommendations, relevants)]

    return mean(aps) if aps else 0.0


# =============================================================================
# Hit Rate
# =============================================================================


def hit_rate(
    recommended: List[Any], relevant: List[Any], k: Optional[int] = None
) -> float:
    """
    Calculate Hit Rate at K (whether any relevant item is in top-K).

    Hit@K = 1 if any relevant item in top-K, else 0

    Args:
        recommended: List of recommended items
        relevant: List of relevant items
        k: Number of top recommendations to consider

    Returns:
        1.0 if hit, 0.0 otherwise

    Example:
        >>> recommended = [1, 2, 3, 4, 5]
        >>> relevant = [3, 7, 9]
        >>> hit_rate(recommended, relevant, k=5)
        1.0
    """
    if k is not None:
        recommended = recommended[:k]

    relevant_set = set(relevant)

    for item in recommended:
        if item in relevant_set:
            return 1.0

    return 0.0


def mean_hit_rate(
    recommendations: List[List[Any]], relevants: List[List[Any]], k: int
) -> float:
    """
    Calculate Mean Hit Rate at K across multiple users.

    Args:
        recommendations: List of recommendation lists
        relevants: List of relevant item lists
        k: Number of top recommendations to consider

    Returns:
        Mean Hit Rate (0 to 1)
    """
    if len(recommendations) != len(relevants):
        raise ValueError("Number of users must match")

    hits = [hit_rate(rec, rel, k) for rec, rel in zip(recommendations, relevants)]

    return mean(hits) if hits else 0.0


# =============================================================================
# Mean Reciprocal Rank (MRR)
# =============================================================================


def reciprocal_rank(recommended: List[Any], relevant: List[Any]) -> float:
    """
    Calculate Reciprocal Rank.

    RR = 1 / (rank of first relevant item)

    Args:
        recommended: List of recommended items
        relevant: List of relevant items

    Returns:
        Reciprocal rank (0 to 1)

    Example:
        >>> recommended = [1, 2, 3, 4, 5]
        >>> relevant = [3, 7]
        >>> reciprocal_rank(recommended, relevant)
        0.3333333333333333
    """
    relevant_set = set(relevant)

    for i, item in enumerate(recommended, 1):
        if item in relevant_set:
            return 1.0 / i

    return 0.0


def mean_reciprocal_rank(
    recommendations: List[List[Any]], relevants: List[List[Any]]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR = mean(RR for each user)

    Args:
        recommendations: List of recommendation lists
        relevants: List of relevant item lists

    Returns:
        MRR score (0 to 1)
    """
    if len(recommendations) != len(relevants):
        raise ValueError("Number of users must match")

    rrs = [reciprocal_rank(rec, rel) for rec, rel in zip(recommendations, relevants)]

    return mean(rrs) if rrs else 0.0


# =============================================================================
# Coverage Metrics
# =============================================================================


def catalog_coverage(recommendations: List[List[Any]], catalog: List[Any]) -> float:
    """
    Calculate Catalog Coverage.

    Coverage = (unique items recommended) / (total items in catalog)

    Args:
        recommendations: List of recommendation lists (all users)
        catalog: Complete catalog of items

    Returns:
        Coverage score (0 to 1)
    """
    if not catalog:
        return 0.0

    recommended_items = set()
    for rec_list in recommendations:
        recommended_items.update(rec_list)

    return len(recommended_items) / len(catalog)


def user_coverage(
    recommendations: List[List[Any]], min_recommendations: int = 1
) -> float:
    """
    Calculate User Coverage.

    Coverage = users with >= min_recommendations / total users

    Args:
        recommendations: List of recommendation lists
        min_recommendations: Minimum recommendations required

    Returns:
        User coverage score (0 to 1)
    """
    if not recommendations:
        return 0.0

    covered = sum(1 for rec in recommendations if len(rec) >= min_recommendations)
    return covered / len(recommendations)


# =============================================================================
# Diversity Metrics
# =============================================================================


def intra_list_diversity(
    recommendations: List[Any], item_features: Dict[Any, List[float]]
) -> float:
    """
    Calculate Intra-List Diversity (average pairwise distance).

    ILD = (2 / n(n-1)) * sum(dist(i, j)) for all pairs

    Args:
        recommendations: List of recommended items
        item_features: Dict mapping items to feature vectors

    Returns:
        Intra-list diversity score (0 to 1)
    """
    from pyeval.utils.math_ops import euclidean_distance

    items_with_features = [item for item in recommendations if item in item_features]

    n = len(items_with_features)
    if n < 2:
        return 0.0

    total_dist = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            feat_i = item_features[items_with_features[i]]
            feat_j = item_features[items_with_features[j]]

            dist = euclidean_distance(feat_i, feat_j)
            # Normalize to 0-1 range
            max_dist = math.sqrt(len(feat_i))
            total_dist += dist / max_dist if max_dist > 0 else dist
            count += 1

    return total_dist / count if count > 0 else 0.0


def personalization(recommendations: List[List[Any]]) -> float:
    """
    Calculate Personalization (dissimilarity between user recommendations).

    Higher = more personalized recommendations across users.

    Args:
        recommendations: List of recommendation lists (one per user)

    Returns:
        Personalization score (0 to 1)
    """
    from pyeval.utils.math_ops import jaccard_similarity

    n_users = len(recommendations)
    if n_users < 2:
        return 0.0

    total_dissim = 0.0
    count = 0

    for i in range(n_users):
        for j in range(i + 1, n_users):
            sim = jaccard_similarity(set(recommendations[i]), set(recommendations[j]))
            total_dissim += 1 - sim
            count += 1

    return total_dissim / count if count > 0 else 0.0


# =============================================================================
# Novelty and Serendipity
# =============================================================================


def novelty(
    recommendations: List[List[Any]], item_popularity: Dict[Any, float]
) -> float:
    """
    Calculate Novelty (recommending unpopular/long-tail items).

    Novelty = avg(-log2(popularity(item)))

    Args:
        recommendations: List of recommendation lists
        item_popularity: Dict mapping items to popularity scores (0-1)

    Returns:
        Novelty score (higher = more novel)
    """
    all_novelties = []

    for rec_list in recommendations:
        for item in rec_list:
            pop = item_popularity.get(item, 0.5)  # Default to medium popularity
            if pop > 0:
                novelty_score = -math.log2(pop)
            else:
                novelty_score = 10.0  # Very novel if popularity is 0
            all_novelties.append(novelty_score)

    return mean(all_novelties) if all_novelties else 0.0


# =============================================================================
# Advanced Recommender Metrics
# =============================================================================


def serendipity(
    recommended: List[Any], relevant: List[Any], expected: List[Any]
) -> float:
    """
    Calculate Serendipity (unexpected but relevant recommendations).

    Serendipity = (relevant items NOT in expected) / total recommendations

    Args:
        recommended: List of recommended items
        relevant: List of relevant items (ground truth)
        expected: List of items user would expect (e.g., most popular)

    Returns:
        Serendipity score (0 to 1)
    """
    if not recommended:
        return 0.0

    relevant_set = set(relevant)
    expected_set = set(expected)

    serendipitous = 0
    for item in recommended:
        if item in relevant_set and item not in expected_set:
            serendipitous += 1

    return serendipitous / len(recommended)


def gini_index(recommendations: List[List[Any]], catalog: List[Any]) -> float:
    """
    Calculate Gini Index (inequality of item recommendation frequency).

    Gini = 0 means perfectly uniform distribution
    Gini = 1 means maximum inequality (all recommendations to one item)

    Args:
        recommendations: List of recommendation lists
        catalog: Complete catalog of items

    Returns:
        Gini index (0 to 1)
    """
    # Count frequency of each item
    item_counts = {item: 0 for item in catalog}

    for rec_list in recommendations:
        for item in rec_list:
            if item in item_counts:
                item_counts[item] += 1

    counts = sorted(item_counts.values())
    n = len(counts)

    if n == 0:
        return 0.0

    total = sum(counts)
    if total == 0:
        return 0.0

    # Gini coefficient calculation
    gini_sum = 0.0
    for i, count in enumerate(counts):
        gini_sum += (2 * (i + 1) - n - 1) * count

    gini = gini_sum / (n * total)

    return gini


def expected_percentile_ranking(
    recommended: List[Any], relevant: Dict[Any, float]
) -> float:
    """
    Calculate Expected Percentile Ranking.

    Measures how high relevant items are ranked on average.
    EPR of 0 = all relevant items at top, EPR of 100 = at bottom

    Args:
        recommended: List of recommended items
        relevant: Dict mapping items to relevance scores

    Returns:
        Expected percentile ranking (0 to 100)
    """
    if not recommended or not relevant:
        return 50.0

    n = len(recommended)
    weighted_rank_sum = 0.0
    weight_sum = 0.0

    for i, item in enumerate(recommended):
        if item in relevant:
            percentile = (i / n) * 100
            weight = relevant[item]
            weighted_rank_sum += percentile * weight
            weight_sum += weight

    if weight_sum == 0:
        return 50.0

    return weighted_rank_sum / weight_sum


def auc_score(recommended: List[Any], relevant: List[Any], catalog: List[Any]) -> float:
    """
    Calculate Area Under ROC Curve for recommendations.

    AUC = P(random relevant item ranked higher than random irrelevant item)

    Args:
        recommended: List of recommended items
        relevant: List of relevant items
        catalog: Complete catalog of items

    Returns:
        AUC score (0 to 1)
    """
    relevant_set = set(relevant)
    recommended_set = set(recommended)

    # Items not recommended
    not_recommended = set(catalog) - recommended_set

    # Create ranking (recommended items first, then others)
    ranking = {item: i for i, item in enumerate(recommended)}
    next_rank = len(recommended)
    for item in not_recommended:
        ranking[item] = next_rank
        next_rank += 1

    # Calculate AUC
    relevant_in_catalog = [item for item in catalog if item in relevant_set]
    irrelevant_in_catalog = [item for item in catalog if item not in relevant_set]

    if not relevant_in_catalog or not irrelevant_in_catalog:
        return 0.5

    wins = 0
    total = 0

    for rel_item in relevant_in_catalog:
        for irr_item in irrelevant_in_catalog:
            rel_rank = ranking.get(rel_item, len(ranking))
            irr_rank = ranking.get(irr_item, len(ranking))

            if rel_rank < irr_rank:
                wins += 1
            elif rel_rank == irr_rank:
                wins += 0.5
            total += 1

    return wins / total if total > 0 else 0.5


def inter_list_diversity(recommendations: List[List[Any]]) -> float:
    """
    Calculate Inter-List Diversity (diversity across users).

    Measures how different recommendations are across users.

    Args:
        recommendations: List of recommendation lists (one per user)

    Returns:
        Inter-list diversity (0 to 1)
    """
    n_users = len(recommendations)
    if n_users < 2:
        return 1.0

    total_dissim = 0.0
    count = 0

    for i in range(n_users):
        for j in range(i + 1, n_users):
            set_i = set(recommendations[i])
            set_j = set(recommendations[j])

            union = len(set_i | set_j)
            intersection = len(set_i & set_j)

            if union > 0:
                dissim = 1 - (intersection / union)
            else:
                dissim = 1.0

            total_dissim += dissim
            count += 1

    return total_dissim / count if count > 0 else 1.0


def entropy_diversity(recommendations: List[List[Any]], catalog: List[Any]) -> float:
    """
    Calculate Entropy-based Diversity.

    Higher entropy = more diverse recommendations.

    Args:
        recommendations: List of recommendation lists
        catalog: Complete catalog of items

    Returns:
        Normalized entropy (0 to 1)
    """
    # Count item frequencies
    item_counts = {}
    total = 0

    for rec_list in recommendations:
        for item in rec_list:
            item_counts[item] = item_counts.get(item, 0) + 1
            total += 1

    if total == 0:
        return 0.0

    # Calculate entropy
    entropy = 0.0
    for count in item_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Normalize by max entropy
    max_entropy = math.log2(len(catalog)) if len(catalog) > 1 else 1.0

    return entropy / max_entropy if max_entropy > 0 else 0.0


def surprisal(recommended: List[Any], item_popularity: Dict[Any, float]) -> float:
    """
    Calculate Surprisal (information content of recommendations).

    Surprisal = -log2(P(item))

    Args:
        recommended: List of recommended items
        item_popularity: Dict mapping items to popularity/probability

    Returns:
        Average surprisal
    """
    if not recommended:
        return 0.0

    total_surprisal = 0.0
    count = 0

    for item in recommended:
        pop = item_popularity.get(item, 0.01)  # Default low probability
        if pop > 0:
            total_surprisal -= math.log2(pop)
            count += 1

    return total_surprisal / count if count > 0 else 0.0


def accuracy_at_k(
    recommended: List[Any], relevant: List[Any], k: int
) -> Dict[str, float]:
    """
    Calculate multiple accuracy metrics at K.

    Args:
        recommended: List of recommended items
        relevant: List of relevant items
        k: Number of top recommendations

    Returns:
        Dictionary with precision, recall, F1, and hit at K
    """
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    f1 = f1_at_k(recommended, relevant, k)
    hit = hit_rate(recommended, relevant, k)

    return {"precision": p, "recall": r, "f1": f1, "hit_rate": hit}


def ranking_correlation(
    recommended: List[Any], ground_truth_ranking: List[Any]
) -> Dict[str, float]:
    """
    Calculate ranking correlation metrics.

    Args:
        recommended: List of recommended items (in order)
        ground_truth_ranking: Ideal ranking of items

    Returns:
        Dictionary with Kendall's tau and Spearman's rho approximations
    """
    # Create rank mappings
    rec_rank = {item: i for i, item in enumerate(recommended)}
    truth_rank = {item: i for i, item in enumerate(ground_truth_ranking)}

    # Common items
    common = set(recommended) & set(ground_truth_ranking)

    if len(common) < 2:
        return {"kendall_tau": 0.0, "spearman_rho": 0.0}

    # Get ranks for common items
    rec_ranks = [rec_rank[item] for item in common]
    truth_ranks = [truth_rank[item] for item in common]

    # Kendall's tau: count concordant and discordant pairs
    n = len(common)
    concordant = 0
    discordant = 0

    items_list = list(common)
    for i in range(n):
        for j in range(i + 1, n):
            rec_diff = rec_rank[items_list[i]] - rec_rank[items_list[j]]
            truth_diff = truth_rank[items_list[i]] - truth_rank[items_list[j]]

            if rec_diff * truth_diff > 0:
                concordant += 1
            elif rec_diff * truth_diff < 0:
                discordant += 1

    pairs = n * (n - 1) / 2
    kendall = (concordant - discordant) / pairs if pairs > 0 else 0.0

    # Spearman's rho: correlation of ranks
    mean_rec = sum(rec_ranks) / n
    mean_truth = sum(truth_ranks) / n

    cov = sum(
        (rec_ranks[i] - mean_rec) * (truth_ranks[i] - mean_truth) for i in range(n)
    )
    std_rec = math.sqrt(sum((r - mean_rec) ** 2 for r in rec_ranks))
    std_truth = math.sqrt(sum((r - mean_truth) ** 2 for r in truth_ranks))

    if std_rec > 0 and std_truth > 0:
        spearman = cov / (std_rec * std_truth)
    else:
        spearman = 0.0

    return {"kendall_tau": kendall, "spearman_rho": spearman}


def beyond_accuracy_metrics(
    recommendations: List[List[Any]],
    relevants: List[List[Any]],
    catalog: List[Any],
    item_popularity: Dict[Any, float],
    k: int = 10,
) -> Dict[str, float]:
    """
    Calculate beyond-accuracy metrics for recommendation evaluation.

    Args:
        recommendations: List of recommendation lists
        relevants: List of relevant item lists
        catalog: Complete catalog of items
        item_popularity: Dict mapping items to popularity
        k: Number of top recommendations

    Returns:
        Dictionary with coverage, diversity, novelty metrics
    """
    return {
        "catalog_coverage": catalog_coverage(recommendations, catalog),
        "personalization": personalization(recommendations),
        "novelty": novelty(recommendations, item_popularity),
        "entropy_diversity": entropy_diversity(recommendations, catalog),
        "gini_index": gini_index(recommendations, catalog),
        "inter_list_diversity": inter_list_diversity(recommendations),
    }


# =============================================================================
# Recommender Metrics Class
# =============================================================================


@dataclass
class RecommenderMetrics:
    """Container for recommender system evaluation metrics."""

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    map_score: float = 0.0
    hit_rate: float = 0.0
    mrr: float = 0.0

    @classmethod
    def compute(
        cls, recommended: List[Any], relevant: List[Any], k: int = 10
    ) -> "RecommenderMetrics":
        """
        Compute all recommender metrics for a single user.

        Args:
            recommended: List of recommended items
            relevant: List of relevant items
            k: Number of top recommendations to consider

        Returns:
            RecommenderMetrics object
        """
        return cls(
            precision_at_k=precision_at_k(recommended, relevant, k),
            recall_at_k=recall_at_k(recommended, relevant, k),
            ndcg_at_k=ndcg_at_k(recommended, relevant, k),
            map_score=average_precision(recommended, relevant),
            hit_rate=hit_rate(recommended, relevant, k),
            mrr=reciprocal_rank(recommended, relevant),
        )

    @classmethod
    def compute_batch(
        cls, recommendations: List[List[Any]], relevants: List[List[Any]], k: int = 10
    ) -> "RecommenderMetrics":
        """
        Compute mean recommender metrics across multiple users.

        Args:
            recommendations: List of recommendation lists
            relevants: List of relevant item lists
            k: Number of top recommendations to consider

        Returns:
            RecommenderMetrics object with mean scores
        """
        return cls(
            precision_at_k=mean_precision_at_k(recommendations, relevants, k),
            recall_at_k=mean_recall_at_k(recommendations, relevants, k),
            ndcg_at_k=mean_ndcg_at_k(recommendations, relevants, k),
            map_score=mean_average_precision(recommendations, relevants),
            hit_rate=mean_hit_rate(recommendations, relevants, k),
            mrr=mean_reciprocal_rank(recommendations, relevants),
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "map": self.map_score,
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
        }
