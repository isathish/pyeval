"""
RAG (Retrieval-Augmented Generation) Metrics - Pure Python Implementation
==========================================================================

Evaluation metrics for RAG systems including context relevance, 
answer correctness, retrieval quality, and groundedness.
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import Counter
import re
import math

from pyeval.utils.text_ops import (
    tokenize,
    ngrams,
    sentence_split,
    normalize_text,
    remove_stopwords,
    text_similarity,
    STOPWORDS
)
from pyeval.utils.math_ops import mean, jaccard_similarity


# =============================================================================
# Context Relevance
# =============================================================================

def context_relevance(question: str, contexts: List[str],
                      method: str = 'keyword') -> Dict[str, Any]:
    """
    Calculate relevance of retrieved contexts to the question.
    
    Args:
        question: User question
        contexts: List of retrieved context passages
        method: 'keyword' or 'semantic' relevance calculation
        
    Returns:
        Dictionary with relevance scores
        
    Example:
        >>> q = "What is machine learning?"
        >>> contexts = ["Machine learning is a branch of AI...", "Python is a language..."]
        >>> result = context_relevance(q, contexts)
        >>> result['overall_relevance']
    """
    if not contexts:
        return {
            'overall_relevance': 0.0,
            'context_scores': [],
            'relevant_contexts': 0
        }
    
    # Extract question keywords
    q_tokens = set(tokenize(question, lowercase=True, remove_punct=True))
    q_tokens -= STOPWORDS
    
    # Extract question entities (capitalized words)
    q_entities = set(re.findall(r'\b[A-Z][a-z]+\b', question))
    
    context_scores = []
    
    for ctx in contexts:
        # Tokenize context
        ctx_tokens = set(tokenize(ctx, lowercase=True, remove_punct=True))
        ctx_tokens -= STOPWORDS
        
        # Calculate keyword overlap
        if q_tokens:
            keyword_score = len(q_tokens & ctx_tokens) / len(q_tokens)
        else:
            keyword_score = 0.0
        
        # Calculate entity overlap
        ctx_entities = set(re.findall(r'\b[A-Z][a-z]+\b', ctx))
        if q_entities:
            entity_score = len(q_entities & ctx_entities) / len(q_entities)
        else:
            entity_score = keyword_score
        
        # N-gram overlap for better precision
        q_bigrams = set(ngrams(list(q_tokens), 2))
        ctx_bigrams = set(ngrams(list(ctx_tokens), 2))
        
        if q_bigrams:
            bigram_score = len(q_bigrams & ctx_bigrams) / len(q_bigrams)
        else:
            bigram_score = keyword_score
        
        # Combined score
        relevance = (keyword_score * 0.4 + entity_score * 0.3 + bigram_score * 0.3)
        context_scores.append(relevance)
    
    # Count relevant contexts (above threshold)
    threshold = 0.2
    relevant_count = sum(1 for s in context_scores if s >= threshold)
    
    # Overall relevance (weighted by position - earlier contexts matter more)
    position_weights = [1.0 / (i + 1) for i in range(len(contexts))]
    weight_sum = sum(position_weights)
    
    weighted_relevance = sum(s * w for s, w in zip(context_scores, position_weights))
    overall_relevance = weighted_relevance / weight_sum if weight_sum > 0 else 0.0
    
    return {
        'overall_relevance': overall_relevance,
        'context_scores': context_scores,
        'relevant_contexts': relevant_count,
        'total_contexts': len(contexts),
        'relevance_ratio': relevant_count / len(contexts) if contexts else 0.0
    }


def context_precision(question: str, contexts: List[str],
                      relevant_threshold: float = 0.3) -> float:
    """
    Calculate precision of retrieved contexts.
    
    Precision = relevant contexts / total retrieved contexts
    
    Args:
        question: User question
        contexts: Retrieved contexts
        relevant_threshold: Threshold for considering a context relevant
        
    Returns:
        Context precision score (0 to 1)
    """
    if not contexts:
        return 0.0
    
    result = context_relevance(question, contexts)
    relevant = sum(1 for s in result['context_scores'] if s >= relevant_threshold)
    
    return relevant / len(contexts)


def context_recall(question: str, contexts: List[str],
                   ground_truth_contexts: List[str]) -> float:
    """
    Calculate recall of retrieved contexts against ground truth.
    
    Recall = (retrieved ∩ relevant) / relevant
    
    Args:
        question: User question
        contexts: Retrieved contexts
        ground_truth_contexts: Known relevant contexts
        
    Returns:
        Context recall score (0 to 1)
    """
    if not ground_truth_contexts:
        return 1.0 if not contexts else 0.0
    
    # Normalize contexts for comparison
    normalized_retrieved = set(normalize_text(c) for c in contexts)
    normalized_truth = set(normalize_text(c) for c in ground_truth_contexts)
    
    # Count matches (using text similarity for fuzzy matching)
    matches = 0
    for truth_ctx in ground_truth_contexts:
        for ret_ctx in contexts:
            sim = text_similarity(truth_ctx, ret_ctx, method='jaccard')
            if sim >= 0.7:  # High similarity threshold
                matches += 1
                break
    
    return matches / len(ground_truth_contexts)


# =============================================================================
# Answer Correctness
# =============================================================================

def answer_correctness(answer: str, ground_truth: str,
                       question: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate correctness of generated answer against ground truth.
    
    Args:
        answer: Generated answer
        ground_truth: Ground truth answer
        question: Optional question for context
        
    Returns:
        Dictionary with correctness metrics
        
    Example:
        >>> answer = "Paris is the capital of France."
        >>> truth = "The capital of France is Paris."
        >>> result = answer_correctness(answer, truth)
        >>> result['correctness']
    """
    # Tokenize
    answer_tokens = set(tokenize(answer, lowercase=True, remove_punct=True))
    answer_tokens -= STOPWORDS
    
    truth_tokens = set(tokenize(ground_truth, lowercase=True, remove_punct=True))
    truth_tokens -= STOPWORDS
    
    # Token overlap (F1-style)
    if not answer_tokens or not truth_tokens:
        return {
            'correctness': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    overlap = len(answer_tokens & truth_tokens)
    precision = overlap / len(answer_tokens)
    recall = overlap / len(truth_tokens)
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    # Entity alignment
    answer_entities = set(re.findall(r'\b[A-Z][a-z]+\b', answer))
    truth_entities = set(re.findall(r'\b[A-Z][a-z]+\b', ground_truth))
    
    if truth_entities:
        entity_recall = len(answer_entities & truth_entities) / len(truth_entities)
    else:
        entity_recall = 1.0
    
    # Number alignment
    answer_numbers = set(re.findall(r'\b\d+\.?\d*\b', answer))
    truth_numbers = set(re.findall(r'\b\d+\.?\d*\b', ground_truth))
    
    if truth_numbers:
        number_recall = len(answer_numbers & truth_numbers) / len(truth_numbers)
    else:
        number_recall = 1.0
    
    # Combined correctness
    correctness = (f1 * 0.5 + entity_recall * 0.3 + number_recall * 0.2)
    
    return {
        'correctness': correctness,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'entity_recall': entity_recall,
        'number_recall': number_recall
    }


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts.
    
    Uses bag-of-words with TF weighting as a simple approximation.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0 to 1)
    """
    tokens1 = tokenize(text1, lowercase=True, remove_punct=True)
    tokens2 = tokenize(text2, lowercase=True, remove_punct=True)
    
    tokens1 = remove_stopwords(tokens1)
    tokens2 = remove_stopwords(tokens2)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Build vocabulary
    vocab = list(set(tokens1) | set(tokens2))
    
    # Create TF vectors
    freq1 = Counter(tokens1)
    freq2 = Counter(tokens2)
    
    vec1 = [freq1.get(w, 0) for w in vocab]
    vec2 = [freq2.get(w, 0) for w in vocab]
    
    # Calculate cosine similarity
    from pyeval.utils.math_ops import cosine_similarity as cos_sim
    return cos_sim(vec1, vec2)


# =============================================================================
# Retrieval Metrics
# =============================================================================

def retrieval_precision(retrieved: List[str], relevant: List[str],
                        k: Optional[int] = None) -> float:
    """
    Calculate retrieval precision.
    
    Precision@K = relevant in top-K / K
    
    Args:
        retrieved: List of retrieved documents
        relevant: List of relevant documents (ground truth)
        k: Consider only top-K results (None for all)
        
    Returns:
        Precision score (0 to 1)
    """
    if k is not None:
        retrieved = retrieved[:k]
    
    if not retrieved:
        return 0.0
    
    relevant_set = set(normalize_text(r) for r in relevant)
    
    hits = 0
    for doc in retrieved:
        normalized_doc = normalize_text(doc)
        # Check for exact or fuzzy match
        if normalized_doc in relevant_set:
            hits += 1
        else:
            # Fuzzy matching
            for rel_doc in relevant:
                if text_similarity(doc, rel_doc, method='jaccard') >= 0.7:
                    hits += 1
                    break
    
    return hits / len(retrieved)


def retrieval_recall(retrieved: List[str], relevant: List[str],
                     k: Optional[int] = None) -> float:
    """
    Calculate retrieval recall.
    
    Recall@K = relevant in top-K / total relevant
    
    Args:
        retrieved: List of retrieved documents
        relevant: List of relevant documents (ground truth)
        k: Consider only top-K results (None for all)
        
    Returns:
        Recall score (0 to 1)
    """
    if not relevant:
        return 1.0 if not retrieved else 0.0
    
    if k is not None:
        retrieved = retrieved[:k]
    
    retrieved_set = set(normalize_text(r) for r in retrieved)
    
    hits = 0
    for doc in relevant:
        normalized_doc = normalize_text(doc)
        if normalized_doc in retrieved_set:
            hits += 1
        else:
            # Fuzzy matching
            for ret_doc in retrieved:
                if text_similarity(doc, ret_doc, method='jaccard') >= 0.7:
                    hits += 1
                    break
    
    return hits / len(relevant)


def retrieval_f1(retrieved: List[str], relevant: List[str],
                 k: Optional[int] = None) -> float:
    """
    Calculate retrieval F1 score.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        retrieved: List of retrieved documents
        relevant: List of relevant documents (ground truth)
        k: Consider only top-K results (None for all)
        
    Returns:
        F1 score (0 to 1)
    """
    precision = retrieval_precision(retrieved, relevant, k)
    recall = retrieval_recall(retrieved, relevant, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def mean_reciprocal_rank(retrieved_lists: List[List[str]],
                         relevant_lists: List[List[str]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = average of 1/rank of first relevant result
    
    Args:
        retrieved_lists: List of retrieved document lists (one per query)
        relevant_lists: List of relevant document lists (one per query)
        
    Returns:
        MRR score (0 to 1)
    """
    if len(retrieved_lists) != len(relevant_lists):
        raise ValueError("Number of queries must match")
    
    reciprocal_ranks = []
    
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        relevant_set = set(normalize_text(r) for r in relevant)
        
        rr = 0.0
        for rank, doc in enumerate(retrieved, 1):
            normalized_doc = normalize_text(doc)
            
            # Check for match
            found = normalized_doc in relevant_set
            if not found:
                # Fuzzy matching
                for rel_doc in relevant:
                    if text_similarity(doc, rel_doc, method='jaccard') >= 0.7:
                        found = True
                        break
            
            if found:
                rr = 1.0 / rank
                break
        
        reciprocal_ranks.append(rr)
    
    return mean(reciprocal_ranks) if reciprocal_ranks else 0.0


# =============================================================================
# Groundedness
# =============================================================================

def groundedness_score(answer: str, contexts: List[str]) -> Dict[str, Any]:
    """
    Calculate groundedness - how well the answer is grounded in the contexts.
    
    Args:
        answer: Generated answer
        contexts: Source contexts
        
    Returns:
        Dictionary with groundedness metrics
        
    Example:
        >>> answer = "Python was created by Guido van Rossum."
        >>> contexts = ["Python is a programming language created by Guido van Rossum in 1991."]
        >>> result = groundedness_score(answer, contexts)
        >>> result['groundedness']
    """
    if not contexts:
        return {
            'groundedness': 0.0,
            'grounded_sentences': 0,
            'total_sentences': 0,
            'ungrounded': []
        }
    
    # Combine contexts
    combined_context = ' '.join(contexts)
    context_tokens = set(tokenize(combined_context, lowercase=True, remove_punct=True))
    context_tokens -= STOPWORDS
    
    # Split answer into sentences
    sentences = sentence_split(answer)
    
    if not sentences:
        return {
            'groundedness': 1.0,
            'grounded_sentences': 0,
            'total_sentences': 0,
            'ungrounded': []
        }
    
    grounded_count = 0
    ungrounded = []
    
    for sent in sentences:
        sent_tokens = set(tokenize(sent, lowercase=True, remove_punct=True))
        sent_tokens -= STOPWORDS
        
        if not sent_tokens:
            grounded_count += 1
            continue
        
        # Calculate grounding score
        overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
        
        # Check entity grounding
        sent_entities = set(re.findall(r'\b[A-Z][a-z]+\b', sent))
        context_entities = set(re.findall(r'\b[A-Z][a-z]+\b', combined_context))
        
        if sent_entities:
            entity_grounding = len(sent_entities & context_entities) / len(sent_entities)
        else:
            entity_grounding = 1.0
        
        grounding = (overlap * 0.6 + entity_grounding * 0.4)
        
        if grounding >= 0.3:
            grounded_count += 1
        else:
            ungrounded.append({
                'sentence': sent,
                'grounding_score': grounding
            })
    
    groundedness = grounded_count / len(sentences)
    
    return {
        'groundedness': groundedness,
        'grounded_sentences': grounded_count,
        'total_sentences': len(sentences),
        'ungrounded': ungrounded
    }


# =============================================================================
# Answer Faithfulness (for RAG)
# =============================================================================

def rag_faithfulness(answer: str, contexts: List[str]) -> Dict[str, Any]:
    """
    Calculate faithfulness of answer to retrieved contexts.
    
    Different from LLM faithfulness - specifically for RAG evaluation.
    
    Args:
        answer: Generated answer
        contexts: Retrieved contexts
        
    Returns:
        Dictionary with faithfulness metrics
    """
    # Extract claims from answer
    sentences = sentence_split(answer)
    
    if not sentences or not contexts:
        return {
            'faithfulness': 1.0 if not sentences else 0.0,
            'faithful_claims': 0,
            'total_claims': 0,
            'details': []
        }
    
    combined_context = ' '.join(contexts)
    
    faithful_count = 0
    details = []
    
    for sent in sentences:
        # Check support in contexts
        max_support = 0.0
        
        for ctx in contexts:
            # Calculate support
            sent_tokens = set(tokenize(sent, lowercase=True, remove_punct=True))
            sent_tokens -= STOPWORDS
            
            ctx_tokens = set(tokenize(ctx, lowercase=True, remove_punct=True))
            ctx_tokens -= STOPWORDS
            
            if sent_tokens:
                support = len(sent_tokens & ctx_tokens) / len(sent_tokens)
                max_support = max(max_support, support)
        
        is_faithful = max_support >= 0.4
        
        if is_faithful:
            faithful_count += 1
        
        details.append({
            'sentence': sent,
            'support': max_support,
            'faithful': is_faithful
        })
    
    faithfulness = faithful_count / len(sentences)
    
    return {
        'faithfulness': faithfulness,
        'faithful_claims': faithful_count,
        'total_claims': len(sentences),
        'details': details
    }


# =============================================================================
# Noise Robustness
# =============================================================================

def noise_robustness(question: str, answer: str,
                     contexts: List[str], noise_contexts: List[str]) -> Dict[str, float]:
    """
    Evaluate robustness of RAG to noisy/irrelevant contexts.
    
    Args:
        question: User question
        answer: Generated answer
        contexts: Relevant contexts
        noise_contexts: Irrelevant/noisy contexts that were also retrieved
        
    Returns:
        Dictionary with robustness metrics
    """
    # Check if answer uses relevant contexts
    relevant_usage = groundedness_score(answer, contexts)
    
    # Check if answer uses noise contexts
    noise_usage = groundedness_score(answer, noise_contexts)
    
    # Good RAG should use relevant contexts and ignore noise
    signal_score = relevant_usage['groundedness']
    noise_score = noise_usage['groundedness']
    
    # Robustness = high signal, low noise
    robustness = signal_score * (1 - noise_score)
    
    return {
        'robustness': robustness,
        'signal_usage': signal_score,
        'noise_usage': noise_score,
        'signal_noise_ratio': signal_score / max(0.01, noise_score)
    }


def context_entity_recall(contexts: List[str], ground_truth: str) -> Dict[str, float]:
    """
    Calculate entity recall from contexts to ground truth.
    
    Measures how many key entities from the ground truth are present in contexts.
    
    Args:
        contexts: Retrieved context passages
        ground_truth: Ground truth answer containing key entities
        
    Returns:
        Dictionary with entity recall metrics
    """
    # Extract entities (capitalized words, numbers, quoted strings)
    def extract_entities(text: str) -> Set[str]:
        entities = set()
        # Proper nouns (capitalized words)
        entities.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        # Numbers with units
        entities.update(re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:km|m|kg|lb|%|year|day|hour)s?)?\b', text))
        # Quoted strings
        entities.update(re.findall(r'"([^"]+)"', text))
        entities.update(re.findall(r"'([^']+)'", text))
        return entities
    
    truth_entities = extract_entities(ground_truth)
    
    if not truth_entities:
        return {
            'entity_recall': 1.0,
            'found_entities': [],
            'missing_entities': [],
            'total_entities': 0
        }
    
    combined_context = ' '.join(contexts)
    context_entities = extract_entities(combined_context)
    context_text_lower = combined_context.lower()
    
    found = []
    missing = []
    
    for entity in truth_entities:
        if entity in context_entities or entity.lower() in context_text_lower:
            found.append(entity)
        else:
            missing.append(entity)
    
    recall = len(found) / len(truth_entities)
    
    return {
        'entity_recall': recall,
        'found_entities': found,
        'missing_entities': missing,
        'total_entities': len(truth_entities)
    }


def answer_attribution(answer: str, contexts: List[str]) -> Dict[str, Any]:
    """
    Attribute parts of the answer to specific contexts.
    
    Identifies which context(s) each sentence in the answer comes from.
    
    Args:
        answer: Generated answer
        contexts: Retrieved contexts
        
    Returns:
        Dictionary with attribution details
    """
    sentences = sentence_split(answer)
    
    attributions = []
    attributed_count = 0
    
    for sent in sentences:
        sent_tokens = set(tokenize(sent, lowercase=True, remove_punct=True))
        sent_tokens -= STOPWORDS
        
        best_context = -1
        best_score = 0.0
        
        for i, ctx in enumerate(contexts):
            ctx_tokens = set(tokenize(ctx, lowercase=True, remove_punct=True))
            ctx_tokens -= STOPWORDS
            
            if sent_tokens and ctx_tokens:
                overlap = len(sent_tokens & ctx_tokens) / len(sent_tokens)
                if overlap > best_score:
                    best_score = overlap
                    best_context = i
        
        is_attributed = best_score >= 0.3
        if is_attributed:
            attributed_count += 1
        
        attributions.append({
            'sentence': sent,
            'context_index': best_context if is_attributed else None,
            'confidence': best_score,
            'is_attributed': is_attributed
        })
    
    attribution_rate = attributed_count / len(sentences) if sentences else 0.0
    
    # Which contexts were used
    used_contexts = set(a['context_index'] for a in attributions if a['context_index'] is not None)
    
    return {
        'attribution_rate': attribution_rate,
        'attributions': attributions,
        'used_context_indices': list(used_contexts),
        'context_utilization': len(used_contexts) / len(contexts) if contexts else 0.0
    }


def context_utilization(answer: str, contexts: List[str]) -> Dict[str, float]:
    """
    Measure how much of each context is utilized in the answer.
    
    Args:
        answer: Generated answer
        contexts: Retrieved contexts
        
    Returns:
        Dictionary with utilization metrics per context
    """
    answer_tokens = set(tokenize(answer, lowercase=True, remove_punct=True))
    answer_tokens -= STOPWORDS
    
    utilizations = []
    
    for i, ctx in enumerate(contexts):
        ctx_tokens = set(tokenize(ctx, lowercase=True, remove_punct=True))
        ctx_tokens -= STOPWORDS
        
        if ctx_tokens:
            # What fraction of context appears in answer
            utilization = len(ctx_tokens & answer_tokens) / len(ctx_tokens)
        else:
            utilization = 0.0
        
        utilizations.append({
            'context_index': i,
            'utilization': utilization
        })
    
    # Overall metrics
    avg_utilization = mean([u['utilization'] for u in utilizations]) if utilizations else 0.0
    max_utilization = max([u['utilization'] for u in utilizations]) if utilizations else 0.0
    utilized_contexts = sum(1 for u in utilizations if u['utilization'] > 0.1)
    
    return {
        'average_utilization': avg_utilization,
        'max_utilization': max_utilization,
        'utilized_context_count': utilized_contexts,
        'total_contexts': len(contexts),
        'per_context': utilizations
    }


def question_answer_relevance(question: str, answer: str) -> Dict[str, float]:
    """
    Measure relevance of answer to the question.
    
    Args:
        question: User question
        answer: Generated answer
        
    Returns:
        Dictionary with relevance metrics
    """
    # Extract question type and key terms
    q_lower = question.lower()
    
    # Question type detection
    q_types = {
        'what': ['what'],
        'who': ['who'],
        'when': ['when'],
        'where': ['where'],
        'why': ['why'],
        'how': ['how'],
        'which': ['which'],
        'yes_no': ['is ', 'are ', 'do ', 'does ', 'can ', 'will ', 'should ']
    }
    
    question_type = 'other'
    for qtype, markers in q_types.items():
        if any(q_lower.startswith(m) or f' {m}' in q_lower for m in markers):
            question_type = qtype
            break
    
    # Answer type matching
    answer_lower = answer.lower()
    type_match_score = 1.0
    
    if question_type == 'yes_no':
        if any(w in answer_lower for w in ['yes', 'no', 'correct', 'incorrect', 'true', 'false']):
            type_match_score = 1.0
        else:
            type_match_score = 0.7
    elif question_type == 'when':
        if re.search(r'\b\d{4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', answer_lower):
            type_match_score = 1.0
        else:
            type_match_score = 0.6
    elif question_type == 'where':
        # Check for location indicators
        if re.search(r'\b(?:in|at|near|from|to)\s+[A-Z]', answer):
            type_match_score = 1.0
        else:
            type_match_score = 0.7
    
    # Keyword coverage
    q_tokens = set(tokenize(question, lowercase=True, remove_punct=True))
    q_tokens -= STOPWORDS
    q_tokens -= {'what', 'who', 'when', 'where', 'why', 'how', 'which', 'is', 'are', 'do', 'does'}
    
    a_tokens = set(tokenize(answer, lowercase=True, remove_punct=True))
    
    keyword_coverage = len(q_tokens & a_tokens) / len(q_tokens) if q_tokens else 0.5
    
    # Overall relevance
    relevance = type_match_score * 0.4 + keyword_coverage * 0.6
    
    return {
        'relevance': relevance,
        'question_type': question_type,
        'type_match_score': type_match_score,
        'keyword_coverage': keyword_coverage
    }


def rag_pipeline_score(question: str, contexts: List[str], answer: str,
                       ground_truth: Optional[str] = None) -> Dict[str, float]:
    """
    Comprehensive RAG pipeline evaluation.
    
    Combines multiple metrics into an overall score.
    
    Args:
        question: User question
        contexts: Retrieved contexts
        answer: Generated answer
        ground_truth: Ground truth answer (optional)
        
    Returns:
        Dictionary with all RAG metrics and overall score
    """
    # Retrieval quality
    ctx_rel = context_relevance(question, contexts)
    
    # Generation quality
    ground = groundedness_score(answer, contexts)
    faith = rag_faithfulness(answer, contexts)
    qa_rel = question_answer_relevance(question, answer)
    
    # Attribution
    attrib = answer_attribution(answer, contexts)
    
    # Correctness (if ground truth available)
    if ground_truth:
        correct = answer_correctness(answer, ground_truth)
        correctness = correct['correctness']
    else:
        correctness = None
    
    # Calculate overall score
    scores = [
        ctx_rel['overall_relevance'] * 0.2,
        ground['groundedness'] * 0.25,
        faith['faithfulness'] * 0.25,
        qa_rel['relevance'] * 0.2,
        attrib['attribution_rate'] * 0.1
    ]
    
    overall = sum(scores)
    
    result = {
        'overall_score': overall,
        'retrieval_score': ctx_rel['overall_relevance'],
        'groundedness': ground['groundedness'],
        'faithfulness': faith['faithfulness'],
        'relevance': qa_rel['relevance'],
        'attribution_rate': attrib['attribution_rate']
    }
    
    if correctness is not None:
        result['correctness'] = correctness
    
    return result


# =============================================================================
# RAG Metrics Class
# =============================================================================

@dataclass
class RAGMetrics:
    """Container for RAG evaluation metrics."""
    
    context_relevance: float = 0.0
    answer_correctness: float = 0.0
    faithfulness: float = 0.0
    groundedness: float = 0.0
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    
    @classmethod
    def compute(cls, question: str, answer: str,
                contexts: List[str],
                ground_truth_answer: Optional[str] = None,
                ground_truth_contexts: Optional[List[str]] = None) -> 'RAGMetrics':
        """
        Compute all RAG metrics.
        
        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth_answer: Ground truth answer (optional)
            ground_truth_contexts: Ground truth relevant contexts (optional)
            
        Returns:
            RAGMetrics object
        """
        # Context relevance
        ctx_rel = context_relevance(question, contexts)
        
        # Faithfulness and groundedness
        faith = rag_faithfulness(answer, contexts)
        ground = groundedness_score(answer, contexts)
        
        # Answer correctness (if ground truth available)
        ans_correct = 0.0
        if ground_truth_answer:
            correct_result = answer_correctness(answer, ground_truth_answer)
            ans_correct = correct_result['correctness']
        
        # Retrieval metrics (if ground truth contexts available)
        ret_prec = 0.0
        ret_rec = 0.0
        if ground_truth_contexts:
            ret_prec = retrieval_precision(contexts, ground_truth_contexts)
            ret_rec = retrieval_recall(contexts, ground_truth_contexts)
        
        return cls(
            context_relevance=ctx_rel['overall_relevance'],
            answer_correctness=ans_correct,
            faithfulness=faith['faithfulness'],
            groundedness=ground['groundedness'],
            retrieval_precision=ret_prec,
            retrieval_recall=ret_rec
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'context_relevance': self.context_relevance,
            'answer_correctness': self.answer_correctness,
            'faithfulness': self.faithfulness,
            'groundedness': self.groundedness,
            'retrieval_precision': self.retrieval_precision,
            'retrieval_recall': self.retrieval_recall
        }
    
    def overall_score(self) -> float:
        """Calculate overall RAG quality score."""
        weights = {
            'context_relevance': 0.15,
            'answer_correctness': 0.25,
            'faithfulness': 0.25,
            'groundedness': 0.2,
            'retrieval_precision': 0.075,
            'retrieval_recall': 0.075
        }
        
        score = sum(getattr(self, metric) * weight 
                   for metric, weight in weights.items())
        
        return score
