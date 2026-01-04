"""
LLM Evaluation Metrics - Pure Python Implementation
====================================================

Evaluation metrics for Large Language Model outputs including
hallucination detection, answer relevancy, faithfulness, and toxicity.
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
    get_word_frequencies,
    text_similarity,
    STOPWORDS
)
from pyeval.utils.math_ops import (
    cosine_similarity,
    jaccard_similarity,
    mean,
    softmax
)


# =============================================================================
# Toxicity Detection
# =============================================================================

# Toxic/profane word patterns (expandable)
TOXIC_PATTERNS = {
    'profanity': [
        r'\b(damn|hell|crap|shit|fuck|ass|bitch)\w*\b',
    ],
    'hate_speech': [
        r'\b(hate|kill|die|murder|attack)\s+(all|every|those)\b',
        r'\b(racist|sexist|homophobic)\b',
    ],
    'threat': [
        r'\b(i\'ll|gonna|will)\s+(kill|hurt|destroy|attack)\b',
        r'\b(threat|threaten)\w*\b',
    ],
    'insult': [
        r'\b(idiot|stupid|dumb|moron|loser|pathetic)\b',
    ],
    'harassment': [
        r'\b(harass|bully|stalk)\w*\b',
    ]
}

# Positive/neutral indicators
NEUTRAL_PATTERNS = [
    r'\bhowever\b',
    r'\bon the other hand\b',
    r'\bin my opinion\b',
    r'\bi think\b',
    r'\bperhaps\b',
    r'\bmight\b',
]


def toxicity_score(text: str, detailed: bool = False) -> Dict[str, Any]:
    """
    Calculate toxicity score for text.
    
    This is a rule-based approach. For production use, consider
    using a trained model.
    
    Args:
        text: Text to analyze
        detailed: Return detailed breakdown by category
        
    Returns:
        Dictionary with toxicity score and details
        
    Example:
        >>> result = toxicity_score("You are an idiot!")
        >>> result['toxicity']  # Higher score = more toxic
    """
    text_lower = text.lower()
    
    category_scores = {}
    total_matches = 0
    
    for category, patterns in TOXIC_PATTERNS.items():
        matches = 0
        for pattern in patterns:
            matches += len(re.findall(pattern, text_lower))
        category_scores[category] = matches
        total_matches += matches
    
    # Count neutral indicators (reduce toxicity score)
    neutral_count = sum(len(re.findall(p, text_lower)) for p in NEUTRAL_PATTERNS)
    
    # Calculate base toxicity
    word_count = len(tokenize(text, remove_punct=True))
    if word_count == 0:
        base_toxicity = 0.0
    else:
        # Toxicity ratio adjusted by text length
        base_toxicity = min(1.0, total_matches / max(1, word_count / 10))
    
    # Apply neutral modifier
    neutral_modifier = max(0.5, 1.0 - neutral_count * 0.1)
    toxicity = base_toxicity * neutral_modifier
    
    result = {
        'toxicity': toxicity,
        'is_toxic': toxicity > 0.3,
        'total_matches': total_matches
    }
    
    if detailed:
        result['category_scores'] = category_scores
        result['neutral_indicators'] = neutral_count
    
    return result


def detect_toxic_spans(text: str) -> List[Dict[str, Any]]:
    """
    Detect toxic spans in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of toxic spans with positions and categories
    """
    text_lower = text.lower()
    spans = []
    
    for category, patterns in TOXIC_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower):
                spans.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'category': category
                })
    
    # Sort by position
    spans.sort(key=lambda x: x['start'])
    
    return spans


# =============================================================================
# Hallucination Detection
# =============================================================================

def _extract_claims(text: str) -> List[str]:
    """
    Extract factual claims from text.
    
    Args:
        text: Text to extract claims from
        
    Returns:
        List of claim strings
    """
    # Split into sentences
    sentences = sentence_split(text)
    
    claims = []
    
    # Patterns indicating factual claims
    claim_patterns = [
        r'^[A-Z][^.!?]*\b(is|are|was|were|has|have|had)\b[^.!?]*[.!?]?$',
        r'^[A-Z][^.!?]*\b(contains?|includes?|consists?)\b[^.!?]*[.!?]?$',
        r'^[A-Z][^.!?]*\b(created?|founded?|discovered?|invented?)\b[^.!?]*[.!?]?$',
        r'^[A-Z][^.!?]*\b(located?|situated?|found)\b[^.!?]*[.!?]?$',
        r'\b\d+\b',  # Contains numbers (often factual)
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        
        # Check if sentence contains factual claim patterns
        for pattern in claim_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                claims.append(sentence)
                break
    
    return claims


def _check_claim_support(claim: str, context: str) -> float:
    """
    Check if a claim is supported by context.
    
    Args:
        claim: Claim to check
        context: Context/source to check against
        
    Returns:
        Support score (0 to 1)
    """
    # Tokenize and get key terms
    claim_tokens = set(tokenize(claim, lowercase=True, remove_punct=True))
    claim_tokens -= STOPWORDS
    
    context_tokens = set(tokenize(context, lowercase=True, remove_punct=True))
    context_tokens -= STOPWORDS
    
    if not claim_tokens:
        return 1.0  # Empty claim is trivially supported
    
    # Calculate overlap
    overlap = len(claim_tokens & context_tokens)
    support = overlap / len(claim_tokens)
    
    # Boost if key entities match
    # Look for capitalized words (potential entities)
    claim_entities = set(re.findall(r'\b[A-Z][a-z]+\b', claim))
    context_entities = set(re.findall(r'\b[A-Z][a-z]+\b', context))
    
    if claim_entities:
        entity_overlap = len(claim_entities & context_entities) / len(claim_entities)
        support = (support + entity_overlap) / 2
    
    return support


def hallucination_score(response: str, context: str,
                        threshold: float = 0.5) -> Dict[str, Any]:
    """
    Calculate hallucination score for LLM response.
    
    Hallucination = claims in response not supported by context.
    
    Args:
        response: LLM response to evaluate
        context: Source/context that response should be grounded in
        threshold: Support threshold below which a claim is hallucinated
        
    Returns:
        Dictionary with hallucination score and details
        
    Example:
        >>> context = "The Eiffel Tower is located in Paris, France."
        >>> response = "The Eiffel Tower is in Paris and was built in 1850."
        >>> result = hallucination_score(response, context)
        >>> result['hallucination_rate']  # Lower is better
    """
    claims = _extract_claims(response)
    
    if not claims:
        return {
            'hallucination_score': 0.0,
            'hallucination_rate': 0.0,
            'total_claims': 0,
            'hallucinated_claims': [],
            'supported_claims': []
        }
    
    hallucinated = []
    supported = []
    
    for claim in claims:
        support = _check_claim_support(claim, context)
        
        if support < threshold:
            hallucinated.append({
                'claim': claim,
                'support_score': support
            })
        else:
            supported.append({
                'claim': claim,
                'support_score': support
            })
    
    hallucination_rate = len(hallucinated) / len(claims)
    
    return {
        'hallucination_score': hallucination_rate,
        'hallucination_rate': hallucination_rate,
        'total_claims': len(claims),
        'hallucinated_claims': hallucinated,
        'supported_claims': supported
    }


def factual_consistency(response: str, source: str) -> float:
    """
    Calculate factual consistency between response and source.
    
    Args:
        response: Generated response
        source: Source document
        
    Returns:
        Consistency score (0 to 1, higher is more consistent)
    """
    result = hallucination_score(response, source)
    return 1.0 - result['hallucination_rate']


# =============================================================================
# Answer Relevancy
# =============================================================================

def _extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Extract top keywords from text using TF-based scoring.
    
    Args:
        text: Text to extract keywords from
        top_k: Number of keywords to return
        
    Returns:
        List of keywords
    """
    tokens = tokenize(text, lowercase=True, remove_punct=True)
    tokens = remove_stopwords(tokens)
    
    # Count frequencies
    freq = Counter(tokens)
    
    # Get top keywords
    top_words = [word for word, _ in freq.most_common(top_k)]
    
    return top_words


def answer_relevancy(question: str, answer: str,
                     context: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate answer relevancy score.
    
    Measures how relevant the answer is to the question.
    
    Args:
        question: Question asked
        answer: Answer provided
        context: Optional context (if available)
        
    Returns:
        Dictionary with relevancy scores
        
    Example:
        >>> q = "What is the capital of France?"
        >>> a = "The capital of France is Paris."
        >>> result = answer_relevancy(q, a)
        >>> result['relevancy']  # Higher is more relevant
    """
    # Extract question keywords (what the question is about)
    q_keywords = set(_extract_keywords(question, top_k=5))
    
    # Tokenize answer
    answer_tokens = set(tokenize(answer, lowercase=True, remove_punct=True))
    answer_tokens -= STOPWORDS
    
    # Calculate keyword coverage
    if q_keywords:
        keyword_coverage = len(q_keywords & answer_tokens) / len(q_keywords)
    else:
        keyword_coverage = 0.5
    
    # Calculate semantic overlap (using Jaccard as proxy)
    q_tokens = set(tokenize(question, lowercase=True, remove_punct=True)) - STOPWORDS
    semantic_overlap = jaccard_similarity(q_tokens, answer_tokens)
    
    # Check for question type alignment
    question_types = {
        'what': ['is', 'are', 'means', 'definition'],
        'who': ['person', 'name', 'people', 'individual'],
        'when': ['date', 'year', 'time', 'century', 'period'],
        'where': ['location', 'place', 'country', 'city'],
        'why': ['because', 'reason', 'due to', 'cause'],
        'how': ['method', 'process', 'way', 'steps'],
    }
    
    type_bonus = 0.0
    q_lower = question.lower()
    a_lower = answer.lower()
    
    for q_type, indicators in question_types.items():
        if q_lower.startswith(q_type):
            for indicator in indicators:
                if indicator in a_lower:
                    type_bonus = 0.1
                    break
            break
    
    # Final relevancy score
    relevancy = (keyword_coverage * 0.4 + semantic_overlap * 0.4 + 
                 type_bonus + 0.1)  # Base score
    relevancy = min(1.0, relevancy)
    
    result = {
        'relevancy': relevancy,
        'keyword_coverage': keyword_coverage,
        'semantic_overlap': semantic_overlap,
        'type_alignment': type_bonus > 0
    }
    
    # Include context relevancy if provided
    if context:
        context_tokens = set(tokenize(context, lowercase=True, remove_punct=True))
        context_tokens -= STOPWORDS
        context_relevancy = jaccard_similarity(answer_tokens, context_tokens)
        result['context_relevancy'] = context_relevancy
    
    return result


# =============================================================================
# Faithfulness
# =============================================================================

def faithfulness_score(response: str, source: str,
                       granularity: str = 'sentence') -> Dict[str, Any]:
    """
    Calculate faithfulness score - how faithful the response is to the source.
    
    Args:
        response: Generated response
        source: Source document/context
        granularity: 'sentence' or 'claim' level analysis
        
    Returns:
        Dictionary with faithfulness scores
        
    Example:
        >>> source = "Python was created by Guido van Rossum in 1991."
        >>> response = "Python is a programming language created by Guido van Rossum."
        >>> result = faithfulness_score(response, source)
        >>> result['faithfulness']  # Higher is more faithful
    """
    if granularity == 'sentence':
        units = sentence_split(response)
    else:
        units = _extract_claims(response)
    
    if not units:
        return {
            'faithfulness': 1.0,
            'total_units': 0,
            'faithful_units': 0,
            'unfaithful_units': []
        }
    
    source_lower = source.lower()
    source_tokens = set(tokenize(source, lowercase=True, remove_punct=True))
    source_tokens -= STOPWORDS
    
    faithful_count = 0
    unfaithful = []
    
    for unit in units:
        unit_tokens = set(tokenize(unit, lowercase=True, remove_punct=True))
        unit_tokens -= STOPWORDS
        
        if not unit_tokens:
            faithful_count += 1
            continue
        
        # Calculate support
        overlap = len(unit_tokens & source_tokens) / len(unit_tokens)
        
        # Check for entity alignment
        unit_entities = set(re.findall(r'\b[A-Z][a-z]+\b', unit))
        source_entities = set(re.findall(r'\b[A-Z][a-z]+\b', source))
        
        entity_support = 1.0
        if unit_entities:
            entity_support = len(unit_entities & source_entities) / len(unit_entities)
        
        # Combined faithfulness for this unit
        unit_faithfulness = (overlap * 0.6 + entity_support * 0.4)
        
        if unit_faithfulness >= 0.4:
            faithful_count += 1
        else:
            unfaithful.append({
                'text': unit,
                'faithfulness': unit_faithfulness
            })
    
    faithfulness = faithful_count / len(units)
    
    return {
        'faithfulness': faithfulness,
        'total_units': len(units),
        'faithful_units': faithful_count,
        'unfaithful_units': unfaithful
    }


# =============================================================================
# Coherence
# =============================================================================

def coherence_score(text: str) -> Dict[str, float]:
    """
    Calculate coherence score for text.
    
    Measures logical flow and consistency of the text.
    
    Args:
        text: Text to evaluate
        
    Returns:
        Dictionary with coherence metrics
    """
    sentences = sentence_split(text)
    
    if len(sentences) < 2:
        return {
            'coherence': 1.0,
            'local_coherence': 1.0,
            'global_coherence': 1.0,
            'num_sentences': len(sentences)
        }
    
    # Local coherence: similarity between adjacent sentences
    local_scores = []
    for i in range(len(sentences) - 1):
        sim = text_similarity(sentences[i], sentences[i + 1], method='jaccard')
        local_scores.append(sim)
    
    local_coherence = mean(local_scores) if local_scores else 1.0
    
    # Global coherence: consistency of topics throughout
    all_keywords = []
    for sent in sentences:
        keywords = _extract_keywords(sent, top_k=3)
        all_keywords.extend(keywords)
    
    keyword_freq = Counter(all_keywords)
    if keyword_freq:
        # Check if there are recurring themes
        max_freq = max(keyword_freq.values())
        recurring_ratio = sum(1 for f in keyword_freq.values() if f > 1) / len(keyword_freq)
        global_coherence = min(1.0, (max_freq / len(sentences)) * 0.5 + recurring_ratio * 0.5)
    else:
        global_coherence = 0.5
    
    # Transition word bonus
    transition_words = [
        'however', 'therefore', 'furthermore', 'moreover', 'additionally',
        'consequently', 'meanwhile', 'similarly', 'in contrast', 'for example',
        'specifically', 'in addition', 'as a result', 'on the other hand'
    ]
    
    text_lower = text.lower()
    transition_count = sum(1 for tw in transition_words if tw in text_lower)
    transition_bonus = min(0.2, transition_count * 0.05)
    
    # Final coherence
    coherence = min(1.0, (local_coherence * 0.4 + global_coherence * 0.4 + 
                         transition_bonus + 0.2))  # Base score
    
    return {
        'coherence': coherence,
        'local_coherence': local_coherence,
        'global_coherence': global_coherence,
        'num_sentences': len(sentences),
        'transition_words': transition_count
    }


# =============================================================================
# Consistency Score
# =============================================================================

def consistency_score(responses: List[str], question: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate consistency score across multiple responses to the same prompt.
    
    Measures how consistent the model's responses are when asked the same question.
    
    Args:
        responses: Multiple responses to the same prompt
        question: Original question/prompt (optional, for context)
        
    Returns:
        Dictionary with consistency metrics
        
    Example:
        >>> responses = [
        ...     "Python was created in 1991.",
        ...     "Python was created in 1991 by Guido van Rossum.",
        ...     "Python was invented in 1991."
        ... ]
        >>> result = consistency_score(responses)
        >>> result['consistency']  # Higher = more consistent
    """
    if len(responses) < 2:
        return {
            'consistency': 1.0,
            'factual_consistency': 1.0,
            'semantic_similarity': 1.0,
            'lexical_overlap': 1.0
        }
    
    # Tokenize all responses
    response_tokens = [
        set(tokenize(r, lowercase=True, remove_punct=True)) - STOPWORDS 
        for r in responses
    ]
    
    # Lexical overlap (Jaccard similarity across all pairs)
    lexical_scores = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            if response_tokens[i] and response_tokens[j]:
                jaccard = len(response_tokens[i] & response_tokens[j]) / len(response_tokens[i] | response_tokens[j])
                lexical_scores.append(jaccard)
    
    lexical_overlap = mean(lexical_scores) if lexical_scores else 0.0
    
    # Semantic similarity using text_similarity
    semantic_scores = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            sim = text_similarity(responses[i], responses[j], method='jaccard')
            semantic_scores.append(sim)
    
    semantic_similarity = mean(semantic_scores) if semantic_scores else 0.0
    
    # Factual consistency - check for contradicting facts
    # Extract numbers, dates, proper nouns from each response
    def extract_facts(text: str) -> Set[str]:
        facts = set()
        # Numbers
        facts.update(re.findall(r'\b\d+(?:\.\d+)?\b', text))
        # Dates
        facts.update(re.findall(r'\b\d{4}\b', text))
        facts.update(re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d+', text))
        # Proper nouns (capitalized words)
        facts.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        return facts
    
    response_facts = [extract_facts(r) for r in responses]
    
    # Measure fact agreement
    if any(response_facts):
        all_facts = set().union(*response_facts)
        if all_facts:
            fact_agreement_scores = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    common_facts = response_facts[i] & response_facts[j]
                    all_pair_facts = response_facts[i] | response_facts[j]
                    if all_pair_facts:
                        fact_agreement_scores.append(len(common_facts) / len(all_pair_facts))
            factual_consistency = mean(fact_agreement_scores) if fact_agreement_scores else 0.5
        else:
            factual_consistency = 1.0
    else:
        factual_consistency = 1.0
    
    # Overall consistency
    consistency = (factual_consistency * 0.4 + semantic_similarity * 0.35 + lexical_overlap * 0.25)
    
    return {
        'consistency': consistency,
        'factual_consistency': factual_consistency,
        'semantic_similarity': semantic_similarity,
        'lexical_overlap': lexical_overlap
    }


# =============================================================================
# Completeness
# =============================================================================

def completeness_score(question: str, answer: str,
                       expected_aspects: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate completeness score - how completely the answer addresses the question.
    
    Args:
        question: Question asked
        answer: Answer provided
        expected_aspects: Optional list of expected aspects to cover
        
    Returns:
        Dictionary with completeness metrics
    """
    # Extract question aspects
    q_tokens = set(tokenize(question, lowercase=True, remove_punct=True))
    q_tokens -= STOPWORDS
    
    a_tokens = set(tokenize(answer, lowercase=True, remove_punct=True))
    a_tokens -= STOPWORDS
    
    # Coverage of question terms
    term_coverage = len(q_tokens & a_tokens) / len(q_tokens) if q_tokens else 1.0
    
    # Check for expected aspects
    aspect_coverage = 1.0
    covered_aspects = []
    missing_aspects = []
    
    if expected_aspects:
        for aspect in expected_aspects:
            aspect_tokens = set(tokenize(aspect, lowercase=True, remove_punct=True))
            if aspect_tokens & a_tokens:
                covered_aspects.append(aspect)
            else:
                missing_aspects.append(aspect)
        
        aspect_coverage = len(covered_aspects) / len(expected_aspects)
    
    # Check answer length adequacy
    answer_length = len(a_tokens)
    question_complexity = len(q_tokens)
    
    # Simple heuristic: longer questions might need longer answers
    expected_min_length = max(5, question_complexity * 2)
    length_adequacy = min(1.0, answer_length / expected_min_length)
    
    # Final completeness
    completeness = (term_coverage * 0.3 + aspect_coverage * 0.4 + 
                   length_adequacy * 0.3)
    
    return {
        'completeness': completeness,
        'term_coverage': term_coverage,
        'aspect_coverage': aspect_coverage,
        'length_adequacy': length_adequacy,
        'covered_aspects': covered_aspects,
        'missing_aspects': missing_aspects
    }


# =============================================================================
# Fluency
# =============================================================================

def fluency_score(text: str) -> Dict[str, float]:
    """
    Calculate fluency score for text.
    
    Measures grammatical correctness and readability.
    
    Args:
        text: Text to evaluate
        
    Returns:
        Dictionary with fluency metrics
    """
    sentences = sentence_split(text)
    
    if not sentences:
        return {
            'fluency': 0.0,
            'avg_sentence_length': 0,
            'vocabulary_richness': 0.0,
            'readability': 0.0
        }
    
    # Average sentence length (proxy for complexity)
    tokens = tokenize(text, remove_punct=True)
    avg_sent_len = len(tokens) / len(sentences)
    
    # Ideal sentence length is 15-20 words
    if 10 <= avg_sent_len <= 25:
        length_score = 1.0
    elif avg_sent_len < 10:
        length_score = avg_sent_len / 10
    else:
        length_score = max(0.3, 1.0 - (avg_sent_len - 25) / 50)
    
    # Vocabulary richness (type-token ratio)
    unique_tokens = set(t.lower() for t in tokens)
    if tokens:
        ttr = len(unique_tokens) / len(tokens)
    else:
        ttr = 0
    
    vocabulary_richness = min(1.0, ttr * 2)  # Scale up since TTR is usually < 0.5
    
    # Check for common grammar issues (simple heuristics)
    text_lower = text.lower()
    
    grammar_issues = 0
    
    # Double spaces
    grammar_issues += len(re.findall(r'  +', text))
    
    # Missing punctuation at end
    for sent in sentences:
        if sent and sent[-1] not in '.!?':
            grammar_issues += 1
    
    # Repeated words
    for i in range(len(tokens) - 1):
        if tokens[i].lower() == tokens[i + 1].lower():
            grammar_issues += 1
    
    grammar_score = max(0.0, 1.0 - grammar_issues * 0.1)
    
    # Final fluency
    fluency = (length_score * 0.3 + vocabulary_richness * 0.3 + 
               grammar_score * 0.4)
    
    return {
        'fluency': fluency,
        'avg_sentence_length': avg_sent_len,
        'vocabulary_richness': vocabulary_richness,
        'grammar_score': grammar_score,
        'readability': length_score
    }


def bias_detection(text: str, bias_categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detect potential bias in text.
    
    This is a rule-based approach using lexicon matching.
    For production use, consider using a trained model.
    
    Args:
        text: Text to analyze
        bias_categories: Categories to check (None for all)
        
    Returns:
        Dictionary with bias scores and detected terms
    """
    # Bias lexicons (simplified)
    BIAS_LEXICONS = {
        'gender': {
            'stereotypes': [
                r'\b(all|every)\s+(men|women|boys|girls)\s+(are|should|must)\b',
                r'\b(naturally|inherently)\s+(better|worse)\b',
                r'\b(real|true)\s+(man|woman)\b',
            ],
            'exclusionary': [
                r'\b(mankind|manpower|chairman)\b',
            ]
        },
        'age': {
            'stereotypes': [
                r'\b(old|elderly)\s+people\s+(can\'t|cannot|don\'t|unable)\b',
                r'\b(young|millennial)\s+(lazy|entitled|irresponsible)\b',
            ]
        },
        'racial': {
            'stereotypes': [
                r'\b(all|every|typical)\s+[A-Z][a-z]+(s)?\s+(are|always)\b',
            ]
        },
        'disability': {
            'ableist': [
                r'\b(retarded|crippled|handicapped|lame|dumb)\b',
            ]
        }
    }
    
    text_lower = text.lower()
    
    if bias_categories is None:
        bias_categories = list(BIAS_LEXICONS.keys())
    
    results = {}
    total_score = 0.0
    detected_terms = []
    
    for category in bias_categories:
        if category not in BIAS_LEXICONS:
            continue
        
        category_matches = 0
        for subcat, patterns in BIAS_LEXICONS[category].items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                category_matches += len(matches)
                detected_terms.extend(matches)
        
        results[category] = category_matches
        total_score += category_matches
    
    # Normalize score
    word_count = len(tokenize(text))
    bias_score = min(1.0, total_score / max(1, word_count / 20))
    
    return {
        'bias_score': bias_score,
        'has_bias': bias_score > 0.1,
        'category_scores': results,
        'detected_terms': detected_terms
    }


def instruction_following_score(instruction: str, response: str) -> Dict[str, float]:
    """
    Evaluate how well a response follows the given instruction.
    
    Args:
        instruction: Instruction/prompt given
        response: Model's response
        
    Returns:
        Dictionary with instruction following metrics
    """
    # Extract instruction elements
    instruction_lower = instruction.lower()
    
    # Check for length requirements
    length_keywords = {
        'brief': (1, 50),
        'short': (1, 100),
        'concise': (1, 100),
        'detailed': (100, float('inf')),
        'comprehensive': (150, float('inf')),
        'elaborate': (150, float('inf')),
    }
    
    response_words = len(tokenize(response))
    length_score = 1.0
    
    for keyword, (min_len, max_len) in length_keywords.items():
        if keyword in instruction_lower:
            if min_len <= response_words <= max_len:
                length_score = 1.0
            elif response_words < min_len:
                length_score = response_words / min_len
            else:
                length_score = max(0.5, max_len / response_words)
            break
    
    # Check for format requirements
    format_keywords = {
        'list': r'(?:^\s*[-*•]\s+|\d+\.\s+)',
        'bullet': r'(?:^\s*[-*•]\s+)',
        'numbered': r'(?:^\s*\d+[.)]\s+)',
        'steps': r'(?:step\s*\d+|^\s*\d+[.)]\s+)',
    }
    
    format_score = 1.0
    for keyword, pattern in format_keywords.items():
        if keyword in instruction_lower:
            matches = re.findall(pattern, response, re.MULTILINE)
            if matches:
                format_score = 1.0
            else:
                format_score = 0.5
            break
    
    # Check for content requirements (keywords from instruction)
    instruction_tokens = set(tokenize(instruction, lowercase=True, remove_punct=True))
    instruction_tokens -= STOPWORDS
    instruction_tokens -= {'please', 'write', 'explain', 'describe', 'tell', 'give'}
    
    response_tokens = set(tokenize(response, lowercase=True, remove_punct=True))
    
    if instruction_tokens:
        topic_coverage = len(instruction_tokens & response_tokens) / len(instruction_tokens)
    else:
        topic_coverage = 0.5
    
    # Overall score
    overall = (length_score * 0.3 + format_score * 0.3 + topic_coverage * 0.4)
    
    return {
        'instruction_following': overall,
        'length_score': length_score,
        'format_score': format_score,
        'topic_coverage': topic_coverage
    }


def multi_turn_coherence(conversation: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Evaluate coherence across multiple turns in a conversation.
    
    Args:
        conversation: List of turns, each with 'role' and 'content' keys
        
    Returns:
        Dictionary with multi-turn coherence metrics
    """
    if len(conversation) < 2:
        return {
            'coherence': 1.0,
            'topic_consistency': 1.0,
            'reference_tracking': 1.0
        }
    
    # Extract assistant responses
    responses = [turn['content'] for turn in conversation if turn.get('role') == 'assistant']
    
    if len(responses) < 2:
        return {
            'coherence': 1.0,
            'topic_consistency': 1.0,
            'reference_tracking': 1.0
        }
    
    # Topic consistency: measure keyword overlap between consecutive responses
    topic_scores = []
    for i in range(1, len(responses)):
        prev_tokens = set(tokenize(responses[i-1], lowercase=True, remove_punct=True)) - STOPWORDS
        curr_tokens = set(tokenize(responses[i], lowercase=True, remove_punct=True)) - STOPWORDS
        
        if prev_tokens and curr_tokens:
            overlap = len(prev_tokens & curr_tokens) / len(prev_tokens | curr_tokens)
            topic_scores.append(overlap)
    
    topic_consistency = mean(topic_scores) if topic_scores else 1.0
    
    # Reference tracking: check if pronouns/references are resolvable
    ref_patterns = [r'\b(it|this|that|these|those|they|them)\b', r'\b(he|she|him|her)\b']
    ref_score = 1.0
    
    for i, response in enumerate(responses[1:], 1):
        prev_content = ' '.join(t['content'] for t in conversation[:i*2])
        
        for pattern in ref_patterns:
            refs = re.findall(pattern, response.lower())
            if refs:
                # Simple check: was there context provided?
                if len(tokenize(prev_content)) < 10:
                    ref_score -= 0.1
    
    ref_score = max(0.0, ref_score)
    
    # Overall coherence
    coherence = (topic_consistency * 0.6 + ref_score * 0.4)
    
    return {
        'coherence': coherence,
        'topic_consistency': topic_consistency,
        'reference_tracking': ref_score
    }


def summarization_quality(source: str, summary: str) -> Dict[str, float]:
    """
    Evaluate quality of text summarization.
    
    Args:
        source: Source text
        summary: Generated summary
        
    Returns:
        Dictionary with summarization quality metrics
    """
    from pyeval.nlp import rouge_score
    
    source_tokens = tokenize(source, lowercase=True, remove_punct=True)
    summary_tokens = tokenize(summary, lowercase=True, remove_punct=True)
    
    # Compression ratio
    compression = len(summary_tokens) / len(source_tokens) if source_tokens else 0
    
    # Coverage: what fraction of source content is in summary
    source_set = set(source_tokens) - STOPWORDS
    summary_set = set(summary_tokens) - STOPWORDS
    
    coverage = len(source_set & summary_set) / len(source_set) if source_set else 0
    
    # ROUGE scores
    rouge = rouge_score(source, summary)
    
    # Information density (unique content words per token)
    density = len(summary_set) / len(summary_tokens) if summary_tokens else 0
    
    # Redundancy (repeated content)
    summary_counts = Counter(summary_tokens)
    redundancy = sum(1 for c in summary_counts.values() if c > 1) / len(summary_set) if summary_set else 0
    
    # Overall quality
    quality = (rouge['rouge1']['f1'] * 0.3 + coverage * 0.25 + 
               density * 0.2 + (1 - redundancy) * 0.15 + 
               (1 if 0.1 < compression < 0.5 else 0.5) * 0.1)
    
    return {
        'quality': quality,
        'compression_ratio': compression,
        'coverage': coverage,
        'density': density,
        'redundancy': redundancy,
        'rouge1_f1': rouge['rouge1']['f1'],
        'rougeL_f1': rouge['rougeL']['f1']
    }


def response_diversity(responses: List[str]) -> Dict[str, float]:
    """
    Evaluate diversity across multiple generated responses.
    
    Args:
        responses: List of generated responses
        
    Returns:
        Dictionary with diversity metrics
    """
    if len(responses) < 2:
        return {
            'diversity': 1.0,
            'self_bleu': 0.0,
            'distinct_1': 1.0,
            'distinct_2': 1.0
        }
    
    # Self-BLEU (lower is more diverse)
    from pyeval.nlp import bleu_score
    
    self_bleu_scores = []
    for i, resp in enumerate(responses):
        refs = [responses[j] for j in range(len(responses)) if j != i]
        result = bleu_score(refs, resp, smoothing=True)
        self_bleu_scores.append(result['bleu'])
    
    avg_self_bleu = mean(self_bleu_scores)
    
    # Distinct-N across all responses
    all_tokens = []
    for resp in responses:
        all_tokens.extend(tokenize(resp, lowercase=True, remove_punct=True))
    
    all_bigrams = list(zip(all_tokens[:-1], all_tokens[1:]))
    
    distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
    
    # Overall diversity (higher is better)
    diversity = (1 - avg_self_bleu) * 0.5 + distinct_1 * 0.25 + distinct_2 * 0.25
    
    return {
        'diversity': diversity,
        'self_bleu': avg_self_bleu,
        'distinct_1': distinct_1,
        'distinct_2': distinct_2
    }


# =============================================================================
# LLM Metrics Class
# =============================================================================

@dataclass
class LLMMetrics:
    """Container for LLM evaluation metrics."""
    
    hallucination: float = 0.0
    relevancy: float = 0.0
    faithfulness: float = 0.0
    coherence: float = 0.0
    toxicity: float = 0.0
    fluency: float = 0.0
    
    @classmethod
    def compute(cls, response: str, 
                question: Optional[str] = None,
                context: Optional[str] = None) -> 'LLMMetrics':
        """
        Compute all LLM metrics.
        
        Args:
            response: LLM response to evaluate
            question: Question asked (optional)
            context: Source context (optional)
            
        Returns:
            LLMMetrics object
        """
        # Always compute
        coherence_result = coherence_score(response)
        toxicity_result = toxicity_score(response)
        fluency_result = fluency_score(response)
        
        hallucination = 0.0
        faithfulness = 0.0
        relevancy = 0.0
        
        # Compute if context available
        if context:
            hall_result = hallucination_score(response, context)
            hallucination = hall_result['hallucination_score']
            
            faith_result = faithfulness_score(response, context)
            faithfulness = faith_result['faithfulness']
        
        # Compute if question available
        if question:
            rel_result = answer_relevancy(question, response, context)
            relevancy = rel_result['relevancy']
        
        return cls(
            hallucination=hallucination,
            relevancy=relevancy,
            faithfulness=faithfulness,
            coherence=coherence_result['coherence'],
            toxicity=toxicity_result['toxicity'],
            fluency=fluency_result['fluency']
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'hallucination': self.hallucination,
            'relevancy': self.relevancy,
            'faithfulness': self.faithfulness,
            'coherence': self.coherence,
            'toxicity': self.toxicity,
            'fluency': self.fluency
        }
    
    def overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate overall quality score.
        
        Args:
            weights: Custom weights for each metric
            
        Returns:
            Overall score (0 to 1, higher is better)
        """
        if weights is None:
            weights = {
                'hallucination': -0.25,  # Negative because lower is better
                'relevancy': 0.25,
                'faithfulness': 0.2,
                'coherence': 0.15,
                'toxicity': -0.05,  # Negative because lower is better
                'fluency': 0.1
            }
        
        score = 0.0
        for metric, weight in weights.items():
            value = getattr(self, metric, 0.0)
            if weight < 0:
                # For metrics where lower is better
                score += abs(weight) * (1 - value)
            else:
                score += weight * value
        
        return max(0.0, min(1.0, score))
