"""
Text Operations - Pure Python Implementation
============================================

Text processing functions for NLP evaluation without external dependencies.
"""

import re
import string
from typing import List, Set, Tuple, Optional
from collections import Counter


# Common English stopwords
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "might",
    "more",
    "most",
    "must",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "on",
    "or",
    "other",
    "our",
    "ours",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
}

# Porter Stemmer suffix rules (simplified)
PORTER_SUFFIXES = [
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("bli", "ble"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
    ("logi", "log"),
    ("ness", ""),
    ("ment", ""),
    ("ing", ""),
    ("ed", ""),
    ("er", ""),
    ("ly", ""),
    ("s", ""),
]

# Simple lemmatization rules
LEMMA_RULES = {
    "running": "run",
    "ran": "run",
    "runs": "run",
    "swimming": "swim",
    "swam": "swim",
    "swims": "swim",
    "better": "good",
    "best": "good",
    "worse": "bad",
    "worst": "bad",
    "children": "child",
    "men": "man",
    "women": "woman",
    "mice": "mouse",
    "feet": "foot",
    "teeth": "tooth",
    "geese": "goose",
    "leaves": "leaf",
    "lives": "life",
    "wolves": "wolf",
    "knives": "knife",
    "wives": "wife",
    "was": "be",
    "were": "be",
    "been": "be",
    "being": "be",
    "am": "be",
    "is": "be",
    "are": "be",
    "has": "have",
    "had": "have",
    "having": "have",
    "does": "do",
    "did": "do",
    "doing": "do",
    "done": "do",
    "goes": "go",
    "went": "go",
    "gone": "go",
    "going": "go",
    "says": "say",
    "said": "say",
    "saying": "say",
    "makes": "make",
    "made": "make",
    "making": "make",
    "takes": "take",
    "took": "take",
    "taken": "take",
    "taking": "take",
    "comes": "come",
    "came": "come",
    "coming": "come",
    "sees": "see",
    "saw": "see",
    "seen": "see",
    "seeing": "see",
    "knows": "know",
    "knew": "know",
    "known": "know",
    "knowing": "know",
    "thinks": "think",
    "thought": "think",
    "thinking": "think",
    "gives": "give",
    "gave": "give",
    "given": "give",
    "giving": "give",
    "gets": "get",
    "got": "get",
    "gotten": "get",
    "getting": "get",
}


def tokenize(
    text: str, lowercase: bool = True, remove_punct: bool = False
) -> List[str]:
    """
    Tokenize text into words.

    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_punct: Remove punctuation tokens

    Returns:
        List of tokens

    Example:
        >>> tokenize("Hello, World!")
        ['hello', ',', 'world', '!']
    """
    if lowercase:
        text = text.lower()

    # Split on whitespace and punctuation while keeping punctuation as tokens
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text)

    if remove_punct:
        tokens = [t for t in tokens if t not in string.punctuation]

    return tokens


def ngrams(sequence: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Generate n-grams from a sequence.

    Args:
        sequence: List of tokens
        n: Size of n-grams

    Returns:
        List of n-gram tuples

    Example:
        >>> ngrams(['a', 'b', 'c', 'd'], 2)
        [('a', 'b'), ('b', 'c'), ('c', 'd')]
    """
    if n <= 0:
        raise ValueError("n must be positive")

    if len(sequence) < n:
        return []

    return [tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)]


def word_ngrams(text: str, n: int, lowercase: bool = True) -> List[Tuple[str, ...]]:
    """
    Generate word n-grams from text.

    Args:
        text: Input text
        n: Size of n-grams
        lowercase: Convert to lowercase

    Returns:
        List of word n-gram tuples

    Example:
        >>> word_ngrams("The quick brown fox", 2)
        [('the', 'quick'), ('quick', 'brown'), ('brown', 'fox')]
    """
    tokens = tokenize(text, lowercase=lowercase, remove_punct=True)
    return ngrams(tokens, n)


def char_ngrams(text: str, n: int, include_spaces: bool = False) -> List[str]:
    """
    Generate character n-grams from text.

    Args:
        text: Input text
        n: Size of n-grams
        include_spaces: Include space characters

    Returns:
        List of character n-grams

    Example:
        >>> char_ngrams("hello", 2)
        ['he', 'el', 'll', 'lo']
    """
    if not include_spaces:
        text = text.replace(" ", "")

    if len(text) < n:
        return []

    return [text[i : i + n] for i in range(len(text) - n + 1)]


def stem(word: str) -> str:
    """
    Apply Porter-like stemming to a word.

    Args:
        word: Input word

    Returns:
        Stemmed word

    Example:
        >>> stem("running")
        'run'
    """
    word = word.lower()

    # Skip short words
    if len(word) <= 2:
        return word

    # Apply suffix rules
    for suffix, replacement in PORTER_SUFFIXES:
        if word.endswith(suffix):
            new_word = word[: -len(suffix)] + replacement
            if len(new_word) >= 2:
                return new_word

    return word


def lemmatize(word: str) -> str:
    """
    Apply simple lemmatization to a word.

    Args:
        word: Input word

    Returns:
        Lemmatized word

    Example:
        >>> lemmatize("running")
        'run'
    """
    word_lower = word.lower()

    # Check dictionary first
    if word_lower in LEMMA_RULES:
        return LEMMA_RULES[word_lower]

    # Apply common rules
    # Plural nouns ending in -ies -> -y
    if word_lower.endswith("ies") and len(word_lower) > 4:
        return word_lower[:-3] + "y"

    # Plural nouns ending in -es
    if word_lower.endswith("es") and len(word_lower) > 3:
        if (
            word_lower.endswith("sses")
            or word_lower.endswith("xes")
            or word_lower.endswith("ches")
            or word_lower.endswith("shes")
        ):
            return word_lower[:-2]

    # Plural nouns ending in -s
    if (
        word_lower.endswith("s")
        and not word_lower.endswith("ss")
        and len(word_lower) > 3
    ):
        return word_lower[:-1]

    # Verbs ending in -ing
    if word_lower.endswith("ing") and len(word_lower) > 5:
        # Check for doubled consonant (running -> run)
        if len(word_lower) > 6 and word_lower[-4] == word_lower[-5]:
            return word_lower[:-4]
        # Check for silent e (making -> make)
        base = word_lower[:-3]
        if base + "e" in LEMMA_RULES.values() or len(base) > 2:
            return base + "e" if base[-1] not in "aeiou" else base
        return base

    # Past tense ending in -ed
    if word_lower.endswith("ed") and len(word_lower) > 4:
        # Check for doubled consonant
        if len(word_lower) > 5 and word_lower[-3] == word_lower[-4]:
            return word_lower[:-3]
        return word_lower[:-2]

    return word_lower


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from text.

    Args:
        text: Input text

    Returns:
        Text without punctuation

    Example:
        >>> remove_punctuation("Hello, World!")
        'Hello World'
    """
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stopwords(
    tokens: List[str], stopwords: Optional[Set[str]] = None
) -> List[str]:
    """
    Remove stopwords from a list of tokens.

    Args:
        tokens: List of tokens
        stopwords: Set of stopwords (uses default if None)

    Returns:
        Tokens without stopwords

    Example:
        >>> remove_stopwords(['the', 'quick', 'brown', 'fox'])
        ['quick', 'brown', 'fox']
    """
    if stopwords is None:
        stopwords = STOPWORDS

    return [t for t in tokens if t.lower() not in stopwords]


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_numbers: bool = False,
    remove_extra_spaces: bool = True,
) -> str:
    """
    Normalize text with various options.

    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_punct: Remove punctuation
        remove_numbers: Remove numeric characters
        remove_extra_spaces: Collapse multiple spaces

    Returns:
        Normalized text

    Example:
        >>> normalize_text("  Hello,   World!  123  ")
        'hello world 123'
    """
    if lowercase:
        text = text.lower()

    if remove_punct:
        text = remove_punctuation(text)

    if remove_numbers:
        text = re.sub(r"\d+", "", text)

    if remove_extra_spaces:
        text = " ".join(text.split())

    return text


def word_count(text: str) -> int:
    """
    Count the number of words in text.

    Args:
        text: Input text

    Returns:
        Word count
    """
    return len(tokenize(text, remove_punct=True))


def sentence_split(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences

    Example:
        >>> sentence_split("Hello. How are you? I'm fine!")
        ['Hello.', 'How are you?', "I'm fine!"]
    """
    # Simple sentence splitter based on punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def get_word_frequencies(
    text: str, lowercase: bool = True, remove_stopwords_flag: bool = False
) -> Counter:
    """
    Get word frequencies from text.

    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_stopwords_flag: Remove common stopwords

    Returns:
        Counter with word frequencies
    """
    tokens = tokenize(text, lowercase=lowercase, remove_punct=True)

    if remove_stopwords_flag:
        tokens = remove_stopwords(tokens)

    return Counter(tokens)


def longest_common_subsequence(s1: str, s2: str) -> str:
    """
    Find the longest common subsequence of two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Longest common subsequence
    """
    m, n = len(s1), len(s2)

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find the LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            lcs.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return "".join(reversed(lcs))


def longest_common_substring(s1: str, s2: str) -> str:
    """
    Find the longest common substring of two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Longest common substring
    """
    m, n = len(s1), len(s2)

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_pos = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_pos = i

    return s1[end_pos - max_len : end_pos]


def text_similarity(text1: str, text2: str, method: str = "jaccard") -> float:
    """
    Calculate text similarity between two texts.

    Args:
        text1: First text
        text2: Second text
        method: Similarity method ("jaccard", "overlap", "dice")

    Returns:
        Similarity score (0 to 1)
    """
    tokens1 = set(tokenize(text1, remove_punct=True))
    tokens2 = set(tokenize(text2, remove_punct=True))

    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)

    if method == "jaccard":
        union = len(tokens1 | tokens2)
        return intersection / union

    elif method == "overlap":
        return intersection / min(len(tokens1), len(tokens2))

    elif method == "dice":
        return (2 * intersection) / (len(tokens1) + len(tokens2))

    else:
        raise ValueError(f"Unknown similarity method: {method}")
