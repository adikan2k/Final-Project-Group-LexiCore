"""
Text preprocessing utilities for NLP pipeline.

This module provides functions for cleaning and preprocessing text data
from academic papers, including abstracts and titles.

Functions:
    - clean_text: Remove URLs, emails, extra whitespace
    - remove_special_chars: Remove special characters
    - tokenize_text: Tokenize using spaCy
    - remove_stopwords: Remove common stopwords
    - lemmatize_tokens: Lemmatize tokens
    - preprocess_text: Complete preprocessing pipeline

Usage:
    from src.preprocessing.preprocess import preprocess_text
    
    result = preprocess_text("Your text here")
    print(result['processed_text'])
"""

import re
import spacy
from nltk.corpus import stopwords

# load spacy model (make sure it's installed: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

# get stopwords from nltk
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("NLTK stopwords not found. Please run: nltk.download('stopwords')")
    stop_words = set()


def clean_text(text):
    """
    Remove URLs, emails, and extra whitespace from text.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def remove_special_chars(text):
    """
    Remove special characters but keep basic punctuation.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Text with special characters removed
    """
    # keep letters, numbers, and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    
    # remove multiple punctuation
    text = re.sub(r'([.,!?-])\1+', r'\1', text)
    
    return text


def tokenize_text(text):
    """
    Tokenize text using spaCy.
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of tokens
    """
    if nlp is None:
        # fallback to simple split if spacy not available
        return text.split()
    
    doc = nlp(text)
    return [token.text for token in doc]


def remove_stopwords(tokens):
    """
    Remove stopwords from token list.
    
    Args:
        tokens (list): List of tokens
    
    Returns:
        list: Filtered tokens without stopwords
    """
    return [token for token in tokens if token.lower() not in stop_words]


def lemmatize_tokens(tokens):
    """
    Lemmatize tokens using spaCy.
    
    Args:
        tokens (list): List of tokens
    
    Returns:
        list: Lemmatized tokens
    """
    if nlp is None:
        # return original tokens if spacy not available
        return tokens
    
    # join tokens back to text for spacy processing
    text = ' '.join(tokens)
    doc = nlp(text)
    
    # get lemmas
    return [token.lemma_ for token in doc]


def preprocess_text(text, lowercase=True, remove_stops=True, lemmatize=True):
    """
    Complete preprocessing pipeline for text.
    
    This function applies a series of preprocessing steps:
    1. Clean text (remove URLs, emails, etc.)
    2. Remove special characters
    3. Convert to lowercase (optional)
    4. Tokenize
    5. Remove stopwords (optional)
    6. Lemmatize (optional)
    7. Filter short tokens
    
    Args:
        text (str): Input text string
        lowercase (bool): Convert to lowercase (default: True)
        remove_stops (bool): Remove stopwords (default: True)
        lemmatize (bool): Apply lemmatization (default: True)
    
    Returns:
        dict: Dictionary containing:
            - cleaned_text: Text after cleaning and normalization
            - tokens: List of processed tokens
            - processed_text: Tokens joined back into text
    
    Example:
        >>> result = preprocess_text("Natural Language Processing is amazing!")
        >>> print(result['processed_text'])
        'natural language processing amazing'
    """
    if not isinstance(text, str) or len(text) == 0:
        return {
            'cleaned_text': '',
            'tokens': [],
            'processed_text': ''
        }
    
    # step 1: clean text
    text = clean_text(text)
    text = remove_special_chars(text)
    
    # step 2: lowercase
    if lowercase:
        text = text.lower()
    
    cleaned_text = text
    
    # step 3: tokenize
    tokens = tokenize_text(text)
    
    # step 4: remove stopwords
    if remove_stops:
        tokens = remove_stopwords(tokens)
    
    # step 5: lemmatize
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    
    # step 6: filter short and non-alphanumeric tokens
    tokens = [t for t in tokens if len(t) > 2 and t.isalnum()]
    
    # join back to text
    processed_text = ' '.join(tokens)
    
    return {
        'cleaned_text': cleaned_text,
        'tokens': tokens,
        'processed_text': processed_text
    }


def batch_preprocess(texts, **kwargs):
    """
    Preprocess multiple texts at once.
    
    Args:
        texts (list): List of text strings
        **kwargs: Arguments to pass to preprocess_text
    
    Returns:
        list: List of preprocessing results
    """
    return [preprocess_text(text, **kwargs) for text in texts]


if __name__ == "__main__":
    # test the functions
    test_text = """
    Natural Language Processing (NLP) is amazing! 
    Visit https://example.com for more info.
    We're studying various techniques including tokenization and lemmatization.
    """
    
    result = preprocess_text(test_text)
    
    print("Original text:")
    print(test_text)
    print("\nCleaned text:")
    print(result['cleaned_text'])
    print("\nTokens:")
    print(result['tokens'])
    print("\nProcessed text:")
    print(result['processed_text'])
