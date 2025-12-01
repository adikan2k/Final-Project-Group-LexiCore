"""
Text cleaning and normalization utilities for academic papers.
Handles common issues in academic text preprocessing.
"""

import re
import string
from typing import List, Optional, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
# Optional spaCy import
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextCleaner:
    """
    Comprehensive text cleaning and preprocessing for academic papers.
    """
    
    def __init__(self, language: str = 'english', use_spacy: bool = False):
        """
        Initialize text cleaner.
        
        Args:
            language: Language for stopwords and processing
            use_spacy: Whether to use spaCy for advanced NLP features
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Academic-specific patterns
        self.citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
            r'\[[^\]]*\d{4}[^\]]*\]',  # [Author, 2023]
            r'\b\w+\s+et\s+al\.?,?\s+\d{4}\b',  # Smith et al., 2023
            r'\([^)]*et\s+al\.?[^)]*\)',  # (Smith et al.)
        ]
        
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Load spaCy model if requested and available
        self.nlp = None
        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        elif use_spacy and not SPACY_AVAILABLE:
            print("Warning: spaCy not available. Install with: pip install spacy")
    
    def remove_citations(self, text: str) -> str:
        """Remove academic citations from text."""
        for pattern in self.citation_patterns:
            text = re.sub(pattern, '', text)
        return text
    
    def remove_urls_emails(self, text: str) -> str:
        """Remove URLs and email addresses."""
        text = re.sub(self.url_pattern, '', text)
        text = re.sub(self.email_pattern, '', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and remove extra spaces."""
        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def remove_special_chars(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters, optionally keeping basic punctuation."""
        if keep_punctuation:
            # Keep basic punctuation: . , ! ? ; :
            text = re.sub(r'[^\w\s.,!?;:\-]', '', text)
        else:
            # Remove all punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def expand_contractions(self, text: str) -> str:
        """Expand common English contractions."""
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "'s": " is"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
            
        return text
    
    def clean_academic_text(
        self, 
        text: str,
        remove_citations: bool = True,
        remove_urls: bool = True,
        lowercase: bool = True,
        expand_contractions: bool = True,
        remove_special: bool = True,
        keep_punctuation: bool = True
    ) -> str:
        """
        Comprehensive cleaning for academic text.
        
        Args:
            text: Input text to clean
            remove_citations: Remove academic citations
            remove_urls: Remove URLs and emails
            lowercase: Convert to lowercase
            expand_contractions: Expand contractions
            remove_special: Remove special characters
            keep_punctuation: Keep basic punctuation when removing special chars
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove citations
        if remove_citations:
            text = self.remove_citations(text)
        
        # Remove URLs and emails
        if remove_urls:
            text = self.remove_urls_emails(text)
        
        # Expand contractions
        if expand_contractions:
            text = self.expand_contractions(text)
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Remove special characters
        if remove_special:
            text = self.remove_special_chars(text, keep_punctuation)
        
        # Normalize whitespace
        text = self.normalize_whitespace(text)
        
        return text
    
    def tokenize_text(
        self, 
        text: str, 
        method: str = 'nltk',
        remove_stopwords: bool = True,
        min_length: int = 2
    ) -> List[str]:
        """
        Tokenize text using specified method.
        
        Args:
            text: Text to tokenize
            method: 'nltk' or 'spacy'
            remove_stopwords: Remove stopwords
            min_length: Minimum token length
            
        Returns:
            List of tokens
        """
        if method == 'spacy' and self.nlp:
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_space]
        else:
            tokens = word_tokenize(text)
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip if too short
            if len(token) < min_length:
                continue
            # Skip stopwords if requested
            if remove_stopwords and token.lower() in self.stop_words:
                continue
            # Skip if only punctuation
            if token in string.punctuation:
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def lemmatize_tokens(self, tokens: List[str], method: str = 'nltk') -> List[str]:
        """
        Lemmatize tokens using specified method.
        
        Args:
            tokens: List of tokens
            method: 'nltk' or 'spacy'
            
        Returns:
            Lemmatized tokens
        """
        if method == 'spacy' and self.nlp:
            # Process as single text for better accuracy
            text = ' '.join(tokens)
            doc = self.nlp(text)
            return [token.lemma_ for token in doc if not token.is_space and len(token.lemma_) > 1]
        else:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stem tokens using Porter stemmer."""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(
        self,
        text: str,
        clean_params: Optional[Dict[str, Any]] = None,
        tokenize: bool = True,
        lemmatize: bool = True,
        stem: bool = False
    ) -> Dict[str, Any]:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Input text
            clean_params: Parameters for cleaning (passed to clean_academic_text)
            tokenize: Whether to tokenize
            lemmatize: Whether to lemmatize
            stem: Whether to stem
            
        Returns:
            Dictionary with processed text and metadata
        """
        if clean_params is None:
            clean_params = {}
        
        original_length = len(text) if text else 0
        
        # Clean text
        cleaned_text = self.clean_academic_text(text, **clean_params)
        
        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'original_length': original_length,
            'cleaned_length': len(cleaned_text),
            'compression_ratio': len(cleaned_text) / original_length if original_length > 0 else 0
        }
        
        if tokenize:
            tokens = self.tokenize_text(cleaned_text)
            result['tokens'] = tokens
            result['token_count'] = len(tokens)
            
            if lemmatize:
                lemmatized = self.lemmatize_tokens(tokens)
                result['lemmatized_tokens'] = lemmatized
            
            if stem:
                stemmed = self.stem_tokens(tokens)
                result['stemmed_tokens'] = stemmed
        
        return result