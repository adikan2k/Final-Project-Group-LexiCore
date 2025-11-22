"""
Text summarization for academic papers and research insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime

# Traditional summarization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Neural summarization
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    pipeline, BartTokenizer, BartForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration
)

# NLP utilities
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)


class PaperSummarizer:
    """
    Multi-approach text summarization for academic papers.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize summarizer.
        
        Args:
            device: Device to use for neural models
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model containers
        self.summarization_pipeline = None
        self.bart_model = None
        self.bart_tokenizer = None
        self.t5_model = None
        self.t5_tokenizer = None
        
        # Default models
        self.default_models = {
            'bart': 'facebook/bart-large-cnn',
            't5': 't5-small',
            'pipeline': 'facebook/bart-large-cnn'
        }
        
        # Summarization configurations
        self.summarization_configs = {
            'extractive': {
                'sentence_count': 3,
                'similarity_threshold': 0.1
            },
            'neural': {
                'max_length': 150,
                'min_length': 50,
                'length_penalty': 2.0,
                'num_beams': 4
            }
        }
    
    def load_summarization_pipeline(self, model_name: Optional[str] = None):
        """Load summarization pipeline."""
        if model_name is None:
            model_name = self.default_models['pipeline']
        
        logger.info(f"Loading summarization pipeline: {model_name}")
        self.summarization_pipeline = pipeline(
            "summarization",
            model=model_name,
            device=0 if self.device.type == 'cuda' else -1
        )
    
    def load_bart_model(self, model_name: Optional[str] = None):
        """Load BART model for summarization."""
        if model_name is None:
            model_name = self.default_models['bart']
        
        logger.info(f"Loading BART model: {model_name}")
        self.bart_tokenizer = BartTokenizer.from_pretrained(model_name)
        self.bart_model = BartForConditionalGeneration.from_pretrained(model_name)
        self.bart_model.to(self.device)
        self.bart_model.eval()
    
    def load_t5_model(self, model_name: Optional[str] = None):
        """Load T5 model for summarization."""
        if model_name is None:
            model_name = self.default_models['t5']
        
        logger.info(f"Loading T5 model: {model_name}")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.t5_model.to(self.device)
        self.t5_model.eval()
    
    def extractive_summarization(
        self,
        text: str,
        num_sentences: int = 3,
        method: str = 'textrank'
    ) -> Dict[str, Any]:
        """
        Perform extractive summarization using various methods.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            method: Summarization method ('textrank', 'tfidf', 'frequency')
            
        Returns:
            Summarization results
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return {
                'summary': text,
                'selected_sentences': sentences,
                'method': method,
                'compression_ratio': 1.0
            }
        
        if method == 'textrank':
            summary_sentences = self._textrank_summarization(sentences, num_sentences)
        elif method == 'tfidf':
            summary_sentences = self._tfidf_summarization(sentences, num_sentences)
        elif method == 'frequency':
            summary_sentences = self._frequency_summarization(sentences, num_sentences)
        else:
            raise ValueError(f"Unknown extractive method: {method}")
        
        # Create summary maintaining original order
        selected_indices = []
        for summary_sent in summary_sentences:
            for i, original_sent in enumerate(sentences):
                if summary_sent == original_sent:
                    selected_indices.append(i)
                    break
        
        selected_indices.sort()
        ordered_sentences = [sentences[i] for i in selected_indices]
        summary = ' '.join(ordered_sentences)
        
        return {
            'summary': summary,
            'selected_sentences': ordered_sentences,
            'selected_indices': selected_indices,
            'method': method,
            'compression_ratio': len(summary) / len(text)
        }
    
    def _textrank_summarization(self, sentences: List[str], num_sentences: int) -> List[str]:
        """TextRank-based extractive summarization."""
        # Create similarity matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            sentence_vectors = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(sentence_vectors)
        except ValueError:
            # Fallback to first sentences if TF-IDF fails
            return sentences[:num_sentences]
        
        # Create graph
        graph = nx.Graph()
        graph.add_nodes_from(range(len(sentences)))
        
        # Add edges based on similarity
        threshold = self.summarization_configs['extractive']['similarity_threshold']
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = similarity_matrix[i][j]
                if similarity > threshold:
                    graph.add_edge(i, j, weight=similarity)
        
        # Calculate PageRank scores
        try:
            scores = nx.pagerank(graph, weight='weight')
        except:
            # Fallback if PageRank fails
            scores = {i: 1.0 for i in range(len(sentences))}
        
        # Select top sentences
        ranked_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in ranked_sentences[:num_sentences]]
        
        return [sentences[i] for i in sorted(top_indices)]
    
    def _tfidf_summarization(self, sentences: List[str], num_sentences: int) -> List[str]:
        """TF-IDF based extractive summarization."""
        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        except ValueError:
            # Fallback if TF-IDF fails
            return sentences[:num_sentences]
        
        # Select sentences with highest TF-IDF scores
        top_indices = np.argsort(sentence_scores)[::-1][:num_sentences]
        return [sentences[i] for i in sorted(top_indices)]
    
    def _frequency_summarization(self, sentences: List[str], num_sentences: int) -> List[str]:
        """Frequency-based extractive summarization."""
        # Calculate word frequencies
        words = []
        for sentence in sentences:
            words.extend(word_tokenize(sentence.lower()))
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word.isalnum() and word not in stop_words]
        except:
            words = [word for word in words if word.isalnum()]
        
        word_freq = Counter(words)
        
        # Score sentences based on word frequencies
        sentence_scores = []
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            score = sum(word_freq.get(word, 0) for word in sentence_words if word.isalnum())
            sentence_scores.append(score)
        
        # Select top sentences
        top_indices = np.argsort(sentence_scores)[::-1][:num_sentences]
        return [sentences[i] for i in sorted(top_indices)]
    
    def neural_summarization(
        self,
        text: str,
        model: str = 'pipeline',
        max_length: int = 150,
        min_length: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform neural summarization.
        
        Args:
            text: Input text to summarize
            model: Model to use ('pipeline', 'bart', 't5')
            max_length: Maximum summary length
            min_length: Minimum summary length
            **kwargs: Additional generation parameters
            
        Returns:
            Summarization results
        """
        # Update config with kwargs
        config = self.summarization_configs['neural'].copy()
        config.update(kwargs)
        config['max_length'] = max_length
        config['min_length'] = min_length
        
        if model == 'pipeline':
            if self.summarization_pipeline is None:
                self.load_summarization_pipeline()
            
            # Split long texts if needed
            max_input_length = 1024  # Typical BART limit
            if len(text.split()) > max_input_length:
                # Split into chunks and summarize each
                chunks = self._split_text_for_summarization(text, max_input_length)
                chunk_summaries = []
                
                for chunk in chunks:
                    result = self.summarization_pipeline(
                        chunk,
                        max_length=config['max_length'],
                        min_length=config['min_length'],
                        length_penalty=config['length_penalty'],
                        num_beams=config['num_beams']
                    )
                    chunk_summaries.append(result[0]['summary_text'])
                
                # Combine and re-summarize if needed
                combined_summary = ' '.join(chunk_summaries)
                if len(combined_summary.split()) > max_length:
                    final_result = self.summarization_pipeline(
                        combined_summary,
                        max_length=max_length,
                        min_length=min_length
                    )
                    summary = final_result[0]['summary_text']
                else:
                    summary = combined_summary
            else:
                result = self.summarization_pipeline(
                    text,
                    max_length=config['max_length'],
                    min_length=config['min_length'],
                    length_penalty=config['length_penalty'],
                    num_beams=config['num_beams']
                )
                summary = result[0]['summary_text']
        
        elif model == 'bart':
            if self.bart_model is None:
                self.load_bart_model()
            
            summary = self._generate_bart_summary(text, config)
        
        elif model == 't5':
            if self.t5_model is None:
                self.load_t5_model()
            
            summary = self._generate_t5_summary(text, config)
        
        else:
            raise ValueError(f"Unknown neural model: {model}")
        
        return {
            'summary': summary,
            'method': f'neural_{model}',
            'compression_ratio': len(summary) / len(text),
            'config': config
        }
    
    def _split_text_for_summarization(self, text: str, max_length: int) -> List[str]:
        """Split long text into chunks for summarization."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= max_length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _generate_bart_summary(self, text: str, config: Dict) -> str:
        """Generate summary using BART model."""
        inputs = self.bart_tokenizer.encode(
            text,
            return_tensors='pt',
            max_length=1024,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.bart_model.generate(
                inputs,
                max_length=config['max_length'],
                min_length=config['min_length'],
                length_penalty=config['length_penalty'],
                num_beams=config['num_beams'],
                early_stopping=True
            )
        
        summary = self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def _generate_t5_summary(self, text: str, config: Dict) -> str:
        """Generate summary using T5 model."""
        # T5 requires task prefix
        input_text = f"summarize: {text}"
        
        inputs = self.t5_tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.t5_model.generate(
                inputs,
                max_length=config['max_length'],
                min_length=config['min_length'],
                length_penalty=config['length_penalty'],
                num_beams=config['num_beams'],
                early_stopping=True
            )
        
        summary = self.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def multi_document_summarization(
        self,
        documents: List[str],
        titles: Optional[List[str]] = None,
        method: str = 'extractive',
        num_sentences: int = 5
    ) -> Dict[str, Any]:
        """
        Summarize multiple related documents.
        
        Args:
            documents: List of documents to summarize
            titles: Optional document titles
            method: Summarization method ('extractive', 'neural')
            num_sentences: Number of sentences in final summary
            
        Returns:
            Multi-document summary
        """
        if not documents:
            return {'summary': '', 'method': method}
        
        # Combine all documents
        if titles:
            combined_text = ' '.join([
                f"From {title}: {doc}" for title, doc in zip(titles, documents)
            ])
        else:
            combined_text = ' '.join(documents)
        
        # Apply summarization method
        if method == 'extractive':
            result = self.extractive_summarization(
                combined_text, 
                num_sentences, 
                'textrank'
            )
        elif method == 'neural':
            result = self.neural_summarization(combined_text)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add multi-document specific information
        result.update({
            'num_documents': len(documents),
            'document_lengths': [len(doc.split()) for doc in documents],
            'total_length': len(combined_text.split())
        })
        
        return result
    
    def topic_focused_summarization(
        self,
        text: str,
        topic_keywords: List[str],
        method: str = 'extractive',
        num_sentences: int = 3
    ) -> Dict[str, Any]:
        """
        Generate topic-focused summary.
        
        Args:
            text: Input text
            topic_keywords: Keywords defining the topic of interest
            method: Summarization method
            num_sentences: Number of sentences
            
        Returns:
            Topic-focused summary
        """
        sentences = sent_tokenize(text)
        
        # Score sentences based on topic relevance
        topic_scores = []
        topic_keywords_lower = [kw.lower() for kw in topic_keywords]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for kw in topic_keywords_lower if kw in sentence_lower)
            # Normalize by sentence length
            score = score / len(sentence.split()) if len(sentence.split()) > 0 else 0
            topic_scores.append(score)
        
        # Select sentences with highest topic relevance
        if method == 'extractive':
            # Combine topic relevance with general importance
            general_result = self.extractive_summarization(text, len(sentences))
            
            # Weight by topic relevance
            combined_scores = []
            for i, sentence in enumerate(sentences):
                general_importance = i in general_result['selected_indices']
                combined_score = topic_scores[i] * 2 + (1 if general_importance else 0)
                combined_scores.append((i, combined_score))
            
            # Select top sentences
            top_sentences = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
            selected_indices = sorted([idx for idx, score in top_sentences])
            selected_sentences = [sentences[i] for i in selected_indices]
            summary = ' '.join(selected_sentences)
            
        elif method == 'neural':
            # Create topic-aware prompt
            topic_prompt = f"Focusing on {', '.join(topic_keywords)}: {text}"
            result = self.neural_summarization(topic_prompt)
            summary = result['summary']
            selected_sentences = [summary]
            selected_indices = []
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'summary': summary,
            'selected_sentences': selected_sentences,
            'selected_indices': selected_indices,
            'topic_keywords': topic_keywords,
            'method': f'topic_focused_{method}',
            'compression_ratio': len(summary) / len(text)
        }
    
    def compare_summarization_methods(
        self,
        text: str,
        methods: Optional[List[str]] = None,
        target_length: int = 100
    ) -> Dict[str, Any]:
        """
        Compare different summarization methods on the same text.
        
        Args:
            text: Input text
            methods: List of methods to compare
            target_length: Target summary length in words
            
        Returns:
            Comparison results
        """
        if methods is None:
            methods = ['extractive_textrank', 'extractive_tfidf', 'neural_pipeline']
        
        results = {}
        
        for method in methods:
            try:
                if method.startswith('extractive_'):
                    extractive_method = method.split('_')[1]
                    num_sentences = max(1, target_length // 20)  # Estimate sentences needed
                    result = self.extractive_summarization(
                        text, num_sentences, extractive_method
                    )
                elif method.startswith('neural_'):
                    neural_method = method.split('_')[1] if '_' in method else 'pipeline'
                    result = self.neural_summarization(
                        text, neural_method, max_length=target_length
                    )
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue
                
                # Add quality metrics
                result['word_count'] = len(result['summary'].split())
                result['char_count'] = len(result['summary'])
                results[method] = result
                
            except Exception as e:
                logger.error(f"Error with method {method}: {e}")
                results[method] = {
                    'error': str(e),
                    'summary': '',
                    'method': method
                }
        
        # Compare results
        comparison = {
            'methods_compared': methods,
            'results': results,
            'original_length': len(text.split()),
            'target_length': target_length
        }
        
        return comparison
    
    def batch_summarization(
        self,
        texts: List[str],
        method: str = 'neural',
        batch_size: int = 8,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform batch summarization on multiple texts.
        
        Args:
            texts: List of texts to summarize
            method: Summarization method
            batch_size: Batch size for neural methods
            **kwargs: Additional parameters
            
        Returns:
            List of summarization results
        """
        logger.info(f"Batch summarizing {len(texts)} texts using {method}...")
        
        results = []
        
        if method.startswith('neural'):
            neural_method = method.split('_')[1] if '_' in method else 'pipeline'
            
            # Process in batches for neural methods
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                for text in batch_texts:
                    try:
                        result = self.neural_summarization(text, neural_method, **kwargs)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error summarizing text: {e}")
                        results.append({
                            'summary': '',
                            'error': str(e),
                            'method': method
                        })
        
        elif method.startswith('extractive'):
            extractive_method = method.split('_')[1] if '_' in method else 'textrank'
            
            for text in texts:
                try:
                    result = self.extractive_summarization(text, method=extractive_method, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error summarizing text: {e}")
                    results.append({
                        'summary': '',
                        'error': str(e),
                        'method': method
                    })
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return results
    
    def save_summaries(
        self,
        summaries: List[Dict[str, Any]],
        output_path: Union[str, Path],
        include_metadata: bool = True
    ):
        """Save summarization results."""
        from pathlib import Path
        import json
        
        output_path = Path(output_path)
        
        if include_metadata:
            # Create comprehensive output
            output_data = {
                'summaries': summaries,
                'metadata': {
                    'total_summaries': len(summaries),
                    'methods_used': list(set(s.get('method', 'unknown') for s in summaries)),
                    'generated_at': datetime.now().isoformat()
                }
            }
        else:
            output_data = summaries
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Summaries saved to {output_path}")