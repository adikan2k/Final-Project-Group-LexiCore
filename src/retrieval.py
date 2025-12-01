"""
Information retrieval system for academic papers.
Supports semantic search, keyword search, and hybrid approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import logging
from datetime import datetime
import re
from collections import Counter

# Information retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

# Search and ranking
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class PaperRetriever:
    """
    Multi-modal information retrieval system for academic papers.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize paper retriever.
        
        Args:
            device: Device to use for neural models
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model containers
        self.sentence_model = None
        self.tfidf_vectorizer = None
        self.bm25_model = None
        
        # Index containers
        self.faiss_index = None
        self.paper_embeddings = None
        self.paper_metadata = None
        
        # Search configurations
        self.search_configs = {
            'semantic': {
                'model_name': 'all-MiniLM-L6-v2',
                'similarity_metric': 'cosine',
                'top_k': 50
            },
            'keyword': {
                'max_features': 10000,
                'ngram_range': (1, 2),
                'min_df': 2
            },
            'bm25': {
                'tokenizer': 'simple',
                'k1': 1.2,
                'b': 0.75
            }
        }
    
    def load_sentence_model(self, model_name: Optional[str] = None):
        """Load sentence transformer model."""
        if model_name is None:
            model_name = self.search_configs['semantic']['model_name']
        
        logger.info(f"Loading sentence model: {model_name}")
        self.sentence_model = SentenceTransformer(model_name, device=str(self.device))
    
    def index_papers(
        self,
        papers_df: pd.DataFrame,
        text_column: str = 'combined_text',
        id_column: str = 'paper_id',
        use_faiss: bool = True
    ) -> Dict[str, Any]:
        """
        Index papers for retrieval.
        
        Args:
            papers_df: DataFrame with papers
            text_column: Column containing text to index
            id_column: Column with paper IDs
            use_faiss: Whether to use FAISS for fast similarity search
            
        Returns:
            Indexing results
        """
        logger.info(f"Indexing {len(papers_df)} papers...")
        
        # Store metadata
        self.paper_metadata = papers_df.copy()
        
        # Get texts and ensure they're strings
        texts = papers_df[text_column].fillna('').astype(str).tolist()
        
        # Generate embeddings
        if self.sentence_model is None:
            self.load_sentence_model()
        
        logger.info("Generating embeddings...")
        embeddings = self.sentence_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.paper_embeddings = embeddings
        
        # Build FAISS index
        if use_faiss:
            logger.info("Building FAISS index...")
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings.astype('float32'))
        
        # Build TF-IDF index
        logger.info("Building TF-IDF index...")
        config = self.search_configs['keyword']
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            min_df=config['min_df'],
            stop_words='english'
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25_model = BM25Okapi(tokenized_texts)
        
        results = {
            'n_papers': len(papers_df),
            'embedding_dimension': embeddings.shape[1],
            'tfidf_features': tfidf_matrix.shape[1],
            'faiss_enabled': use_faiss,
            'indexed_at': datetime.now().isoformat()
        }
        
        logger.info("Indexing completed!")
        return results
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        use_faiss: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_faiss: Whether to use FAISS index
            
        Returns:
            Search results
        """
        if self.sentence_model is None:
            raise ValueError("Sentence model not loaded. Call load_sentence_model first.")
        
        if self.paper_embeddings is None:
            raise ValueError("Papers not indexed. Call index_papers first.")
        
        # Encode query
        query_embedding = self.sentence_model.encode([query])
        
        if use_faiss and self.faiss_index is not None:
            # Use FAISS for fast search
            faiss.normalize_L2(query_embedding.astype('float32'))
            scores, indices = self.faiss_index.search(
                query_embedding.astype('float32'), top_k
            )
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                paper_info = self.paper_metadata.iloc[idx].to_dict()
                results.append({
                    'rank': i + 1,
                    'score': float(score),
                    'paper_id': paper_info.get('paper_id'),
                    'title': paper_info.get('title'),
                    'abstract': paper_info.get('abstract', '')[:200] + '...',
                    'metadata': paper_info
                })
        else:
            # Use sklearn cosine similarity
            similarities = cosine_similarity(query_embedding, self.paper_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for i, idx in enumerate(top_indices):
                paper_info = self.paper_metadata.iloc[idx].to_dict()
                results.append({
                    'rank': i + 1,
                    'score': float(similarities[idx]),
                    'paper_id': paper_info.get('paper_id'),
                    'title': paper_info.get('title'),
                    'abstract': paper_info.get('abstract', '')[:200] + '...',
                    'metadata': paper_info
                })
        
        return results
    
    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        method: str = 'tfidf'
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            method: Search method ('tfidf', 'bm25')
            
        Returns:
            Search results
        """
        if method == 'tfidf':
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF not built. Call index_papers first.")
            
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Get TF-IDF matrix for all documents
            doc_tfidf = self.tfidf_vectorizer.transform(
                self.paper_metadata['combined_text'].fillna('').astype(str)
            )
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, doc_tfidf)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for i, idx in enumerate(top_indices):
                paper_info = self.paper_metadata.iloc[idx].to_dict()
                results.append({
                    'rank': i + 1,
                    'score': float(similarities[idx]),
                    'paper_id': paper_info.get('paper_id'),
                    'title': paper_info.get('title'),
                    'abstract': paper_info.get('abstract', '')[:200] + '...',
                    'metadata': paper_info
                })
        
        elif method == 'bm25':
            if self.bm25_model is None:
                raise ValueError("BM25 not built. Call index_papers first.")
            
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_model.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for i, idx in enumerate(top_indices):
                paper_info = self.paper_metadata.iloc[idx].to_dict()
                results.append({
                    'rank': i + 1,
                    'score': float(scores[idx]),
                    'paper_id': paper_info.get('paper_id'),
                    'title': paper_info.get('title'),
                    'abstract': paper_info.get('abstract', '')[:200] + '...',
                    'metadata': paper_info
                })
        
        else:
            raise ValueError(f"Unknown keyword search method: {method}")
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        keyword_method: str = 'bm25'
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword methods.
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword similarity
            keyword_method: Keyword search method
            
        Returns:
            Hybrid search results
        """
        # Perform both searches
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2, keyword_method)
        
        # Create combined scoring
        paper_scores = {}
        
        # Add semantic scores
        for result in semantic_results:
            paper_id = result['paper_id']
            paper_scores[paper_id] = {
                'semantic_score': result['score'],
                'keyword_score': 0.0,
                'metadata': result['metadata']
            }
        
        # Add keyword scores
        for result in keyword_results:
            paper_id = result['paper_id']
            if paper_id not in paper_scores:
                paper_scores[paper_id] = {
                    'semantic_score': 0.0,
                    'keyword_score': result['score'],
                    'metadata': result['metadata']
                }
            else:
                paper_scores[paper_id]['keyword_score'] = result['score']
        
        # Calculate hybrid scores
        for paper_id, scores in paper_scores.items():
            # Normalize scores to [0, 1] range
            semantic_norm = scores['semantic_score']
            keyword_norm = scores['keyword_score']
            
            # Combine scores
            hybrid_score = (
                semantic_weight * semantic_norm + 
                keyword_weight * keyword_norm
            )
            scores['hybrid_score'] = hybrid_score
        
        # Sort by hybrid score and create final results
        sorted_papers = sorted(
            paper_scores.items(),
            key=lambda x: x[1]['hybrid_score'],
            reverse=True
        )
        
        results = []
        for i, (paper_id, scores) in enumerate(sorted_papers[:top_k]):
            metadata = scores['metadata']
            results.append({
                'rank': i + 1,
                'hybrid_score': scores['hybrid_score'],
                'semantic_score': scores['semantic_score'],
                'keyword_score': scores['keyword_score'],
                'paper_id': paper_id,
                'title': metadata.get('title'),
                'abstract': metadata.get('abstract', '')[:200] + '...',
                'metadata': metadata
            })
        
        return results
    
    def filter_results(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply filters to search results.
        
        Args:
            results: Search results
            filters: Dictionary of filters to apply
            
        Returns:
            Filtered results
        """
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            include = True
            
            # Year filter
            if 'year_range' in filters:
                year_range = filters['year_range']
                paper_year = metadata.get('year')
                if paper_year and (paper_year < year_range[0] or paper_year > year_range[1]):
                    include = False
            
            # Venue filter
            if 'venues' in filters:
                allowed_venues = filters['venues']
                paper_venue = metadata.get('venue', '').lower()
                if not any(venue.lower() in paper_venue for venue in allowed_venues):
                    include = False
            
            # Category filter
            if 'categories' in filters:
                allowed_categories = filters['categories']
                paper_categories = metadata.get('categories', [])
                if isinstance(paper_categories, str):
                    paper_categories = [paper_categories]
                if not any(cat in paper_categories for cat in allowed_categories):
                    include = False
            
            # Author filter
            if 'authors' in filters:
                target_authors = [author.lower() for author in filters['authors']]
                paper_authors = metadata.get('authors', [])
                if isinstance(paper_authors, str):
                    paper_authors = [paper_authors]
                paper_authors_lower = [author.lower() for author in paper_authors]
                if not any(author in paper_authors_lower for author in target_authors):
                    include = False
            
            # Minimum score filter
            if 'min_score' in filters:
                min_score = filters['min_score']
                result_score = result.get('score', result.get('hybrid_score', 0))
                if result_score < min_score:
                    include = False
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def get_similar_papers(
        self,
        paper_id: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find papers similar to a given paper.
        
        Args:
            paper_id: ID of the reference paper
            top_k: Number of similar papers to return
            exclude_self: Whether to exclude the reference paper
            
        Returns:
            Similar papers
        """
        # Find the reference paper
        paper_mask = self.paper_metadata['paper_id'] == paper_id
        if not paper_mask.any():
            raise ValueError(f"Paper {paper_id} not found in index")
        
        paper_idx = paper_mask.idxmax()
        reference_embedding = self.paper_embeddings[paper_idx:paper_idx+1]
        
        # Find similar papers
        if self.faiss_index is not None:
            # Use FAISS
            search_k = top_k + 1 if exclude_self else top_k
            scores, indices = self.faiss_index.search(
                reference_embedding.astype('float32'), search_k
            )
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if exclude_self and idx == paper_idx:
                    continue
                
                paper_info = self.paper_metadata.iloc[idx].to_dict()
                results.append({
                    'rank': len(results) + 1,
                    'similarity': float(score),
                    'paper_id': paper_info.get('paper_id'),
                    'title': paper_info.get('title'),
                    'abstract': paper_info.get('abstract', '')[:200] + '...',
                    'metadata': paper_info
                })
        else:
            # Use sklearn cosine similarity
            similarities = cosine_similarity(reference_embedding, self.paper_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1]
            
            if exclude_self:
                top_indices = top_indices[top_indices != paper_idx]
            
            results = []
            for i, idx in enumerate(top_indices[:top_k]):
                paper_info = self.paper_metadata.iloc[idx].to_dict()
                results.append({
                    'rank': i + 1,
                    'similarity': float(similarities[idx]),
                    'paper_id': paper_info.get('paper_id'),
                    'title': paper_info.get('title'),
                    'abstract': paper_info.get('abstract', '')[:200] + '...',
                    'metadata': paper_info
                })
        
        return results[:top_k]
    
    def save_index(self, output_dir: Union[str, Path]):
        """Save the search index to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if self.paper_embeddings is not None:
            np.save(output_dir / 'embeddings.npy', self.paper_embeddings)
        
        # Save FAISS index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(output_dir / 'faiss.index'))
        
        # Save TF-IDF vectorizer
        if self.tfidf_vectorizer is not None:
            import pickle
            with open(output_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        # Save BM25 model
        if self.bm25_model is not None:
            import pickle
            with open(output_dir / 'bm25_model.pkl', 'wb') as f:
                pickle.dump(self.bm25_model, f)
        
        # Save metadata
        if self.paper_metadata is not None:
            self.paper_metadata.to_parquet(output_dir / 'metadata.parquet', index=False)
        
        logger.info(f"Index saved to {output_dir}")
    
    def load_index(self, input_dir: Union[str, Path]):
        """Load the search index from disk."""
        input_dir = Path(input_dir)
        
        # Load embeddings
        embeddings_path = input_dir / 'embeddings.npy'
        if embeddings_path.exists():
            self.paper_embeddings = np.load(embeddings_path)
        
        # Load FAISS index
        faiss_path = input_dir / 'faiss.index'
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        
        # Load TF-IDF vectorizer
        tfidf_path = input_dir / 'tfidf_vectorizer.pkl'
        if tfidf_path.exists():
            import pickle
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        
        # Load BM25 model
        bm25_path = input_dir / 'bm25_model.pkl'
        if bm25_path.exists():
            import pickle
            with open(bm25_path, 'rb') as f:
                self.bm25_model = pickle.load(f)
        
        # Load metadata
        metadata_path = input_dir / 'metadata.parquet'
        if metadata_path.exists():
            self.paper_metadata = pd.read_parquet(metadata_path)
        
        logger.info(f"Index loaded from {input_dir}")
    
    def get_search_analytics(
        self,
        query_logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze search patterns and performance.
        
        Args:
            query_logs: List of search query logs
            
        Returns:
            Analytics results
        """
        if not query_logs:
            return {}
        
        # Extract query terms
        all_queries = [log.get('query', '') for log in query_logs]
        all_terms = []
        for query in all_queries:
            terms = re.findall(r'\b\w+\b', query.lower())
            all_terms.extend(terms)
        
        # Term frequency analysis
        term_counts = Counter(all_terms)
        
        # Query length analysis
        query_lengths = [len(query.split()) for query in all_queries]
        
        # Results analysis
        result_counts = [log.get('results_returned', 0) for log in query_logs]
        
        analytics = {
            'total_queries': len(query_logs),
            'unique_queries': len(set(all_queries)),
            'top_terms': dict(term_counts.most_common(20)),
            'query_stats': {
                'avg_length': np.mean(query_lengths),
                'median_length': np.median(query_lengths),
                'max_length': np.max(query_lengths),
                'min_length': np.min(query_lengths)
            },
            'result_stats': {
                'avg_results': np.mean(result_counts),
                'median_results': np.median(result_counts),
                'zero_result_queries': sum(1 for count in result_counts if count == 0)
            }
        }
        
        return analytics