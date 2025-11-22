"""
Embedding generation for academic papers using multiple approaches.
Supports Word2Vec, BERT, RoBERTa, and Sentence-BERT embeddings.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import logging
import warnings

# Core ML libraries
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Word2Vec and traditional embeddings (optional)
try:
    from gensim.models import Word2Vec
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    Word2Vec = None
    Doc2Vec = None
    TaggedDocument = None

# Transformer-based embeddings
from transformers import (
    AutoTokenizer, AutoModel, 
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel
)
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Multi-modal embedding generator for academic papers.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize embedding generator.
        
        Args:
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model containers
        self.word2vec_model = None
        self.doc2vec_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.roberta_model = None
        self.roberta_tokenizer = None
        self.sentence_bert_model = None
        
        # Model configurations
        self.model_configs = {
            'word2vec': {
                'vector_size': 300,
                'window': 5,
                'min_count': 5,
                'workers': 4,
                'epochs': 10
            },
            'doc2vec': {
                'vector_size': 300,
                'window': 5,
                'min_count': 5,
                'workers': 4,
                'epochs': 10
            },
            'bert': {
                'model_name': 'bert-base-uncased',
                'max_length': 512,
                'batch_size': 16
            },
            'roberta': {
                'model_name': 'roberta-base',
                'max_length': 512,
                'batch_size': 16
            },
            'sentence_bert': {
                'model_name': 'all-MiniLM-L6-v2',
                'batch_size': 32
            }
        }
    
    def train_word2vec(
        self, 
        tokenized_texts: List[List[str]],
        save_path: Optional[str] = None,
        **kwargs
    ):
        """
        Train Word2Vec model on tokenized texts.
        
        Args:
            tokenized_texts: List of tokenized documents
            save_path: Path to save model
            **kwargs: Additional parameters for Word2Vec
            
        Returns:
            Trained Word2Vec model or None if gensim not available
        """
        if not GENSIM_AVAILABLE:
            logger.warning("Gensim not available. Cannot train Word2Vec model.")
            return None
            
        # Update config with kwargs
        config = self.model_configs['word2vec'].copy()
        config.update(kwargs)
        
        logger.info("Training Word2Vec model...")
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            **config
        )
        
        if save_path:
            self.word2vec_model.save(save_path)
            logger.info(f"Word2Vec model saved to {save_path}")
        
        return self.word2vec_model
    
    def train_doc2vec(
        self,
        tokenized_texts: List[List[str]],
        doc_ids: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        **kwargs
    ):
        """
        Train Doc2Vec model on tokenized texts.
        
        Args:
            tokenized_texts: List of tokenized documents
            doc_ids: Optional document IDs
            save_path: Path to save model
            **kwargs: Additional parameters
            
        Returns:
            Trained Doc2Vec model or None if gensim not available
        """
        if not GENSIM_AVAILABLE:
            logger.warning("Gensim not available. Cannot train Doc2Vec model.")
            return None
            
        # Prepare tagged documents
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(tokenized_texts))]
        
        tagged_docs = [
            TaggedDocument(words=doc, tags=[doc_id])
            for doc, doc_id in zip(tokenized_texts, doc_ids)
        ]
        
        # Update config
        config = self.model_configs['doc2vec'].copy()
        config.update(kwargs)
        
        logger.info("Training Doc2Vec model...")
        self.doc2vec_model = Doc2Vec(
            documents=tagged_docs,
            **config
        )
        
        if save_path:
            self.doc2vec_model.save(save_path)
            logger.info(f"Doc2Vec model saved to {save_path}")
        
        return self.doc2vec_model
    
    def load_bert_model(self, model_name: str = None):
        """Load BERT model and tokenizer."""
        if model_name is None:
            model_name = self.model_configs['bert']['model_name']
        
        logger.info(f"Loading BERT model: {model_name}")
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()
    
    def load_roberta_model(self, model_name: str = None):
        """Load RoBERTa model and tokenizer."""
        if model_name is None:
            model_name = self.model_configs['roberta']['model_name']
        
        logger.info(f"Loading RoBERTa model: {model_name}")
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.roberta_model = RobertaModel.from_pretrained(model_name)
        self.roberta_model.to(self.device)
        self.roberta_model.eval()
    
    def load_sentence_bert_model(self, model_name: str = None):
        """Load Sentence-BERT model."""
        if model_name is None:
            model_name = self.model_configs['sentence_bert']['model_name']
        
        logger.info(f"Loading Sentence-BERT model: {model_name}")
        self.sentence_bert_model = SentenceTransformer(model_name, device=str(self.device))
    
    def get_word2vec_embeddings(
        self, 
        tokenized_texts: List[List[str]],
        aggregation: str = 'mean'
    ) -> np.ndarray:
        """
        Get Word2Vec document embeddings by aggregating word vectors.
        
        Args:
            tokenized_texts: List of tokenized documents
            aggregation: How to aggregate word vectors ('mean', 'sum', 'max')
            
        Returns:
            Document embeddings array
        """
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained. Call train_word2vec first.")
        
        embeddings = []
        
        for tokens in tqdm(tokenized_texts, desc="Getting Word2Vec embeddings"):
            word_vectors = []
            for token in tokens:
                if token in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[token])
            
            if word_vectors:
                word_vectors = np.array(word_vectors)
                if aggregation == 'mean':
                    doc_embedding = np.mean(word_vectors, axis=0)
                elif aggregation == 'sum':
                    doc_embedding = np.sum(word_vectors, axis=0)
                elif aggregation == 'max':
                    doc_embedding = np.max(word_vectors, axis=0)
                else:
                    doc_embedding = np.mean(word_vectors, axis=0)
            else:
                # Handle case with no known words
                doc_embedding = np.zeros(self.word2vec_model.vector_size)
            
            embeddings.append(doc_embedding)
        
        return np.array(embeddings)
    
    def get_doc2vec_embeddings(self, doc_ids: List[str]) -> np.ndarray:
        """
        Get Doc2Vec document embeddings.
        
        Args:
            doc_ids: Document IDs used during training
            
        Returns:
            Document embeddings array
        """
        if self.doc2vec_model is None:
            raise ValueError("Doc2Vec model not trained. Call train_doc2vec first.")
        
        embeddings = []
        for doc_id in doc_ids:
            try:
                embedding = self.doc2vec_model.dv[doc_id]
                embeddings.append(embedding)
            except KeyError:
                # Handle unknown document
                embedding = np.zeros(self.doc2vec_model.vector_size)
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def get_bert_embeddings(
        self, 
        texts: List[str],
        pooling: str = 'cls',
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Get BERT embeddings for texts.
        
        Args:
            texts: List of text documents
            pooling: Pooling strategy ('cls', 'mean', 'max')
            batch_size: Batch size for processing
            
        Returns:
            Document embeddings array
        """
        if self.bert_model is None:
            self.load_bert_model()
        
        if batch_size is None:
            batch_size = self.model_configs['bert']['batch_size']
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Getting BERT embeddings"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch
                encoded = self.bert_tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.model_configs['bert']['max_length'],
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get embeddings
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_states = outputs.last_hidden_state
                
                # Apply pooling
                if pooling == 'cls':
                    batch_embeddings = last_hidden_states[:, 0, :]  # CLS token
                elif pooling == 'mean':
                    batch_embeddings = torch.mean(last_hidden_states, dim=1)
                elif pooling == 'max':
                    batch_embeddings = torch.max(last_hidden_states, dim=1)[0]
                else:
                    batch_embeddings = last_hidden_states[:, 0, :]  # Default to CLS
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_roberta_embeddings(
        self, 
        texts: List[str],
        pooling: str = 'cls',
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Get RoBERTa embeddings for texts.
        
        Args:
            texts: List of text documents
            pooling: Pooling strategy ('cls', 'mean', 'max')
            batch_size: Batch size for processing
            
        Returns:
            Document embeddings array
        """
        if self.roberta_model is None:
            self.load_roberta_model()
        
        if batch_size is None:
            batch_size = self.model_configs['roberta']['batch_size']
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Getting RoBERTa embeddings"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch
                encoded = self.roberta_tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.model_configs['roberta']['max_length'],
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get embeddings
                outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_states = outputs.last_hidden_state
                
                # Apply pooling
                if pooling == 'cls':
                    batch_embeddings = last_hidden_states[:, 0, :]  # CLS token
                elif pooling == 'mean':
                    batch_embeddings = torch.mean(last_hidden_states, dim=1)
                elif pooling == 'max':
                    batch_embeddings = torch.max(last_hidden_states, dim=1)[0]
                else:
                    batch_embeddings = last_hidden_states[:, 0, :]  # Default to CLS
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_sentence_bert_embeddings(
        self, 
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Get Sentence-BERT embeddings for texts.
        
        Args:
            texts: List of text documents
            batch_size: Batch size for processing
            
        Returns:
            Document embeddings array
        """
        if self.sentence_bert_model is None:
            self.load_sentence_bert_model()
        
        if batch_size is None:
            batch_size = self.model_configs['sentence_bert']['batch_size']
        
        logger.info("Getting Sentence-BERT embeddings...")
        embeddings = self.sentence_bert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def compare_embeddings(
        self,
        texts: List[str],
        tokenized_texts: Optional[List[List[str]]] = None,
        methods: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compare different embedding methods on the same texts.
        
        Args:
            texts: List of text documents
            tokenized_texts: Pre-tokenized texts for Word2Vec/Doc2Vec
            methods: List of methods to compare
            
        Returns:
            Dictionary mapping method names to embeddings
        """
        if methods is None:
            methods = ['sentence_bert', 'bert', 'roberta']
        
        results = {}
        
        for method in methods:
            logger.info(f"Generating {method} embeddings...")
            
            if method == 'word2vec':
                if tokenized_texts is None:
                    raise ValueError("tokenized_texts required for Word2Vec")
                if self.word2vec_model is None:
                    self.train_word2vec(tokenized_texts)
                results[method] = self.get_word2vec_embeddings(tokenized_texts)
                
            elif method == 'doc2vec':
                if tokenized_texts is None:
                    raise ValueError("tokenized_texts required for Doc2Vec")
                doc_ids = [f"doc_{i}" for i in range(len(texts))]
                if self.doc2vec_model is None:
                    self.train_doc2vec(tokenized_texts, doc_ids)
                results[method] = self.get_doc2vec_embeddings(doc_ids)
                
            elif method == 'bert':
                results[method] = self.get_bert_embeddings(texts)
                
            elif method == 'roberta':
                results[method] = self.get_roberta_embeddings(texts)
                
            elif method == 'sentence_bert':
                results[method] = self.get_sentence_bert_embeddings(texts)
                
            else:
                logger.warning(f"Unknown method: {method}")
        
        return results
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = 'pca',
        n_components: int = 50,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce dimensionality of embeddings.
        
        Args:
            embeddings: High-dimensional embeddings
            method: Reduction method ('pca', 'tsne')
            n_components: Number of output dimensions
            **kwargs: Additional parameters for the method
            
        Returns:
            Reduced embeddings
        """
        if method == 'pca':
            reducer = PCA(n_components=n_components, **kwargs)
            reduced = reducer.fit_transform(embeddings)
            
        elif method == 'tsne':
            # For t-SNE, first reduce with PCA if needed
            if embeddings.shape[1] > 50:
                pca = PCA(n_components=50)
                embeddings = pca.fit_transform(embeddings)
            
            reducer = TSNE(n_components=n_components, **kwargs)
            reduced = reducer.fit_transform(embeddings)
            
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        return reduced
    
    def save_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
        metadata: Optional[Dict] = None
    ):
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Dictionary of embeddings
            output_dir: Output directory
            metadata: Additional metadata to save
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for method, emb_array in embeddings.items():
            # Save embeddings
            emb_path = output_dir / f"{method}_embeddings.npy"
            np.save(emb_path, emb_array)
            
            # Save metadata
            meta_data = {
                'method': method,
                'shape': emb_array.shape,
                'dtype': str(emb_array.dtype),
                'generated_at': pd.Timestamp.now().isoformat()
            }
            if metadata:
                meta_data.update(metadata)
            
            meta_path = output_dir / f"{method}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
        
        logger.info(f"Embeddings saved to {output_dir}")
    
    def load_embeddings(
        self,
        input_dir: Union[str, Path],
        methods: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load embeddings from disk.
        
        Args:
            input_dir: Input directory
            methods: Specific methods to load
            
        Returns:
            Dictionary of loaded embeddings
        """
        input_dir = Path(input_dir)
        embeddings = {}
        
        # Find all embedding files
        embedding_files = list(input_dir.glob("*_embeddings.npy"))
        
        for emb_file in embedding_files:
            method = emb_file.stem.replace('_embeddings', '')
            
            if methods is None or method in methods:
                embeddings[method] = np.load(emb_file)
                logger.info(f"Loaded {method} embeddings: {embeddings[method].shape}")
        
        return embeddings