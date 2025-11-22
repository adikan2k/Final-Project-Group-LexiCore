"""
Document clustering using various algorithms and embeddings.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
from tqdm import tqdm
import logging

# Clustering algorithms
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    SpectralClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score,
    normalized_mutual_info_score
)

# Topic modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


class DocumentClusterer:
    """
    Document clustering with multiple algorithms and evaluation metrics.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize document clusterer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.cluster_models = {}
        self.cluster_results = {}
        self.evaluation_metrics = {}
        
        # Default clustering parameters
        self.clustering_configs = {
            'kmeans': {
                'n_clusters': 10,
                'random_state': random_state,
                'n_init': 10,
                'max_iter': 300
            },
            'dbscan': {
                'eps': 0.5,
                'min_samples': 5
            },
            'agglomerative': {
                'n_clusters': 10,
                'linkage': 'ward'
            },
            'spectral': {
                'n_clusters': 10,
                'random_state': random_state,
                'affinity': 'rbf'
            },
            'gaussian_mixture': {
                'n_components': 10,
                'random_state': random_state,
                'covariance_type': 'full'
            }
        }
    
    def cluster_documents(
        self,
        embeddings: np.ndarray,
        algorithm: str = 'kmeans',
        **kwargs
    ) -> np.ndarray:
        """
        Cluster documents using specified algorithm.
        
        Args:
            embeddings: Document embeddings
            algorithm: Clustering algorithm
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Cluster labels
        """
        # Get default config and update with kwargs
        config = self.clustering_configs.get(algorithm, {}).copy()
        config.update(kwargs)
        
        logger.info(f"Clustering with {algorithm} algorithm...")
        
        if algorithm == 'kmeans':
            model = KMeans(**config)
        elif algorithm == 'dbscan':
            model = DBSCAN(**config)
        elif algorithm == 'agglomerative':
            model = AgglomerativeClustering(**config)
        elif algorithm == 'spectral':
            model = SpectralClustering(**config)
        elif algorithm == 'gaussian_mixture':
            model = GaussianMixture(**config)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        # Fit and predict
        if algorithm == 'gaussian_mixture':
            labels = model.fit_predict(embeddings)
        else:
            labels = model.fit_predict(embeddings)
        
        # Store model and results
        self.cluster_models[algorithm] = model
        self.cluster_results[algorithm] = labels
        
        return labels
    
    def find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        algorithm: str = 'kmeans',
        k_range: Tuple[int, int] = (2, 20),
        metric: str = 'silhouette'
    ) -> Dict[str, Any]:
        """
        Find optimal number of clusters using various metrics.
        
        Args:
            embeddings: Document embeddings
            algorithm: Clustering algorithm
            k_range: Range of cluster numbers to test
            metric: Evaluation metric ('silhouette', 'calinski', 'davies')
            
        Returns:
            Results with optimal k and scores
        """
        k_min, k_max = k_range
        k_values = range(k_min, k_max + 1)
        scores = []
        
        logger.info(f"Finding optimal clusters for {algorithm} using {metric}...")
        
        for k in tqdm(k_values, desc="Testing cluster numbers"):
            if algorithm in ['dbscan']:
                # DBSCAN doesn't use n_clusters
                continue
                
            # Update config with current k
            config = self.clustering_configs.get(algorithm, {}).copy()
            if algorithm == 'gaussian_mixture':
                config['n_components'] = k
            else:
                config['n_clusters'] = k
            
            # Cluster
            if algorithm == 'kmeans':
                model = KMeans(**config)
            elif algorithm == 'agglomerative':
                model = AgglomerativeClustering(**config)
            elif algorithm == 'spectral':
                model = SpectralClustering(**config)
            elif algorithm == 'gaussian_mixture':
                model = GaussianMixture(**config)
            
            labels = model.fit_predict(embeddings)
            
            # Calculate score
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters
                if metric == 'silhouette':
                    score = silhouette_score(embeddings, labels)
                elif metric == 'calinski':
                    score = calinski_harabasz_score(embeddings, labels)
                elif metric == 'davies':
                    score = davies_bouldin_score(embeddings, labels)
                else:
                    score = silhouette_score(embeddings, labels)
            else:
                score = -1  # Invalid clustering
            
            scores.append(score)
        
        # Find optimal k
        if metric == 'davies':
            # Lower is better for Davies-Bouldin
            optimal_idx = np.argmin(scores)
        else:
            # Higher is better for silhouette and Calinski-Harabasz
            optimal_idx = np.argmax(scores)
        
        optimal_k = list(k_values)[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        return {
            'optimal_k': optimal_k,
            'optimal_score': optimal_score,
            'all_scores': dict(zip(k_values, scores)),
            'metric': metric,
            'algorithm': algorithm
        }
    
    def evaluate_clustering(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate clustering results with multiple metrics.
        
        Args:
            embeddings: Document embeddings
            labels: Predicted cluster labels
            true_labels: Ground truth labels (if available)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Internal evaluation metrics
        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = float(silhouette_score(embeddings, labels))
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(embeddings, labels))
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(embeddings, labels))
        
        # External evaluation metrics (if true labels available)
        if true_labels is not None:
            metrics['adjusted_rand_score'] = float(adjusted_rand_score(true_labels, labels))
            metrics['normalized_mutual_info'] = float(normalized_mutual_info_score(true_labels, labels))
        
        # Cluster statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['n_clusters'] = int(len(unique_labels))
        metrics['largest_cluster_size'] = int(counts.max())
        metrics['smallest_cluster_size'] = int(counts.min())
        metrics['avg_cluster_size'] = float(counts.mean())
        metrics['cluster_size_std'] = float(counts.std())
        
        # Noise points (for DBSCAN)
        noise_count = np.sum(labels == -1)
        metrics['noise_points'] = int(noise_count)
        metrics['noise_ratio'] = float(noise_count / len(labels) if len(labels) > 0 else 0)
        
        return metrics
    
    def topic_modeling_lda(
        self,
        texts: List[str],
        n_topics: int = 10,
        max_features: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform topic modeling using Latent Dirichlet Allocation.
        
        Args:
            texts: List of documents
            n_topics: Number of topics
            max_features: Maximum number of features for TF-IDF
            **kwargs: Additional LDA parameters
            
        Returns:
            Topic modeling results
        """
        logger.info(f"Performing LDA topic modeling with {n_topics} topics...")
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Fit LDA model
        lda_config = {
            'n_components': n_topics,
            'random_state': self.random_state,
            'max_iter': 100,
            'learning_decay': 0.7,
            'learning_offset': 50.0
        }
        lda_config.update(kwargs)
        
        lda_model = LatentDirichletAllocation(**lda_config)
        doc_topic_probs = lda_model.fit_transform(doc_term_matrix)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_probs = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'probabilities': top_probs
            })
        
        # Assign dominant topics to documents
        dominant_topics = np.argmax(doc_topic_probs, axis=1)
        
        return {
            'model': lda_model,
            'vectorizer': vectorizer,
            'doc_topic_probs': doc_topic_probs,
            'dominant_topics': dominant_topics,
            'topics': topics,
            'perplexity': lda_model.perplexity(doc_term_matrix),
            'log_likelihood': lda_model.score(doc_term_matrix)
        }
    
    def topic_modeling_nmf(
        self,
        texts: List[str],
        n_topics: int = 10,
        max_features: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform topic modeling using Non-negative Matrix Factorization.
        
        Args:
            texts: List of documents
            n_topics: Number of topics
            max_features: Maximum number of features
            **kwargs: Additional NMF parameters
            
        Returns:
            Topic modeling results
        """
        logger.info(f"Performing NMF topic modeling with {n_topics} topics...")
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Fit NMF model
        nmf_config = {
            'n_components': n_topics,
            'random_state': self.random_state,
            'max_iter': 200,
            'alpha': 0.1,
            'l1_ratio': 0.5
        }
        nmf_config.update(kwargs)
        
        nmf_model = NMF(**nmf_config)
        doc_topic_probs = nmf_model.fit_transform(doc_term_matrix)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': top_weights
            })
        
        # Assign dominant topics to documents
        dominant_topics = np.argmax(doc_topic_probs, axis=1)
        
        # Calculate reconstruction error
        reconstruction_error = nmf_model.reconstruction_err_
        
        return {
            'model': nmf_model,
            'vectorizer': vectorizer,
            'doc_topic_probs': doc_topic_probs,
            'dominant_topics': dominant_topics,
            'topics': topics,
            'reconstruction_error': reconstruction_error
        }
    
    def visualize_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = 'tsne',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize clusters in 2D space.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Cluster labels
            method: Dimensionality reduction method ('tsne', 'pca')
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Reduce dimensions to 2D
        if method == 'tsne':
            if embeddings.shape[1] > 50:
                # Pre-reduce with PCA for t-SNE
                pca = PCA(n_components=50, random_state=self.random_state)
                embeddings_reduced = pca.fit_transform(embeddings)
            else:
                embeddings_reduced = embeddings
                
            reducer = TSNE(
                n_components=2,
                random_state=self.random_state,
                perplexity=min(30, len(embeddings) - 1)
            )
            coords_2d = reducer.fit_transform(embeddings_reduced)
            
        elif method == 'pca':
            pca = PCA(n_components=2, random_state=self.random_state)
            coords_2d = pca.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot clusters
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points (for DBSCAN)
                mask = labels == label
                ax.scatter(
                    coords_2d[mask, 0], coords_2d[mask, 1],
                    c='black', marker='x', s=50, alpha=0.5, label='Noise'
                )
            else:
                mask = labels == label
                ax.scatter(
                    coords_2d[mask, 0], coords_2d[mask, 1],
                    c=[color], s=50, alpha=0.7, label=f'Cluster {label}'
                )
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(title or f'Document Clusters ({method.upper()})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_cluster_summaries(
        self,
        texts: List[str],
        labels: np.ndarray,
        paper_data: Optional[pd.DataFrame] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate summaries for each cluster.
        
        Args:
            texts: Original text documents
            labels: Cluster labels
            paper_data: Optional paper metadata
            
        Returns:
            Dictionary with cluster summaries
        """
        summaries = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            mask = labels == label
            cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
            
            # Basic statistics
            summary = {
                'cluster_id': int(label),
                'size': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(labels) * 100)
            }
            
            # Text statistics
            text_lengths = [len(text.split()) for text in cluster_texts]
            summary['avg_text_length'] = np.mean(text_lengths)
            summary['text_length_std'] = np.std(text_lengths)
            
            # Extract key terms using TF-IDF
            if len(cluster_texts) > 1:
                vectorizer = TfidfVectorizer(
                    max_features=20,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                try:
                    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Get average TF-IDF scores
                    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                    top_indices = mean_scores.argsort()[-10:][::-1]
                    
                    summary['key_terms'] = [
                        {
                            'term': feature_names[idx],
                            'score': float(mean_scores[idx])
                        }
                        for idx in top_indices
                    ]
                except:
                    summary['key_terms'] = []
            else:
                summary['key_terms'] = []
            
            # Paper metadata statistics (if provided)
            if paper_data is not None:
                cluster_papers = paper_data[mask]
                
                # Year distribution
                if 'year' in cluster_papers.columns:
                    year_counts = cluster_papers['year'].value_counts()
                    summary['year_distribution'] = year_counts.to_dict()
                    summary['avg_year'] = cluster_papers['year'].mean()
                
                # Venue distribution
                if 'venue' in cluster_papers.columns:
                    venue_counts = cluster_papers['venue'].value_counts()
                    summary['top_venues'] = venue_counts.head(5).to_dict()
                
                # Category distribution
                if 'categories' in cluster_papers.columns:
                    all_categories = []
                    for categories in cluster_papers['categories']:
                        if isinstance(categories, list):
                            all_categories.extend(categories)
                        elif isinstance(categories, str):
                            all_categories.append(categories)
                    
                    from collections import Counter
                    cat_counts = Counter(all_categories)
                    summary['top_categories'] = dict(cat_counts.most_common(5))
            
            summaries[int(label)] = summary
        
        return summaries
    
    def save_clustering_results(
        self,
        output_dir: Union[str, Path],
        embeddings: np.ndarray,
        labels: np.ndarray,
        texts: Optional[List[str]] = None,
        paper_data: Optional[pd.DataFrame] = None,
        algorithm: str = 'unknown'
    ):
        """
        Save clustering results to disk.
        
        Args:
            output_dir: Output directory
            embeddings: Document embeddings
            labels: Cluster labels
            texts: Original texts
            paper_data: Paper metadata
            algorithm: Clustering algorithm used
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cluster labels
        np.save(output_dir / 'cluster_labels.npy', labels)
        
        # Save evaluation metrics
        metrics = self.evaluate_clustering(embeddings, labels)
        with open(output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save cluster summaries
        if texts is not None:
            summaries = self.get_cluster_summaries(texts, labels, paper_data)
            with open(output_dir / 'cluster_summaries.json', 'w') as f:
                json.dump(summaries, f, indent=2, default=str)
        
        # Save visualization
        try:
            fig = self.visualize_clusters(
                embeddings, labels,
                title=f'{algorithm.title()} Clustering Results'
            )
            fig.savefig(output_dir / 'cluster_visualization.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not save visualization: {e}")
        
        # Save metadata
        metadata = {
            'algorithm': algorithm,
            'n_documents': len(labels),
            'n_clusters': len(np.unique(labels[labels != -1])),
            'embedding_dim': embeddings.shape[1],
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        with open(output_dir / 'clustering_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Clustering results saved to {output_dir}")