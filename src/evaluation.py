"""
Evaluation metrics and testing framework for the scholarly topic navigator.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import json
from pathlib import Path
from datetime import datetime
import time

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score,
    normalized_mutual_info_score
)

# Text evaluation
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Statistical tests
from scipy import stats

logger = logging.getLogger(__name__)


class SystemEvaluator:
    """
    Comprehensive evaluation framework for the scholarly topic navigator.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.evaluation_results = {}
        self.test_data = {}
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def evaluate_preprocessing(
        self,
        original_texts: List[str],
        processed_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate text preprocessing quality.
        
        Args:
            original_texts: Original input texts
            processed_results: Preprocessing results
            
        Returns:
            Evaluation metrics
        """
        metrics = {
            'total_documents': len(original_texts),
            'processing_stats': {},
            'quality_metrics': {}
        }
        
        # Processing statistics
        compression_ratios = []
        token_counts = []
        char_counts = []
        
        for i, (original, processed) in enumerate(zip(original_texts, processed_results)):
            if 'compression_ratio' in processed:
                compression_ratios.append(processed['compression_ratio'])
            
            if 'token_count' in processed:
                token_counts.append(processed['token_count'])
            
            if 'cleaned_length' in processed:
                char_counts.append(processed['cleaned_length'])
        
        if compression_ratios:
            metrics['processing_stats']['compression_ratio'] = {
                'mean': np.mean(compression_ratios),
                'std': np.std(compression_ratios),
                'min': np.min(compression_ratios),
                'max': np.max(compression_ratios)
            }
        
        if token_counts:
            metrics['processing_stats']['token_count'] = {
                'mean': np.mean(token_counts),
                'std': np.std(token_counts),
                'min': np.min(token_counts),
                'max': np.max(token_counts)
            }
        
        # Quality metrics
        non_empty_processed = sum(1 for result in processed_results 
                                 if result.get('cleaned_text', '').strip())
        
        metrics['quality_metrics']['processing_success_rate'] = non_empty_processed / len(original_texts)
        
        return metrics
    
    def evaluate_embeddings(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        similarity_pairs: Optional[List[Tuple[int, int, float]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate embedding quality.
        
        Args:
            embeddings_dict: Dictionary of embedding methods and arrays
            similarity_pairs: Optional human similarity judgments (idx1, idx2, similarity)
            
        Returns:
            Embedding evaluation metrics
        """
        metrics = {}
        
        for method, embeddings in embeddings_dict.items():
            method_metrics = {
                'shape': embeddings.shape,
                'dimensionality': embeddings.shape[1],
                'density': np.mean(embeddings != 0),
                'magnitude_stats': {
                    'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
                    'std_norm': np.std(np.linalg.norm(embeddings, axis=1))
                }
            }
            
            # Check for NaN or infinite values
            method_metrics['quality_checks'] = {
                'nan_count': np.sum(np.isnan(embeddings)),
                'inf_count': np.sum(np.isinf(embeddings)),
                'zero_vectors': np.sum(np.all(embeddings == 0, axis=1))
            }
            
            # Evaluate against similarity pairs if provided
            if similarity_pairs:
                correlations = self._evaluate_similarity_correlation(embeddings, similarity_pairs)
                method_metrics['similarity_correlation'] = correlations
            
            metrics[method] = method_metrics
        
        # Compare methods if multiple available
        if len(embeddings_dict) > 1:
            metrics['method_comparison'] = self._compare_embedding_methods(embeddings_dict)
        
        return metrics
    
    def _evaluate_similarity_correlation(
        self,
        embeddings: np.ndarray,
        similarity_pairs: List[Tuple[int, int, float]]
    ) -> Dict[str, float]:
        """Evaluate correlation with human similarity judgments."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        computed_similarities = []
        human_similarities = []
        
        for idx1, idx2, human_sim in similarity_pairs:
            if idx1 < len(embeddings) and idx2 < len(embeddings):
                emb1 = embeddings[idx1:idx1+1]
                emb2 = embeddings[idx2:idx2+1]
                computed_sim = cosine_similarity(emb1, emb2)[0][0]
                
                computed_similarities.append(computed_sim)
                human_similarities.append(human_sim)
        
        if len(computed_similarities) > 1:
            pearson_corr, _ = stats.pearsonr(computed_similarities, human_similarities)
            spearman_corr, _ = stats.spearmanr(computed_similarities, human_similarities)
            
            return {
                'pearson_correlation': pearson_corr,
                'spearman_correlation': spearman_corr,
                'num_pairs': len(computed_similarities)
            }
        
        return {'correlation': 0.0, 'num_pairs': 0}
    
    def _compare_embedding_methods(
        self,
        embeddings_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compare different embedding methods."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        methods = list(embeddings_dict.keys())
        comparison = {
            'methods_compared': methods,
            'pairwise_correlations': {}
        }
        
        # Calculate pairwise correlations between methods
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                emb1 = embeddings_dict[method1]
                emb2 = embeddings_dict[method2]
                
                # Sample subset for efficiency
                n_samples = min(1000, len(emb1), len(emb2))
                indices = np.random.choice(len(emb1), n_samples, replace=False)
                
                # Calculate cosine similarities for each method
                sim1 = cosine_similarity(emb1[indices])
                sim2 = cosine_similarity(emb2[indices])
                
                # Flatten upper triangular matrices
                upper_indices = np.triu_indices_from(sim1, k=1)
                sim1_flat = sim1[upper_indices]
                sim2_flat = sim2[upper_indices]
                
                # Calculate correlation
                corr, _ = stats.pearsonr(sim1_flat, sim2_flat)
                comparison['pairwise_correlations'][f'{method1}_vs_{method2}'] = corr
        
        return comparison
    
    def evaluate_clustering(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        method_name: str = 'unknown'
    ) -> Dict[str, Any]:
        """
        Evaluate clustering results.
        
        Args:
            embeddings: Document embeddings
            cluster_labels: Predicted cluster labels
            true_labels: Ground truth labels (if available)
            method_name: Name of clustering method
            
        Returns:
            Clustering evaluation metrics
        """
        metrics = {
            'method': method_name,
            'n_clusters': len(np.unique(cluster_labels[cluster_labels != -1])),
            'n_noise_points': np.sum(cluster_labels == -1)
        }
        
        # Internal evaluation metrics
        if len(np.unique(cluster_labels)) > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(embeddings, cluster_labels)
            except:
                metrics['silhouette_score'] = None
            
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, cluster_labels)
            except:
                metrics['calinski_harabasz_score'] = None
            
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, cluster_labels)
            except:
                metrics['davies_bouldin_score'] = None
        
        # External evaluation metrics (if ground truth available)
        if true_labels is not None:
            try:
                metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, cluster_labels)
                metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, cluster_labels)
            except:
                pass
        
        # Cluster size distribution
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        metrics['cluster_size_distribution'] = {
            'mean_size': np.mean(counts),
            'std_size': np.std(counts),
            'min_size': np.min(counts),
            'max_size': np.max(counts),
            'size_distribution': dict(zip(unique_labels.tolist(), counts.tolist()))
        }
        
        return metrics
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        method_name: str = 'unknown'
    ) -> Dict[str, Any]:
        """
        Evaluate classification results.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            class_names: Names of classes
            method_name: Name of classification method
            
        Returns:
            Classification evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics = {
            'method': method_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support.tolist() if hasattr(support, 'tolist') else support
        }
        
        # Detailed classification report
        report = classification_report(y_true, y_pred, output_dict=True, target_names=class_names)
        metrics['detailed_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        if class_names:
            precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
                y_true, y_pred, average=None
            )
            
            metrics['per_class_metrics'] = {}
            for i, class_name in enumerate(class_names):
                metrics['per_class_metrics'][class_name] = {
                    'precision': precision_pc[i],
                    'recall': recall_pc[i],
                    'f1_score': f1_pc[i],
                    'support': int(support_pc[i])
                }
        
        return metrics
    
    def evaluate_retrieval(
        self,
        retrieved_results: List[List[Dict[str, Any]]],
        relevant_docs: List[List[str]],
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, Any]:
        """
        Evaluate information retrieval results.
        
        Args:
            retrieved_results: List of retrieved results per query
            relevant_docs: List of relevant document IDs per query
            k_values: K values for precision@k and recall@k
            
        Returns:
            Retrieval evaluation metrics
        """
        metrics = {
            'num_queries': len(retrieved_results),
            'precision_at_k': {},
            'recall_at_k': {},
            'map_score': 0.0  # Mean Average Precision
        }
        
        all_ap_scores = []  # Average Precision scores for MAP calculation
        
        for k in k_values:
            precisions = []
            recalls = []
            
            for query_results, query_relevant in zip(retrieved_results, relevant_docs):
                # Get top-k results
                top_k_results = query_results[:k]
                retrieved_ids = [result.get('paper_id', '') for result in top_k_results]
                
                # Calculate precision@k
                relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in query_relevant)
                precision_k = relevant_retrieved / k if k > 0 else 0
                precisions.append(precision_k)
                
                # Calculate recall@k
                recall_k = relevant_retrieved / len(query_relevant) if len(query_relevant) > 0 else 0
                recalls.append(recall_k)
            
            metrics['precision_at_k'][f'P@{k}'] = np.mean(precisions)
            metrics['recall_at_k'][f'R@{k}'] = np.mean(recalls)
        
        # Calculate Mean Average Precision (MAP)
        for query_results, query_relevant in zip(retrieved_results, relevant_docs):
            ap_score = self._calculate_average_precision(query_results, query_relevant)
            all_ap_scores.append(ap_score)
        
        metrics['map_score'] = np.mean(all_ap_scores)
        
        return metrics
    
    def _calculate_average_precision(
        self,
        retrieved_results: List[Dict[str, Any]],
        relevant_docs: List[str]
    ) -> float:
        """Calculate Average Precision for a single query."""
        if not relevant_docs:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, result in enumerate(retrieved_results):
            doc_id = result.get('paper_id', '')
            if doc_id in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
    
    def evaluate_summarization(
        self,
        generated_summaries: List[str],
        reference_summaries: List[str],
        original_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate summarization quality.
        
        Args:
            generated_summaries: Generated summaries
            reference_summaries: Reference (ground truth) summaries
            original_texts: Original texts (for compression ratio)
            
        Returns:
            Summarization evaluation metrics
        """
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge_scores = {
            'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
            'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
        }
        
        bleu_scores = []
        
        # Calculate metrics for each summary pair
        for gen_sum, ref_sum in zip(generated_summaries, reference_summaries):
            # ROUGE scores
            scores = scorer.score(ref_sum, gen_sum)
            for rouge_type in rouge_scores:
                rouge_scores[rouge_type]['precision'].append(scores[rouge_type].precision)
                rouge_scores[rouge_type]['recall'].append(scores[rouge_type].recall)
                rouge_scores[rouge_type]['fmeasure'].append(scores[rouge_type].fmeasure)
            
            # BLEU score
            reference_tokens = [ref_sum.split()]
            candidate_tokens = gen_sum.split()
            
            smoothing_function = SmoothingFunction().method4
            bleu_score = sentence_bleu(
                reference_tokens, candidate_tokens,
                smoothing_function=smoothing_function
            )
            bleu_scores.append(bleu_score)
        
        # Aggregate metrics
        metrics = {
            'rouge_scores': {},
            'bleu_score': np.mean(bleu_scores),
            'num_summaries': len(generated_summaries)
        }
        
        for rouge_type in rouge_scores:
            metrics['rouge_scores'][rouge_type] = {
                'precision': np.mean(rouge_scores[rouge_type]['precision']),
                'recall': np.mean(rouge_scores[rouge_type]['recall']),
                'fmeasure': np.mean(rouge_scores[rouge_type]['fmeasure'])
            }
        
        # Compression ratio if original texts provided
        if original_texts:
            compression_ratios = []
            for gen_sum, orig_text in zip(generated_summaries, original_texts):
                ratio = len(gen_sum.split()) / len(orig_text.split()) if len(orig_text.split()) > 0 else 0
                compression_ratios.append(ratio)
            
            metrics['compression_ratio'] = {
                'mean': np.mean(compression_ratios),
                'std': np.std(compression_ratios),
                'min': np.min(compression_ratios),
                'max': np.max(compression_ratios)
            }
        
        return metrics
    
    def evaluate_system_performance(
        self,
        component_results: Dict[str, Dict[str, Any]],
        latency_measurements: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate overall system performance.
        
        Args:
            component_results: Results from each system component
            latency_measurements: Latency measurements per component
            
        Returns:
            System-level evaluation metrics
        """
        system_metrics = {
            'component_performance': component_results,
            'overall_score': 0.0,
            'performance_summary': {}
        }
        
        # Calculate component scores
        component_scores = {}
        
        for component, results in component_results.items():
            if component == 'classification':
                score = results.get('f1_score', 0)
            elif component == 'clustering':
                score = results.get('silhouette_score', 0)
            elif component == 'retrieval':
                score = results.get('map_score', 0)
            elif component == 'summarization':
                score = results.get('rouge_scores', {}).get('rouge1', {}).get('fmeasure', 0)
            else:
                score = 0
            
            component_scores[component] = score
        
        # Calculate weighted overall score
        weights = {
            'classification': 0.25,
            'clustering': 0.2,
            'retrieval': 0.3,
            'summarization': 0.25
        }
        
        overall_score = sum(
            weights.get(comp, 0) * score 
            for comp, score in component_scores.items()
        )
        
        system_metrics['overall_score'] = overall_score
        system_metrics['component_scores'] = component_scores
        
        # Add latency analysis
        if latency_measurements:
            latency_analysis = {}
            for component, measurements in latency_measurements.items():
                latency_analysis[component] = {
                    'mean_latency': np.mean(measurements),
                    'median_latency': np.median(measurements),
                    'p95_latency': np.percentile(measurements, 95),
                    'std_latency': np.std(measurements)
                }
            
            system_metrics['latency_analysis'] = latency_analysis
        
        return system_metrics
    
    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        include_metadata: bool = True
    ):
        """Save evaluation results to file."""
        output_path = Path(output_path)
        
        if include_metadata:
            output_data = {
                'evaluation_results': results,
                'metadata': {
                    'evaluated_at': datetime.now().isoformat(),
                    'evaluator_version': '1.0.0'
                }
            }
        else:
            output_data = results
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {output_path}")


class PerformanceBenchmark:
    """
    Performance benchmarking utilities.
    """
    
    def __init__(self):
        """Initialize benchmark."""
        self.benchmark_results = {}
    
    def benchmark_component(
        self,
        component_function: Callable,
        test_data: Any,
        num_runs: int = 5,
        warmup_runs: int = 1
    ) -> Dict[str, float]:
        """
        Benchmark a system component.
        
        Args:
            component_function: Function to benchmark
            test_data: Test data to pass to function
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results
        """
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                component_function(test_data)
            except:
                pass
        
        # Benchmark runs
        execution_times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            try:
                result = component_function(test_data)
                end_time = time.time()
                execution_times.append(end_time - start_time)
            except Exception as e:
                logger.error(f"Benchmark run failed: {e}")
                execution_times.append(float('inf'))
        
        # Calculate statistics
        valid_times = [t for t in execution_times if t != float('inf')]
        
        if valid_times:
            return {
                'mean_time': np.mean(valid_times),
                'median_time': np.median(valid_times),
                'min_time': np.min(valid_times),
                'max_time': np.max(valid_times),
                'std_time': np.std(valid_times),
                'successful_runs': len(valid_times),
                'total_runs': num_runs
            }
        else:
            return {
                'mean_time': float('inf'),
                'successful_runs': 0,
                'total_runs': num_runs
            }