"""
Text classification and zero-shot classification for academic papers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import logging
from tqdm import tqdm
import pickle

# Traditional ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)

# Deep Learning
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, TrainingArguments, Trainer
)

# Zero-shot classification
from transformers import AutoModelForSequenceClassification as ZeroShotModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PaperClassifier:
    """
    Multi-approach text classification for academic papers.
    """
    
    def __init__(self, device: str = 'auto', random_state: int = 42):
        """
        Initialize paper classifier.
        
        Args:
            device: Device to use for neural models
            random_state: Random state for reproducibility
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.random_state = random_state
        
        # Model containers
        self.traditional_models = {}
        self.neural_models = {}
        self.vectorizers = {}
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        
        # Model configurations
        self.model_configs = {
            'logistic_regression': {
                'random_state': random_state,
                'max_iter': 1000,
                'multi_class': 'auto'
            },
            'svm': {
                'random_state': random_state,
                'probability': True,
                'kernel': 'rbf'
            },
            'random_forest': {
                'random_state': random_state,
                'n_estimators': 100,
                'max_depth': None
            },
            'gradient_boosting': {
                'random_state': random_state,
                'n_estimators': 100,
                'learning_rate': 0.1
            },
            'naive_bayes': {
                'alpha': 1.0
            }
        }
    
    def prepare_features(
        self,
        texts: List[str],
        method: str = 'tfidf',
        max_features: int = 5000,
        fit_vectorizer: bool = True,
        vectorizer_name: str = 'default'
    ) -> np.ndarray:
        """
        Prepare text features for traditional ML models.
        
        Args:
            texts: List of text documents
            method: Feature extraction method ('tfidf', 'bow')
            max_features: Maximum number of features
            fit_vectorizer: Whether to fit a new vectorizer
            vectorizer_name: Name to store/retrieve vectorizer
            
        Returns:
            Feature matrix
        """
        if method == 'tfidf':
            if fit_vectorizer or vectorizer_name not in self.vectorizers:
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8,
                    sublinear_tf=True
                )
                features = vectorizer.fit_transform(texts)
                self.vectorizers[vectorizer_name] = vectorizer
            else:
                vectorizer = self.vectorizers[vectorizer_name]
                features = vectorizer.transform(texts)
        else:
            raise ValueError(f"Unknown feature method: {method}")
        
        return features.toarray()
    
    def train_traditional_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        algorithm: str = 'logistic_regression',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train traditional ML classifier.
        
        Args:
            X: Feature matrix
            y: Target labels
            algorithm: Algorithm to use
            **kwargs: Additional parameters
            
        Returns:
            Training results
        """
        # Get config and update with kwargs
        config = self.model_configs.get(algorithm, {}).copy()
        config.update(kwargs)
        
        logger.info(f"Training {algorithm} classifier...")
        
        # Initialize model
        if algorithm == 'logistic_regression':
            model = LogisticRegression(**config)
        elif algorithm == 'svm':
            model = SVC(**config)
        elif algorithm == 'random_forest':
            model = RandomForestClassifier(**config)
        elif algorithm == 'gradient_boosting':
            model = GradientBoostingClassifier(**config)
        elif algorithm == 'naive_bayes':
            model = MultinomialNB(**config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Train model
        model.fit(X, y)
        self.traditional_models[algorithm] = model
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        results = {
            'algorithm': algorithm,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        self.training_results[algorithm] = results
        return results
    
    def evaluate_classifier(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str = 'traditional'
    ) -> Dict[str, Any]:
        """
        Evaluate trained classifier.
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            model_type: Type of model ('traditional', 'neural')
            
        Returns:
            Evaluation results
        """
        if model_type == 'traditional':
            model = self.traditional_models.get(model_name)
        else:
            model = self.neural_models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probabilities (if available)
        try:
            y_pred_proba = model.predict_proba(X_test)
        except:
            y_pred_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Add AUC metrics for multi-class if probabilities available
        if y_pred_proba is not None:
            try:
                # For multi-class, use one-vs-rest approach
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_auc_score
                
                classes = np.unique(y_test)
                if len(classes) > 2:
                    y_test_bin = label_binarize(y_test, classes=classes)
                    auc_score = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
                else:
                    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                
                results['auc_score'] = auc_score
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
        
        self.evaluation_results[model_name] = results
        return results
    
    def compare_classifiers(
        self,
        texts: List[str],
        labels: List[str],
        test_size: float = 0.2,
        algorithms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple classification algorithms.
        
        Args:
            texts: List of text documents
            labels: List of labels
            test_size: Test set size
            algorithms: List of algorithms to compare
            
        Returns:
            Comparison results
        """
        if algorithms is None:
            algorithms = ['logistic_regression', 'random_forest', 'svm', 'naive_bayes']
        
        logger.info(f"Comparing {len(algorithms)} classifiers...")
        
        # Prepare features
        features = self.prepare_features(texts, vectorizer_name='comparison')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        # Train and evaluate each algorithm
        results = {}
        
        for algorithm in algorithms:
            logger.info(f"Training {algorithm}...")
            
            # Train
            train_result = self.train_traditional_classifier(
                X_train, y_train, algorithm
            )
            
            # Evaluate
            eval_result = self.evaluate_classifier(
                algorithm, X_test, y_test, 'traditional'
            )
            
            # Combine results
            results[algorithm] = {
                'training': train_result,
                'evaluation': eval_result
            }
        
        # Create comparison summary
        comparison = {
            'algorithms_compared': algorithms,
            'dataset_size': len(texts),
            'n_classes': len(np.unique(labels)),
            'test_size': test_size,
            'results': results
        }
        
        # Best model summary
        best_model = max(
            algorithms,
            key=lambda alg: results[alg]['evaluation']['f1_score']
        )
        
        comparison['best_model'] = {
            'algorithm': best_model,
            'f1_score': results[best_model]['evaluation']['f1_score'],
            'accuracy': results[best_model]['evaluation']['accuracy']
        }
        
        return comparison


class ZeroShotClassifier:
    """
    Zero-shot text classification using pre-trained models.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize zero-shot classifier.
        
        Args:
            device: Device to use
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model containers
        self.classification_pipeline = None
        self.sentence_model = None
        
        # Default models
        self.default_models = {
            'zero_shot': 'facebook/bart-large-mnli',
            'sentence_similarity': 'all-MiniLM-L6-v2'
        }
    
    def load_zero_shot_pipeline(self, model_name: Optional[str] = None):
        """Load zero-shot classification pipeline."""
        if model_name is None:
            model_name = self.default_models['zero_shot']
        
        logger.info(f"Loading zero-shot model: {model_name}")
        self.classification_pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if self.device.type == 'cuda' else -1
        )
    
    def load_sentence_model(self, model_name: Optional[str] = None):
        """Load sentence transformer model for similarity-based classification."""
        if model_name is None:
            model_name = self.default_models['sentence_similarity']
        
        logger.info(f"Loading sentence model: {model_name}")
        self.sentence_model = SentenceTransformer(model_name, device=str(self.device))
    
    def classify_zero_shot(
        self,
        texts: List[str],
        candidate_labels: List[str],
        hypothesis_template: str = "This paper is about {}.",
        multi_label: bool = False,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Perform zero-shot classification on texts.
        
        Args:
            texts: List of texts to classify
            candidate_labels: List of possible labels
            hypothesis_template: Template for hypothesis generation
            multi_label: Whether to allow multiple labels per text
            batch_size: Batch size for processing
            
        Returns:
            List of classification results
        """
        if self.classification_pipeline is None:
            self.load_zero_shot_pipeline()
        
        results = []
        
        logger.info(f"Performing zero-shot classification on {len(texts)} texts...")
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Zero-shot classification"):
            batch_texts = texts[i:i+batch_size]
            
            batch_results = self.classification_pipeline(
                batch_texts,
                candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=multi_label
            )
            
            # Handle single text vs batch results
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            
            results.extend(batch_results)
        
        return results
    
    def classify_by_similarity(
        self,
        texts: List[str],
        label_descriptions: Dict[str, str],
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Classify texts using semantic similarity to label descriptions.
        
        Args:
            texts: List of texts to classify
            label_descriptions: Dictionary mapping labels to descriptions
            threshold: Similarity threshold for classification
            
        Returns:
            List of classification results
        """
        if self.sentence_model is None:
            self.load_sentence_model()
        
        logger.info(f"Performing similarity-based classification on {len(texts)} texts...")
        
        # Encode texts and label descriptions
        text_embeddings = self.sentence_model.encode(texts)
        label_embeddings = self.sentence_model.encode(list(label_descriptions.values()))
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(text_embeddings, label_embeddings)
        
        # Generate results
        results = []
        labels = list(label_descriptions.keys())
        
        for i, text in enumerate(texts):
            text_similarities = similarities[i]
            
            # Find best match
            best_idx = np.argmax(text_similarities)
            best_score = text_similarities[best_idx]
            best_label = labels[best_idx]
            
            # Create result
            result = {
                'sequence': text,
                'labels': labels,
                'scores': text_similarities.tolist()
            }
            
            # Add prediction
            if best_score >= threshold:
                result['predicted_label'] = best_label
                result['confidence'] = float(best_score)
            else:
                result['predicted_label'] = 'unknown'
                result['confidence'] = float(best_score)
            
            results.append(result)
        
        return results
    
    def evaluate_zero_shot(
        self,
        texts: List[str],
        true_labels: List[str],
        candidate_labels: List[str],
        method: str = 'pipeline'
    ) -> Dict[str, Any]:
        """
        Evaluate zero-shot classification performance.
        
        Args:
            texts: List of texts
            true_labels: True labels
            candidate_labels: Candidate labels for classification
            method: Classification method ('pipeline', 'similarity')
            
        Returns:
            Evaluation results
        """
        if method == 'pipeline':
            results = self.classify_zero_shot(texts, candidate_labels)
            predicted_labels = [r['labels'][np.argmax(r['scores'])] for r in results]
            
        elif method == 'similarity':
            # Create simple label descriptions
            label_descriptions = {label: f"This is about {label}" for label in candidate_labels}
            results = self.classify_by_similarity(texts, label_descriptions)
            predicted_labels = [r['predicted_label'] for r in results]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate metrics
        from sklearn.metrics import classification_report, accuracy_score
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        
        evaluation = {
            'method': method,
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': predicted_labels,
            'detailed_results': results
        }
        
        return evaluation
    
    def adaptive_classification(
        self,
        texts: List[str],
        base_labels: List[str],
        confidence_threshold: float = 0.7,
        similarity_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Adaptive classification that combines multiple approaches.
        
        Args:
            texts: Texts to classify
            base_labels: Base set of labels
            confidence_threshold: Threshold for zero-shot confidence
            similarity_threshold: Threshold for similarity-based classification
            
        Returns:
            Classification results with adaptation info
        """
        logger.info("Performing adaptive zero-shot classification...")
        
        # Primary classification with zero-shot pipeline
        primary_results = self.classify_zero_shot(texts, base_labels)
        
        # Identify low-confidence predictions
        low_confidence_indices = []
        final_predictions = []
        
        for i, result in enumerate(primary_results):
            max_score = max(result['scores'])
            if max_score >= confidence_threshold:
                # High confidence - use pipeline result
                predicted_label = result['labels'][np.argmax(result['scores'])]
                final_predictions.append({
                    'text_id': i,
                    'predicted_label': predicted_label,
                    'confidence': max_score,
                    'method': 'zero_shot_pipeline'
                })
            else:
                # Low confidence - mark for secondary method
                low_confidence_indices.append(i)
                final_predictions.append({
                    'text_id': i,
                    'predicted_label': None,
                    'confidence': max_score,
                    'method': 'pending'
                })
        
        # Secondary classification for low-confidence cases
        if low_confidence_indices:
            logger.info(f"Applying similarity-based classification to {len(low_confidence_indices)} low-confidence cases...")
            
            low_conf_texts = [texts[i] for i in low_confidence_indices]
            label_descriptions = {label: f"Research about {label}" for label in base_labels}
            
            similarity_results = self.classify_by_similarity(
                low_conf_texts, label_descriptions, similarity_threshold
            )
            
            # Update final predictions
            for idx, sim_result in zip(low_confidence_indices, similarity_results):
                final_predictions[idx].update({
                    'predicted_label': sim_result['predicted_label'],
                    'confidence': sim_result['confidence'],
                    'method': 'similarity_based'
                })
        
        return {
            'predictions': final_predictions,
            'primary_results': primary_results,
            'low_confidence_count': len(low_confidence_indices),
            'methods_used': ['zero_shot_pipeline', 'similarity_based'],
            'thresholds': {
                'confidence_threshold': confidence_threshold,
                'similarity_threshold': similarity_threshold
            }
        }
    
    def save_classification_results(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path]
    ):
        """Save classification results to file."""
        output_path = Path(output_path)
        
        # Create serializable version
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.int64, np.float64)):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Classification results saved to {output_path}")