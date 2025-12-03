"""
Class Definitions - Must Match Day 2 Exactly
These classes are needed for pickle deserialization.
"""

import numpy as np
import torch
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


class BM25Retriever:
    """BM25-based keyword retrieval."""

    def __init__(self, corpus=None):
        self.tokenized_corpus = None
        self.bm25 = None
        self.corpus = corpus

        if corpus is not None:
            self.tokenized_corpus = [doc.lower().split() for doc in corpus]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=10):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(idx, scores[idx]) for idx in top_indices]
        return results


class FAISSRetriever:
    """FAISS-based semantic vector search."""

    def __init__(self, embeddings=None, encoder_model='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(encoder_model)
        self.index = None
        self.dimension = None

        if embeddings is not None:
            self.embeddings = embeddings.astype('float32')
            self.dimension = embeddings.shape[1]
            faiss.normalize_L2(self.embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(self.embeddings)

    def search(self, query, top_k=10):
        query_embedding = self.encoder.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        return results

    def save_index(self, filepath):
        faiss.write_index(self.index, str(filepath))

    def load_index(self, filepath):
        self.index = faiss.read_index(str(filepath))
        self.dimension = self.index.d


class HybridRetriever:
    """Hybrid retrieval combining BM25 and semantic search."""

    def __init__(self, bm25_retriever, faiss_retriever, bm25_weight=0.3, semantic_weight=0.7):
        self.bm25 = bm25_retriever
        self.faiss = faiss_retriever
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    def search(self, query, top_k=10, expand_k=50):
        bm25_results = self.bm25.search(query, top_k=expand_k)
        faiss_results = self.faiss.search(query, top_k=expand_k)

        bm25_scores = {idx: score for idx, score in bm25_results}
        faiss_scores = {idx: score for idx, score in faiss_results}
        all_indices = set(bm25_scores.keys()) | set(faiss_scores.keys())

        bm25_max = max(bm25_scores.values()) if bm25_scores else 1
        faiss_max = max(faiss_scores.values()) if faiss_scores else 1

        combined_results = []
        for idx in all_indices:
            bm25_score = bm25_scores.get(idx, 0) / bm25_max if bm25_max > 0 else 0
            faiss_score = faiss_scores.get(idx, 0) / faiss_max if faiss_max > 0 else 0
            combined_score = self.bm25_weight * bm25_score + self.semantic_weight * faiss_score
            combined_results.append((idx, combined_score, bm25_score, faiss_score))

        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]

    def search_with_metadata(self, query, df, top_k=10):
        results = self.search(query, top_k=top_k)
        
        if 'original_abstract' in df.columns:
            abs_col = 'original_abstract'
        elif 'abstract' in df.columns:
            abs_col = 'abstract'
        else:
            abs_col = 'processed_text'

        enriched_results = []
        for idx, combined_score, bm25_score, semantic_score in results:
            paper = df.iloc[idx]
            enriched_results.append({
                'index': idx,
                'paper_id': paper['paper_id'],
                'title': paper['title'],
                'abstract': paper[abs_col],
                'authors': paper.get('authors', 'N/A'),
                'year': paper.get('year', 'N/A'),
                'category': paper.get('category', 'N/A'),
                'combined_score': combined_score,
                'bm25_score': bm25_score,
                'semantic_score': semantic_score
            })
        return enriched_results


class EmbeddingClassifier:
    """Fast classification using sentence embeddings + logistic regression."""

    def __init__(self, encoder_model='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(encoder_model)
        self.classifier = LogisticRegression(max_iter=1000, multi_class='multinomial', n_jobs=-1)
        self.label_encoder = None
        self.classes_ = None
        self.batch_size = 32 if not torch.cuda.is_available() else 64

    def fit(self, texts, labels, label_encoder=None):
        print(f"Encoding training texts...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True, batch_size=self.batch_size)
        print("Training classifier...")
        self.classifier.fit(embeddings, labels)
        self.label_encoder = label_encoder
        if label_encoder:
            self.classes_ = list(label_encoder.classes_)
        print("✓ Training complete")

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        predictions = self.classifier.predict(embeddings)
        if self.label_encoder:
            return [self.classes_[p] for p in predictions]
        return predictions

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        return self.classifier.predict_proba(embeddings)

    def predict_with_confidence(self, text):
        probs = self.predict_proba(text)[0]
        pred_idx = np.argmax(probs)
        return {
            'predicted_class': self.classes_[pred_idx] if self.classes_ else pred_idx,
            'confidence': float(probs[pred_idx]),
            'all_probabilities': {self.classes_[i]: float(p) for i, p in enumerate(probs)} if self.classes_ else {}
        }
