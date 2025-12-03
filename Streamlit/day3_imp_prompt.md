# 📚 Scholarly Topic Navigator - Day 3 Implementation Prompt

## Complete Technical Specification for Integration, UI, and Explainability

---

## ⚠️ CRITICAL FILE SAFETY RULES

**READ THIS FIRST - NON-NEGOTIABLE:**

1. **CREATE a new folder called `Day3`** inside the project root directory
2. **ALL Day 3 code files MUST be created ONLY inside the `Day3` folder**
3. **DO NOT modify, edit, delete, or overwrite ANY files outside of the `Day3` folder**
4. **DO NOT touch any files in:**
   - `Complete_day2_deliverables/` folder
   - `data/` folder (except reading from it)
   - `code/` folder
   - `src/` folder
   - `notebooks/` folder
   - `simple_demo.py`
   - `README.md`
   - `LICENSE`
   - Any other existing files

5. **You may READ from:**
   - `Complete_day2_deliverables/` - to understand Day 2 class structures
   - `data/processed/` - to load the processed data
   - `data/retrieval/` - to load retrieval indices
   - `data/embeddings/` - to load embedding files
   - `models/` - to load trained models

---

## 📁 Required Folder Structure

```
Final-Project-Group-LexiCore/
│
├── Day3/                          ← CREATE THIS FOLDER (All Day 3 code goes here)
│   ├── Day_3_Complete_Pipeline.ipynb   ← Main Jupyter notebook
│   ├── app.py                          ← Streamlit application
│   ├── utils/                          ← Utility modules
│   │   ├── __init__.py
│   │   ├── class_definitions.py        ← Exact class copies from Day 2
│   │   ├── summarization.py            ← Summarization engine
│   │   ├── explainability.py           ← LIME/SHAP explainers
│   │   └── visualizations.py           ← Visualization helpers
│   └── requirements_day3.txt           ← Day 3 specific requirements
│
├── Complete_day2_deliverables/    ← DO NOT MODIFY (Read Only)
│   ├── (Day 2 notebooks and code)
│   └── ...
│
├── data/                          ← DO NOT MODIFY (Read Only for loading)
│   ├── processed/
│   │   └── papers_with_topics.parquet
│   ├── retrieval/
│   │   ├── faiss_index.bin
│   │   ├── faiss_id_mapping.pkl
│   │   └── bm25_retriever.pkl
│   └── embeddings/
│       └── sbert_abstract_embeddings.npy
│
├── models/                        ← DO NOT MODIFY (Read Only for loading)
│   └── embedding_classifier.pkl
│
└── (other existing folders - DO NOT TOUCH)
```

---

## 📋 Day 3 Master Plan (From JSON Specification)

### Goal
Synthesize the Intelligence Layer (Day 2) into a user-facing application with Summarization and Explainability features.

### Academic Value
Demonstrates system integration, model interpretability (LIME), and comparative summarization (Extractive vs. Abstractive).

---

## 📦 Prerequisites & Dependencies

### Required Libraries (Day 3 Specific)

```python
# Install command (run at the beginning of the notebook)
!pip install streamlit sumy lime transformers torch matplotlib plotly \
    sentence-transformers rank_bm25 faiss-cpu nltk wordcloud shap \
    --quiet
```

### NLTK Data Downloads

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

---

## 🔧 Task 1: Define Class Wrappers

### Critical Requirement
You MUST re-declare the **exact same classes** from Day 2 in your Day 3 code so that `pickle` can deserialize the saved objects without errors.

### ⚠️ Important: Locate Day 2 Class Definitions

Before writing any code, you MUST:

1. **Open and read** the Day 2 notebook from `Complete_day2_deliverables/`
2. **Find the exact class definitions** for:
   - `BM25Retriever`
   - `FAISSRetriever`
   - `HybridRetriever`
   - `EmbeddingClassifier`

3. **Copy them EXACTLY** - do not change:
   - Class names
   - Method signatures
   - Method names
   - Attribute names
   - Import statements used inside the classes

### Template (Adjust Based on Actual Day 2 Code)

```python
# ============================================================
# CLASS DEFINITIONS - MUST MATCH DAY 2 EXACTLY
# ============================================================

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

class BM25Retriever:
    """
    BM25 keyword-based retriever.
    COPY THE EXACT IMPLEMENTATION FROM DAY 2.
    """
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
        return [(idx, scores[idx]) for idx in top_indices]


class FAISSRetriever:
    """
    FAISS semantic vector retriever.
    COPY THE EXACT IMPLEMENTATION FROM DAY 2.
    """
    def __init__(self, embeddings=None, encoder_model='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(encoder_model)
        self.index = None
        if embeddings is not None:
            self.embeddings = embeddings.astype('float32')
            self.dimension = embeddings.shape[1]
            faiss.normalize_L2(self.embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(self.embeddings)
    
    def search(self, query, top_k=10):
        query_vec = self.encoder.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
    
    def load_index(self, filepath):
        self.index = faiss.read_index(filepath)


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and FAISS.
    COPY THE EXACT IMPLEMENTATION FROM DAY 2.
    """
    def __init__(self, bm25_retriever, faiss_retriever, bm25_weight=0.3, semantic_weight=0.7):
        self.bm25 = bm25_retriever
        self.faiss = faiss_retriever
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
    
    def search(self, query, top_k=10):
        bm25_res = self.bm25.search(query, top_k=50)
        faiss_res = self.faiss.search(query, top_k=50)
        
        bm25_scores = {idx: score for idx, score in bm25_res}
        faiss_scores = {idx: score for idx, score in faiss_res}
        
        all_indices = set(bm25_scores.keys()) | set(faiss_scores.keys())
        
        bm25_max = max(bm25_scores.values()) if bm25_scores else 1
        faiss_max = max(faiss_scores.values()) if faiss_scores else 1
        
        combined = []
        for idx in all_indices:
            b_score = bm25_scores.get(idx, 0) / bm25_max
            f_score = faiss_scores.get(idx, 0) / faiss_max
            final = self.bm25_weight * b_score + self.semantic_weight * f_score
            combined.append((idx, final))
        
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]


class EmbeddingClassifier:
    """
    SBERT-based embedding classifier.
    COPY THE EXACT IMPLEMENTATION FROM DAY 2.
    """
    def __init__(self, encoder_model='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(encoder_model)
        self.classifier = None
        self.label_encoder = None
        self.classes_ = None
        self.batch_size = 32  # If Day 2 had adaptive batch sizing
    
    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        return self.classifier.predict_proba(embeddings)
    
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        return self.classifier.predict(embeddings)
```

---

## 🔧 Task 2: Implement Summarization Engine

### 2.1 Extractive Summarization (TextRank)

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def get_extractive_summary(text, num_sentences=3, language='english'):
    """
    Generate extractive summary using TextRank algorithm.
    
    Args:
        text: Input text to summarize
        num_sentences: Number of sentences in summary
        language: Language for tokenization
    
    Returns:
        str: Extractive summary
    """
    try:
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        stemmer = Stemmer(language)
        summarizer = TextRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(language)
        
        summary_sentences = summarizer(parser.document, num_sentences)
        return ' '.join([str(sentence) for sentence in summary_sentences])
    except Exception as e:
        print(f"Extractive summarization error: {e}")
        # Fallback: return first N sentences
        sentences = text.split('.')[:num_sentences]
        return '. '.join(sentences) + '.'
```

### 2.2 Abstractive Summarization (BART)

```python
from transformers import pipeline
import torch

def load_abstractive_summarizer():
    """
    Load BART summarization pipeline with hardware optimization.
    """
    device = 0 if torch.cuda.is_available() else -1
    
    summarizer = pipeline(
        'summarization',
        model='facebook/bart-large-cnn',
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return summarizer


def get_abstractive_summary(text, summarizer, max_length=130, min_length=30):
    """
    Generate abstractive summary using BART.
    
    Args:
        text: Input text to summarize
        summarizer: Loaded BART pipeline
        max_length: Maximum summary length
        min_length: Minimum summary length
    
    Returns:
        str: Abstractive summary
    """
    try:
        # BART has a max input length of 1024 tokens
        # Truncate input if too long
        text_truncated = text[:1024]
        
        result = summarizer(
            text_truncated,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return result[0]['summary_text']
    except Exception as e:
        print(f"Abstractive summarization error: {e}")
        return get_extractive_summary(text, num_sentences=2)


def generate_multi_level_summary(text, summarizer):
    """
    Generate summaries at multiple granularities.
    
    Returns:
        dict: Contains 1-sentence, 3-sentence, and 5-bullet summaries
    """
    return {
        'one_sentence': get_abstractive_summary(text, summarizer, max_length=50, min_length=20),
        'three_sentence': get_extractive_summary(text, num_sentences=3),
        'five_bullet': generate_bullet_insights(text, summarizer)
    }


def generate_bullet_insights(text, summarizer, num_bullets=5):
    """
    Generate bullet-point insights from text.
    """
    # Get extractive sentences
    extractive = get_extractive_summary(text, num_sentences=num_bullets)
    sentences = extractive.split('. ')
    
    bullets = []
    for i, sent in enumerate(sentences[:num_bullets]):
        if sent.strip():
            bullets.append(f"• {sent.strip()}")
    
    return '\n'.join(bullets)
```

---

## 🔧 Task 3: Implement Explainability (LIME)

### LIME Text Explainer

```python
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import numpy as np

class ClassificationExplainer:
    """
    LIME-based explainability for text classification.
    """
    
    def __init__(self, classifier, class_names):
        """
        Initialize explainer.
        
        Args:
            classifier: Trained EmbeddingClassifier with predict_proba method
            class_names: List of class labels
        """
        self.classifier = classifier
        self.class_names = class_names
        self.explainer = LimeTextExplainer(
            class_names=class_names,
            split_expression=r'\W+',  # Split on non-word characters
            bow=True
        )
    
    def explain_prediction(self, text, num_features=10, num_samples=500):
        """
        Generate LIME explanation for a prediction.
        
        Args:
            text: Input text to explain
            num_features: Number of top features to show
            num_samples: Number of perturbations for LIME
        
        Returns:
            explanation: LIME explanation object
        """
        explanation = self.explainer.explain_instance(
            text,
            self.classifier.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        return explanation
    
    def visualize_explanation(self, explanation, figsize=(10, 6)):
        """
        Create matplotlib visualization of LIME explanation.
        
        Returns:
            fig: matplotlib figure
        """
        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(figsize)
        plt.tight_layout()
        return fig
    
    def get_word_importance(self, explanation):
        """
        Extract word importance scores from explanation.
        
        Returns:
            list: List of (word, importance_score) tuples
        """
        return explanation.as_list()
    
    def highlight_text(self, text, explanation, predicted_class_idx=None):
        """
        Generate HTML with highlighted important words.
        
        Returns:
            str: HTML string with color-coded words
        """
        if predicted_class_idx is None:
            predicted_class_idx = 0
        
        word_weights = dict(explanation.as_list(label=predicted_class_idx))
        words = text.split()
        
        html_parts = []
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word.lower() in word_weights:
                weight = word_weights[clean_word.lower()]
                if weight > 0:
                    color = f'rgba(0, 255, 0, {min(abs(weight), 1)})'  # Green for positive
                else:
                    color = f'rgba(255, 0, 0, {min(abs(weight), 1)})'  # Red for negative
                html_parts.append(f'<span style="background-color: {color}">{word}</span>')
            else:
                html_parts.append(word)
        
        return ' '.join(html_parts)
```

### SHAP Explainer (Optional Enhancement)

```python
# Note: SHAP requires more setup but provides global feature importance

try:
    import shap
    
    def get_shap_explanation(classifier, texts, background_texts=None):
        """
        Generate SHAP explanations for predictions.
        
        Args:
            classifier: Trained classifier
            texts: Texts to explain
            background_texts: Background dataset for SHAP
        
        Returns:
            shap_values: SHAP explanation values
        """
        if background_texts is None:
            background_texts = texts[:100]  # Use subset as background
        
        # Create SHAP explainer
        explainer = shap.Explainer(
            classifier.predict_proba,
            masker=shap.maskers.Text(tokenizer=r'\W+'),
            output_names=classifier.classes_
        )
        
        shap_values = explainer(texts[:10])  # Limit for speed
        return shap_values

except ImportError:
    print("SHAP not installed. Using LIME only.")
```

---

## 🔧 Task 4: Build Streamlit Dashboard

### Complete `app.py` File

**IMPORTANT:** Create this file at `Day3/app.py`

```python
"""
Scholarly Topic Navigator - Streamlit Dashboard
Day 3: Integration, UI, and Explainability

IMPORTANT: This file must be in the Day3/ folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import faiss
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from wordcloud import WordCloud
import nltk

# ============================================================
# NLTK SETUP
# ============================================================
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# ============================================================
# PATH CONFIGURATION
# ============================================================
# Paths relative to project root (parent of Day3/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================
# CLASS DEFINITIONS (MUST MATCH DAY 2 EXACTLY)
# Copy these from Complete_day2_deliverables/
# ============================================================

class BM25Retriever:
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
        return [(idx, scores[idx]) for idx in top_indices]


class FAISSRetriever:
    def __init__(self, embeddings=None, encoder_model='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(encoder_model)
        self.index = None
        if embeddings is not None:
            self.embeddings = embeddings.astype('float32')
            self.dimension = embeddings.shape[1]
            faiss.normalize_L2(self.embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(self.embeddings)
    
    def search(self, query, top_k=10):
        query_vec = self.encoder.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
    
    def load_index(self, filepath):
        self.index = faiss.read_index(str(filepath))


class HybridRetriever:
    def __init__(self, bm25_retriever, faiss_retriever, bm25_weight=0.3, semantic_weight=0.7):
        self.bm25 = bm25_retriever
        self.faiss = faiss_retriever
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
    
    def search(self, query, top_k=10):
        bm25_res = self.bm25.search(query, top_k=50)
        faiss_res = self.faiss.search(query, top_k=50)
        
        bm25_scores = {idx: score for idx, score in bm25_res}
        faiss_scores = {idx: score for idx, score in faiss_res}
        
        all_indices = set(bm25_scores.keys()) | set(faiss_scores.keys())
        
        bm25_max = max(bm25_scores.values()) if bm25_scores else 1
        faiss_max = max(faiss_scores.values()) if faiss_scores else 1
        
        combined = []
        for idx in all_indices:
            b_score = bm25_scores.get(idx, 0) / bm25_max
            f_score = faiss_scores.get(idx, 0) / faiss_max
            final = self.bm25_weight * b_score + self.semantic_weight * f_score
            combined.append((idx, final))
        
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]


class EmbeddingClassifier:
    def __init__(self, encoder_model='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(encoder_model)
        self.classifier = None
        self.label_encoder = None
        self.classes_ = None
    
    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        return self.classifier.predict_proba(embeddings)
    
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        return self.classifier.predict(embeddings)


# ============================================================
# SUMMARIZATION FUNCTIONS
# ============================================================

def get_extractive_summary(text, num_sentences=3):
    """TextRank extractive summarization."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer('english'))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return ' '.join([str(s) for s in summary])
    except Exception as e:
        # Fallback
        sentences = text.split('.')[:num_sentences]
        return '. '.join(sentences) + '.'


# ============================================================
# EXPLAINABILITY FUNCTIONS
# ============================================================

def explain_prediction(classifier, text, class_names, num_features=6, num_samples=100):
    """Generate LIME explanation."""
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        text,
        classifier.predict_proba,
        num_features=num_features,
        num_samples=num_samples
    )
    return exp


# ============================================================
# DATA LOADING (Cached)
# ============================================================

@st.cache_resource
def load_system():
    """Load all system components."""
    
    # Load DataFrame
    df_path = DATA_DIR / "processed" / "papers_with_topics.parquet"
    df = pd.read_parquet(df_path)
    
    # Load BM25 Retriever
    bm25_path = DATA_DIR / "retrieval" / "bm25_retriever.pkl"
    with open(bm25_path, 'rb') as f:
        bm25_retriever = pickle.load(f)
    
    # Load FAISS Retriever
    faiss_retriever = FAISSRetriever()
    faiss_path = DATA_DIR / "retrieval" / "faiss_index.bin"
    faiss_retriever.load_index(faiss_path)
    
    # Create Hybrid Retriever
    hybrid_retriever = HybridRetriever(bm25_retriever, faiss_retriever)
    
    # Load Classifier
    clf_path = MODELS_DIR / "embedding_classifier.pkl"
    with open(clf_path, 'rb') as f:
        clf_data = pickle.load(f)
        # Handle different pickle formats
        if isinstance(clf_data, dict):
            classifier = clf_data.get('classifier', clf_data)
        else:
            classifier = clf_data
    
    # Load BART Summarizer
    device = 0 if torch.cuda.is_available() else -1
    bart_summarizer = pipeline(
        'summarization',
        model='facebook/bart-large-cnn',
        device=device
    )
    
    return df, hybrid_retriever, classifier, bart_summarizer


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_category_distribution(df):
    """Create category distribution chart."""
    cat_counts = df['category'].value_counts()
    fig = px.bar(
        x=cat_counts.index,
        y=cat_counts.values,
        labels={'x': 'Category', 'y': 'Count'},
        title='Paper Distribution by Category'
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def create_topic_wordcloud(df, topic_column='bertopic_topic'):
    """Create word cloud for topics."""
    if topic_column in df.columns:
        topic_counts = df[topic_column].value_counts()
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white'
        ).generate_from_frequencies(topic_counts.to_dict())
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    return None


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # Page Configuration
    st.set_page_config(
        page_title="Scholarly Topic Navigator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 2rem;
        }
        .result-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #1E3A8A;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">📚 Scholarly Topic Navigator</div>', unsafe_allow_html=True)
    st.markdown("*An intelligent research paper discovery and analysis system*")
    
    # Load System
    try:
        with st.spinner("Loading system components..."):
            df, retriever, classifier, summarizer = load_system()
        st.success(f"✅ System loaded! Index contains **{len(df):,}** papers.")
    except Exception as e:
        st.error(f"❌ Failed to load system: {e}")
        st.info("Please ensure all Day 2 outputs are in the correct locations.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        num_results = st.slider("Number of Results", 3, 20, 5)
        
        st.header("📊 Dataset Overview")
        st.metric("Total Papers", f"{len(df):,}")
        st.metric("Categories", df['category'].nunique())
        
        if 'bertopic_topic' in df.columns:
            st.metric("Topics (BERTopic)", df['bertopic_topic'].nunique())
        
        st.header("🎨 Visualizations")
        if st.checkbox("Show Category Distribution"):
            fig = create_category_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
    
    # Main Search Interface
    st.header("🔍 Search Papers")
    
    query = st.text_input(
        "Enter your research query:",
        placeholder="e.g., transformer attention mechanism for NLP"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    # Search Results
    if search_button and query:
        st.header(f"📄 Results for: *{query}*")
        
        with st.spinner("Searching..."):
            results = retriever.search(query, top_k=num_results)
        
        for rank, (idx, score) in enumerate(results, 1):
            row = df.iloc[idx]
            
            # Get abstract (handle different column names)
            abstract = row.get('original_abstract', row.get('abstract', 'No abstract available'))
            if pd.isna(abstract):
                abstract = 'No abstract available'
            
            with st.expander(f"**{rank}. {row['title']}** (Score: {score:.3f})", expanded=(rank <= 3)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Metadata
                    st.markdown(f"""
                    **Category:** `{row['category']}` | 
                    **Topic:** `{row.get('bertopic_topic', 'N/A')}` |
                    **Year:** `{row.get('year', 'N/A')}`
                    """)
                    
                    # Abstract
                    st.markdown("**Abstract:**")
                    st.write(abstract[:500] + "..." if len(str(abstract)) > 500 else abstract)
                    
                    # Summarization
                    st.markdown("---")
                    sum_col1, sum_col2 = st.columns(2)
                    
                    with sum_col1:
                        if st.button(f"📝 Extractive Summary", key=f"ext_{idx}"):
                            with st.spinner("Generating extractive summary..."):
                                ext_summary = get_extractive_summary(str(abstract))
                            st.info(f"**Extractive Summary:**\n\n{ext_summary}")
                    
                    with sum_col2:
                        if st.button(f"🤖 Abstractive Summary", key=f"abs_{idx}"):
                            with st.spinner("Generating abstractive summary..."):
                                try:
                                    abs_result = summarizer(str(abstract)[:1024], max_length=100, min_length=30, do_sample=False)
                                    abs_summary = abs_result[0]['summary_text']
                                except Exception as e:
                                    abs_summary = f"Error: {e}"
                            st.success(f"**Abstractive Summary:**\n\n{abs_summary}")
                
                with col2:
                    # Explainability
                    st.markdown("**Explainability**")
                    if st.checkbox(f"🔍 Explain Classification", key=f"exp_{idx}"):
                        with st.spinner("Running LIME..."):
                            try:
                                exp = explain_prediction(
                                    classifier,
                                    str(abstract)[:500],
                                    list(classifier.classes_) if hasattr(classifier, 'classes_') else ['Unknown']
                                )
                                fig = exp.as_pyplot_figure()
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Explanation error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Scholarly Topic Navigator v1.0 | Day 3 Implementation</p>
        <p>Team: Aditya, Trisha, Pramod</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
```

---

## 🔧 Task 5: Deployment (Colab/Kaggle)

### Running Streamlit in Colab

```python
# Cell 1: Get public IP
!wget -q -O - ipv4.icanhazip.com

# Cell 2: Install localtunnel
!npm install -g localtunnel

# Cell 3: Run Streamlit with localtunnel
!streamlit run Day3/app.py &>/content/logs.txt & npx localtunnel --port 8501
```

### Alternative: Using ngrok

```python
# Install ngrok
!pip install pyngrok

from pyngrok import ngrok

# Start Streamlit in background
!nohup streamlit run Day3/app.py --server.port 8501 &

# Create tunnel
public_url = ngrok.connect(8501)
print(f"Access your app at: {public_url}")
```

---

## 📓 Jupyter Notebook Structure

### Create `Day3/Day_3_Complete_Pipeline.ipynb`

The notebook should have the following sections:

```
1. Setup & Installation
   - Install dependencies
   - Import libraries
   - NLTK downloads

2. Configuration & Paths
   - Define paths to Day 2 outputs
   - Verify all files exist

3. Class Definitions (Task 1)
   - Copy EXACT classes from Day 2
   - BM25Retriever, FAISSRetriever, HybridRetriever, EmbeddingClassifier

4. Load Day 2 Components
   - Load DataFrame
   - Load Retrievers
   - Load Classifier
   - Verify loaded objects

5. Summarization Engine (Task 2)
   - Implement extractive summarization (TextRank)
   - Implement abstractive summarization (BART)
   - Test on sample papers

6. Explainability Module (Task 3)
   - Implement LIME explainer
   - Visualize explanations
   - Test on sample predictions

7. End-to-End Pipeline Integration
   - Query → Retrieve → Classify → Summarize → Explain
   - Create pipeline function

8. Visualizations
   - Category distributions
   - Topic visualizations
   - Retrieval score distributions
   - Word clouds

9. Export & Save
   - Save app.py
   - Create requirements.txt
   - Test Streamlit launch

10. Final Verification
    - Verify all components work
    - Check file outputs
    - Generate summary report
```

---

## ⚠️ Critical Code from Day 2 to Reference

Before implementing Day 3, you MUST review these specific sections from Day 2 notebooks in `Complete_day2_deliverables/`:

1. **Cell containing `class BM25Retriever`** - Copy exactly
2. **Cell containing `class FAISSRetriever`** - Copy exactly  
3. **Cell containing `class HybridRetriever`** - Copy exactly
4. **Cell containing `class EmbeddingClassifier`** - Copy exactly
5. **Cell where `embedding_classifier.pkl` is saved** - Note the structure
6. **Cell where `bm25_retriever.pkl` is saved** - Note the structure
7. **Cell where FAISS index is saved** - Note the filename

---

## 📋 Verification Checklist

Before submitting, verify:

- [ ] All code files are in `Day3/` folder only
- [ ] No files outside `Day3/` were modified
- [ ] Class definitions match Day 2 exactly
- [ ] All pickle files load without errors
- [ ] Extractive summarization works
- [ ] Abstractive summarization works
- [ ] LIME explanation generates plots
- [ ] Streamlit app launches without errors
- [ ] Search returns relevant results
- [ ] All visualizations render correctly

---

## 🚀 Quick Start Commands

```bash
# Navigate to project root
cd Final-Project-Group-LexiCore

# Create Day3 folder
mkdir -p Day3/utils

# Run Streamlit (after creating app.py)
streamlit run Day3/app.py
```

---

## 📝 Notes for Implementation

1. **Memory Management**: The BART model is large (~1.6GB). Load it once using `@st.cache_resource`

2. **Error Handling**: Wrap all model inference in try-except blocks

3. **Path Handling**: Use `pathlib.Path` for cross-platform compatibility

4. **GPU Detection**: Check `torch.cuda.is_available()` before GPU operations

5. **Pickle Compatibility**: If pickle fails, the class definition doesn't match Day 2

---

*Document prepared for Scholarly Topic Navigator - Day 3 Implementation*
*Team: Aditya, Trisha, Pramod*
