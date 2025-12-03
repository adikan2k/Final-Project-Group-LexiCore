
import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from lime.lime_text import LimeTextExplainer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from wordcloud import WordCloud
import nltk
import io
import base64

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Scholarly Topic Navigator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR BEAUTIFUL UI
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
        border-right: 2px solid #e94560;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e94560 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.2);
    }
    
    /* Search input */
    .stTextInput > div > div > input {
        background-color: #16213e;
        color: white;
        border: 2px solid #e94560;
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #e94560 0%, #0f3460 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(233, 69, 96, 0.4);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e94560;
        border-radius: 10px;
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #16213e;
        border-radius: 10px;
        color: white;
        border: 1px solid #e94560;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #e94560 0%, #0f3460 100%);
    }
    
    /* Cards */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e94560;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background-color: rgba(233, 69, 96, 0.1);
        border: 1px solid #e94560;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATH CONFIGURATION (UPDATED FOR LOCAL USE)
# ============================================================
import os

# Use '.' (current directory) so it works on Mac, Windows, and Linux
INPUT_DIR = Path(".")
WORKING_DIR = Path(".")
VIZ_DIR = WORKING_DIR / "visualizations"

# Ensure visualization directory exists locally to prevent errors
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CLASS DEFINITIONS
# ============================================================
class BM25Retriever:
    def __init__(self, corpus=None):
        self.bm25 = None
        if corpus:
            self.bm25 = BM25Okapi([doc.lower().split() for doc in corpus])
    def search(self, query, top_k=10):
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(i, scores[i]) for i in top_idx]

class FAISSRetriever:
    def __init__(self, embeddings=None):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        if embeddings is not None:
            emb = embeddings.astype("float32")
            faiss.normalize_L2(emb)
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
    def search(self, query, top_k=10):
        qv = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(qv)
        scores, idx = self.index.search(qv, top_k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])]

class HybridRetriever:
    def __init__(self, bm25, faiss_ret):
        self.bm25, self.faiss = bm25, faiss_ret
    def search(self, query, top_k=10):
        b_res = dict(self.bm25.search(query, 50))
        f_res = dict(self.faiss.search(query, 50))
        all_idx = set(b_res) | set(f_res)
        b_max = max(b_res.values()) if b_res else 1
        f_max = max(f_res.values()) if f_res else 1
        combined = [(i, 0.3*(b_res.get(i,0)/b_max) + 0.7*(f_res.get(i,0)/f_max)) for i in all_idx]
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

class EmbeddingClassifier:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.classifier = None
        self.classes_ = None
    def predict_proba(self, texts):
        if isinstance(texts, str): texts = [texts]
        emb = self.encoder.encode(texts, show_progress_bar=False)
        return self.classifier.predict_proba(emb)

def get_extractive_summary(text, n=3):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer(Stemmer("english"))
        summarizer.stop_words = get_stop_words("english")
        return " ".join(str(s) for s in summarizer(parser.document, n))
    except:
        return ". ".join(text.split(".")[:n]) + "."

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_resource
def load_all_data():
    # Load DataFrame (prefer one with categories)
    if (WORKING_DIR / "papers_with_categories.parquet").exists():
        df = pd.read_parquet(WORKING_DIR / "papers_with_categories.parquet")
    else:
        df = pd.read_parquet(INPUT_DIR / "cleaned_papers.parquet")
    
    # Load embeddings
    emb = np.load(INPUT_DIR / "sbert_abstract_embeddings.npy")
    
    # Determine abstract column
    abs_col = "original_abstract" if "original_abstract" in df.columns else "abstract"
    
    # Build retrievers
    corpus = (df["title"] + " " + df[abs_col].fillna("")).tolist()
    bm25 = BM25Retriever(corpus)
    faiss_ret = FAISSRetriever(emb)
    hybrid = HybridRetriever(bm25, faiss_ret)
    
    return df, hybrid, abs_col, emb

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def create_category_chart(df):
    if 'category' not in df.columns:
        return None
    cat_counts = df['category'].value_counts().reset_index()
    cat_counts.columns = ['Category', 'Count']
    fig = px.bar(cat_counts, x='Count', y='Category', orientation='h',
                 color='Count', color_continuous_scale='viridis',
                 title='Paper Distribution by Category')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400
    )
    return fig

def create_year_chart(df):
    if 'year' not in df.columns:
        return None
    year_counts = df['year'].value_counts().sort_index().reset_index()
    year_counts.columns = ['Year', 'Count']
    fig = px.bar(year_counts, x='Year', y='Count',
                 color='Count', color_continuous_scale='plasma',
                 title='Papers by Publication Year')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=350
    )
    return fig

def create_source_pie(df):
    if 'source' not in df.columns:
        return None
    source_counts = df['source'].value_counts().reset_index()
    source_counts.columns = ['Source', 'Count']
    fig = px.pie(source_counts, values='Count', names='Source',
                 title='Data Sources', hole=0.4,
                 color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    return fig

def create_confidence_hist(df):
    if 'category_confidence' not in df.columns:
        return None
    fig = px.histogram(df, x='category_confidence', nbins=30,
                       title='Classification Confidence Distribution',
                       color_discrete_sequence=['#e94560'])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=300
    )
    return fig

def create_wordcloud_image(df, category=None):
    if category and 'category' in df.columns:
        text = df[df['category'] == category]['processed_text'].fillna('').str.cat(sep=' ')
    else:
        text = df['processed_text'].fillna('').str.cat(sep=' ')[:50000]
    
    if len(text) < 100:
        return None
    
    wc = WordCloud(width=800, height=400, background_color='#1a1a2e',
                   colormap='cool', max_words=100).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1a1a2e', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='font-size: 3rem; background: linear-gradient(90deg, #e94560, #0f3460); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            📚 Scholarly Topic Navigator
        </h1>
        <p style='color: #aaa; font-size: 1.2rem;'>
            Intelligent Research Paper Discovery & Analysis System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    try:
        with st.spinner("🔄 Loading system components..."):
            df, retriever, abs_col, embeddings = load_all_data()
        st.success(f"✅ System loaded! **{len(df):,}** papers indexed.")
    except Exception as e:
        st.error(f"❌ Failed to load: {e}")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Controls")
        
        # Metrics
        st.markdown("### 📊 Dataset Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Papers", f"{len(df):,}")
        with col2:
            if 'category' in df.columns:
                st.metric("🏷️ Categories", df['category'].nunique())
        
        if 'year' in df.columns:
            st.metric("📅 Year Range", f"{df['year'].min()}-{df['year'].max()}")
        
        st.markdown("---")
        
        # Search settings
        st.markdown("### 🔍 Search Settings")
        num_results = st.slider("Number of Results", 3, 20, 5)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        page = st.radio("Go to:", ["🔍 Search", "📊 Analytics", "📈 Visualizations"])
    
    # Main content based on navigation
    if page == "🔍 Search":
        render_search_page(df, retriever, abs_col, num_results)
    elif page == "📊 Analytics":
        render_analytics_page(df, abs_col)
    else:
        render_visualizations_page(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎓 Scholarly Topic Navigator v2.0 | Team: Aditya, Trisha, Pramod</p>
        <p>Built with ❤️ using Streamlit, FAISS, SBERT, and Transformers</p>
    </div>
    """, unsafe_allow_html=True)

def render_search_page(df, retriever, abs_col, num_results):
    st.markdown("## 🔍 Search Papers")
    
    # Search input
    query = st.text_input("Enter your research query:", 
                          placeholder="e.g., transformer attention mechanism for NLP")
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        lucky_btn = st.button("🎲 Random", use_container_width=True)
    
    if lucky_btn:
        query = np.random.choice([
            "deep learning neural networks",
            "natural language processing",
            "computer vision image recognition",
            "reinforcement learning",
            "transformer attention mechanism"
        ])
        st.info(f"Random query: **{query}**")
    
    if (search_btn or lucky_btn) and query:
        with st.spinner("🔄 Searching..."):
            results = retriever.search(query, num_results)
        
        st.markdown(f"### 📄 Results for: *{query}*")
        st.markdown(f"Found **{len(results)}** relevant papers")
        
        for rank, (idx, score) in enumerate(results, 1):
            row = df.iloc[idx]
            abstract = str(row.get(abs_col, "No abstract available"))
            
            with st.expander(f"**{rank}. {row['title'][:80]}...** (Score: {score:.3f})", expanded=(rank==1)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Metadata badges
                    badges = []
                    if 'category' in df.columns:
                        badges.append(f"🏷️ `{row.get('category', 'N/A')}`")
                    if 'year' in df.columns:
                        badges.append(f"📅 `{row.get('year', 'N/A')}`")
                    if 'source' in df.columns:
                        badges.append(f"📚 `{row.get('source', 'N/A')}`")
                    st.markdown(" | ".join(badges))
                    
                    # Abstract
                    st.markdown("**Abstract:**")
                    display_text = abstract[:500] + "..." if len(abstract) > 500 else abstract
                    st.write(display_text)
                    
                    # Summarization buttons
                    st.markdown("---")
                    sum_col1, sum_col2 = st.columns(2)
                    with sum_col1:
                        if st.button(f"📝 Extractive Summary", key=f"ext_{idx}"):
                            summary = get_extractive_summary(abstract)
                            st.info(f"**Summary:** {summary}")
                
                with col2:
                    st.markdown("**Relevance Score**")
                    st.progress(min(score, 1.0))
                    st.metric("Score", f"{score:.3f}")

def render_analytics_page(df, abs_col):
    st.markdown("## 📊 Dataset Analytics")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Total Papers", f"{len(df):,}")
    with col2:
        if 'category' in df.columns:
            st.metric("🏷️ Categories", df['category'].nunique())
    with col3:
        if 'source' in df.columns:
            st.metric("📚 Sources", df['source'].nunique())
    with col4:
        if 'year' in df.columns:
            st.metric("📅 Years", f"{df['year'].nunique()}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_category_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_year_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_source_pie(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_confidence_hist(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

def render_visualizations_page(df):
    st.markdown("## 📈 Visualizations Gallery")
    
    tabs = st.tabs(["🏷️ Categories", "☁️ Word Clouds", "📊 Statistics"])
    
    with tabs[0]:
        st.markdown("### Category Distribution")
        if 'category' in df.columns:
            fig = create_category_chart(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Category details
            st.markdown("### Category Details")
            cat_counts = df['category'].value_counts()
            for cat, count in cat_counts.items():
                pct = count / len(df) * 100
                st.markdown(f"**{cat}**: {count:,} papers ({pct:.1f}%)")
                st.progress(pct / 100)
        else:
            st.warning("No category data available. Run Zero-Shot Classification first.")
    
    with tabs[1]:
        st.markdown("### Word Clouds")
        
        if 'category' in df.columns:
            selected_cat = st.selectbox("Select Category:", 
                                       ["All"] + df['category'].value_counts().head(10).index.tolist())
            
            with st.spinner("Generating word cloud..."):
                cat = None if selected_cat == "All" else selected_cat
                wc_buf = create_wordcloud_image(df, cat)
                if wc_buf:
                    st.image(wc_buf, caption=f"Word Cloud: {selected_cat}")
        else:
            with st.spinner("Generating word cloud..."):
                wc_buf = create_wordcloud_image(df)
                if wc_buf:
                    st.image(wc_buf, caption="Overall Word Cloud")
    
    with tabs[2]:
        st.markdown("### Dataset Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Column Information")
            st.dataframe(pd.DataFrame({
                'Column': df.columns,
                'Non-Null': df.count().values,
                'Dtype': df.dtypes.values
            }))
        
        with col2:
            st.markdown("#### Sample Data")
            st.dataframe(df.head(5))

if __name__ == "__main__":
    main()

