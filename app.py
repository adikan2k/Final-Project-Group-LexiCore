import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer


try:
    from run_pipeline import run_digest_query, digest_to_dataframe
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False


@st.cache_data
def load_corpus():
    df = pd.read_parquet("data/processed/cleaned_papers.parquet")
    return df

@st.cache_resource
def load_classifier():
    clf_path = Path("models/embedding_classifier.pkl")
    if not clf_path.exists():
        raise FileNotFoundError("models/embedding_classifier.pkl not found. "
                                "Make sure you committed the classifier file.")
    with open(clf_path, "rb") as f:
        clf_obj = pickle.load(f)
    return clf_obj

class EmbeddingClassifier:
    def __init__(self, encoder_model="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(encoder_model)
        self.classifier = None
        self.classes_ = None

    def attach(self, clf_obj):
        
        self.classifier = clf_obj["classifier"]
        self.classes_ = clf_obj["classes"]

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embs = self.encoder.encode(texts, show_progress_bar=False)
        return self.classifier.predict_proba(embs)


@st.cache_resource
def get_explainer(_classifier_wrapper):
    return LimeTextExplainer(class_names=_classifier_wrapper.classes_)



def simple_retrieval(df, query, top_k=5):
    """Very simple lexical retrieval using title+abstract contains query."""
    q = query.lower()
    combined = (df["title"].astype(str) + " " +
                df["original_abstract"].astype(str)).str.lower()
    mask = combined.str.contains(q)
    cand = df[mask].copy()
    if len(cand) == 0:
        return df.sample(min(top_k, len(df)))
    return cand.head(top_k)

def main():
    st.set_page_config(page_title="Scholarly Topic Navigator", layout="wide")
    st.title(" Scholarly Topic Navigator – Day 3 UI")
    st.markdown(
        "End-to-end **research digest** with summarization + classifier "
        "explainability (LIME) built in Streamlit."
    )

    
    with st.spinner("Loading corpus and classifier..."):
        df_corpus = load_corpus()
        clf_obj = load_classifier()
        classifier_wrapper = EmbeddingClassifier()
        classifier_wrapper.attach(clf_obj)
        explainer = get_explainer(classifier_wrapper)

    st.sidebar.header("Corpus Stats")
    st.sidebar.write(f"Total papers: `{len(df_corpus)}`")
    st.sidebar.write(f"Label column: `venue`")
    st.sidebar.write(f"Num classes: `{len(classifier_wrapper.classes_)}`")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### How to use")
    st.sidebar.markdown(
        "1. Type a query\n"
        "2. See top matching papers\n"
        "3. Expand a paper to see summaries\n"
        "4. Click **Explain prediction** to view LIME explanation"
    )

  
    query = st.text_input(
        " Enter search query",
        value="transformer models",
        help="Try topics like 'graph neural networks', 'medical NLP', etc."
    )

    top_k = st.slider("Top K results", min_value=3, max_value=10, value=5, step=1)

    if st.button("Run search"):
        if not query.strip():
            st.warning("Please enter a query first.")
            st.stop()

        st.subheader(f"Results for: `{query}`")

        
        if HAS_PIPELINE:
            
            with st.spinner("Running full pipeline (hybrid retrieval + summarization)..."):
                digest = run_digest_query(query, top_k=top_k)
                df_digest = digest_to_dataframe(digest)
        else:
           
            with st.spinner("Running simple lexical retrieval..."):
                df_digest = simple_retrieval(df_corpus, query, top_k=top_k)
                
                if "one_sentence_summary" not in df_digest.columns:
                    df_digest["one_sentence_summary"] = (
                        df_digest["original_abstract"].str.split(".").str[0].fillna("")
                    )

        
        df_digest = df_digest.reset_index(drop=True)
        df_digest["rank"] = df_digest.index + 1

        
        for _, row in df_digest.iterrows():
            title = row.get("title", "Untitled paper")
            paper_id = row.get("paper_id", "unknown")
            venue = row.get("venue", "unknown")
            year = int(row.get("year", 0)) if "year" in row and pd.notna(row["year"]) else "N/A"

            with st.expander(f"#{row['rank']} – {title}  ({venue}, {year})"):
                abstract = row.get("original_abstract", "")
                st.markdown("**Abstract**")
                st.write(abstract)

               
                one_sent = row.get("one_sentence_summary", None)
                three_sent = row.get("three_sentence_summary", None)
                five_bullets = row.get("five_bullet_summary", None)

                if one_sent or three_sent or five_bullets:
                    st.markdown("---")
                    st.markdown("###  Summaries")

                    if one_sent:
                        st.markdown("**1-sentence summary**")
                        st.info(one_sent)

                    if three_sent:
                        st.markdown("**3-sentence summary**")
                        st.write(three_sent)

                    if five_bullets:
                        st.markdown("**5 key bullet insights**")
                        if isinstance(five_bullets, (list, tuple)):
                            for b in five_bullets:
                                st.markdown(f"- {b}")
                        else:
                            st.write(five_bullets)

               
                topic_label = row.get("topic_label", None)
                topic_id = row.get("topic_id", None)
                if topic_label or topic_id is not None:
                    st.markdown("---")
                    st.markdown("###  Topic assignment")
                    if topic_id is not None:
                        st.write(f"Topic ID: `{topic_id}`")
                    if topic_label:
                        st.write(f"Topic label: **{topic_label}**")

                
                st.markdown("---")
                st.markdown("###  Explain model prediction")

                explain_text = abstract or title
                if not explain_text:
                    st.info("No text available to explain.")
                else:
                    if st.button(f"Explain prediction for {paper_id}", key=f"explain-{paper_id}"):
                        with st.spinner("Running LIME explanation..."):
                            exp = explainer.explain_instance(
                                explain_text,
                                classifier_wrapper.predict_proba,
                                num_features=10
                            )
                            fig = exp.as_pyplot_figure()
                            st.pyplot(fig)
                            st.caption(
                                "Green = words that support the predicted venue; "
                                "red = words that push against it."
                            )


if __name__ == "__main__":
    main()
