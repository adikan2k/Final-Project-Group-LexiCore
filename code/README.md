# Code Directory

This folder contains the Jupyter notebooks that implement the Scholarly Topic Navigator pipeline.

---

## Notebooks Overview

| Notebook | Description |
|----------|-------------|
| `Week1_implementation_AdityaKanbargi.ipynb` | Data ingestion from arXiv, text cleaning, preprocessing, and initial embedding generation |
| `Retrieval_Topic_Modeling.ipynb` | Core pipeline for topic modeling (BERTopic), document clustering, and information retrieval (BM25 + FAISS) |
| `Week_3_Integration_UI_Explainability_Trisha.ipynb` | Zero-shot classification, summarization, LIME explainability, and UI integration |

---

## Execution Order

Run the notebooks in the following sequence:

### Step 1: Data Ingestion & Preprocessing
```
Week1_implementation_AdityaKanbargi.ipynb
```
- Downloads papers from arXiv API
- Cleans and normalizes text (titles, abstracts)
- Generates initial embeddings (Word2Vec, SBERT, SciBERT)
- Outputs: `Ouputs/cleaned_papers.parquet`, `Ouputs/*_embeddings.npy`

### Step 2: Topic Modeling & Retrieval
```
Retrieval_Topic_Modeling.ipynb
```
- Builds topic models using BERTopic
- Clusters documents with UMAP + HDBSCAN
- Sets up retrieval indices (BM25, FAISS)
- Outputs: Topic visualizations, retrieval indices

### Step 3: Classification, Explainability & UI
```
Week_3_Integration_UI_Explainability_Trisha.ipynb
```
- Implements zero-shot classification for paper categorization
- Adds LIME explanations for transparency
- Integrates with UI components
- Outputs: Classification results, LIME visualizations

---

## Running the UI

After running all notebooks, launch the web interface:

```bash
cd ../data/Trisha_Week_3
python app.py
```

---

## Dependencies

All notebooks require the packages listed in `../requirements.txt`. Install with:

```bash
pip install -r ../requirements.txt
python -m spacy download en_core_web_sm
```
