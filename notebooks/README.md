# Data Ingestion Notebooks

This directory contains Jupyter notebooks for Day 1 data ingestion tasks.

## Notebooks Overview

### 01_arxiv_ingestion.ipynb
Fetches papers from ArXiv API for NLP-related categories (cs.CL, cs.LG, stat.ML).

**Key Features:**
- Uses `arxiv` Python library for API access
- Extracts: title, authors, abstract, categories, publication dates
- Handles ~300-500 papers per category
- Exports to JSON and Parquet formats

**Output:** `data/raw/arxiv_papers.parquet`

---

### 02_acl_anthology_ingestion.ipynb
Downloads and parses ACL Anthology BibTeX dump.

**Key Features:**
- Downloads complete ACL Anthology bibliography
- Parses ~70K+ conference/journal papers
- Filters to recent papers (2015+)
- Normalizes venue and author information

**Output:** `data/raw/acl_anthology_papers.parquet`

**Note:** Not all papers have abstracts in BibTeX. May need additional scraping if abstracts are critical.

---

### 03_s2orc_ingestion.ipynb
Fetches papers from Semantic Scholar API (S2ORC subset).

**Key Features:**
- Uses free Semantic Scholar API
- Searches multiple NLP topics
- Includes citation counts and field classifications
- Gets ~1000+ papers with rich metadata

**Output:** `data/raw/s2orc_papers.parquet`

**Note:** API has rate limits. Add delays between requests. For larger datasets, consider S2 bulk download.

---

### 04_unified_metadata.ipynb
Combines all three data sources into unified format.

**Key Features:**
- Loads data from all sources
- Normalizes to consistent schema
- Deduplicates across sources
- Adds computed fields (text lengths, author counts, etc.)
- Exports in multiple formats

**Outputs:**
- `data/processed/unified_papers.parquet` (recommended for downstream tasks)
- `data/processed/unified_papers.json`
- `data/processed/unified_papers.csv`
- `data/processed/dataset_summary.json`

---

## Running the Notebooks

### On Google Colab (Recommended)

1. Upload notebooks to your Google Drive or open directly from GitHub
2. Run cells sequentially
3. Mount Google Drive when prompted (for data persistence)
4. Each notebook is self-contained with package installation

### Locally

```bash
# create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# install dependencies
pip install pandas arxiv requests beautifulsoup4 pyarrow jupyter

# start jupyter
jupyter notebook
```

---

## Unified Schema

The final unified dataset has this structure:

| Field | Type | Description |
|-------|------|-------------|
| `paper_id` | string | Unique identifier (prefixed with source) |
| `title` | string | Paper title |
| `authors` | list | List of author names |
| `abstract` | string | Paper abstract |
| `venue` | string | Publication venue/conference |
| `year` | int | Publication year |
| `categories` | list | Topic categories/fields |
| `source` | string | Data source (arxiv/acl/s2orc) |
| `metadata` | dict | Source-specific additional fields |

---

## Expected Output Sizes

After running all notebooks:

- ArXiv: ~900-1500 papers
- ACL: ~40K-60K papers (2015+)
- S2ORC: ~1000-1500 papers
- **Unified (after deduplication):** ~42K-63K papers

File sizes (approximate):
- JSON files: 50-150 MB
- Parquet files: 10-40 MB
- CSV files: 60-180 MB

---

## Troubleshooting

### API Rate Limits
- ArXiv: No strict limits, but add 1-3 second delays
- Semantic Scholar: 100 requests/5 minutes. Increase `time.sleep()` if needed

### Missing Abstracts
- ArXiv: All papers have abstracts
- ACL: ~30-40% might lack abstracts in BibTeX
- S2ORC: API returns abstracts for most papers

### Out of Memory
- Reduce `max_results_per_category` in ArXiv notebook
- Filter ACL by year range (e.g., only 2020+)
- Process S2ORC in smaller batches

### File Not Found Errors
- Ensure notebooks are run in sequence (01 → 02 → 03 → 04)
- Check that `data/raw/` directory exists
- Verify previous notebooks completed successfully

---

## Next Steps

After completing data ingestion:
1. **Preprocessing** (notebook 02_preprocessing_tests.ipynb)
   - Text cleaning and normalization
   - Tokenization
   - Removing special characters
   
2. **Embeddings** (notebook 03_embeddings_comparison.ipynb)
   - Generate Word2Vec embeddings
   - Create BERT/RoBERTa embeddings
   - Compare embedding quality

3. **Classification** (notebook 04_classification.ipynb)
   - Train topic classifiers
   - Zero-shot classification
   - Evaluation metrics

---

## Team Responsibilities

**Aditya Kanbargi** - Data Engineering
- [x] Repository structure setup
- [x] ArXiv ingestion
- [x] ACL Anthology ingestion
- [x] S2ORC ingestion
- [x] Unified metadata creation

---

## References

- ArXiv API: https://info.arxiv.org/help/api/index.html
- ACL Anthology: https://aclanthology.org/
- Semantic Scholar API: https://api.semanticscholar.org/
- S2ORC Paper: https://aclanthology.org/2020.acl-main.447/
