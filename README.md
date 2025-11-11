Scholarly Topic Navigator — Explainable Research Digest Pipeline

Automated, explainable NLP system to reduce academic information overload for faculty labs.

Overview

This project tackles the surge of NLP publications (arXiv, ACL, EMNLP, etc.) by building an automated pipeline that surfaces timely, relevant papers with transparent recommendation logic. It combines modern neural methods with standard NLP preprocessing to enable clustering, classification, retrieval, and summarization, while incorporating faculty feedback for continuous improvement.

Methods & Tools:
Models: Word2Vec embeddings; transformer-based architectures (BERT/RoBERTa); zero-shot classification.
Libraries: TensorFlow, Hugging Face Transformers, spaCy.
Core NLP Tasks: text preprocessing & normalization; document clustering; text classification; information retrieval; summarization; entity recognition; zero-shot classification for adaptive recommendations.


Evaluation:
Classification: Precision, Recall, F1-score.
Topic Modeling: topic coherence, perplexity.
System-Level: coverage, latency.
Qualitative: faculty surveys to assess usefulness and transparency.


Datasets:
arXiv Computer Science Collections (cs.CL, cs.LG, stat.ML): real-time intake (abstracts, categories, authors, submission dates).
ACL Anthology: structured metadata (venue, citations, affiliations) for ground truth and quality assessment.
S2ORC (Semantic Scholar Open Research Corpus): full text + citation graphs for citation-based explanations and trend detection.


Links:
arXiv bulk data: https://arxiv.org/help/bulk_data
ACL Anthology: https://aclanthology.org/
S2ORC: https://allenai.org/data/s2orc


Goal
Effectively identify relevant papers while minimizing overload and providing transparent, actionable insights to the academic community.

Team:
Aditya Kanbargi
Trisha Singh
Pramod Krishnachari



