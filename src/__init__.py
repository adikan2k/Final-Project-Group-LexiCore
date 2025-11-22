"""
Scholarly Topic Navigator - Core modules for automated research digest pipeline.
"""

from .preprocessing import TextCleaner, PaperProcessor
from .embeddings import EmbeddingGenerator, DocumentClusterer
from .classification import PaperClassifier, ZeroShotClassifier
from .retrieval import PaperRetriever
from .summarization import PaperSummarizer
from .evaluation import SystemEvaluator, PerformanceBenchmark

__version__ = "0.1.0"
__all__ = [
    'TextCleaner', 'PaperProcessor',
    'EmbeddingGenerator', 'DocumentClusterer',
    'PaperClassifier', 'ZeroShotClassifier',
    'PaperRetriever', 'PaperSummarizer',
    'SystemEvaluator', 'PerformanceBenchmark'
]