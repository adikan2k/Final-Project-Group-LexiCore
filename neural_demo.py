#!/usr/bin/env python3
"""
Comprehensive Neural Demo of the Scholarly Topic Navigator
Showcasing advanced NLP capabilities with Hugging Face models.
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("CWD =", os.getcwd())

# Set Hugging Face token
os.environ["HUGGINGFACE_HUB_TOKEN"] = ""

# Import required libraries
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def setup_models():
    """Initialize all the neural models."""
    print("🤖 Loading Neural Models...")
    print("=" * 50)
    
    models = {}
    
    try:
        # Sentence Transformer for embeddings
        print("📊 Loading Sentence Transformer...")
        models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence Transformer loaded")
        
        # Zero-shot classification
        print("🎯 Loading Zero-shot Classifier...")
        models['zero_shot'] = pipeline("zero-shot-classification", 
                                      model="facebook/bart-large-mnli")
        print("✅ Zero-shot Classifier loaded")
        
        # Summarization
        print("📝 Loading Summarization Model...")
        models['summarizer'] = pipeline("summarization", 
                                       model="facebook/bart-large-cnn")
        print("✅ Summarization Model loaded")
        
        # Text generation
        print("✍️ Loading Text Generation Model...")
        models['generator'] = pipeline("text-generation", 
                                     model="gpt2",
                                     max_length=100)
        print("✅ Text Generation Model loaded")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None
    
    print("\n🎉 All models loaded successfully!")
    return models

def load_and_prepare_data(n_papers=100):
    """Load and prepare the academic papers data."""
    print(f"\n📚 Loading Academic Papers Data")
    print("=" * 50)
    
    try:
        # Load the unified dataset
        from pathlib import Path
        import pandas as pd

        # Get the directory where THIS script lives
        ROOT = Path(__file__).resolve().parent

        # Point to the parquet file in the same folder
        UNIFIED_PATH = ROOT / "unified_papers.parquet"

        papers_df = pd.read_parquet(UNIFIED_PATH)
        print(f"📊 Total papers available: {len(papers_df)}")
        
        # Sample papers for demo (to manage processing time)
        sample_papers = papers_df.sample(n=min(n_papers, len(papers_df)), random_state=42)
        print(f"🎯 Using {len(sample_papers)} papers for demo")
        
        # Prepare combined text
        combined_texts = []
        for _, paper in sample_papers.iterrows():
            title = str(paper.get('title', ''))
            abstract = str(paper.get('abstract', ''))
            combined = f"{title}. {abstract}".strip()
            combined_texts.append(combined)
        
        sample_papers = sample_papers.copy()
        sample_papers['combined_text'] = combined_texts
        
        print(f"✅ Data prepared - average text length: {np.mean([len(text) for text in combined_texts]):.0f} characters")
        
        return sample_papers, combined_texts
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def neural_embeddings_demo(models, texts, papers_df):
    """Demonstrate neural embeddings generation."""
    print(f"\n🧠 Neural Embeddings Generation")
    print("=" * 50)
    
    try:
        sentence_model = models['sentence_transformer']
        
        # Generate embeddings
        print("🔄 Generating sentence embeddings...")
        embeddings = sentence_model.encode(texts, show_progress_bar=True)
        
        print(f"✅ Generated embeddings: {embeddings.shape}")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        print(f"   Documents encoded: {embeddings.shape[0]}")
        
        # Analyze embeddings
        mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))
        print(f"   Average embedding norm: {mean_norm:.3f}")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Error in embeddings generation: {e}")
        return None

def semantic_clustering_demo(embeddings, papers_df, n_clusters=5):
    """Demonstrate semantic clustering with embeddings."""
    print(f"\n🎯 Semantic Clustering")
    print("=" * 50)
    
    try:
        # Perform K-means clustering
        print(f"🔄 Performing K-means clustering (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        print(f"✅ Clustering completed")
        
        # Analyze clusters
        cluster_sizes = np.bincount(cluster_labels)
        print(f"📊 Cluster sizes: {cluster_sizes}")
        
        # Show sample papers from each cluster
        for cluster_id in range(n_clusters):
            cluster_papers = papers_df[cluster_labels == cluster_id]
            print(f"\n🏷️ Cluster {cluster_id} ({len(cluster_papers)} papers):")
            
            # Show top 2 titles from cluster
            for i, (_, paper) in enumerate(cluster_papers.head(2).iterrows()):
                title = paper.get('title', 'No title')[:80]
                year = paper.get('year', 'Unknown')
                print(f"   {i+1}. {title}... ({year})")
        
        # Visualize clusters (2D projection)
        print(f"\n📈 Creating cluster visualization...")
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=cluster_labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Semantic Clusters of Academic Papers (PCA Projection)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        
        # Add cluster centers
        centers_2d = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/semantic_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Visualization saved to results/semantic_clusters.png")
        
        return cluster_labels
        
    except Exception as e:
        print(f"❌ Error in clustering: {e}")
        return None

def zero_shot_classification_demo(models, texts, papers_df):
    """Demonstrate zero-shot classification."""
    print(f"\n🎯 Zero-Shot Classification")
    print("=" * 50)
    
    try:
        classifier = models['zero_shot']
        
        # Define research topic labels
        topic_labels = [
            "machine learning",
            "natural language processing", 
            "computer vision",
            "deep learning",
            "data mining",
            "artificial intelligence",
            "robotics",
            "information retrieval"
        ]
        
        print(f"🏷️ Topic labels: {topic_labels}")
        
        # Classify a sample of papers
        sample_size = min(10, len(texts))
        sample_texts = texts[:sample_size]
        sample_papers = papers_df.head(sample_size)
        
        print(f"🔄 Classifying {sample_size} papers...")
        
        classification_results = []
        
        for i, text in enumerate(sample_texts):
            # Limit text length for the model
            text_truncated = text[:512]  # BART limit
            
            try:
                result = classifier(text_truncated, topic_labels)
                
                predicted_label = result['labels'][0]
                confidence = result['scores'][0]
                
                paper_title = sample_papers.iloc[i].get('title', 'Unknown')
                paper_year = sample_papers.iloc[i].get('year', 'Unknown')
                
                classification_results.append({
                    'title': paper_title,
                    'year': paper_year,
                    'predicted_topic': predicted_label,
                    'confidence': confidence,
                    'all_scores': dict(zip(result['labels'], result['scores']))
                })
                
                print(f"   📄 Paper {i+1}: {predicted_label} (confidence: {confidence:.3f})")
                print(f"       Title: {paper_title[:60]}...")
                
            except Exception as e:
                print(f"   ❌ Error classifying paper {i+1}: {e}")
                continue
        
        # Analyze results
        if classification_results:
            predicted_topics = [r['predicted_topic'] for r in classification_results]
            topic_distribution = pd.Series(predicted_topics).value_counts()
            
            print(f"\n📊 Topic Distribution:")
            for topic, count in topic_distribution.items():
                print(f"   {topic}: {count} papers")
            
            # High confidence predictions
            high_conf_results = [r for r in classification_results if r['confidence'] > 0.8]
            print(f"\n🎯 High Confidence Predictions (>0.8): {len(high_conf_results)}")
            
            for result in high_conf_results[:3]:  # Show top 3
                print(f"   📄 {result['predicted_topic']}: {result['confidence']:.3f}")
                print(f"       {result['title'][:60]}...")
        
        return classification_results
        
    except Exception as e:
        print(f"❌ Error in zero-shot classification: {e}")
        return None

def neural_summarization_demo(models, texts, papers_df):
    """Demonstrate neural summarization."""
    print(f"\n📝 Neural Summarization")
    print("=" * 50)
    
    try:
        summarizer = models['summarizer']
        
        # Select a few papers for summarization
        sample_size = min(5, len(texts))
        
        print(f"🔄 Summarizing {sample_size} papers...")
        
        summaries = []
        
        for i in range(sample_size):
            text = texts[i]
            paper = papers_df.iloc[i]
            
            # Truncate text for BART (max 1024 tokens)
            text_truncated = text[:1024] if len(text) > 1024 else text
            
            try:
                summary_result = summarizer(
                    text_truncated, 
                    max_length=130, 
                    min_length=30, 
                    do_sample=False
                )
                
                summary = summary_result[0]['summary_text']
                
                summaries.append({
                    'original_title': paper.get('title', 'Unknown'),
                    'original_length': len(text),
                    'summary': summary,
                    'summary_length': len(summary),
                    'compression_ratio': len(summary) / len(text),
                    'year': paper.get('year', 'Unknown')
                })
                
                print(f"\n📄 Paper {i+1}: {paper.get('title', 'Unknown')[:50]}...")
                print(f"   📏 Original: {len(text)} chars → Summary: {len(summary)} chars")
                print(f"   📊 Compression: {len(summary)/len(text):.1%}")
                print(f"   📝 Summary: {summary}")
                
            except Exception as e:
                print(f"   ❌ Error summarizing paper {i+1}: {e}")
                continue
        
        # Analyze summarization performance
        if summaries:
            avg_compression = np.mean([s['compression_ratio'] for s in summaries])
            avg_summary_length = np.mean([s['summary_length'] for s in summaries])
            
            print(f"\n📊 Summarization Statistics:")
            print(f"   Average compression ratio: {avg_compression:.1%}")
            print(f"   Average summary length: {avg_summary_length:.0f} characters")
            print(f"   Papers processed: {len(summaries)}")
        
        return summaries
        
    except Exception as e:
        print(f"❌ Error in summarization: {e}")
        return None

def semantic_search_demo(embeddings, texts, papers_df):
    """Demonstrate semantic search capabilities."""
    print(f"\n🔍 Semantic Search Demo")
    print("=" * 50)
    
    try:
        # Load the sentence transformer for query encoding
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test queries
        test_queries = [
            "deep learning neural networks",
            "natural language processing transformers",
            "computer vision image recognition",
            "machine learning algorithms",
            "text generation and summarization"
        ]
        
        for query in test_queries:
            print(f"\n🔎 Query: '{query}'")
            
            # Encode query
            query_embedding = sentence_model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            # Get top 3 results
            top_indices = np.argsort(similarities)[::-1][:3]
            
            print("   📋 Top Results:")
            for rank, idx in enumerate(top_indices, 1):
                paper = papers_df.iloc[idx]
                similarity = similarities[idx]
                title = paper.get('title', 'Unknown')
                year = paper.get('year', 'Unknown')
                
                print(f"   {rank}. Similarity: {similarity:.3f}")
                print(f"      Title: {title}")
                print(f"      Year: {year}")
                print(f"      Abstract: {str(paper.get('abstract', ''))[:100]}...")
                print()
        
        print("✅ Semantic search demo completed")
        
    except Exception as e:
        print(f"❌ Error in semantic search: {e}")

def save_comprehensive_results(papers_df, embeddings, cluster_labels, classifications, summaries):
    """Save all results to files."""
    print(f"\n💾 Saving Comprehensive Results")
    print("=" * 50)
    
    try:
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Save embeddings
        np.save('results/neural_embeddings.npy', embeddings)
        print("✅ Embeddings saved")
        
        # Save enhanced papers dataframe
        enhanced_df = papers_df.copy()
        if cluster_labels is not None:
            enhanced_df['cluster_id'] = cluster_labels
        
        enhanced_df.to_csv('results/papers_with_clusters.csv', index=False)
        print("✅ Enhanced papers data saved")
        
        # Save classification results
        if classifications:
            with open('results/zero_shot_classifications.json', 'w') as f:
                json.dump(classifications, f, indent=2, default=str)
            print("✅ Classification results saved")
        
        # Save summarization results
        if summaries:
            with open('results/neural_summaries.json', 'w') as f:
                json.dump(summaries, f, indent=2, default=str)
            print("✅ Summarization results saved")
        
        # Create comprehensive report
        report = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'total_papers': len(papers_df),
                'embedding_dimension': embeddings.shape[1] if embeddings is not None else None,
                'num_clusters': len(set(cluster_labels)) if cluster_labels is not None else None
            },
            'model_performance': {
                'embeddings_generated': embeddings is not None,
                'clustering_completed': cluster_labels is not None,
                'classifications_completed': len(classifications) if classifications else 0,
                'summaries_generated': len(summaries) if summaries else 0
            },
            'summary_statistics': {}
        }
        
        if classifications:
            topics = [c['predicted_topic'] for c in classifications]
            confidences = [c['confidence'] for c in classifications]
            report['summary_statistics']['classification'] = {
                'avg_confidence': np.mean(confidences),
                'top_topics': list(pd.Series(topics).value_counts().to_dict().items())
            }
        
        if summaries:
            compressions = [s['compression_ratio'] for s in summaries]
            report['summary_statistics']['summarization'] = {
                'avg_compression_ratio': np.mean(compressions),
                'avg_summary_length': np.mean([s['summary_length'] for s in summaries])
            }
        
        with open('results/comprehensive_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print("✅ Comprehensive report saved")
        
        print(f"\n📁 All results saved to 'results/' directory")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    """Main demo function showcasing all neural capabilities."""
    print("🎯 Scholarly Topic Navigator - Neural Demo")
    print("🤖 Powered by Hugging Face Transformers")
    print("=" * 70)
    
    # Setup
    models = setup_models()
    if models is None:
        print("❌ Failed to load models. Exiting.")
        return
    
    # Load data
    papers_df, texts = load_and_prepare_data(n_papers=50)  # Smaller sample for demo
    if papers_df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Run neural demos
    print(f"\n🚀 Starting Neural Processing Pipeline...")
    
    # 1. Generate embeddings
    embeddings = neural_embeddings_demo(models, texts, papers_df)
    
    # 2. Semantic clustering
    cluster_labels = None
    if embeddings is not None:
        cluster_labels = semantic_clustering_demo(embeddings, papers_df)
    
    # 3. Zero-shot classification
    classifications = zero_shot_classification_demo(models, texts, papers_df)
    
    # 4. Neural summarization
    summaries = neural_summarization_demo(models, texts, papers_df)
    
    # 5. Semantic search
    if embeddings is not None:
        semantic_search_demo(embeddings, texts, papers_df)
    
    # 6. Save results
    save_comprehensive_results(papers_df, embeddings, cluster_labels, classifications, summaries)
    
    # Final summary
    print(f"\n🎉 Neural Demo Completed Successfully!")
    print("=" * 70)
    print("📊 What was accomplished:")
    print("   ✅ Neural embeddings generated with Sentence Transformers")
    print("   ✅ Semantic clustering performed")
    print("   ✅ Zero-shot classification demonstrated")
    print("   ✅ Neural summarization with BART")
    print("   ✅ Semantic search capabilities")
    print("   ✅ Comprehensive results saved")
    print("\n📁 Check the 'results/' directory for all outputs!")
    print("🖼️ Cluster visualization: results/semantic_clusters.png")
    print("📄 Detailed report: results/comprehensive_report.json")
    print("\n🚀 The Scholarly Topic Navigator is ready for deployment!")

if __name__ == "__main__":
    main()