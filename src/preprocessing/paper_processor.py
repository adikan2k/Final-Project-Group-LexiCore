"""
Paper-specific preprocessing for academic documents.
Handles the unified paper dataset format.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
from pathlib import Path
import json
from tqdm import tqdm
from .text_cleaner import TextCleaner


class PaperProcessor:
    """
    Processor for academic paper datasets with preprocessing capabilities.
    """
    
    def __init__(self, language: str = 'english', use_spacy: bool = True):
        """
        Initialize paper processor.
        
        Args:
            language: Language for text processing
            use_spacy: Whether to use spaCy for advanced NLP
        """
        self.text_cleaner = TextCleaner(language=language, use_spacy=use_spacy)
        self.processed_fields = ['title', 'abstract']
        
    def load_papers(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load papers from various formats.
        
        Args:
            file_path: Path to paper dataset file
            
        Returns:
            DataFrame with papers
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def validate_paper_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate and analyze paper dataset.
        
        Args:
            df: Papers DataFrame
            
        Returns:
            Validation report
        """
        report = {
            'total_papers': len(df),
            'required_columns': ['paper_id', 'title', 'abstract'],
            'missing_columns': [],
            'missing_values': {},
            'text_statistics': {}
        }
        
        # Check required columns
        for col in report['required_columns']:
            if col not in df.columns:
                report['missing_columns'].append(col)
        
        # Check missing values
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                report['missing_values'][col] = int(missing)
        
        # Text statistics for text fields
        for field in ['title', 'abstract']:
            if field in df.columns:
                text_col = df[field].fillna('')
                lengths = text_col.str.len()
                report['text_statistics'][field] = {
                    'min_length': int(lengths.min()),
                    'max_length': int(lengths.max()),
                    'mean_length': float(lengths.mean()),
                    'median_length': float(lengths.median()),
                    'empty_count': int((text_col == '').sum())
                }
        
        return report
    
    def preprocess_paper_field(
        self,
        text: str,
        field_type: str = 'abstract',
        custom_params: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Preprocess a single paper text field.
        
        Args:
            text: Text to preprocess
            field_type: Type of field ('title', 'abstract', 'content')
            custom_params: Custom preprocessing parameters
            
        Returns:
            Preprocessing results
        """
        # Default parameters based on field type
        default_params = {
            'title': {
                'remove_citations': True,
                'remove_urls': True,
                'lowercase': True,
                'expand_contractions': True,
                'remove_special': True,
                'keep_punctuation': False
            },
            'abstract': {
                'remove_citations': True,
                'remove_urls': True,
                'lowercase': True,
                'expand_contractions': True,
                'remove_special': True,
                'keep_punctuation': True
            },
            'content': {
                'remove_citations': False,  # Keep some citations for context
                'remove_urls': True,
                'lowercase': True,
                'expand_contractions': True,
                'remove_special': True,
                'keep_punctuation': True
            }
        }
        
        # Get parameters for field type
        params = default_params.get(field_type, default_params['abstract'])
        if custom_params:
            params.update(custom_params)
        
        # Preprocess text
        return self.text_cleaner.preprocess_text(
            text,
            clean_params=params,
            tokenize=True,
            lemmatize=True,
            stem=False
        )
    
    def process_papers_batch(
        self,
        df: pd.DataFrame,
        fields: Optional[List[str]] = None,
        batch_size: int = 1000,
        save_intermediate: bool = True,
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process papers in batches with preprocessing.
        
        Args:
            df: Papers DataFrame
            fields: Fields to process (default: ['title', 'abstract'])
            batch_size: Size of processing batches
            save_intermediate: Save intermediate results
            output_dir: Directory to save intermediate results
            
        Returns:
            Processed DataFrame
        """
        if fields is None:
            fields = ['title', 'abstract']
        
        # Validate input
        missing_fields = [f for f in fields if f not in df.columns]
        if missing_fields:
            raise ValueError(f"Missing fields in DataFrame: {missing_fields}")
        
        # Create output directory if needed
        if save_intermediate and output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        processed_df = df.copy()
        
        # Process each field
        for field in fields:
            print(f"Processing field: {field}")
            
            # Create new columns for processed data
            processed_df[f'{field}_cleaned'] = ''
            processed_df[f'{field}_tokens'] = None
            processed_df[f'{field}_lemmatized'] = None
            processed_df[f'{field}_token_count'] = 0
            processed_df[f'{field}_char_count'] = 0
            
            # Process in batches
            total_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
            
            for batch_idx in tqdm(range(total_batches), desc=f"Processing {field}"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(df))
                
                batch_df = df.iloc[start_idx:end_idx]
                
                # Process each text in batch
                for idx, row in batch_df.iterrows():
                    text = row[field]
                    if pd.isna(text) or text == '':
                        continue
                    
                    # Preprocess text
                    result = self.preprocess_paper_field(text, field)
                    
                    # Store results
                    processed_df.loc[idx, f'{field}_cleaned'] = result['cleaned_text']
                    processed_df.at[idx, f'{field}_tokens'] = result.get('tokens', [])
                    processed_df.at[idx, f'{field}_lemmatized'] = result.get('lemmatized_tokens', [])
                    processed_df.loc[idx, f'{field}_token_count'] = result.get('token_count', 0)
                    processed_df.loc[idx, f'{field}_char_count'] = result.get('cleaned_length', 0)
                
                # Save intermediate results
                if save_intermediate and output_dir and batch_idx % 10 == 0:
                    temp_file = Path(output_dir) / f"processed_batch_{field}_{batch_idx}.parquet"
                    batch_result = processed_df.iloc[start_idx:end_idx]
                    batch_result.to_parquet(temp_file, index=False)
        
        return processed_df
    
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional text-based features for papers.
        
        Args:
            df: DataFrame with processed papers
            
        Returns:
            DataFrame with additional features
        """
        feature_df = df.copy()
        
        # Basic text statistics
        if 'title_cleaned' in df.columns:
            feature_df['title_word_count'] = df['title_cleaned'].str.split().str.len()
            
        if 'abstract_cleaned' in df.columns:
            feature_df['abstract_word_count'] = df['abstract_cleaned'].str.split().str.len()
            feature_df['abstract_sentence_count'] = df['abstract_cleaned'].str.count(r'[.!?]+')
            
        # Combined text features
        if 'title_cleaned' in df.columns and 'abstract_cleaned' in df.columns:
            feature_df['combined_text'] = (
                df['title_cleaned'].fillna('') + ' ' + df['abstract_cleaned'].fillna('')
            ).str.strip()
            feature_df['total_word_count'] = feature_df['combined_text'].str.split().str.len()
        
        # Author statistics
        if 'authors' in df.columns:
            feature_df['author_count'] = df['authors'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        # Category features
        if 'categories' in df.columns:
            feature_df['category_count'] = df['categories'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        return feature_df
    
    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        output_path: Union[str, Path],
        format: str = 'parquet',
        include_metadata: bool = True
    ):
        """
        Save processed paper data with metadata.
        
        Args:
            df: Processed DataFrame
            output_path: Output file path
            format: Output format ('parquet', 'json', 'csv')
            include_metadata: Include processing metadata
        """
        output_path = Path(output_path)
        
        # Save main data
        if format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'csv':
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save metadata
        if include_metadata:
            metadata = {
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'total_papers': len(df),
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'preprocessing_info': {
                    'text_cleaner_version': '0.1.0',
                    'processed_fields': [col for col in df.columns if col.endswith('_cleaned')],
                    'feature_fields': [col for col in df.columns if 'count' in col or 'feature' in col]
                }
            }
            
            metadata_path = output_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def get_processing_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate summary of processed papers.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Processing summary
        """
        summary = {
            'total_papers': len(df),
            'processed_fields': [col.replace('_cleaned', '') for col in df.columns if col.endswith('_cleaned')],
            'average_statistics': {}
        }
        
        # Calculate averages for numeric fields
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'count' in col or 'length' in col:
                summary['average_statistics'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return summary