import os
import json
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path

class DataLoader:
    """Load and preprocess review data"""
    
    @staticmethod
    def load_csv(filepath: str, text_column: str = "review_text", 
                 rating_column: str = "rating", limit: int = None) -> Tuple[List[str], List[int]]:
        """
        Load reviews from CSV file
        
        Args:
            filepath: Path to CSV file
            text_column: Column name containing review text
            rating_column: Column name containing rating
            limit: Maximum number of reviews to load
        
        Returns:
            Tuple of (reviews, ratings)
        """
        try:
            df = pd.read_csv(filepath)
            
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV")
            
            # Clean text
            reviews = df[text_column].fillna("").astype(str).tolist()
            
            # Load ratings if available
            ratings = []
            if rating_column in df.columns:
                ratings = df[rating_column].tolist()
            
            if limit:
                reviews = reviews[:limit]
                ratings = ratings[:limit] if ratings else []
            
            return reviews, ratings
        
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return [], []
    
    @staticmethod
    def load_json(filepath: str, text_field: str = "text", limit: int = None) -> List[str]:
        """
        Load reviews from JSON file (list of objects or strings)
        
        Args:
            filepath: Path to JSON file
            text_field: Field name containing review text (if JSON is list of objects)
            limit: Maximum number of reviews to load
        
        Returns:
            List of reviews
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            reviews = []
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        reviews.append(item)
                    elif isinstance(item, dict) and text_field in item:
                        reviews.append(str(item[text_field]))
            
            if limit:
                reviews = reviews[:limit]
            
            return reviews
        
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return []
    
    @staticmethod
    def load_txt(filepath: str, skip_empty: bool = True, limit: int = None) -> List[str]:
        """
        Load reviews from text file (one review per line)
        
        Args:
            filepath: Path to text file
            skip_empty: Skip empty lines
            limit: Maximum number of reviews to load
        
        Returns:
            List of reviews
        """
        try:
            reviews = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line or not skip_empty:
                        reviews.append(line)
            
            if limit:
                reviews = reviews[:limit]
            
            return reviews
        
        except Exception as e:
            print(f"Error loading text file: {e}")
            return []
    
    @staticmethod
    def load_from_directory(directory: str, extension: str = "*.csv", 
                           limit: int = None) -> List[str]:
        """
        Load reviews from all files in a directory
        
        Args:
            directory: Path to directory
            extension: File extension pattern (*.csv, *.json, *.txt)
            limit: Maximum total reviews to load
        
        Returns:
            List of reviews
        """
        reviews = []
        
        try:
            path = Path(directory)
            
            for filepath in path.glob(extension):
                if filepath.suffix == '.csv':
                    file_reviews, _ = DataLoader.load_csv(str(filepath), limit=limit)
                elif filepath.suffix == '.json':
                    file_reviews = DataLoader.load_json(str(filepath), limit=limit)
                elif filepath.suffix == '.txt':
                    file_reviews = DataLoader.load_txt(str(filepath), limit=limit)
                else:
                    continue
                
                reviews.extend(file_reviews)
                
                if limit and len(reviews) >= limit:
                    reviews = reviews[:limit]
                    break
            
            return reviews
        
        except Exception as e:
            print(f"Error loading from directory: {e}")
            return []
    
    @staticmethod
    def split_into_sentences(review: str, min_length: int = 10) -> List[str]:
        """
        Split review into sentences
        
        Args:
            review: Review text
            min_length: Minimum sentence length
        
        Returns:
            List of sentences
        """
        import re
        
        # Simple sentence splitter
        sentences = re.split(r'[.!?]+', review)
        sentences = [s.strip() for s in sentences if len(s.strip()) > min_length]
        
        return sentences
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Basic text preprocessing
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        return text.strip()

class RatingToSentiment:
    """Convert rating scores to sentiment labels"""
    
    @staticmethod
    def convert(rating: float, scale: int = 5) -> str:
        """
        Convert numerical rating to sentiment label
        
        Args:
            rating: Numerical rating
            scale: Rating scale (typically 5 for Amazon reviews)
        
        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'
        """
        if scale == 5:
            if rating >= 4:
                return "positive"
            elif rating <= 2:
                return "negative"
            else:
                return "neutral"
        
        elif scale == 10:
            if rating >= 7:
                return "positive"
            elif rating <= 4:
                return "negative"
            else:
                return "neutral"
        
        else:
            # Generic for any scale
            midpoint = scale / 2
            if rating > midpoint + 1:
                return "positive"
            elif rating < midpoint - 1:
                return "negative"
            else:
                return "neutral"
    
    @staticmethod
    def convert_batch(ratings: List[float], scale: int = 5) -> List[str]:
        """
        Convert batch of ratings to sentiments
        
        Args:
            ratings: List of ratings
            scale: Rating scale
        
        Returns:
            List of sentiment labels
        """
        return [RatingToSentiment.convert(r, scale) for r in ratings]
