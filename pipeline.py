from typing import List, Dict, Optional
import json
import os
from datetime import datetime

from config import Config, PipelineConfig
from ollama_client import OllamaClient
from sarcasm_detector import SarcasmDetector
from sentiment_classifier import SentimentClassifier
from summarizer import TextSummarizer
from evaluator import Evaluator

class NLPPipeline:
    """Main pipeline orchestrating sarcasm detection, sentiment classification, and summarization"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = Config(config_path)
        self.client = OllamaClient(self.config.model_config)
        
        # Initialize components
        self.sarcasm_detector = None
        self.sentiment_classifier = None
        self.summarizer_without_sentiment = None
        self.summarizer_with_sentiment = None
        
        self._initialize_components()
        
        # Ensure output directory exists
        os.makedirs(self.config.pipeline_config.output_dir, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize NLP components"""
        technique = self.config.pipeline_config.prompting_technique
        
        if self.config.pipeline_config.use_sarcasm_detection:
            self.sarcasm_detector = SarcasmDetector(
                self.client,
                prompting_technique=technique
            )
        
        self.sentiment_classifier = SentimentClassifier(
            self.client,
            prompting_technique=technique
        )
        
        self.summarizer_without_sentiment = TextSummarizer(
            self.client,
            prompting_technique=technique,
            use_sentiment_info=False
        )
        
        self.summarizer_with_sentiment = TextSummarizer(
            self.client,
            prompting_technique=technique,
            use_sentiment_info=True
        )
    
    def check_setup(self) -> bool:
        """Check if Ollama is running and required models are available"""
        print("Checking Ollama connection...")
        
        if not self.client.check_connection():
            print("❌ Error: Ollama server is not running!")
            print("   Make sure Ollama is started with: ollama serve")
            return False
        
        print("✓ Ollama connection established")
        
        available_models = self.client.list_models()
        print(f"\nAvailable models: {available_models}")
        
        required_models = [
            self.config.pipeline_config.sarcasm_model,
            self.config.pipeline_config.sentiment_model,
            self.config.pipeline_config.summarization_model,
        ]
        
        for model in required_models:
            if not any(model in m for m in available_models):
                print(f"⚠ Warning: Model '{model}' not found in available models")
        
        return True
    
    def process_review(self, review: str) -> Dict:
        """
        Process a single review through the pipeline
        
        Args:
            review: Review text to process
        
        Returns:
            Dictionary with all pipeline results
        """
        result = {
            "original_review": review,
            "timestamp": datetime.now().isoformat(),
            "pipeline_variant": "full"
        }
        
        # Step 1: Sarcasm detection (if enabled)
        if self.config.pipeline_config.use_sarcasm_detection and self.sarcasm_detector:
            sarcasm_result = self.sarcasm_detector.detect(
                review, 
                model=self.config.pipeline_config.sarcasm_model
            )
            result["sarcasm_detection"] = sarcasm_result
        else:
            result["sarcasm_detection"] = None
        
        # Step 2: Sentiment classification
        sentiment_result = self.sentiment_classifier.classify(
            review,
            model=self.config.pipeline_config.sentiment_model
        )
        result["sentiment_classification"] = sentiment_result
        
        # Step 3: Summarization without sentiment
        summary_without_sentiment = self.summarizer_without_sentiment.summarize(
            review,
            sentiment=None,
            model=self.config.pipeline_config.summarization_model
        )
        result["summarization_without_sentiment"] = summary_without_sentiment
        
        # Step 4: Summarization with sentiment (if enabled)
        if self.config.pipeline_config.use_sentiment_for_summarization:
            sentiment = sentiment_result["sentiment"]
            summary_with_sentiment = self.summarizer_with_sentiment.summarize(
                review,
                sentiment=sentiment,
                model=self.config.pipeline_config.summarization_model
            )
            result["summarization_with_sentiment"] = summary_with_sentiment
        else:
            result["summarization_with_sentiment"] = None
        
        return result
    
    def process_batch(self, reviews: List[str]) -> List[Dict]:
        """
        Process multiple reviews through the pipeline
        
        Args:
            reviews: List of review texts
        
        Returns:
            List of pipeline results
        """
        results = []
        print(f"\nProcessing {len(reviews)} reviews...")
        
        for i, review in enumerate(reviews, 1):
            print(f"  [{i}/{len(reviews)}] Processing review...")
            result = self.process_review(review)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], filename: str = "results.json"):
        """
        Save pipeline results to file
        
        Args:
            results: List of pipeline results
            filename: Output filename
        """
        output_path = os.path.join(self.config.pipeline_config.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Results saved to: {output_path}")
    
    def print_result(self, result: Dict):
        """Pretty print a single result"""
        print("\n" + "="*80)
        print("REVIEW:", result['original_review'][:100] + "...")
        
        if result.get('sarcasm_detection'):
            print(f"\nSarcasm: {result['sarcasm_detection']['is_sarcastic']}")
        
        print(f"\nSentiment: {result['sentiment_classification']['sentiment'].upper()}")
        
        print(f"\nSummary (standard): {result['summarization_without_sentiment']['summary']}")
        
        if result.get('summarization_with_sentiment'):
            print(f"Summary (sentiment-aware): {result['summarization_with_sentiment']['summary']}")
        print("="*80)
