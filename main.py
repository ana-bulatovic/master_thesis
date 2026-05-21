#!/usr/bin/env python3
"""
Main script for running the NLP pipeline for sarcasm detection,
sentiment classification, and abstractive summarization.

This script demonstrates the full pipeline as described in the thesis:
1. Sarcasm Detection
2. Sentiment Classification
3. Abstractive Summarization (with and without sentiment information)
"""

import sys
import json
from typing import List
from pipeline import NLPPipeline
from evaluator import Evaluator

# Sample test reviews (Serbian Amazon reviews examples)
SAMPLE_REVIEWS = [
    "Ovaj proizvod je čudesna kupnja! Odličan kvalitet i brza dostava. Velmi sam zadovoljan.",
    "Ponos sam što sam kupio ovo! Nije ni loše, nije specijalno. Prosečan proizvod.",
    "Najgori proizvod koji sam ikada kupio! Otišao je u kvar posle dva dana. Totalno razočaran.",
    "Divno, prosljeđen mi je proizvod u komadima. 'Odličan' kvalitet!",
    "Proizvod je dobar, funkcira kako treba. Nema razloga za žalbu.",
]

def demonstrate_pipeline():
    """Demonstrate the full NLP pipeline"""
    
    print("="*80)
    print("NLP PIPELINE: Sarcasm Detection, Sentiment Classification & Summarization")
    print("Master Thesis - Application of Large Language Models")
    print("="*80)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = NLPPipeline()
    
    # Check Ollama setup
    print("\n2. Checking Ollama setup...")
    if not pipeline.check_setup():
        print("\n❌ Setup failed. Please ensure Ollama is running.")
        print("   Installation and setup guide is in the README.md file.")
        return False
    
    print("\n✓ Setup successful!")
    
    # Process sample reviews
    print("\n3. Processing sample reviews...")
    results = pipeline.process_batch(SAMPLE_REVIEWS)
    
    # Display results
    print("\n4. Displaying results...")
    for i, result in enumerate(results, 1):
        print(f"\n--- Review {i} ---")
        pipeline.print_result(result)
    
    # Save results
    print("\n5. Saving results...")
    pipeline.save_results(results, "pipeline_results.json")
    
    return True

def demonstrate_individual_tasks():
    """Demonstrate individual NLP tasks"""
    
    print("\n" + "="*80)
    print("INDIVIDUAL TASK DEMONSTRATION")
    print("="*80)
    
    pipeline = NLPPipeline()
    
    if not pipeline.check_setup():
        return False
    
    test_review = "Ovaj proizvod je čudesna kupnja! Funkcioniše savršeno."
    
    print(f"\nTest review: {test_review}\n")
    
    # Task 1: Sarcasm Detection
    print("--- Task 1: Sarcasm Detection ---")
    if pipeline.sarcasm_detector:
        sarcasm_result = pipeline.sarcasm_detector.detect(test_review)
        print(f"Is sarcastic: {sarcasm_result['is_sarcastic']}")
        print(f"Response: {sarcasm_result['response']}")
    
    # Task 2: Sentiment Classification
    print("\n--- Task 2: Sentiment Classification ---")
    sentiment_result = pipeline.sentiment_classifier.classify(test_review)
    print(f"Sentiment: {sentiment_result['sentiment']}")
    print(f"Response: {sentiment_result['response']}")
    
    # Task 3a: Summarization (without sentiment)
    print("\n--- Task 3a: Summarization (Standard) ---")
    summary_result = pipeline.summarizer_without_sentiment.summarize(test_review)
    print(f"Summary: {summary_result['summary']}")
    
    # Task 3b: Summarization (with sentiment)
    print("\n--- Task 3b: Summarization (Sentiment-Aware) ---")
    summary_sentiment_result = pipeline.summarizer_with_sentiment.summarize(
        test_review, 
        sentiment=sentiment_result['sentiment']
    )
    print(f"Summary: {summary_sentiment_result['summary']}")
    
    return True

def demonstrate_different_prompting_techniques():
    """Demonstrate different prompting techniques"""
    
    print("\n" + "="*80)
    print("PROMPTING TECHNIQUES COMPARISON")
    print("="*80)
    
    test_review = "Odličan proizvod, ali je malo skupi. Svakako bih ga preporučio!"
    
    techniques = ["zero-shot", "few-shot", "chain-of-thought"]
    
    print(f"\nTest review: {test_review}\n")
    
    for technique in techniques:
        print(f"\n--- {technique.upper()} ---")
        
        pipeline = NLPPipeline()
        pipeline.config.pipeline_config.prompting_technique = technique
        pipeline._initialize_components()
        
        if not pipeline.check_setup():
            continue
        
        # Test sentiment classification with different techniques
        result = pipeline.sentiment_classifier.classify(test_review)
        print(f"Sentiment: {result['sentiment']}")
        print(f"Response: {result['response'][:150]}...")

def run_evaluation_example():
    """Demonstrate evaluation metrics"""
    
    print("\n" + "="*80)
    print("EVALUATION METRICS DEMONSTRATION")
    print("="*80)
    
    # Example: Classification evaluation
    print("\n--- Classification Metrics ---")
    predictions = ["positive", "negative", "positive", "neutral", "negative"]
    references = ["positive", "negative", "neutral", "neutral", "negative"]
    
    metrics = Evaluator.evaluate_classification(predictions, references)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    
    # Example: Summarization evaluation
    print("\n--- Summarization Metrics ---")
    predictions_sum = [
        "Odličan proizvod, preporučujem ga.",
        "Loš proizvod, ne kupujte ga."
    ]
    references_sum = [
        "Odličan proizvod, veoma ga preporučujem.",
        "Loš i neupotrebljiv proizvod, izbjegavajte."
    ]
    
    eval_results = Evaluator.evaluate_summarization(
        predictions_sum, 
        references_sum,
        use_rouge=True,
        use_bertscore=False
    )
    
    print(f"ROUGE Scores:")
    for key, value in eval_results['rouge'].items():
        print(f"  {key}: {value:.3f}")

def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "individual":
            demonstrate_individual_tasks()
        elif command == "techniques":
            demonstrate_different_prompting_techniques()
        elif command == "evaluate":
            run_evaluation_example()
        else:
            print("Unknown command. Available commands:")
            print("  python main.py individual     - Run individual task demonstrations")
            print("  python main.py techniques     - Compare prompting techniques")
            print("  python main.py evaluate       - Demonstrate evaluation metrics")
            print("  python main.py (default)      - Run full pipeline")
    else:
        demonstrate_pipeline()

if __name__ == "__main__":
    main()
