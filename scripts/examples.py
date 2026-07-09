#!/usr/bin/env python3
"""
Example script showing how to use the pipeline with your own data.

This demonstrates:
1. Loading reviews from various sources
2. Processing them through the pipeline
3. Evaluating results
"""

import json
from pathlib import Path
from pipeline import NLPPipeline
from data_loader import DataLoader, RatingToSentiment
from evaluator import Evaluator

def example_1_simple_reviews():
    """Example 1: Process simple review texts"""
    print("=" * 80)
    print("EXAMPLE 1: Simple Review Processing")
    print("=" * 80)
    
    reviews = [
        "Odličan proizvod, veoma sam zadovoljan kupnjom!",
        "Loš kvalitet, nije vredan novca.",
        "Proizvod je prosečan, nema ništa posebno.",
    ]
    
    pipeline = NLPPipeline()
    if not pipeline.check_setup():
        return
    
    results = pipeline.process_batch(reviews)
    
    # Display and save
    for result in results:
        pipeline.print_result(result)
    
    pipeline.save_results(results, "example1_results.json")
    print("\n✓ Results saved to results/example1_results.json")

def example_2_csv_data():
    """Example 2: Load and process reviews from CSV"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Loading from CSV File")
    print("=" * 80)
    
    # Create sample CSV file if it doesn't exist
    sample_csv = "sample_reviews.csv"
    if not Path(sample_csv).exists():
        with open(sample_csv, 'w', encoding='utf-8') as f:
            f.write("review_text,rating\n")
            f.write('"Odličan proizvod, preporučujem ga svima!",5\n')
            f.write('"Veoma loš, otišao je u kvar posle dana.",1\n')
            f.write('"Nije loše, ali nije specijalno.",3\n')
    
    # Load reviews
    print(f"Loading reviews from {sample_csv}...")
    reviews, ratings = DataLoader.load_csv(
        sample_csv,
        text_column="review_text",
        rating_column="rating"
    )
    
    # Convert ratings to sentiment labels
    sentiments = RatingToSentiment.convert_batch(ratings, scale=5)
    
    print(f"Loaded {len(reviews)} reviews")
    print(f"Ratings: {ratings}")
    print(f"Sentiments from ratings: {sentiments}\n")
    
    # Process pipeline
    pipeline = NLPPipeline()
    if not pipeline.check_setup():
        return
    
    results = pipeline.process_batch(reviews)
    pipeline.save_results(results, "example2_csv_results.json")
    
    print("✓ Results saved to results/example2_csv_results.json")

def example_3_json_data():
    """Example 3: Load and process reviews from JSON"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Loading from JSON File")
    print("=" * 80)
    
    # Create sample JSON file
    sample_json = "sample_reviews.json"
    sample_data = [
        {"text": "Odličan proizvod, preporučujem ga!", "rating": 5},
        {"text": "Loš kvalitet i brzo se pokvarilo.", "rating": 1},
        {"text": "Prosečan proizvod, ništa specijalno.", "rating": 3},
    ]
    
    with open(sample_json, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # Load reviews
    print(f"Loading reviews from {sample_json}...")
    reviews = DataLoader.load_json(sample_json, text_field="text")
    
    print(f"Loaded {len(reviews)} reviews\n")
    
    # Process pipeline
    pipeline = NLPPipeline()
    if not pipeline.check_setup():
        return
    
    results = pipeline.process_batch(reviews)
    pipeline.save_results(results, "example3_json_results.json")
    
    print("✓ Results saved to results/example3_json_results.json")

def example_4_evaluation():
    """Example 4: Evaluate results"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Evaluating Classification Results")
    print("=" * 80)
    
    # Simulate predictions vs ground truth
    predictions = ["positive", "negative", "positive", "neutral", "negative"]
    ground_truth = ["positive", "negative", "neutral", "neutral", "negative"]
    
    print("Predictions:", predictions)
    print("Ground Truth:", ground_truth)
    print()
    
    # Compute metrics
    metrics = Evaluator.evaluate_classification(predictions, ground_truth)
    
    print("Classification Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1']:.3f}")

def example_5_custom_config():
    """Example 5: Use custom configuration"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Using Custom Configuration")
    print("=" * 80)
    
    # Create custom config
    custom_config = {
        "model": {
            "name": "mistral",
            "temperature": 0.5,
        },
        "pipeline": {
            "sarcasm_model": "mistral",
            "sentiment_model": "mistral",
            "summarization_model": "mistral",
            "prompting_technique": "chain-of-thought",
            "use_sarcasm_detection": True,
            "use_sentiment_for_summarization": True,
        }
    }
    
    # Save custom config
    import yaml
    with open("custom_config.yaml", 'w') as f:
        yaml.dump(custom_config, f)
    
    print("Created custom_config.yaml with:")
    print(f"  Model: mistral")
    print(f"  Technique: chain-of-thought")
    
    # Load with custom config
    pipeline = NLPPipeline("custom_config.yaml")
    
    print("\n✓ Pipeline initialized with custom config")
    print(f"  Model: {pipeline.config.pipeline_config.sentiment_model}")
    print(f"  Technique: {pipeline.config.pipeline_config.prompting_technique}")

def example_6_different_techniques():
    """Example 6: Compare different prompting techniques"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Comparing Prompting Techniques")
    print("=" * 80)
    
    test_review = "Odličan proizvod, ali je skupi!"
    print(f"Test Review: {test_review}\n")
    
    techniques = ["zero-shot", "few-shot", "chain-of-thought"]
    results_by_technique = {}
    
    for technique in techniques:
        print(f"Testing with {technique.upper()}...")
        
        pipeline = NLPPipeline()
        pipeline.config.pipeline_config.prompting_technique = technique
        pipeline._initialize_components()
        
        if not pipeline.check_setup():
            continue
        
        result = pipeline.process_review(test_review)
        results_by_technique[technique] = result
        
        sentiment = result['sentiment_classification']['sentiment']
        print(f"  → Sentiment: {sentiment}")
    
    print("\n✓ Comparison complete")
    
    # Save comparison
    with open("results/technique_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(results_by_technique, f, ensure_ascii=False, indent=2)

def main():
    """Run examples"""
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        examples = {
            "1": example_1_simple_reviews,
            "2": example_2_csv_data,
            "3": example_3_json_data,
            "4": example_4_evaluation,
            "5": example_5_custom_config,
            "6": example_6_different_techniques,
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Unknown example: {example_num}")
            print("Available examples: 1, 2, 3, 4, 5, 6")
    else:
        print("Usage: python examples.py <number>")
        print("\nAvailable examples:")
        print("  1 - Simple review processing")
        print("  2 - Load from CSV file")
        print("  3 - Load from JSON file")
        print("  4 - Evaluate classification results")
        print("  5 - Use custom configuration")
        print("  6 - Compare prompting techniques")
        print("\nExample: python examples.py 1")

if __name__ == "__main__":
    main()
