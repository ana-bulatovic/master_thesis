# NLP Pipeline: Sarcasm Detection, Sentiment Classification & Summarization

**Master Thesis Project** - Application of Large Language Models for Natural Language Processing Tasks

This project implements a multi-stage NLP pipeline for processing product reviews, combining:
1. **Sarcasm Detection** - Identifying sarcasm in reviews
2. **Sentiment Classification** - Classifying reviews as positive, negative, or neutral
3. **Abstractive Summarization** - Generating concise summaries of reviews

---

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Evaluation Metrics](#evaluation-metrics)
- [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline processes customer reviews through three interconnected NLP tasks:

```
Review Input
    ↓
[Sarcasm Detection] → Identify if review contains sarcasm
    ↓
[Sentiment Classification] → Classify as positive/negative/neutral
    ↓
[Summarization] → Generate summary (with and without sentiment info)
    ↓
Results with Evaluation Metrics
```

### Key Features

- **Multiple LLM Models**: Support for Meta Llama, Alibaba Qwen, Mistral, and other models via Ollama
- **Flexible Prompting**: Zero-shot, Few-shot, and Chain-of-Thought prompting techniques
- **Sentiment-Aware Summarization**: Compare summaries with and without sentiment context
- **Comprehensive Evaluation**: ROUGE, BERTScore, Accuracy, Precision, Recall, F1 metrics
- **Pipeline Variants**: Test different combinations of tasks
- **Batch Processing**: Process multiple reviews efficiently

---

## Prerequisites

### System Requirements

- **Python 3.8+**
- **8GB+ RAM** (recommended for LLM inference)
- **GPU** (optional but recommended for faster processing)

### Required Software

1. **Ollama** - Open source LLM framework
   - Download from: https://ollama.ai
   - Install for your OS (Windows, macOS, Linux)

---

## Installation

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd master-rad

# Or navigate to your project folder
cd "Master rad"
```

### Step 2: Create Python Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues installing `torch`, you may want to install it separately:

```bash
# For CPU only (smaller, ~150MB)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8 (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install and Start Ollama

```bash
# 1. Install Ollama from https://ollama.ai/download/windows

# 2. Start Ollama server
# Option A: If 'ollama' command works in terminal:
ollama serve

# Option B: If 'ollama' command doesn't work, use provided batch files:
# In one terminal:
start_ollama.bat

# In another terminal:
start_ollama_models.bat pull llama2

# 3. Download required models
ollama pull llama2
ollama pull mistral
ollama pull neural-chat
```

**Note:** If you get "'ollama' is not recognized" error, use the batch files instead. Run `python fix_ollama.py` to diagnose PATH issues.

Leave this terminal running - Ollama server must be active for the pipeline to work.

---

## Configuration

### Config File: `config.yaml`

Edit `config.yaml` to customize the pipeline:

```yaml
model:
  name: "llama2"  # Default LLM model
  temperature: 0.7  # Lower = more deterministic, Higher = more creative
  num_predict: 128  # Max tokens to generate

pipeline:
  sarcasm_model: "llama2"
  sentiment_model: "llama2"
  summarization_model: "llama2"
  
  # Prompting technique: zero-shot, few-shot, chain-of-thought
  prompting_technique: "few-shot"
  
  # Enable/disable pipeline components
  use_sarcasm_detection: true
  use_sentiment_for_summarization: true
  
  # Output directory for results
  output_dir: "./results"
```

### Available Models in Ollama

```bash
# Pull additional models:
ollama pull llama2
ollama pull mistral
ollama pull neural-chat
ollama pull orca-mini
ollama pull dolphin-mixtral
```

View all downloaded models:
```bash
ollama list
```

---

## Usage

### Basic Usage: Run Full Pipeline

```bash
# With Ollama server running, open a new terminal:
python main.py
```

This will:
1. Check Ollama connection
2. Process sample reviews through the pipeline
3. Display results for each review
4. Save results to `results/pipeline_results.json`

### Web UI Interface ⭐

**NEW!** Use the interactive web interface for easier configuration and visualization:

```bash
# Install additional dependencies (if not already done)
pip install streamlit plotly matplotlib

# Run the web interface
streamlit run app.py
```

The web UI provides:
- **Interactive Configuration**: Select models, techniques, and pipeline components
- **Real-time Processing**: Process reviews with live progress updates
- **Visual Results**: See results with sentiment distribution charts
- **Metrics Dashboard**: View evaluation metrics with interactive plots
- **Export Options**: Download results as JSON or CSV
- **Batch Processing**: Handle multiple reviews at once

**Features:**
- 🔧 **Model Selection**: Choose from available Ollama models
- 📝 **Technique Selection**: Zero-shot, Few-shot, Chain-of-thought
- ⚙️ **Pipeline Configuration**: Enable/disable sarcasm detection, sentiment classification, summarization
- 📊 **Visual Metrics**: Charts for sentiment distribution and evaluation metrics
- 💾 **Export Results**: JSON and CSV export options

### Command Options

```bash
# Run individual task demonstration
python main.py individual

# Compare different prompting techniques
python main.py techniques

# Demonstrate evaluation metrics
python main.py evaluate
```

### Using the Pipeline in Your Code

```python
from pipeline import NLPPipeline

# Initialize pipeline
pipeline = NLPPipeline()

# Check setup
if not pipeline.check_setup():
    print("Ollama not running")
    exit()

# Process a single review
review = "Your review text here"
result = pipeline.process_review(review)

# Print result
pipeline.print_result(result)

# Save results
pipeline.save_results([result], "my_results.json")
```

### Process Batch of Reviews

```python
from pipeline import NLPPipeline

pipeline = NLPPipeline()

reviews = [
    "First review text",
    "Second review text",
    "Third review text",
]

results = pipeline.process_batch(reviews)
pipeline.save_results(results, "batch_results.json")
```

### Custom Configuration

```python
from pipeline import NLPPipeline

# Load custom config
pipeline = NLPPipeline(config_path="custom_config.yaml")

# Or modify config programmatically
pipeline.config.pipeline_config.prompting_technique = "chain-of-thought"
pipeline._initialize_components()

# Process reviews
result = pipeline.process_review("Your review")
```

---

## Project Structure

```
Master rad/
├── main.py                     # Entry point script
├── pipeline.py                 # Main NLP pipeline orchestrator
├── config.py                   # Configuration management
├── config.yaml                 # Configuration file
├── ollama_client.py            # Ollama API client
├── sarcasm_detector.py         # Sarcasm detection module
├── sentiment_classifier.py     # Sentiment classification module
├── summarizer.py               # Text summarization module
├── evaluator.py                # Evaluation metrics module
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── results/                    # Output directory (created automatically)
│   └── pipeline_results.json  # Pipeline results
└── data/                       # Input data directory (optional)
    └── reviews.csv            # Your review data
```

---

## Features

### 1. Sarcasm Detection

**Purpose**: Identifies whether a review contains sarcasm, which can significantly affect sentiment classification accuracy.

**Example**:
```
Review: "Great product, stopped working after one day!"
Is Sarcastic: Yes
```

**Prompting Techniques**:
- **Zero-shot**: No examples, just ask if text is sarcastic
- **Few-shot**: Provide examples of sarcastic and non-sarcastic reviews
- **Chain-of-Thought**: Step-by-step reasoning about sarcasm indicators

### 2. Sentiment Classification

**Purpose**: Classify reviews into positive, negative, or neutral sentiment.

**Example**:
```
Review: "Excellent product, highly recommended!"
Sentiment: Positive
```

**Classes**: 
- Positive
- Negative  
- Neutral

### 3. Abstractive Summarization

**Purpose**: Generate concise summaries of reviews.

**Variants**:

a) **Standard Summarization** (without sentiment info):
```
Review: "This phone has excellent camera but poor battery life. Overall good but not perfect."
Summary: "Phone with good camera but limited battery"
```

b) **Sentiment-Aware Summarization** (with sentiment):
```
Review: "This phone has excellent camera but poor battery life"
Sentiment: Mixed (detected as Negative overall)
Summary: "Good camera doesn't compensate for poor battery - avoid"
```

---

## Evaluation Metrics

### Classification Metrics

For sarcasm detection and sentiment classification:

- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted positives, how many are correct
- **Recall**: Of actual positives, how many were detected
- **F1-Score**: Harmonic mean of precision and recall

### Summarization Metrics

For abstractive summarization evaluation:

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **ROUGE-1**: Unigram overlap between generated and reference
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

#### BERTScore
- Semantic similarity using contextual embeddings
- Better captures meaning beyond surface-level n-grams
- **Precision, Recall, F1**: Standard metrics

### Using Evaluator

```python
from evaluator import Evaluator

# Evaluate classification
predictions = ["positive", "negative", "positive"]
references = ["positive", "positive", "negative"]
metrics = Evaluator.evaluate_classification(predictions, references)
print(f"F1-Score: {metrics['f1']:.3f}")

# Evaluate summarization
predictions_sum = ["Summary 1", "Summary 2"]
references_sum = ["Reference 1", "Reference 2"]
eval_results = Evaluator.evaluate_summarization(predictions_sum, references_sum)
print(f"ROUGE-1 F1: {eval_results['rouge']['rouge1_f']:.3f}")
```

---

## Troubleshooting

### "Ollama server is not running"

**Solution**:
```bash
# Open a terminal and run:
ollama serve

# Keep this terminal open while running pipeline
```

### "Model not found" error

**Solution**:
```bash
# List available models
ollama list

# Pull missing model
ollama pull llama2

# Update config.yaml with correct model name
```

### Slow inference

**Solutions**:
1. Use smaller model: `ollama pull orca-mini`
2. Reduce `num_predict` in config.yaml
3. Use GPU-enabled Ollama
4. Process fewer reviews at once

### Memory issues

**Solutions**:
1. Use smaller model (orca-mini instead of llama2)
2. Reduce batch size in config
3. Close other applications
4. Use CPU-only torch if GPU is available but limited memory

### "ImportError: No module named 'rouge_score'"

**Solution**:
```bash
pip install rouge-score
```

### "ModuleNotFoundError: No module named 'transformers'"

**Solution**:
```bash
pip install transformers
```

---

## Data Format

### Input

The pipeline accepts review text as strings:

```python
reviews = [
    "First product review text...",
    "Second product review text...",
    "Third product review text..."
]
```

### Output Format

Each result contains:

```json
{
  "original_review": "Review text...",
  "timestamp": "2024-01-15T10:30:45.123456",
  "pipeline_variant": "full",
  "sarcasm_detection": {
    "text": "Review text...",
    "is_sarcastic": false,
    "response": "No",
    "technique": "few-shot"
  },
  "sentiment_classification": {
    "text": "Review text...",
    "sentiment": "positive",
    "response": "positive",
    "technique": "few-shot"
  },
  "summarization_without_sentiment": {
    "original_text": "Review text...",
    "summary": "Generated summary...",
    "sentiment": null,
    "sentiment_aware": false,
    "technique": "few-shot"
  },
  "summarization_with_sentiment": {
    "original_text": "Review text...",
    "summary": "Sentiment-aware summary...",
    "sentiment": "positive",
    "sentiment_aware": true,
    "technique": "few-shot"
  }
}
```

---

## Advanced Usage

### Using Different Models

```python
from pipeline import NLPPipeline

pipeline = NLPPipeline()

# Change model for specific task
pipeline.config.pipeline_config.sentiment_model = "mistral"
pipeline._initialize_components()

review = "Your review"
result = pipeline.process_review(review)
```

### Comparing Prompting Techniques

```python
from pipeline import NLPPipeline

techniques = ["zero-shot", "few-shot", "chain-of-thought"]
review = "Your review"

for technique in techniques:
    pipeline = NLPPipeline()
    pipeline.config.pipeline_config.prompting_technique = technique
    pipeline._initialize_components()
    
    result = pipeline.process_review(review)
    print(f"{technique}: {result['sentiment_classification']['sentiment']}")
```

### Sentiment-Aware Pipeline Variants

```python
# Variant 1: Only sentiment → summarization
pipeline.config.pipeline_config.use_sarcasm_detection = False
pipeline.config.pipeline_config.use_sentiment_for_summarization = True

# Variant 2: All three: sarcasm → sentiment → summarization
pipeline.config.pipeline_config.use_sarcasm_detection = True
pipeline.config.pipeline_config.use_sentiment_for_summarization = True

# Variant 3: Independent summarization (no sentiment)
pipeline.config.pipeline_config.use_sarcasm_detection = False
pipeline.config.pipeline_config.use_sentiment_for_summarization = False
```

---

## Performance Optimization

### Running on GPU

For better performance with larger models:

```bash
# Ensure your GPU has CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Use smaller models for faster inference
ollama pull orca-mini
# Update config.yaml: model.name = "orca-mini"
```

### Batch Processing

For processing large numbers of reviews:

```python
from pipeline import NLPPipeline
import json

pipeline = NLPPipeline()

# Load reviews from file
with open('reviews.json', 'r') as f:
    reviews = json.load(f)

# Process in batches
batch_size = 100
for i in range(0, len(reviews), batch_size):
    batch = reviews[i:i+batch_size]
    results = pipeline.process_batch(batch)
    pipeline.save_results(results, f"results_batch_{i//batch_size}.json")
```

---

## References

### Thesis Components

- **Sarcasm Detection**: Using LLMs to identify implicit negative sentiment
- **Sentiment Classification**: Multi-class classification (positive/negative/neutral)
- **Abstractive Summarization**: Generating concise summaries, with sentiment-aware variant
- **Evaluation**: ROUGE, BERTScore, standard classification metrics
- **Datasets**: Amazon Reviews Multi dataset, public sarcasm datasets

### Key Models

- Meta Llama 3.1 / Llama 2
- Alibaba Qwen 2.5
- Mistral 7B
- OpenChat, Neural Chat

### External Tools

- **Ollama**: https://ollama.ai - Local LLM serving
- **ROUGE**: Package for summarization evaluation
- **BERTScore**: Semantic similarity scoring
- **Scikit-learn**: Classification metrics

---

## Contributing

To modify or extend the pipeline:

1. Add new prompting techniques in respective modules
2. Implement new evaluation metrics in `evaluator.py`
3. Add new NLP tasks following the module pattern
4. Update configuration schema in `config.py`

---

## License

This is a Master's Thesis project. Please refer to your institution's guidelines for usage and distribution.

---

## Contact & Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Verify Ollama is running: `ollama serve`
3. Check Python environment: `python --version` (should be 3.8+)
4. Verify dependencies: `pip list | grep -E "(transformers|torch|requests)"`

---

**Last Updated**: January 2024
**Python Version**: 3.8+
**Ollama Version**: Latest
