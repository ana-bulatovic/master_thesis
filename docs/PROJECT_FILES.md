# 📦 Project Files Summary

## Created Project Structure

```
Master rad/
├── 📄 tema_za_master_rad.txt          (Original thesis topic)
├── 
├── 🐍 PYTHON CODE FILES
├── ├── main.py                         ⭐ Entry point - Run this to start
├── ├── app.py                           🌐 Streamlit Web UI - Interactive interface
├── ├── run_ui.py                        🚀 Quick launcher for web UI
├── ├── pipeline.py                     Main NLP pipeline orchestrator
├── ├── config.py                       Configuration management
├── ├── ollama_client.py                Ollama LLM API client
├── ├── sarcasm_detector.py             Sarcasm detection module
├── ├── sentiment_classifier.py         Sentiment classification module
├── ├── summarizer.py                   Text summarization module
├── ├── evaluator.py                    Evaluation metrics module
├── ├── data_loader.py                  Data loading utilities
├── └── examples.py                     Usage examples
├── 
├── 📋 DOCUMENTATION
├── ├── README.md                       📖 Complete documentation
├── ├── QUICKSTART.md                   ⚡ Quick start (5 minutes)
├── ├── PROJECT_FILES.md                This file
├── ├── config.yaml                     ⚙️ Configuration file
├── └── requirements.txt                Python dependencies
├── 
├── 📂 AUTO-CREATED DIRECTORIES
├── ├── results/                        Output results (JSON)
├── └── data/                           Input data (optional)
└── (git, venv, etc.)                   Virtual environment
```

---

## 🎯 Quick Navigation

### To Get Started:
1. **QUICKSTART.md** - 5-minute setup guide
2. **README.md** - Full documentation
3. **main.py** - Run the pipeline

### To Understand the Code:
1. **pipeline.py** - Main orchestrator
2. **config.py** - Configuration
3. **examples.py** - Usage examples

### To Process Your Data:
1. **data_loader.py** - Load CSV, JSON, TXT
2. **examples.py** - Example code

### To Evaluate Results:
1. **evaluator.py** - Metrics computation
2. **examples.py** - Evaluation examples

---

## 📝 File Descriptions

### Core Pipeline Files

#### `main.py` - Entry Point ⭐
- **Purpose**: Main script to run the pipeline
- **Usage**: `python main.py [command]`
- **Commands**: 
  - No args: Run full pipeline
  - `individual`: Test individual tasks
  - `techniques`: Compare prompting techniques
  - `evaluate`: Show evaluation metrics

#### `pipeline.py` - Pipeline Orchestrator
- **Purpose**: Main NLP pipeline class
- **Key Class**: `NLPPipeline`
- **Methods**:
  - `process_review()` - Process single review
  - `process_batch()` - Process multiple reviews
  - `check_setup()` - Verify Ollama connection
  - `save_results()` - Save to JSON
  - `print_result()` - Pretty print results

#### `config.py` - Configuration Management
- **Purpose**: Handle pipeline configuration
- **Classes**: `Config`, `ModelConfig`, `PipelineConfig`
- **File**: `config.yaml` (YAML configuration)

#### `ollama_client.py` - LLM Interface
- **Purpose**: Communicate with Ollama server
- **Class**: `OllamaClient`
- **Methods**:
  - `generate()` - Generate text
  - `chat()` - Chat interface
  - `check_connection()` - Verify server
  - `list_models()` - List available models

### NLP Task Modules

#### `sarcasm_detector.py` - Sarcasm Detection
- **Purpose**: Detect sarcasm in reviews
- **Class**: `SarcasmDetector`
- **Methods**:
  - `detect()` - Classify single review
  - `detect_batch()` - Classify multiple
- **Techniques**: zero-shot, few-shot, chain-of-thought

#### `sentiment_classifier.py` - Sentiment Classification
- **Purpose**: Classify sentiment (positive/negative/neutral)
- **Class**: `SentimentClassifier`
- **Methods**:
  - `classify()` - Classify single review
  - `classify_batch()` - Classify multiple
- **Techniques**: zero-shot, few-shot, chain-of-thought

#### `summarizer.py` - Text Summarization
- **Purpose**: Generate abstractive summaries
- **Class**: `TextSummarizer`
- **Methods**:
  - `summarize()` - Summarize single review
  - `summarize_batch()` - Summarize multiple
- **Variants**: With/without sentiment info

#### `evaluator.py` - Evaluation Metrics
- **Purpose**: Compute evaluation metrics
- **Class**: `Evaluator`
- **Methods**:
  - `evaluate_classification()` - Compute Acc, P, R, F1
  - `compute_rouge()` - ROUGE scores
  - `compute_bert_score()` - BERTScore
  - `evaluate_summarization()` - Full evaluation

### Utility Files

#### `data_loader.py` - Data Loading
- **Classes**: `DataLoader`, `RatingToSentiment`
- **Supported Formats**: CSV, JSON, TXT
- **Methods**:
  - `load_csv()` - Load from CSV
  - `load_json()` - Load from JSON
  - `load_txt()` - Load from text file
  - `load_from_directory()` - Load from folder
  - `split_into_sentences()` - Text preprocessing
  - `convert()` - Rating to sentiment

#### `examples.py` - Usage Examples
- **Purpose**: Demonstrate pipeline usage
- **Examples**:
  1. Simple review processing
  2. Load from CSV
  3. Load from JSON
  4. Evaluate results
  5. Custom configuration
  6. Compare techniques
- **Usage**: `python examples.py <number>`

### Configuration Files

#### `config.yaml` - Configuration
- **Purpose**: Pipeline and model settings
- **Sections**:
  - `model`: LLM settings (temperature, etc.)
  - `pipeline`: Task configuration and options

#### `requirements.txt` - Dependencies
- **Purpose**: List of Python packages
- **Usage**: `pip install -r requirements.txt`
- **Packages**: transformers, torch, rouge-score, bert-score, etc.

### Documentation Files

#### `README.md` - Complete Guide 📖
- **Length**: Comprehensive (3000+ lines)
- **Sections**:
  - Overview & features
  - Installation guide
  - Configuration
  - Usage examples
  - Project structure
  - Evaluation metrics
  - Troubleshooting
  - Advanced usage
  - References

#### `QUICKSTART.md` - Fast Setup ⚡
- **Length**: Brief (200 lines)
- **For**: Users wanting to get started immediately
- **Time**: 5 minutes to running

#### `PROJECT_FILES.md` - This File
- **Purpose**: Describe all project files
- **For**: Understanding project structure

---

## 🚀 Getting Started

### 1. First Time Setup (5 minutes)
```bash
# See QUICKSTART.md for detailed steps
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
ollama serve                    # In another terminal
python main.py
```

### 2. Run the Pipeline
```bash
python main.py
```

### 3. Process Your Data
```bash
python examples.py 2            # Load from CSV example
# Or modify examples.py to use your data
```

### 4. View Results
```
results/pipeline_results.json   # Pipeline output
results/technique_comparison.json  # Technique comparison
```

---

## 📊 File Statistics

| Category | Count | Purpose |
|----------|-------|---------|
| Python Modules | 9 | Core functionality |
| Config Files | 2 | Settings & dependencies |
| Documentation | 3 | Guides & references |
| **Total** | **14** | Complete project |

---

## 🔧 Dependencies

### Python Packages
- `transformers` - Hugging Face transformers
- `torch` - PyTorch
- `requests` - HTTP client
- `pandas` - Data processing
- `scikit-learn` - Classification metrics
- `rouge-score` - Summarization evaluation
- `bert-score` - BERT-based scoring
- `pyyaml` - YAML parsing

### External Software
- **Ollama** - LLM server (download from ollama.ai)
- **Python 3.8+** - Programming language
- **Git** (optional) - Version control

---

## 🎓 Thesis Implementation

This project implements your master's thesis:

**Title**: "Application of Large Language Models for Sarcasm Detection, Sentiment Classification, and Summarization of User Reviews"

### Implemented Components:
- ✅ Sarcasm Detection
- ✅ Sentiment Classification (3 classes)
- ✅ Abstractive Summarization
- ✅ Sentiment-aware summarization
- ✅ Pipeline orchestration
- ✅ Evaluation metrics (ROUGE, BERTScore, F1, etc.)
- ✅ Prompting techniques (zero-shot, few-shot, chain-of-thought)
- ✅ Multiple LLM models (via Ollama)
- ✅ Batch processing
- ✅ Data loading utilities

### Features by Task:

#### Sarcasm Detection
- Input: Review text
- Output: Is sarcastic (Yes/No)
- Models: Any Ollama LLM

#### Sentiment Classification
- Input: Review text
- Output: Sentiment (Positive/Negative/Neutral)
- Models: Any Ollama LLM

#### Summarization
- Input: Review text (+ optional sentiment)
- Output: Summary (1-2 sentences)
- Variants: Standard & sentiment-aware
- Models: Any Ollama LLM

---

## 📦 What You Can Do With This Project

1. **Process your own reviews**
   - Load from CSV, JSON, or TXT
   - Get sarcasm, sentiment, and summaries

2. **Compare LLM models**
   - Test different models: llama2, mistral, qwen
   - Compare performance

3. **Test prompting techniques**
   - Zero-shot, few-shot, chain-of-thought
   - Evaluate impact on results

4. **Conduct experiments**
   - Evaluate sentiment-aware summarization
   - Measure sarcasm impact on sentiment
   - Compute classification metrics

5. **Extend functionality**
   - Add new NLP tasks
   - Implement new evaluation metrics
   - Support new data formats

---

## 📞 Need Help?

1. **Quick questions**: See **QUICKSTART.md**
2. **Detailed help**: See **README.md** 
3. **Code examples**: Run **examples.py**
4. **Issues**: Check Troubleshooting in README.md

---

## ✅ Verification Checklist

After installation, verify everything works:

- [ ] Python 3.8+ installed: `python --version`
- [ ] Ollama installed: `ollama --version`
- [ ] Virtual environment: `venv` folder exists
- [ ] Dependencies: `pip list | findstr torch transformers`
- [ ] Ollama server running: `ollama serve` (separate terminal)
- [ ] Pipeline works: `python main.py`
- [ ] Results saved: `results/pipeline_results.json` exists

---

**Project Setup Complete! 🎉**

Start with **streamlit run app.py** for the best experience!
