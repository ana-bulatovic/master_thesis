# ⚡ QUICK START GUIDE

## 🚀 Get Running in 5 Minutes

### Prerequisites Check
```bash
# Check Python version (should be 3.8+)
python --version

# You need Ollama installed from https://ollama.ai
```

### 1. Setup (First Time Only)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Ollama Server

**Option A: If 'ollama' command works:**
```bash
ollama serve
```

**Option B: If 'ollama' command doesn't work (use batch files):**
```bash
# Start server
start_ollama.bat

# In another terminal, download model
start_ollama_models.bat pull llama2
```

**Keep Ollama running!** The server must be active for the pipeline to work.

### 3. Launch Web Interface ⭐

**In your original terminal (with activated venv), run:**
```bash
streamlit run app.py
```

**Or use the quick launcher:**
```bash
python run_ui.py
```

**This opens your browser with the interactive UI!**

---

## 🌐 Web Interface Features

The web UI provides everything you need:

### ⚙️ Configuration Panel (Sidebar)
- **Model Selection**: Choose from available Ollama models
- **Prompting Technique**: Zero-shot, Few-shot, or Chain-of-thought
- **Pipeline Components**: Enable/disable sarcasm detection, sentiment, summarization
- **Model Parameters**: Temperature, max tokens

### 📝 Input Section
- **Single Review**: Process one review at a time
- **Batch Reviews**: Process multiple reviews (one per line)
- **File Upload**: Upload CSV, JSON, or TXT files

### 📊 Results & Visualization
- **Detailed Results**: See sarcasm, sentiment, and summaries for each review
- **Sentiment Charts**: Pie chart showing sentiment distribution
- **Metrics Dashboard**: Accuracy, Precision, Recall, F1-score, ROUGE scores

### 💾 Export Options
- **JSON Export**: Full detailed results
- **CSV Export**: Tabular format for analysis

---

## 📱 Alternative: Command Line

If you prefer command line:

```bash
python main.py                    # Full pipeline
python main.py individual         # Individual tasks
python main.py techniques         # Compare techniques
python main.py evaluate           # Show metrics
```

---

## 🐛 Troubleshooting

### Error: "Ollama server is not running"
→ Make sure you ran `ollama serve` in another terminal

### Error: "Model not found"
→ Run: `ollama pull llama2`

### Error: "ModuleNotFoundError"
→ Run: `pip install -r requirements.txt`

### Slow processing
→ Use smaller model: `ollama pull orca-mini`
→ Update config.yaml: `name: "orca-mini"`

---

## 💻 Use in Your Own Code

```python
from pipeline import NLPPipeline

# Initialize
pipeline = NLPPipeline()

# Process review
review = "This product is amazing!"
result = pipeline.process_review(review)

# See results
pipeline.print_result(result)

# Save results
pipeline.save_results([result])
```

---

## 📂 Output

Results are saved to: `results/pipeline_results.json`

Each result contains:
- Original review
- Sarcasm detection (Yes/No)
- Sentiment (positive/negative/neutral)
- Summary (standard + sentiment-aware)

---

## 🎓 Next Steps

1. Read full `README.md` for detailed documentation
2. Explore `config.yaml` for configuration options
3. Check `main.py` for more usage examples
4. Process your own reviews (CSV, JSON, TXT format)

---

## 📞 Need Help?

See **Troubleshooting** section in `README.md`
