# 🎯 Complete Step-by-Step UI Guide

## 📋 Before Starting

Make sure:
- ✅ Ollama is running (`ollama serve` or `start_ollama.bat`)
- ✅ A model is downloaded (`llama2:latest` shown in terminal)
- ✅ Virtual environment is activated: `venv\Scripts\activate`
- ✅ All dependencies installed: `pip install -r requirements.txt`

---

## 🚀 Step 1: Start the UI

In your terminal with activated virtual environment:

```powershell
streamlit run app.py
```

Or use the quick launcher:
```powershell
python run_ui.py
```

A browser window will open automatically with the interface.

---

## ⚙️ Step 2: Configure (MUST DO THIS FIRST!)

### Left Sidebar - Configuration Panel

#### 1️⃣ **Model Selection**
- Dropdown: "LLM Model"
- Select: `llama2:latest` (or whichever model you have)
- ✅ Check status shows: **✅ Connected**

#### 2️⃣ **Model Parameters**
Three sliders to adjust:

**Temperature** (0.0 → 2.0)
- 0.0 = Deterministic (always same answer)
- 0.7 = Balanced (default) ← **RECOMMENDED**
- 2.0 = Very creative (random answers)

**Max Tokens** (64 → 512)
- 128 = Short responses (default) ← **START HERE**
- 256 = Medium responses
- 512 = Long responses

#### 3️⃣ **Prompting Technique**
Choose one:
- ❌ **Zero-shot**: No examples (sometimes poor results)
- ✅ **Few-shot**: With examples (BEST) ← **RECOMMENDED**
- 🤔 **Chain-of-thought**: Step-by-step (slower but detailed)

#### 4️⃣ **Pipeline Components**
Check/uncheck what you want:

- ✅ **Sarcasm Detection**: Detect sarcasm in reviews
- ✅ **Sentiment Classification**: Classify as positive/negative/neutral
- ✅ **Sentiment-Aware Summarization**: Use sentiment in summary

**Tip:** Start with all 3 checked.

#### 5️⃣ **Apply Configuration**
- Click the blue button: **🔄 Apply Configuration**
- Wait for it to process
- Sidebar should update and show: **✅ Connected**

---

## 📝 Step 3: Enter Your Reviews

### Main Content Area - Choose Input Method

Three ways to input reviews:

#### **Option A: Single Review** (Best for Testing)

1. Select radio button: **Single Review**
2. Click the text box
3. Type your review:

```
"This product is absolutely amazing! It broke after 2 days though. 
Total waste of money but the customer service was fantastic!"
```

4. ✅ Text appears in the box

#### **Option B: Batch Reviews** (Multiple Reviews)

1. Select radio button: **Batch Reviews**
2. Enter multiple reviews, one per line:

```
This product is great and arrived quickly!
Terrible quality, waste of money.
It's okay, nothing special.
Very satisfied with my purchase.
Absolute garbage, don't buy!
```

3. ✅ Each line is processed separately

#### **Option C: Upload File** (Bulk Processing)

1. Select radio button: **Upload File**
2. Click "Browse files"
3. Choose a file:
   - **Text file (.txt)**: One review per line
   - **CSV (.csv)**: Must have a column with review text
   - **JSON (.json)**: Array of objects with `text` field

4. If CSV: Select which column contains reviews
5. ✅ Shows "Loaded X reviews from file"

---

## 🚀 Step 4: Process Reviews

### Process Button

1. Look for blue button: **🚀 Process Reviews**
   - Only enabled if you have reviews entered
   - Shows a loading spinner

2. Click it and wait...
   - First time: 30-60 seconds (slower)
   - Subsequent times: 10-30 seconds
   - Spinner shows: "Processing reviews... This may take a few minutes."

3. ✅ Success message: "✅ Processed X reviews successfully!"

---

## 📊 Step 5: View Results

### Results Display Section

#### **📈 Sentiment Distribution** (Top)
- Pie chart showing % Positive/Negative/Neutral
- Visual summary of overall sentiment

#### **📊 Classification Metrics**
Four key metrics displayed:

| Metric | What it means |
|--------|---------------|
| **Accuracy** | How many correct predictions (0-1) |
| **Precision** | Of positive predictions, how many were correct |
| **Recall** | Of actual positives, how many did we catch |
| **F1-Score** | Balanced measure (0-1, higher is better) |

#### **📋 Detailed Results**
Click each review to expand and see:

1. **Original Review**: Your input text
2. **Sarcasm**: "Yes" or "No" + confidence
3. **Sentiment**: Shows as colored text
   - 🟢 POSITIVE (green)
   - 🔴 NEGATIVE (red)
   - 🟠 NEUTRAL (orange)
4. **Standard Summary**: Generated summary
5. **Sentiment-Aware Summary**: Summary with emotion context

---

## 💾 Step 6: Export Results

### Export Options (Bottom of Page)

Two buttons:

#### **📄 Export as JSON**
- Saves all details to: `results_YYYYMMDD_HHMMSS.json`
- Includes everything: original text, all predictions, summaries
- Best for: detailed analysis

#### **📊 Export as CSV**
- Saves to: `results_YYYYMMDD_HHMMSS.csv`
- Tabular format (rows & columns)
- Best for: Excel, analysis tools

---

## 🔄 Step 7: Process More Reviews

1. Go back to Step 3: Enter New Reviews
2. Change configuration if needed (optional)
3. Click **🔄 Apply Configuration** if you changed it
4. Enter new reviews
5. Click **🚀 Process Reviews**
6. Results update automatically!

---

## 📝 Complete Example Workflow

### Scenario: Analyze 3 Product Reviews

**1. Start UI:**
```powershell
streamlit run app.py
```

**2. Configure (Left Sidebar):**
- Model: `llama2:latest` ✅
- Temperature: 0.7
- Max Tokens: 128
- Technique: Few-shot
- All components checked: ✅
- Click: **🔄 Apply Configuration**

**3. Enter Reviews (Main Area):**
- Select: **Batch Reviews**
- Enter:
```
This product works perfectly! Highly recommend.
Complete waste of money, broke after one day.
Average product, does what it says.
```

**4. Process:**
- Click: **🚀 Process Reviews**
- Wait for ✅ success message

**5. View Results:**
- See pie chart (sentiment distribution)
- See metrics (accuracy, F1-score)
- Click each review to expand details
- See sarcasm, sentiment, summaries

**6. Export:**
- Click: **📄 Export as JSON** or **📊 Export as CSV**
- Files saved to same folder as app.py

---

## ⚠️ Troubleshooting

### "Pipeline not initialized"
→ **Solution**: Click **🔄 Apply Configuration** in sidebar

### "❌ Not Connected" (Ollama status)
→ **Solution**: 
- Make sure Ollama is running: `ollama serve`
- Check no port conflicts: `netstat -ano | findstr :11434`

### Processing is very slow
→ **Solutions**:
- Decrease Max Tokens (use 64 instead of 128)
- Use smaller model: `ollama pull orca-mini`
- Wait (first run is always slower)

### "Error: Model not found"
→ **Solution**: Download model
```powershell
C:\Users\ANA\AppData\Local\Programs\Ollama\ollama.exe pull llama2
```

### Results don't look good
→ **Solutions**:
- Try **Few-shot** or **Chain-of-thought** technique
- Increase Temperature to 0.8-0.9
- Use better model (mistral, llama3 if disk space allows)

---

## 💡 Tips for Best Results

1. **Clear Reviews**: Short, clear reviews work better than rambling ones
2. **Few-shot Best**: Always use "Few-shot" for better quality
3. **Temperature**: 0.7-0.8 is sweet spot (not too random, not too stiff)
4. **Batch Processing**: Process 5-10 reviews at once for better metrics
5. **Model Selection**: Mistral > Llama2 (if you have it), but Llama2 still good

---

## 🎓 What Each Component Does

### 🤖 Sarcasm Detection
- Identifies if text contains sarcasm
- Example: "Great product! Broke immediately." = Sarcasm detected

### 😊 Sentiment Classification
- Classifies emotion: Positive, Negative, Neutral
- Example: "Amazing quality!" = Positive

### 📝 Summarization
- Generates short summary
- **Standard**: Normal summary
- **Sentiment-Aware**: Emphasizes emotional tone

---

## 🚀 You're Ready!

Just follow these steps in order:
1. ✅ Configure (sidebar)
2. ✅ Enter reviews (main area)
3. ✅ Process (click button)
4. ✅ View results (charts & details)
5. ✅ Export (JSON/CSV)

**Happy analyzing!** 🎉
