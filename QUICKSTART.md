# Quick Start Guide

## Installation & Setup (2 minutes)

### 1. Set API Key
```bash
export NVIDIA_API_KEY="your-api-key"
```

### 2. Install Dependencies
```bash
pip install -e .
```

### 3. Run Streamlit App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Using the App

### Step 1: Upload PDFs
- Click the file uploader in the main area
- Select one or more PDF files
- Wait for "Loaded X PDF(s)" message

### Step 2: Configure Settings (Optional)
Use the sidebar to adjust:
- **Max tokens per chunk**: How large each text piece can be
- **Min tokens per chunk**: Minimum size before merging chunks
- **Extract images**: Toggle to save images from PDFs

### Step 3: Generate Q&A Pairs
- Enter number of chunks to process (start with 5)
- Click "🚀 Generate Q&A Pairs"
- Wait for processing to complete

### Step 4: Download Results
- Review generated pairs in the table
- Click "📥 Download CSV" to save
- File saved to `data/eval-sets/`

---

## Configuration Recommendations

### For Quick Testing
```
- Max tokens: 400
- Min tokens: 100
- Chunks to process: 3-5
```

### For Production
```
- Max tokens: 600
- Min tokens: 150
- Chunks to process: All available
```

### For Large PDFs (>50 pages)
```
- Max tokens: 800
- Min tokens: 200
- Process in batches of 10-20 chunks
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "API Key not found" | Run: `export NVIDIA_API_KEY="your-key"` |
| App crashes on upload | File might be corrupted, try another PDF |
| Q&A generation fails | Reduce chunks to process, check API quota |
| Very slow processing | Reduce max tokens or process fewer chunks |

---

## File Locations

- **App code**: `app.py`
- **Generated datasets**: `data/eval-sets/*.csv`
- **Notebooks**: `notebooks/`
- **Config**: `.streamlit/config.toml`

---

## Next Steps

1. Try with a small 5-10 page PDF first
2. Review generated Q&A pairs quality
3. Adjust chunking parameters if needed
4. Process larger datasets
5. Use CSV for RAG evaluation

---

## Example Workflow

```bash
# Terminal 1: Set up environment
export NVIDIA_API_KEY="your-key"
cd /path/to/project
source .venv/bin/activate

# Terminal 1: Start Streamlit
streamlit run app.py

# Browser: Open http://localhost:8501
# 1. Upload your PDF
# 2. Click "Generate Q&A Pairs"
# 3. Download results as CSV
```

---

## Need Help?

- Check the **logs** at bottom of Streamlit app
- Review **README.md** for detailed documentation
- Check **app.py** comments for implementation details
