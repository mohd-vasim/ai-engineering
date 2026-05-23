# RAG Evaluation Dataset Synthesizer

A tool to automatically generate Q&A pairs from PDF documents for evaluating Retrieval-Augmented Generation (RAG) systems using AI.

## Features

- 📄 **PDF Processing**: Convert PDF files to markdown format
- ✂️ **Intelligent Chunking**: Smart text chunking with configurable token limits
- 🤖 **AI-Powered Q&A Generation**: Generate realistic Q&A pairs grounded in document content
- 🎨 **Web Interface**: Streamlit app for easy interaction
- 📊 **Data Export**: Download generated datasets as CSV
- 📓 **Jupyter Notebooks**: Interactive notebooks for exploration and development

## Project Structure

```
├── app.py                          # Streamlit web application
├── main.py                         # Command-line interface
├── utils.py                        # Text chunking utilities
├── notebooks/
│   ├── synthesize_from_scratch.ipynb      # Development notebook
│   └── synthesize_with_deepeval.ipynb     # DeepEval integration notebook
├── data/
│   ├── eval-sets/                 # Generated evaluation datasets (CSV)
│   └── images/                    # Extracted images from PDFs
└── pyproject.toml                 # Project dependencies
```

## Setup

### Prerequisites

- Python 3.12+
- NVIDIA API Key (set as `NVIDIA_API_KEY` environment variable)

### Installation

1. **Install dependencies**:
   ```bash
   pip install -e .
   ```

2. **Set environment variable**:
   ```bash
   export NVIDIA_API_KEY="your-api-key-here"
   ```

## Usage

### Option 1: Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

The web interface provides:
- 📤 PDF file upload
- ⚙️ Configurable chunking parameters
- 🚀 One-click Q&A generation
- 📥 CSV download

### Option 2: Jupyter Notebook

```bash
jupyter notebook notebooks/synthesize_from_scratch.ipynb
```

For advanced workflows with DeepEval integration:
```bash
jupyter notebook notebooks/synthesize_with_deepeval.ipynb
```

### Option 3: Command Line

```bash
python main.py --pdf-files <file1.pdf> <file2.pdf> --output output.csv
```

## Configuration

### Chunking Parameters

Adjust in the Streamlit sidebar or pass to `chunk_text()`:

- **Max tokens per chunk**: Maximum size of each text chunk (100-2000)
- **Min tokens per chunk**: Minimum size to avoid tiny fragments (50-500)
- **Extract images**: Whether to extract images from PDFs

### API Configuration

- **Base URL**: `https://integrate.api.nvidia.com/v1`
- **Embedding Model**: `nvidia/nv-embed-v1`
- **Chat Model**: `openai/gpt-oss-120b`

## Output Format

Generated Q&A pairs are saved as CSV with the following columns:

| Column | Description |
|--------|-------------|
| `id` | Unique identifier for the Q&A pair |
| `query` | User question |
| `ai_response` | Ground-truth answer from the document |
| `context` | Original text chunk used for generation |

Example filename: `rag_eval_dataset_20260523_164119.csv`

## Workflow

1. **Upload PDFs** → Web interface or command line
2. **Convert to Markdown** → Extract text and images
3. **Chunk Text** → Split into manageable pieces
4. **Generate Q&A** → AI creates realistic question-answer pairs
5. **Export Dataset** → Download as CSV for RAG evaluation

## API Integration

The project uses:
- **NVIDIA API** for embeddings and chat models
- **LangChain** for agent framework
- **PyMuPDF4LLM** for PDF processing
- **Tiktoken** for token counting

## Example

```python
from utils import chunk_text
from app import generate_qa_pairs

# Chunk your text
chunks = chunk_text(text, config={"max_tokens": 600, "min_tokens": 150})

# Generate Q&A pairs
qa_pairs = generate_qa_pairs(chunks[:5])

# Export to CSV
df.to_csv("dataset.csv", index=False)
```

## Troubleshooting

### "NVIDIA_API_KEY not set"
- Ensure your API key is exported: `export NVIDIA_API_KEY="your-key"`
- Check it's valid: `echo $NVIDIA_API_KEY`

### Slow PDF Processing
- Reduce the number of chunks to process
- Try disabling image extraction
- Reduce max tokens per chunk

### Out of Memory
- Process fewer PDFs at once
- Reduce max tokens per chunk
- Use a smaller subset of chunks

## Performance Tips

1. **Start with small PDFs** (5-20 pages) to test
2. **Process a subset of chunks** (5-10) before full run
3. **Enable caching** in Streamlit for repeated runs
4. **Adjust chunking** based on your document type

## License

MIT

## Contributing

Contributions welcome! Please submit issues and pull requests.

## Support

For questions or issues, please open a GitHub issue or contact the maintainer.
