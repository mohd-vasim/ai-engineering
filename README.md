# AI Engineering

A monorepo covering the full AI/LLM engineering pipeline — from data preparation and fine-tuning to agentic patterns, RAG, inference, and deployment.

---

## 01. Data Preparation for LLM Training

Notebook implementing **6 tokenization techniques from scratch**: whitespace/regex, character tokenization, word/character n-grams, Byte Pair Encoding (BPE) with full training loop and inference, byte-level BPE, and WordPiece/Unigram comparisons. Interview-oriented with focus on BPE.

- **Deps:** Python >=3.12, numpy, ipykernel

---

## 02. LLM Fine-Tuning

Fine-tuning Llama 2 using **QLoRA** and **LoRA** with Hugging Face `transformers` + `peft`. Covers 4-bit quantization, LoRA adapter training, and inference with merged adapters.

---

## 03. LLM Quickstart

Quickstart notebooks for cloud LLM APIs:

| Notebook | Coverage |
|----------|----------|
| `01-python_quickstart_gemini.ipynb` | Google Gemini SDK — text generation, multimodal (text+image), chat, embeddings |
| `02-amazon-bedrock.ipynb` | Amazon Bedrock via LangChain — multiple models (Gemma, Qwen, Claude), latency metrics |

---

## 04. AI Agents

7 notebooks implementing the **Anthropic agent design patterns** using LangChain + LangGraph:

| # | Pattern | Description |
|---|---------|-------------|
| 01 | **Augmented LLM** | Structured output (Pydantic), retrieval, tool binding |
| 02 | **Prompt Chaining** | StateGraph pipeline: generate joke → check punchline → improve → polish |
| 03 | **Parallelization** | Fan-out: generate story + joke + poem on the same topic |
| 04 | **Routing** | Intent-based routing to poem/story/joke handlers via structured output |
| 05 | **Orchestrator-Worker** | Planner generates report sections, workers execute each section |
| 06 | **Generator-Evaluator** | Joke generation loop with evaluation/feedback until "funny" |
| 07 | **Human-in-the-Loop** | Interrupts agent execution before sensitive SQL operations, requires human approval |

All use `openai/gpt-oss-120b` via Novita AI.

---

## 05. Projects

### 01 — Synthesize LLM Dataset
Dataset synthesis pipeline for LLM fine-tuning. *(scaffolded)*

### 02 — Synthesize RAG Evaluation Dataset
**Streamlit + NVIDIA API** tool to generate Q&A pairs from PDFs for RAG evaluation.

- Upload PDF → convert to markdown (`pymupdf4llm`) → chunk with `tiktoken` → generate Q&A via LangChain agent (NVIDIA `gpt-oss-120b`) → export CSV
- Smart chunking: paragraph extraction, sentence-boundary splitting, greedy merge with token limits
- **Deps:** streamlit, langchain, langgraph, pymupdf4llm, chromadb, deepeval, pandas

### 03 — Fastest RAG
Streamlit RAG app with **binary-quantized vectors** for sub-100ms retrieval.

- Embeddings: `BAAI/bge-large-en-v1.5` → binary quantization (float32 → uint8, 32x reduction)
- Vector DB: Milvus Lite with `BINARY_VECTOR` type, `BIN_FLAT` index, `HAMMING` distance
- LLM: Novita AI `openai/gpt-oss-120b`
- Displays retrieval latency per query
- **Deps:** streamlit, pymilvus, llama-index, llama-index-embeddings-huggingface, llama-index-llms-novita

### 04 — Multimodal RAG
Multimodal document assistant — understands both text and images from PDFs.

- Stack: LangChain, Milvus, BGE-VL embeddings, Qdrant, Nvidia NIM
- Docker Compose with full Milvus standalone (etcd + minio + milvus)
- **Deps:** langchain, langchain-huggingface, langchain-nvidia-ai-endpoints, onnxruntime, pymupdf, qdrant-client, sentence-transformers

### 05 — SLM Fine-Tuning with Chat History
Fine-tune small language models on ChatGPT conversation history.

- **ChatGPT-to-Alpaca converter** (`src/chatgpt_converter/converter.py`): parses ChatGPT JSON exports (tree structure with parent/children/message nodes) into Alpaca-format DataFrame
- Multi-page Streamlit app: upload, explore/search records, dataset statistics
- Uses **Unsloth** for efficient LoRA fine-tuning
- **Deps:** streamlit, pandas, torch, unsloth>=2026.3.4

### 06 — Claude Code
Tutorial resources for Claude Code. *(stub)*

### 07 — OpenCode Quickstart
Todo list app (Streamlit + SQLite) demonstrating the **OpenCode** AI coding workflow.

- CRUD operations, priority color-coding, progress metrics

---

## 06. Interview Resources

- **LLM Interview Questions** — curated list from GeeksforGeeks and community resources
- **AI Agent Interview Questions** — agentic design pattern Q&A

---

## 07. Books

References and book resources. *(stub)*

---

## 08. llama.cpp

5 notebooks for local LLM inference with `llama-cpp-python`:

| # | Topic | Model |
|---|-------|-------|
| 01 | Setup & basic inference | llama-2-7b-chat.Q4_K_M.gguf |
| 02 | Agent with tools (Tavily search) | llama-2-7b-chat.Q4_K_M.gguf |
| 03 | LangChain integration (Colab) | llama-2-7b-chat.Q4_K_M.gguf |
| 04 | Hugging Face transformers + GGUF | llama-2-7b-chat.Q4_K_M.gguf |
| 05 | Llama 3 with llama.cpp | Meta-Llama-3-8B-Instruct.Q2_K.gguf |

- **Deps:** llama-cpp-python, langchain, langchain-community, huggingface-hub

---

## 09. Ollama / GGUF

Deep-dive into **GGML → GGUF format evolution** and text generation with GGUF models.

Educational: GGML limitations, GGUF advantages (single-file, mmap, extensibility, quantization), metadata structure, tokenizer integration. Colab notebook with practical generation.

---

## 10. Course Resources — DeepLearning.AI

4 notebooks from the **"Generative AI with LLMs"** course:

| Notebook | Topic | Key Libraries |
|----------|-------|---------------|
| `quick_start_flan_T5.ipynb` | FLAN-T5 quick start | transformers, datasets |
| `Lab_1_summarize_dialogue.ipynb` | Zero-shot dialogue summarization | transformers, datasets, GenerationConfig |
| `lab_2_instruction_finetuning_dialogue_summarization.ipynb` | LoRA fine-tuning for summarization | Trainer, PEFT, LoRA, ROUGE eval |
| `lab_3_flan_t5_rlhf.ipynb` | RLHF with FLAN-T5 | TRL, PPO, LoRA, dialogsum dataset |

---

## 11. Google ADK (Agent Development Kit)

Tutorial notebooks for building agents with `google-adk`:

| Notebook | Topic |
|----------|-------|
| `01-adk-agent.ipynb` | "Day Trip Genie" — first agent with Google Search tool, Gemini 2.5 Flash, session management |
| `02-adk-agent-tools.ipynb` | Custom tools — real-time weather API via National Weather Service (no API key), `FunctionTool` pattern |

- **Deps:** google-adk>=1.28.0, google-generativeai, python-dotenv

---

## Utilities

| Script | Description |
|--------|-------------|
| `convert_to_md.py` | Convert `.ipynb` files to Markdown, preserving folder structure |
| `convert_to_pdf.py` | Convert `.ipynb` files to PDF |
| `Makefile` | Git subtree management for deploying sub-projects independently |

---

## Key Technologies

| Area | Technologies |
|------|-------------|
| Frameworks | LangChain, LangGraph, LlamaIndex, Hugging Face Transformers, google-adk |
| Vector DBs | Milvus (Lite + Standalone), Qdrant |
| LLM Providers | Novita AI, NVIDIA API, Google Gemini, Amazon Bedrock, OpenAI-compatible |
| Fine-Tuning | PEFT/LoRA, QLoRA, Unsloth, TRL (RLHF) |
| Inference | llama.cpp, Ollama, GGUF |
| Infrastructure | Docker, Docker Compose, Streamlit, uv |

---

## Quick Reference — Git Submodules

### Add a Submodule
```bash
git submodule add <repository-url> <path>
```

**Example:**
```bash
git submodule add https://github.com/user/repo.git vendor/repo
```

**Commit the changes:**
```bash
git add .gitmodules path/to/submodule
git commit -m "Add submodule: vendor/repo"
```

### Clone Repo with Submodules
```bash
# Fresh clone with all submodules
git clone --recurse-submodules <repository-url>

# Or if already cloned without submodules
git submodule update --init --recursive
```

### Update Submodule to Latest Commit
```bash
cd path/to/submodule
git pull origin main  # or your branch
cd ..
git add path/to/submodule
git commit -m "Update submodule: vendor/repo"
```

### Remove a Submodule
```bash
# 1. Unregister the submodule
git submodule deinit -f path/to/submodule

# 2. Remove from git index
git rm -f path/to/submodule

# 3. Remove the submodule directory entry from .gitmodules
# (edit .gitmodules manually or use --all flag)
git config --file .gitmodules --remove-section submodule.path/to/submodule

# 4. Stage and commit
git add .gitmodules
git commit -m "Remove submodule: vendor/repo"

# 5. Clean up
rm -rf .git/modules/path/to/submodule
```

### Check Submodule Status
```bash
# List all submodules
git config --file .gitmodules --name-only --get-regexp path

# Show submodule commits
git submodule foreach git log --oneline -1

# Show which submodules are dirty
git status
```

### Switch Submodule Branch (Detached HEAD)
By default, submodules are in detached HEAD state. To track a branch:

```bash
cd path/to/submodule
git checkout <branch-name>
cd ..
git add path/to/submodule
git commit -m "Update submodule to track branch: <branch-name>"
```
