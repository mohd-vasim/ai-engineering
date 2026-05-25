"""Streamlit app for LLM training dataset synthesis"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import pymupdf4llm
import streamlit as st
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    chunk_markdown_by_topic,
    AtomicFacts,
    AtomicFact,
    AlpacaQAPairs,
    AlpacaQAPair,
    SYSTEM_PROMPT_EXTRACT_FACTS,
    SYSTEM_PROMPT_GENERATE_QA,
)

st.set_page_config(
    page_title="LLM Training Dataset Synthesizer",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# Initialize Session State
# ============================================================
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("NVIDIA_API_KEY", "")
if "base_url" not in st.session_state:
    st.session_state.base_url = os.getenv(
        "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
if "chat_model_name" not in st.session_state:
    st.session_state.chat_model_name = os.getenv(
        "CHAT_MODEL_NAME", "openai/gpt-oss-120b"
    )
if "md_texts" not in st.session_state:
    st.session_state.md_texts = {}
if "topic_wise" not in st.session_state:
    st.session_state.topic_wise = []
if "golden_facts" not in st.session_state:
    st.session_state.golden_facts = []
if "all_pairs" not in st.session_state:
    st.session_state.all_pairs = []


# ============================================================
# Helper Functions
# ============================================================
def get_datetime():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def convert_pdf_to_md(files: List[str], write_images: bool = False) -> dict:
    content = {}
    for file in files:
        filename = os.path.basename(file)
        img_path = f"data/images/{filename}" if write_images else None
        try:
            content[filename] = pymupdf4llm.to_markdown(
                file, image_path=img_path, write_images=write_images
            )
            if write_images and img_path and os.path.exists(img_path):
                content[img_path] = [
                    os.path.join(img_path, image_path)
                    for image_path in os.listdir(img_path)
                ]
        except Exception as e:
            st.error(f"Error converting {filename}: {str(e)}")
    return content


def initialize_chat_model():
    return ChatOpenAI(
        model=st.session_state.chat_model_name,
        api_key=st.session_state.api_key,
        base_url=st.session_state.base_url,
    )


def load_env_file(uploaded_file):
    try:
        content = uploaded_file.read().decode("utf-8")
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key == "NVIDIA_API_KEY" or key == "API_KEY":
                    st.session_state.api_key = value
                elif key == "NVIDIA_BASE_URL" or key == "BASE_URL":
                    st.session_state.base_url = value
                elif key == "CHAT_MODEL_NAME":
                    st.session_state.chat_model_name = value
        return True
    except Exception as e:
        st.error(f"Error loading .env file: {str(e)}")
        return False


# ============================================================
# Main App
# ============================================================
def main():
    st.title("📊 LLM Training Dataset Synthesizer")
    st.markdown(
        "Generate instruction‑response pairs from PDF documents for LLM fine‑tuning. "
        "Pipeline: PDF → Markdown → Topic Chunks → Atomic Facts → Q&A Pairs."
    )

    # ----------------------------------------------------------
    # Sidebar Configuration
    # ----------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("📁 Environment Setup")
        env_file = st.file_uploader(
            "Upload .env file (optional)",
            type="txt",
            help="Upload a .env file with NVIDIA_API_KEY, NVIDIA_BASE_URL, and model names",
        )
        if env_file is not None:
            if load_env_file(env_file):
                st.success("✅ .env file loaded")

        st.subheader("🔌 OpenAI-Compatible API")
        st.caption("Configure your OpenAI-compatible endpoint")

        st.session_state.api_key = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            help="Your API key (e.g., NVIDIA_API_KEY)",
        )
        st.session_state.base_url = st.text_input(
            "Base URL",
            value=st.session_state.base_url,
            help="OpenAI-compatible endpoint URL",
        )
        st.session_state.chat_model_name = st.text_input(
            "Chat Model",
            value=st.session_state.chat_model_name,
            help="Model name for chat (e.g., openai/gpt-oss-120b)",
        )

        if not st.session_state.api_key:
            st.warning("⚠️ API Key is required")
        if not st.session_state.base_url:
            st.warning("⚠️ Base URL is required")
        if not st.session_state.chat_model_name:
            st.warning("⚠️ Chat Model name is required")

        st.divider()
        st.subheader("✂️ Chunking Settings")
        min_words = st.slider(
            "Min words per chunk",
            min_value=20,
            max_value=500,
            value=100,
            step=10,
            help="Sections below this word count get merged with the next section",
        )
        max_words = st.slider(
            "Max words per chunk",
            min_value=200,
            max_value=3000,
            value=1000,
            step=100,
            help="Sections above this word count get split",
        )

        write_images = st.checkbox("Extract images from PDFs", value=False)

    # ----------------------------------------------------------
    # Main Content
    # ----------------------------------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📄 Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Select PDF files to process",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files",
        )

    with col2:
        st.subheader("📋 Processing Status")
        status_placeholder = st.empty()

    if uploaded_files:
        st.markdown("---")

        temp_dir = Path("temp_pdfs")
        temp_dir.mkdir(exist_ok=True)

        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(str(file_path))

        status_placeholder.info(f"📁 Loaded {len(file_paths)} PDF(s)")

        if st.button("🔄 Step 1: Convert PDFs → Topic Chunks", use_container_width=True):
            with st.spinner("🔄 Converting PDFs to markdown..."):
                try:
                    st.session_state.md_texts = convert_pdf_to_md(
                        file_paths, write_images=write_images
                    )
                    status_placeholder.success(
                        f"✅ Converted {len(st.session_state.md_texts)} file(s)"
                    )
                except Exception as e:
                    st.error(f"Error converting PDFs: {str(e)}")
                    st.stop()

            with st.spinner("✂️ Chunking into topics..."):
                topic_wise = []
                for text in st.session_state.md_texts.values():
                    if isinstance(text, str):
                        topic_wise.extend(
                            chunk_markdown_by_topic(
                                text, min_words=min_words, max_words=max_words
                            )
                        )
                st.session_state.topic_wise = topic_wise
                status_placeholder.info(
                    f"📦 Created {len(topic_wise)} topic chunk(s)"
                )

        # Show chunks once they exist
        if st.session_state.topic_wise:
            st.write(f"**Total chunks:** {len(st.session_state.topic_wise)}")

            with st.expander("👀 Preview chunks"):
                for i, chunk in enumerate(
                    st.session_state.topic_wise[:5]
                ):
                    st.write(f"**Chunk {i+1}:** {chunk['title']}")
                    st.text_area(
                        f"Content {i+1}",
                        chunk["content"][:500]
                        + ("..." if len(chunk["content"]) > 500 else ""),
                        height=100,
                        disabled=True,
                        key=f"chunk_{i}",
                    )

            st.markdown("---")

            # Stage 2: Extract Atomic Facts
            api_ok = (
                st.session_state.api_key
                and st.session_state.base_url
                and st.session_state.chat_model_name
            )

            col_facts, col_facts_cfg = st.columns([3, 1])
            with col_facts_cfg:
                num_chunks = st.number_input(
                    "Chunks to process",
                    min_value=1,
                    max_value=len(st.session_state.topic_wise),
                    value=min(5, len(st.session_state.topic_wise)),
                    help="Process first N chunks through the pipeline",
                )

            with col_facts:
                if api_ok:
                    if st.button(
                        "🔍 Step 2: Extract Atomic Facts", use_container_width=True
                    ):
                        run_fact_extraction(num_chunks, status_placeholder)
                else:
                    st.error(
                        "❌ Configure API settings in sidebar before generating"
                    )

        # Stage 3: Generate Q&A Pairs
        if st.session_state.golden_facts:
            st.markdown("---")
            if api_ok:
                if st.button(
                    "🚀 Step 3: Generate Q&A Pairs", use_container_width=True
                ):
                    run_qa_generation(
                        st.session_state.golden_facts, status_placeholder
                    )

        # Results: Preview + Download
        if st.session_state.all_pairs:
            st.markdown("---")
            df = pd.DataFrame(st.session_state.all_pairs)

            st.subheader("📊 Generated Dataset")
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False)
            json_data = df.to_json(orient="records", indent=2)
            ts = get_datetime()

            col_csv, col_json = st.columns(2)
            with col_csv:
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"synthetic_dataset_{ts}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_json:
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"synthetic_dataset_{ts}.json",
                    mime="application/json",
                    use_container_width=True,
                )
    else:
        st.info("👆 Upload PDF files to get started")


# ============================================================
# Pipeline Stages
# ============================================================
def run_fact_extraction(num_chunks: int, status_placeholder):
    chat_model = initialize_chat_model()
    chain = chat_model.with_structured_output(AtomicFacts)

    golden_facts = []
    chunks = st.session_state.topic_wise[:num_chunks]
    progress_bar = st.progress(0)

    for idx, chunk in enumerate(chunks):
        status_placeholder.info(
            f"⏳ Extracting facts from chunk {idx + 1}/{num_chunks}..."
        )
        try:
            response = chain.invoke([
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_EXTRACT_FACTS,
                },
                {
                    "role": "user",
                    "content": f"Title: {chunk['title']}\n\nContent:\n{chunk['content']}",
                },
            ])
            facts = [f.fact for f in response.facts]
            golden_facts.append(facts)
        except Exception as e:
            st.warning(f"⚠️ Error on chunk {idx + 1}: {str(e)}")
            golden_facts.append([])

        progress = (idx + 1) / num_chunks
        progress_bar.progress(progress)

    st.session_state.golden_facts = golden_facts
    total = sum(len(f) for f in golden_facts)
    status_placeholder.success(f"✅ Extracted {total} atomic facts from {num_chunks} chunk(s)")


def run_qa_generation(golden_facts: List[List[str]], status_placeholder):
    chat_model = initialize_chat_model()
    chain = chat_model.with_structured_output(AlpacaQAPairs)

    all_pairs = []
    num_chunks = len(golden_facts)
    progress_bar = st.progress(0)

    for idx, chunk_facts in enumerate(golden_facts):
        if not chunk_facts:
            continue
        status_placeholder.info(
            f"⏳ Generating Q&A from chunk {idx + 1}/{num_chunks}..."
        )
        try:
            facts_text = "\n".join(f"- {f}" for f in chunk_facts)
            response = chain.invoke([
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_GENERATE_QA,
                },
                {
                    "role": "user",
                    "content": f"Atomic facts:\n\n{facts_text}",
                },
            ])
            all_pairs.extend(p.model_dump() for p in response.pairs)
        except Exception as e:
            st.warning(f"⚠️ Error on chunk {idx + 1}: {str(e)}")

        progress = (idx + 1) / num_chunks
        progress_bar.progress(progress)

    st.session_state.all_pairs = all_pairs
    status_placeholder.success(f"✅ Generated {len(all_pairs)} Q&A pair(s)")


if __name__ == "__main__":
    main()
