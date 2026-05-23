"""Streamlit app for RAG evaluation dataset synthesis"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List
from uuid import uuid4

import pandas as pd
import pymupdf4llm
import streamlit as st
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils import chunk_text

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
if "embed_model_name" not in st.session_state:
    st.session_state.embed_model_name = os.getenv(
        "EMBED_MODEL_NAME", "nvidia/nv-embed-v1"
    )


# ============================================================
# Pydantic Models
# ============================================================
class QAPair(BaseModel):
    """QAPair model for RAG evaluation"""

    id: str = Field(description="Unique identifier")
    query: str = Field(description="User query")
    ai_response: str = Field(description="AI response")
    context: str = Field(description="Context used by AI for response")


# ============================================================
# Helper Functions
# ============================================================
def get_datetime():
    """Get current datetime as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def convert_pdf_to_md(files: List[str], write_images: bool = False) -> dict:
    """Convert PDF files to markdown"""
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
    """Initialize LangChain chat model"""
    return ChatOpenAI(
        model=st.session_state.chat_model_name,
        api_key=st.session_state.api_key,
        base_url=st.session_state.base_url,
    )


def load_env_file(uploaded_file):
    """Load environment variables from uploaded .env file"""
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
                elif key == "EMBED_MODEL_NAME":
                    st.session_state.embed_model_name = value

        return True
    except Exception as e:
        st.error(f"Error loading .env file: {str(e)}")
        return False


# ============================================================
# Streamlit App
# ============================================================
def main():
    st.set_page_config(
        page_title="RAG Evaluation Dataset Synthesizer",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 RAG Evaluation Dataset Synthesizer")
    st.markdown(
        "Generate Q&A pairs from PDF documents for RAG system evaluation using AI."
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # .env File Uploader
        st.subheader("📁 Environment Setup")
        env_file = st.file_uploader(
            "Upload .env file (optional)",
            type="txt",
            help="Upload a .env file with NVIDIA_API_KEY, NVIDIA_BASE_URL, and model names",
        )
        if env_file is not None:
            if load_env_file(env_file):
                st.success("✅ .env file loaded")

        # OpenAI-Compatible API Configuration
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
            help="OpenAI-compatible endpoint URL (e.g., https://integrate.api.nvidia.com/v1)",
        )

        st.session_state.chat_model_name = st.text_input(
            "Chat Model",
            value=st.session_state.chat_model_name,
            help="Model name for chat (e.g., openai/gpt-oss-120b)",
        )

        st.session_state.embed_model_name = st.text_input(
            "Embedding Model",
            value=st.session_state.embed_model_name,
            help="Model name for embeddings (e.g., nvidia/nv-embed-v1)",
        )

        # Validate API configuration
        if not st.session_state.api_key:
            st.warning("⚠️ API Key is required")
        if not st.session_state.base_url:
            st.warning("⚠️ Base URL is required")
        if not st.session_state.chat_model_name:
            st.warning("⚠️ Chat Model name is required")

        # Chunking configuration
        st.divider()
        st.subheader("✂️ Chunking Settings")
        max_tokens = st.slider(
            "Max tokens per chunk",
            min_value=100,
            max_value=2000,
            value=600,
            step=50,
        )
        min_tokens = st.slider(
            "Min tokens per chunk",
            min_value=50,
            max_value=500,
            value=150,
            step=50,
        )

        write_images = st.checkbox("Extract images from PDFs", value=False)

        chunk_config = {
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "encoding_name": "cl100k_base",
            "overlap_tokens": 0,
        }

    # Main content
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

    # Process PDFs
    if uploaded_files:
        st.markdown("---")

        # Save uploaded files temporarily
        temp_dir = Path("temp_pdfs")
        temp_dir.mkdir(exist_ok=True)

        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(str(file_path))

        status_placeholder.info(f"📁 Loaded {len(file_paths)} PDF(s)")

        # Convert PDFs to markdown
        with st.spinner("🔄 Converting PDFs to markdown..."):
            try:
                content = convert_pdf_to_md(file_paths, write_images=write_images)
                status_placeholder.success(f"✅ Converted {len(content)} file(s)")
            except Exception as e:
                st.error(f"Error converting PDFs: {str(e)}")
                st.stop()

        # Chunk text
        with st.spinner("✂️ Chunking text..."):
            chunks = []
            for value in content.values():
                if isinstance(value, str):
                    chunks.extend(chunk_text(value, config=chunk_config))
            status_placeholder.info(f"📦 Created {len(chunks)} chunk(s)")

        st.write(f"**Total chunks:** {len(chunks)}")

        # Display sample chunks
        with st.expander("👀 Preview chunks"):
            sample_size = min(3, len(chunks))
            for i, chunk in enumerate(chunks[:sample_size]):
                st.write(f"**Chunk {i+1}:**")
                st.text_area(
                    f"Content {i+1}",
                    chunk[:500] + "..." if len(chunk) > 500 else chunk,
                    height=100,
                    disabled=True,
                    key=f"chunk_{i}",
                )

        # Generate Q&A pairs
        st.markdown("---")
        col_gen, col_cfg = st.columns([3, 1])

        with col_cfg:
            num_samples = st.number_input(
                "Number of chunks to process",
                min_value=1,
                max_value=len(chunks),
                value=min(5, len(chunks)),
                help="Process first N chunks",
            )

        with col_gen:
            # Check API configuration before allowing generation
            if (
                st.session_state.api_key
                and st.session_state.base_url
                and st.session_state.chat_model_name
            ):
                if st.button("🚀 Generate Q&A Pairs", use_container_width=True):
                    generate_qa_pairs(chunks, num_samples, status_placeholder)
            else:
                st.error("❌ Configure API settings in sidebar before generating")

    else:
        st.info("👆 Upload PDF files to get started")


def generate_qa_pairs(chunks: List[str], num_samples: int, status_placeholder):
    """Generate Q&A pairs from chunks"""

    system_prompt = """
You are an AI assistant specialized in creating evaluation data for retrieval‑augmented generation (RAG) systems.
You will receive a TEXT chunk.
Your task is to create a realistic, standalone question that can be answered **only** using the information in the TEXT.
Then, provide the correct answer to that question.
The question must sound like something a real user would ask (vary the phrasing: factual, comparative, list, definition, yes/no, etc.).
The answer must be entirely grounded in the TEXT – do not add external knowledge.

Output a Pydantic model instance with exactly these fields:
- query: the generated question.
- ai_response: the ground‑truth answer.
- context: the exact TEXT you were given (copy it verbatim, or first 500 chars if very long).

Generate a valid Pydantic model instance following the schema provided.
"""

    try:
        chat_model = initialize_chat_model()
        agent = create_agent(
            model=chat_model,
            system_prompt=system_prompt,
            response_format=QAPair,
        )

        agent_output = []
        progress_bar = st.progress(0)

        for idx, chunk in enumerate(chunks[:num_samples]):
            status_placeholder.info(f"⏳ Processing chunk {idx + 1}/{num_samples}...")

            try:
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": chunk}]}
                )
                agent_output.append(response)
                progress = (idx + 1) / num_samples
                progress_bar.progress(progress)
            except Exception as e:
                st.warning(f"⚠️ Error processing chunk {idx + 1}: {str(e)}")
                continue

        status_placeholder.success(f"✅ Generated {len(agent_output)} Q&A pair(s)")

        # Convert to DataFrame
        if agent_output:
            data = []
            for item in agent_output:
                if "structured_response" in item:
                    # Generate UUID in application code, not from LLM
                    qa_data = item["structured_response"].model_dump()
                    qa_data["id"] = str(uuid4())
                    data.append(qa_data)

            if data:
                df = pd.DataFrame(data)

                # Display results
                st.markdown("---")
                st.subheader("📊 Generated Q&A Pairs")
                st.dataframe(df, use_container_width=True)

                # Download CSV
                csv_filename = f"rag_eval_dataset_{get_datetime()}.csv"
                csv_data = df.to_csv(index=False)

                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True,
                    help="Download to your default Downloads folder",
                )
            else:
                st.error("❌ No valid Q&A pairs generated")
        else:
            st.error("❌ Failed to generate Q&A pairs")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback

        st.text(traceback.format_exc())


if __name__ == "__main__":
    main()
