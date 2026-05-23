"""Utils"""

import re
from typing import List, Dict, Any
import tiktoken

# ------------------------------------------------------------
# Configuration defaults
# ------------------------------------------------------------
DEFAULT_CONFIG = {
    "max_tokens": 600,
    "min_tokens": 150,
    "encoding_name": "cl100k_base",   # GPT-3.5/4
    "overlap_tokens": 0,              # not implemented yet
}

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def get_tokenizer(encoding_name: str):
    return tiktoken.get_encoding(encoding_name)

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))

# ------------------------------------------------------------
# Paragraph helpers
# ------------------------------------------------------------
def extract_paragraphs(text: str) -> List[str]:
    """Split raw text into non-empty paragraphs (separated by blank lines)."""
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

# ------------------------------------------------------------
# Oversized paragraph splitting (by sentence)
# ------------------------------------------------------------
def split_by_sentences(text: str) -> List[str]:
    """Naive sentence split: break on .!? followed by space and capital letter."""
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

def split_oversized_paragraph(paragraph: str, max_tokens: int, tokenizer) -> List[str]:
    """
    If a paragraph exceeds max_tokens, split it into sub-chunks that each
    fit within max_tokens, respecting sentence boundaries where possible.
    """
    sentences = split_by_sentences(paragraph)
    chunks = []
    current = ""
    current_len = 0

    for sent in sentences:
        sent_len = count_tokens(sent, tokenizer)
        # Sentence itself is too long → forced token-level chopping
        if sent_len > max_tokens:
            if current:
                chunks.append(current)
                current = ""
                current_len = 0
            # Truncate by token count
            tokens = tokenizer.encode(sent)
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i:i+max_tokens]
                chunks.append(tokenizer.decode(chunk_tokens))
            continue

        if current_len + sent_len <= max_tokens:
            current = (current + " " + sent).strip() if current else sent
            current_len += sent_len
        else:
            chunks.append(current)
            current = sent
            current_len = sent_len

    if current:
        chunks.append(current)
    return chunks

# ------------------------------------------------------------
# Core chunking: greedy merge of paragraphs
# ------------------------------------------------------------
def greedy_merge_paragraphs(paragraphs: List[str], max_tokens: int, tokenizer) -> List[str]:
    """Merge paragraphs into chunks that respect max_tokens, splitting oversized ones."""
    chunks = []
    current_lines = []
    current_len = 0

    for para in paragraphs:
        para_len = count_tokens(para, tokenizer)

        # Oversized paragraph → split it
        if para_len > max_tokens:
            # flush current chunk
            if current_lines:
                chunks.append(" ".join(current_lines))
                current_lines = []
                current_len = 0
            # add the sub-chunks from this paragraph
            sub_chunks = split_oversized_paragraph(para, max_tokens, tokenizer)
            chunks.extend(sub_chunks)
            continue

        # Normal paragraph: try to add to current chunk
        if current_len + para_len <= max_tokens:
            current_lines.append(para)
            current_len += para_len
        else:
            # finish current chunk and start new one
            chunks.append(" ".join(current_lines))
            current_lines = [para]
            current_len = para_len

    # Don't forget the last chunk
    if current_lines:
        chunks.append(" ".join(current_lines))

    return chunks

# ------------------------------------------------------------
# Post-processing: merge small trailing chunks
# ------------------------------------------------------------
def merge_small_chunks(chunks: List[str], min_tokens: int, max_tokens: int, tokenizer) -> List[str]:
    """
    Merge chunks that are smaller than min_tokens into the previous chunk
    if the combined size does not exceed max_tokens.
    """
    if not chunks:
        return []
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        chunk_len = count_tokens(chunk, tokenizer)
        if chunk_len < min_tokens:
            prev_len = count_tokens(merged[-1], tokenizer)
            if prev_len + chunk_len <= max_tokens:
                merged[-1] = merged[-1] + " " + chunk
                continue
        merged.append(chunk)
    return merged

# ------------------------------------------------------------
# Main entry point: config-driven
# ------------------------------------------------------------
def chunk_text(text: str, config: Dict[str, Any] = None) -> List[str]:
    """
    Split raw text into chunks using the provided configuration.

    Args:
        text: raw text from document.
        config: dict with keys: max_tokens, min_tokens, encoding_name, overlap_tokens.
                Defaults to DEFAULT_CONFIG.

    Returns:
        list of chunk strings.
    """
    if config is None:
        config = DEFAULT_CONFIG

    max_tokens = config["max_tokens"]
    min_tokens = config["min_tokens"]
    encoding_name = config["encoding_name"]
    # overlap not implemented yet

    tokenizer = get_tokenizer(encoding_name)
    paragraphs = extract_paragraphs(text)

    # Step 1: greedy merging into chunks that fit max_tokens
    raw_chunks = greedy_merge_paragraphs(paragraphs, max_tokens, tokenizer)

    # Step 2: merge small trailing chunks where possible
    final_chunks = merge_small_chunks(raw_chunks, min_tokens, max_tokens, tokenizer)

    return final_chunks