"""Utility functions"""

import re
from typing import List, Dict
from pydantic import BaseModel, Field
import mistune

# ----------------------------
# 1. Parse Markdown into heading–content sections
# ----------------------------
def _extract_text(token):
    if 'raw' in token:
        return token['raw']
    if 'text' in token:
        return token['text']
    if 'children' in token:
        return ''.join(_extract_text(c) for c in token['children'])
    return ''


def parse_markdown_to_sections(md_text: str) -> List[Dict[str, str]]:
    """
    Parses a Markdown string into a list of sections, each containing:
        - 'title': the nearest heading (e.g., '## Introduction')
        - 'content': the raw text under that heading (including sub‑content)
    """
    ast = mistune.create_markdown(renderer='ast')
    tokens = ast(md_text)

    sections = []
    current_heading = "Untitled"
    current_content_lines = []

    for token in tokens:
        if token['type'] == 'heading':
            if current_content_lines:
                sections.append({
                    'title': current_heading,
                    'content': '\n'.join(current_content_lines).strip()
                })
            current_heading = _extract_text(token).strip()
            current_content_lines = []
        elif token['type'] == 'paragraph':
            text = _extract_text(token)
            if text:
                current_content_lines.append(text)

    if current_content_lines:
        sections.append({
            'title': current_heading,
            'content': '\n'.join(current_content_lines).strip()
        })

    return sections


# ----------------------------
# 2. Word‑count helper
# ----------------------------
def word_count(text: str) -> int:
    """Counts words in a text (simple whitespace split)."""
    return len(text.split())


# ----------------------------
# 3. Merge short sections
# ----------------------------
def merge_short_sections(sections: List[Dict], min_words: int = 100) -> List[Dict]:
    """
    Merges a section with the next one if its word count is below min_words.
    This avoids having tiny, context‑less chunks.
    """
    if not sections:
        return []

    merged = []
    buffer = None  # will hold a section that is too short

    for sec in sections:
        wc = word_count(sec['content'])
        if wc < min_words:
            # Too short – hold it to merge with the next
            if buffer is None:
                buffer = sec
            else:
                # Already have a short section, merge current into buffer
                buffer['content'] += '\n\n' + sec['content']
        else:
            # Current section is long enough
            if buffer is not None:
                # Merge the short buffer into this long section
                buffer['content'] += '\n\n' + sec['content']
                buffer['title'] += ' | ' + sec['title']  # keep both titles
                merged.append(buffer)
                buffer = None
            else:
                merged.append(sec)

    # Append any remaining buffer
    if buffer is not None:
        merged.append(buffer)

    return merged


# ----------------------------
# 4. Split long sections
# ----------------------------
def split_long_sections(
    sections: List[Dict],
    max_words: int = 1000,
    sub_heading_pattern: str = r'^#{1,6}\s'  # markdown heading lines
) -> List[Dict]:
    """
    Splits sections that exceed max_words.
    - First tries to split at any Markdown sub‑headings.
    - If none, falls back to splitting at double‑line‑breaks (paragraph gaps).
    """
    result = []

    for sec in sections:
        wc = word_count(sec['content'])
        if wc <= max_words:
            result.append(sec)
            continue

        # -- Split at sub‑headings --
        # Look for lines that start with one or more '#' and a space
        sub_parts = re.split(r'\n(?=^#{1,6}\s)', sec['content'], flags=re.MULTILINE)
        if len(sub_parts) > 1:
            # Each sub‑part may have its own mini‑heading line
            for part in sub_parts:
                # Extract the first line as title if it looks like a heading
                lines = part.strip().split('\n')
                part_title = sec['title']
                part_content = part.strip()
                if lines and re.match(sub_heading_pattern, lines[0]):
                    part_title += ' > ' + lines[0].lstrip('#').strip()
                    part_content = '\n'.join(lines[1:]).strip()
                if word_count(part_content) > 0:
                    result.append({
                        'title': part_title,
                        'content': part_content
                    })
        else:
            # No sub‑headings – split by double newlines (paragraph breaks)
            paragraphs = re.split(r'\n\s*\n', sec['content'])
            for i, para in enumerate(paragraphs):
                if para.strip():
                    result.append({
                        'title': f"{sec['title']} (part {i+1})",
                        'content': para.strip()
                    })

    return result


# ----------------------------
# 6. Pipeline orchestrator
# ----------------------------
def chunk_markdown_by_topic(
    md_text: str,
    min_words: int = 100,
    max_words: int = 1000
) -> List[Dict[str, str]]:
    """
    Full Stage 1 pipeline:
    1. Parse Markdown into sections.
    2. Merge too‑short sections.
    3. Split too‑long sections.
    Returns a list of topic blocks with 'title' and 'content'.
    """
    sections = parse_markdown_to_sections(md_text)
    merged = merge_short_sections(sections, min_words=min_words)
    split = split_long_sections(merged, max_words=max_words)
    return split


# ----------------------------
# 5. Atomic Fact Extraction (Stage 2)
# ----------------------------
class AtomicFact(BaseModel):
    fact: str = Field(
        description="A single self-contained, declarative factual statement"
    )


class AtomicFacts(BaseModel):
    facts: list[AtomicFact] = Field(
        description="List of atomic facts extracted from the text chunk"
    )


SYSTEM_PROMPT_EXTRACT_FACTS = """\
You are a meticulous fact extractor for building a high-quality training dataset.

Read the text and extract **meaningful, substantive facts** that represent the core domain knowledge. Focus on:

- Definitions, mechanisms, processes, categorizations, comparisons, causal relationships, and named techniques/concepts explicitly stated.
- Actionable information such as what a concept is, how it works, when to use it, what its properties are, or how it differs from another concept.

RULES:
1. Each fact must be a single, self-contained declarative sentence.
2. State the fact directly — NEVER use meta-framing like "The text says...", "The author states...", "The article mentions...", "According to...".
3. Skip trivial/metadata: dates, author names, reading time, picture dimensions, page structure, layout information, navigation elements.
4. Skip redundant restatements of the same idea. If two sentences convey the same core fact, keep only one.
5. Do not add, infer, or speculate beyond what is explicitly stated.
6. Extract only facts that would be useful as ground-truth context for answering a question about this topic."""


# ----------------------------
# 6. Alpaca Q&A models and prompts (Stage 3)
# ----------------------------
class AlpacaQAPair(BaseModel):
    instruction: str = Field(
        description="The question / user instruction"
    )
    input: str = Field(
        description="Optional context (empty string for single-turn)"
    )
    output: str = Field(
        description="The answer grounded in the provided facts"
    )


class AlpacaQAPairs(BaseModel):
    pairs: list[AlpacaQAPair] = Field(
        description="List of diverse instruction-response pairs"
    )


SYSTEM_PROMPT_GENERATE_QA = """\
You are an expert at creating diverse, high-quality instruction data.

Given a list of atomic facts, generate a set of diverse instruction-response pairs.

Requirements:
- Each question must be answerable entirely from the provided facts — do not use external knowledge.
- Vary question types across the set: include "what", "how", "why", "compare/contrast", "list/enumerate", "explain", "define", and "true/false" or "yes/no" questions.
- Vary difficulty: some questions should be simple fact retrieval (one fact), others should require synthesizing 2-3 facts together.
- Each answer must be **200–300 words long** and explain the concept thoroughly: define it, provide context, include supporting details, and connect related facts — all strictly from the provided facts.
- Do not pad with filler or repetition. Every sentence should add substantive information.
- State answers directly without meta-framing like "Based on the facts...".
- Do NOT generate questions about the text itself (e.g. "What does the article say about X?"). Generate questions a real user would ask.
- Output each pair with instruction (the question), input (empty string), and output (the answer)."""