"""ChatGPT Conversation Converter.

Convert ChatGPT exported JSON files to DataFrame/CSV for LLM fine-tuning.
"""

from .converter import ChatGPTConverter

__all__ = ["ChatGPTConverter"]