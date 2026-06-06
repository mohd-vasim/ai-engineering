"""Streamlit app for ChatGPT to Alpaca converter."""

import streamlit as st

# Page config
st.set_page_config(
    page_title="ChatGPT to Alpaca Converter",
    page_icon="💬",
    layout="wide"
)

st.title("💬 ChatGPT to Alpaca Dataset Converter")
st.markdown("""
Convert your ChatGPT conversation exports to **Alpaca format** for LLM fine-tuning.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")