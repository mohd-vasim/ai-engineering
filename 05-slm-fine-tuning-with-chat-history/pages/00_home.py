"""Home page - Welcome screen."""

import streamlit as st


def main():
    st.header("🏠 Home")
    
    st.markdown("""
    ## Welcome to ChatGPT to Alpaca Converter!
    
    This app helps you convert ChatGPT conversation exports to **Alpaca format** 
    for fine-tuning Small Language Models (SLMs).
    
    ### How it works:
    
    1. **Upload** - Upload your ChatGPT JSON export file
    2. **Convert** - The app parses the conversation tree and extracts Q&A pairs
    3. **Download** - Export as CSV or JSON (Alpaca format)
    
    ### What is Alpaca Format?
    
    Alpaca is a popular instruction-following dataset format:
    
    ```json
    {
      "instruction": "The user's question or prompt",
      "input": "",  // Optional context
      "output": "The expected response"
    }
    ```
    
    ### Get Started
    
    Go to the **Upload** page in the sidebar to convert your ChatGPT data!
    
    ---
    
    ### Need to export your ChatGPT data?
    
    1. Go to ChatGPT
    2. Click on your profile → Settings → Data controls
    3. Click "Export" to download your conversation data
    """)


if __name__ == "__main__":
    main()