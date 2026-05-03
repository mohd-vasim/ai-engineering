"""Upload and Convert ChatGPT JSON page."""

import streamlit as st
import pandas as pd
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatgpt_converter import ChatGPTConverter


def main():
    st.header("📤 Upload ChatGPT Export")
    st.markdown("Upload your ChatGPT JSON export file to convert it to a DataFrame.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a JSON file",
        type=['json'],
        help="Upload the ChatGPT conversation export file (conversations.json)"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            import tempfile
            import json
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                # Write uploaded content to temp file
                content = uploaded_file.getvalue()
                tmp.write(content.decode('utf-8'))
                tmp_path = tmp.name
            
            # Convert using our class
            converter = ChatGPTConverter()
            df = converter.convert(tmp_path)
            
            # Store in session state
            st.session_state['converter'] = converter
            st.session_state['df'] = df
            st.session_state['file_uploaded'] = True
            
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
            st.success(f"✅ Successfully converted! Found {len(df)} conversation pairs.")
            
            # Show statistics
            st.subheader("📊 Dataset Statistics")
            stats = converter.get_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Pairs", stats['total_pairs'])
            with col2:
                st.metric("Avg Instruction Length", f"{stats['avg_instruction_length']:.0f} chars")
            with col3:
                st.metric("Avg Output Length", f"{stats['avg_output_length']:.0f} chars")
            with col4:
                st.metric("Max Output Length", f"{stats['max_output_length']:.0f} chars")
            
            # Show DataFrame
            st.subheader("📋 Converted Data")
            
            # Show row count selector
            rows_to_show = st.slider("Rows to display", min_value=5, max_value=len(df), value=min(10, len(df)))
            
            # Display dataframe
            st.dataframe(
                df.head(rows_to_show),
                use_container_width=True,
                height=400
            )
            
            # Show column info
            st.subheader("📌 Columns")
            st.write(f"Columns: {list(df.columns)}")
            
            # Show session grouping info
            st.subheader("🔗 Session Grouping")
            sessions_per_conv = df.groupby('conversation_id')['session_id'].count()
            st.write(f"Unique conversations: {df['conversation_id'].nunique()}")
            st.write(f"Total sessions: {len(df)}")
            
            # Download buttons
            st.subheader("💾 Download")
            
            col1, col2 = st.columns(2)
            
            # CSV download
            csv = df.to_csv(index=False)
            col1.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="chatgpt_conversations.csv",
                mime="text/csv"
            )
            
            # JSON download (Alpaca format)
            json_data = df.to_json(orient='records', indent=2)
            col2.download_button(
                label="📥 Download as JSON (Alpaca)",
                data=json_data,
                file_name="alpaca_dataset.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Make sure your JSON file is in the correct ChatGPT export format.")
    
    else:
        st.info("👆 Please upload a ChatGPT JSON export file to get started.")
        
        # Show example format
        with st.expander("ℹ️ Expected JSON format"):
            st.markdown("""
            The app expects a ChatGPT conversation export in JSON format with this structure:
            
            ```json
            [
              {
                "id": "conversation-id",
                "conversation_id": "conversation-id", 
                "mapping": {
                  "node-id-1": {
                    "id": "node-id-1",
                    "parent": "parent-id",
                    "children": ["child-id-1"],
                    "message": {
                      "author": {"role": "user"},
                      "content": {
                        "content_type": "text",
                        "parts": ["message text"]
                      }
                    }
                  }
                }
              }
            ]
            ```
            """)


if __name__ == "__main__":
    main()