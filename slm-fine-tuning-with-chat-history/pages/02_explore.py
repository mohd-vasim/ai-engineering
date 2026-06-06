"""Data Explorer page - View and analyze converted data."""

import streamlit as st
import pandas as pd


def main():
    st.header("🔍 Data Explorer")
    
    # Check if data exists in session state
    if 'df' not in st.session_state or st.session_state.get('df') is None:
        st.info("No data loaded. Please upload a ChatGPT JSON file first.")
        if st.button("Go to Upload Page"):
            st.switch_page("pages/01_upload.py")
        return
    
    df = st.session_state['df']
    
    st.markdown(f"**Total records:** {len(df)}")
    
    # View mode selection
    view_mode = st.radio(
        "View Mode",
        ["📋 All Records", "💬 By Conversation"],
        horizontal=True
    )
    
    if view_mode == "📋 All Records":
        _show_all_records(df)
    else:
        _show_by_conversation(df)


def _show_all_records(df: pd.DataFrame):
    """Show all records in a table with search."""
    
    # Search functionality
    search_term = st.text_input("🔍 Search in instructions or outputs", placeholder="Type to search...")
    
    if search_term:
        mask = df['instruction'].str.contains(search_term, case=False, na=False) | \
               df['output'].str.contains(search_term, case=False, na=False)
        filtered_df = df[mask]
        st.write(f"Found {len(filtered_df)} matching records")
    else:
        filtered_df = df
    
    # Show dataframe
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )
    
    # View individual record
    st.subheader("📄 View Record Details")
    
    if len(filtered_df) > 0:
        # Create display options
        options = []
        for idx, row in filtered_df.iterrows():
            preview = row['instruction'][:60] + "..." if len(row['instruction']) > 60 else row['instruction']
            options.append(f"Session {row['session_id']}: {preview}")
        
        record_idx = st.selectbox(
            "Select a record to view",
            range(len(filtered_df)),
            format_func=lambda x: options[x]
        )
        
        record = filtered_df.iloc[record_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💬 Instruction")
            st.text_area("", record['instruction'], height=250, key="inst_view", disabled=True)
        
        with col2:
            st.markdown("### 🤖 Output")
            st.text_area("", record['output'], height=250, key="out_view", disabled=True)
        
        # Show metadata
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.caption(f"**Conversation ID:**\n{record['conversation_id']}")
        with col_info2:
            st.caption(f"**Session ID:** {record['session_id']}")
        with col_info3:
            st.caption(f"**Instruction length:** {len(record['instruction'])} chars")


def _show_by_conversation(df: pd.DataFrame):
    """Show conversations grouped by conversation_id with full chat UI."""
    
    # Get unique conversations
    conversations = df['conversation_id'].unique()
    
    # Create display options for conversations
    conv_options = []
    for conv_id in conversations:
        conv_df = df[df['conversation_id'] == conv_id]
        # Get first instruction as preview
        first_inst = conv_df.iloc[0]['instruction']
        preview = first_inst[:50] + "..." if len(first_inst) > 50 else first_inst
        conv_options.append({
            'id': conv_id,
            'preview': preview,
            'sessions': len(conv_df)
        })
    
    # Conversation selector
    st.subheader("💬 Select a Conversation")
    
    selected_conv_idx = st.selectbox(
        "Choose conversation",
        range(len(conv_options)),
        format_func=lambda x: f"Conversation {x+1}: {conv_options[x]['preview']} ({conv_options[x]['sessions']} messages)"
    )
    
    selected_conv_id = conv_options[selected_conv_idx]['id']
    conv_df = df[df['conversation_id'] == selected_conv_id].sort_values('session_id')
    
    st.markdown("---")
    st.markdown(f"### Conversation: `{selected_conv_id}`")
    st.markdown(f"**{len(conv_df)} messages in this conversation**")
    
    # Show full conversation in chat format
    for idx, row in conv_df.iterrows():
        session_num = row['session_id']
        
        # User message (instruction)
        with st.container():
            st.markdown("💬 **You:**")
            st.info(row['instruction'], icon="💬")
        
        # Assistant message (output)
        with st.container():
            st.markdown("🤖 **Assistant:**")
            st.success(row['output'], icon="🤖")
        
        st.caption(f"Session ID: {session_num}")
        st.markdown("---")
    
    # Statistics for this conversation
    st.subheader("📊 Conversation Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", len(conv_df))
    with col2:
        st.metric("Avg Instruction Length", f"{conv_df['instruction'].str.len().mean():.0f} chars")
    with col3:
        st.metric("Avg Output Length", f"{conv_df['output'].str.len().mean():.0f} chars")


if __name__ == "__main__":
    main()