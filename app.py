"""
Aviation Chatbot - Streamlit Web Application
A RAG-based chatbot for aviation documentation with PDF upload capability
"""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import time

from src.rag_engine import get_rag_engine
from src.db_utils import get_total_chunks, get_document_stats, delete_document
from src.ingest import ingest_pdf
from src.config import PROJECT_ROOT


# Page configuration
st.set_page_config(
    page_title="Aviation Chatbot",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set theme to dark mode for better contrast
st.markdown("""
<style>
    /* Overall app styling */
    .main {
        background-color: #0E1117;
    }
    /* Ensure text is visible */
    .stMarkdown, .stText, p, span, div {
        color: #FAFAFA !important;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for better visibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E40AF;
        text-align: center;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #1F2937;
        color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
    }
    .stats-box {
        background-color: #1F2937;
        color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #374151;
    }
    /* Fix chat message styling */
    .stChatMessage {
        background-color: #1F2937;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    /* Make sure text is visible */
    .stMarkdown {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    with st.spinner("🔄 Initializing RAG Engine..."):
        st.session_state.rag_engine = get_rag_engine()

if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0


def display_message(role, content, sources=None):
    """Display a chat message with optional sources"""
    with st.chat_message(role):
        st.markdown(content)
        
        if sources:
            with st.expander("📚 View Sources", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong style="color: #60A5FA;">Source {i}:</strong> 
                        <span style="color: #F9FAFB;">{source['document']} (Page {source['page']})</span><br>
                        <strong style="color: #60A5FA;">Relevance:</strong> 
                        <span style="color: #F9FAFB;">{source['similarity']:.3f}</span><br>
                        <strong style="color: #60A5FA;">Preview:</strong> 
                        <span style="color: #D1D5DB;">{source['content'][:200]}...</span>
                    </div>
                    """, unsafe_allow_html=True)


def save_chat_history():
    """Save chat history to file"""
    history_file = PROJECT_ROOT / "data" / "chat_history.json"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, indent=2, ensure_ascii=False)


# Main header
st.markdown('<div class="main-header">✈️ Aviation Chatbot</div>', unsafe_allow_html=True)
st.markdown("**Ask questions about airport operations, SCADA systems, and aviation regulations**")

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    # Database statistics
    try:
        total_chunks = get_total_chunks()
        doc_stats = get_document_stats()
        
        st.markdown(f"""
        <div class="stats-box">
            <strong style="color: #60A5FA;">📚 Knowledge Base</strong><br><br>
            <span style="color: #F9FAFB;">Total Chunks: <strong>{total_chunks:,}</strong></span><br>
            <span style="color: #F9FAFB;">Documents: <strong>{len(doc_stats)}</strong></span>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📄 Documents in Database")
        for doc_name, count in doc_stats:
            st.write(f"• {doc_name}: {count:,} chunks")
    
    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        st.info("💡 Make sure PostgreSQL Docker container is running:\n`docker start aviation-postgres`")
    
    st.divider()
    
    # PDF Upload Section
    st.header("📤 Upload New PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("🚀 Process PDF", type="primary"):
            # Save uploaded file
            upload_dir = PROJECT_ROOT / "data" / "uploaded_pdfs"
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            pdf_path = upload_dir / uploaded_file.name
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Ingest the PDF with progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(message, progress):
                status_text.text(message)
                progress_bar.progress(progress)
            
            result = ingest_pdf(pdf_path, skip_if_exists=False, progress_callback=update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            if result["status"] == "success":
                st.success(f"✅ {result['message']}")
                st.info(f"📄 Pages: {result['pages']} | Chunks: {result['chunks']}")
                time.sleep(1)
                st.rerun()  # Refresh to show new document
            elif result["status"] == "skipped":
                st.warning(f"⚠️ {result['message']}")
            else:
                st.error(f"❌ {result['message']}")
    
    st.divider()
    
    # Document Management
    st.header("🗑️ Document Management")
    if doc_stats:
        doc_to_delete = st.selectbox(
            "Select document to delete",
            options=[doc[0] for doc in doc_stats]
        )
        
        if st.button("🗑️ Delete Document", type="secondary"):
            if st.session_state.get("confirm_delete") == doc_to_delete:
                with st.spinner(f"Deleting {doc_to_delete}..."):
                    deleted = delete_document(doc_to_delete)
                st.success(f"✅ Deleted {deleted} chunks from {doc_to_delete}")
                st.session_state.confirm_delete = None
                st.rerun()
            else:
                st.session_state.confirm_delete = doc_to_delete
                st.warning("⚠️ Click again to confirm deletion")
    
    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    top_k = st.slider("Number of sources to retrieve", min_value=3, max_value=15, value=8)
    
    st.info("💡 **Tips for Better Answers:**\n- More sources = more detailed answers\n- Technical docs work best with 8-10 sources")
    
    st.info("💡 **Rate Limit Info:**\nFree tier: 15 requests/min\nWait 4s between queries")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("💾 Save Chat History"):
        save_chat_history()
        st.success("✅ Chat history saved!")


# Main chat interface
st.divider()

# Display chat history
for message in st.session_state.messages:
    display_message(
        message["role"],
        message["content"],
        message.get("sources")
    )

# Chat input
if prompt := st.chat_input("Ask a question about aviation..."):
    # Check rate limiting (4 seconds between requests = max 15/min)
    current_time = time.time()
    time_since_last_query = current_time - st.session_state.last_query_time
    
    if time_since_last_query < 4:
        wait_time = 4 - time_since_last_query
        st.warning(f"⏳ Please wait {wait_time:.1f} more seconds to avoid rate limits...")
        time.sleep(wait_time)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_message("user", prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            
            try:
                # Get response from RAG engine
                response = st.session_state.rag_engine.query(prompt, top_k=top_k)
                
                # Update last query time
                st.session_state.last_query_time = time.time()
                
                elapsed_time = time.time() - start_time
                
                # Display answer
                answer = response["answer"]
                st.markdown(answer)
                
                # Display performance info
                st.caption(f"⏱️ Response time: {elapsed_time:.2f}s | 📚 Sources: {response['num_sources']}")
                
                # Display sources
                if response["sources"]:
                    with st.expander("📚 View Sources", expanded=False):
                        for i, source in enumerate(response["sources"], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong style="color: #60A5FA;">Source {i}:</strong> 
                                <span style="color: #F9FAFB;">{source['document']} (Page {source['page']})</span><br>
                                <strong style="color: #60A5FA;">Relevance:</strong> 
                                <span style="color: #F9FAFB;">{source['similarity']:.3f}</span><br>
                                <strong style="color: #60A5FA;">Preview:</strong> 
                                <span style="color: #D1D5DB;">{source['content'][:200]}...</span>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": response["sources"]
                })
            
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #9CA3AF; font-size: 0.9rem; padding: 1rem;">
    Aviation Chatbot v1.0 | Powered by PostgreSQL + pgvector + Gemini AI
</div>
""", unsafe_allow_html=True)