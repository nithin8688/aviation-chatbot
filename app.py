"""
Document QA Chatbot - Streamlit Web Application
A RAG-based chatbot for document Q&A with PDF upload capability

VERSION 1.2.1 - API Usage Display Fix
──────────────────────────────────────
Fixed: API usage counter now updates immediately after each query
Added: Auto-refresh of sidebar stats
"""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime, timedelta
import time
import threading

from src.rag_engine import get_rag_engine
from src.db_utils import get_total_chunks, get_document_stats, delete_document
from src.ingest import ingest_pdf
from src.config import PROJECT_ROOT


# ============================================================================
# IMPROVED RATE LIMITER (with usage tracking)
# ============================================================================
class ImprovedRateLimiter:
    """
    Enhanced rate limiter for Gemini API with dual limits:
    - 15 requests per minute (short-term)
    - 1500 requests per day (long-term)
    
    Shows user-friendly messages and progress during waits.
    """

    def __init__(self, 
                 requests_per_minute: int = 15,
                 requests_per_day: int = 1500):
        self.rpm_limit = requests_per_minute
        self.rpd_limit = requests_per_day
        
        # Short-term tracking (1 minute window)
        self.minute_timestamps = []
        
        # Long-term tracking (24 hour window)
        self.day_timestamps = []
        
        self.lock = threading.Lock()
    
    def acquire(self, show_progress=True):
        """
        Block until a request slot is available.
        Returns (success: bool, wait_time: float, limit_type: str)
        """
        while True:
            with self.lock:
                now = time.time()
                
                # Clean old timestamps
                self.minute_timestamps = [
                    t for t in self.minute_timestamps 
                    if now - t < 60.0
                ]
                
                day_ago = now - (24 * 60 * 60)
                self.day_timestamps = [
                    t for t in self.day_timestamps
                    if t > day_ago
                ]
                
                # Check both limits
                minute_available = len(self.minute_timestamps) < self.rpm_limit
                day_available = len(self.day_timestamps) < self.rpd_limit
                
                if minute_available and day_available:
                    # Slot available!
                    self.minute_timestamps.append(now)
                    self.day_timestamps.append(now)
                    return True, 0, None
                
                # Calculate wait time
                minute_wait = 0
                day_wait = 0
                limit_type = None
                
                if not minute_available:
                    minute_wait = 60.0 - (now - self.minute_timestamps[0])
                    limit_type = "minute"
                
                if not day_available:
                    day_wait = (24 * 60 * 60) - (now - self.day_timestamps[0])
                    limit_type = "day" if day_wait > minute_wait else limit_type
                
                wait_time = max(minute_wait, day_wait)
            
            if show_progress and wait_time > 0:
                return False, wait_time, limit_type
            
            time.sleep(min(wait_time, 1.0))
    
    def get_usage_stats(self):
        """Get current usage statistics"""
        with self.lock:
            now = time.time()
            
            # Clean old timestamps
            self.minute_timestamps = [
                t for t in self.minute_timestamps 
                if now - t < 60.0
            ]
            day_ago = now - (24 * 60 * 60)
            self.day_timestamps = [
                t for t in self.day_timestamps
                if t > day_ago
            ]
            
            return {
                "minute_used": len(self.minute_timestamps),
                "minute_limit": self.rpm_limit,
                "day_used": len(self.day_timestamps),
                "day_limit": self.rpd_limit
            }


# Module-level singleton
_rate_limiter = ImprovedRateLimiter(
    requests_per_minute=15,
    requests_per_day=1500
)


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Document QA Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# THEME-ADAPTIVE CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #3B82F6;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .source-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
        opacity: 0.95;
    }
    
    .stats-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #3B82F6;
        opacity: 0.95;
    }
    
    .source-label {
        color: #3B82F6;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# RAG ENGINE
# ============================================================================
@st.cache_resource
def load_rag_engine():
    """Load once per Streamlit process"""
    return get_rag_engine()


# ============================================================================
# SESSION STATE
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None

if "prev_doc_selection" not in st.session_state:
    st.session_state.prev_doc_selection = None

# Track if we just processed a query (triggers sidebar refresh)
if "query_processed" not in st.session_state:
    st.session_state.query_processed = False


# ============================================================================
# HELPERS
# ============================================================================
def display_message(role, content, sources=None):
    """Display a chat message with optional source cards."""
    with st.chat_message(role):
        st.markdown(content)

        if sources:
            with st.expander("📚 View Sources", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <span class="source-label">Source {i}:</span> {source['document']} (Page {source['page']})<br>
                        <span class="source-label">Relevance:</span> {source['similarity']:.3f}<br>
                        <span class="source-label">Preview:</span> {source['content'][:200]}...
                    </div>
                    """, unsafe_allow_html=True)


def save_chat_history():
    """Persist the current chat to a JSON file on disk."""
    history_file = PROJECT_ROOT / "data" / "chat_history.json"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, indent=2, ensure_ascii=False)


# ============================================================================
# PAGE HEADER
# ============================================================================
st.markdown('<div class="main-header">📚 Document QA Chatbot</div>', unsafe_allow_html=True)
st.markdown("**Ask questions about the documents in your knowledge base**")


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📊 System Status")

    # ── database stats ────────────────────────────────────────────────
    try:
        total_chunks = get_total_chunks()
        doc_stats    = get_document_stats()

        # Get API usage stats (ALWAYS fetch fresh data)
        usage = _rate_limiter.get_usage_stats()

        st.markdown(f"""
        <div class="stats-box">
            <span class="source-label">📚 Knowledge Base</span><br><br>
            Total Chunks: <strong>{total_chunks:,}</strong><br>
            Documents: <strong>{len(doc_stats)}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # API Usage Stats - Shows real-time usage
        st.markdown(f"""
        <div class="stats-box">
            <span class="source-label">🔑 API Usage (Gemini)</span><br><br>
            Per Minute: <strong>{usage['minute_used']}/{usage['minute_limit']}</strong><br>
            Per Day: <strong>{usage['day_used']}/{usage['day_limit']}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Show last update time for debugging
        st.caption(f"📍 Last updated: {datetime.now().strftime('%H:%M:%S')}")

        st.subheader("📄 Documents in Database")
        for doc_name, count in doc_stats:
            st.write(f"• {doc_name}: {count:,} chunks")

    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        st.info("💡 Make sure PostgreSQL is running and secrets are configured")
        doc_stats = []

    st.divider()

    # ── PDF upload ────────────────────────────────────────────────────
    st.header("📤 Upload New PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > 25:
            st.warning(f"⚠️ Large file ({file_size_mb:.1f}MB)! May exceed memory limits on free tier. Consider splitting into smaller files.")
        
        if st.button("🚀 Process PDF", type="primary"):
            upload_dir = PROJECT_ROOT / "data" / "uploaded_pdfs"
            upload_dir.mkdir(parents=True, exist_ok=True)

            pdf_path = upload_dir / uploaded_file.name
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            progress_bar  = st.progress(0)
            status_text   = st.empty()

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
                st.rerun()
            elif result["status"] == "skipped":
                st.warning(f"⚠️ {result['message']}")
            else:
                st.error(f"❌ {result['message']}")

    st.divider()

    # ── document management ───────────────────────────────────────────
    st.header("🗑️ Document Management")
    if doc_stats:
        doc_options  = [doc[0] for doc in doc_stats]
        doc_to_delete = st.selectbox("Select document to delete", options=doc_options)

        if doc_to_delete != st.session_state.prev_doc_selection:
            st.session_state.confirm_delete  = None
            st.session_state.prev_doc_selection = doc_to_delete

        if st.button("🗑️ Delete Document", type="secondary"):
            if st.session_state.confirm_delete == doc_to_delete:
                with st.spinner(f"Deleting {doc_to_delete}..."):
                    deleted = delete_document(doc_to_delete)
                st.success(f"✅ Deleted {deleted} chunks from {doc_to_delete}")
                st.session_state.confirm_delete     = None
                st.session_state.prev_doc_selection = None
                st.rerun()
            else:
                st.session_state.confirm_delete = doc_to_delete
                st.warning("⚠️ Click again to confirm deletion")

    st.divider()

    # ── settings ──────────────────────────────────────────────────────
    st.header("⚙️ Settings")
    top_k = st.slider("Number of sources to retrieve", min_value=3, max_value=15, value=5)

    st.info("💡 **Tips:**\n- More sources = more detailed\n- 5-8 sources recommended")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("💾 Save Chat History"):
        save_chat_history()
        st.success("✅ Chat history saved!")


# ============================================================================
# MAIN CHAT AREA
# ============================================================================
st.divider()

# Replay saved messages
for message in st.session_state.messages:
    display_message(
        message["role"],
        message["content"],
        message.get("sources")
    )

# Input box
if prompt := st.chat_input("Ask a question about your documents..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_message("user", prompt)

    with st.chat_message("assistant"):
        
        # ── RATE LIMIT CHECK ───────────────────────────────────────
        success, wait_time, limit_type = _rate_limiter.acquire(show_progress=True)
        
        if not success and wait_time > 0:
            if limit_type == "minute":
                st.warning(f"⏳ Rate limit reached: 15 requests/minute. Waiting {wait_time:.0f} seconds...")
            else:
                st.error(f"🚫 Daily limit reached: 1500 requests/day. Please try again tomorrow.")
            
            # Progress bar
            if wait_time <= 120:
                progress_bar = st.progress(0)
                status = st.empty()
                
                for i in range(int(wait_time)):
                    remaining = wait_time - i
                    progress = i / wait_time
                    progress_bar.progress(progress)
                    status.text(f"⏳ Waiting... {remaining:.0f}s remaining")
                    time.sleep(1)
                
                progress_bar.empty()
                status.empty()
                st.rerun()  # Refresh to retry
            else:
                st.stop()
        
        # ── QUERY EXECUTION ─────────────────────────────────────────
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()

            try:
                rag = load_rag_engine()
                response = rag.query(prompt, top_k=top_k)
                elapsed_time = time.time() - start_time

                # Display answer
                answer = response["answer"]
                st.markdown(answer)

                st.caption(f"⏱️ Response time: {elapsed_time:.2f}s | 📚 Sources: {response['num_sources']}")

                # Source cards
                if response["sources"]:
                    with st.expander("📚 View Sources", expanded=False):
                        for i, source in enumerate(response["sources"], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <span class="source-label">Source {i}:</span> {source['document']} (Page {source['page']})<br>
                                <span class="source-label">Relevance:</span> {source['similarity']:.3f}<br>
                                <span class="source-label">Preview:</span> {source['content'][:200]}...
                            </div>
                            """, unsafe_allow_html=True)

                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": response["sources"]
                })
                
                # Mark that we processed a query (triggers sidebar refresh)
                st.session_state.query_processed = True
                
                # Force sidebar refresh by rerunning
                st.rerun()

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style="text-align: center; opacity: 0.6; font-size: 0.9rem; padding: 1rem;">
    Document QA Chatbot v1.2.1 | Powered by PostgreSQL + pgvector + Gemini AI
</div>
""", unsafe_allow_html=True)