"""
Document QA Chatbot - Streamlit Web Application
A RAG-based chatbot for document Q&A with PDF upload capability

WHAT CHANGED vs the previous version
─────────────────────────────────────
1. RAG engine cached at process level
   • Replaced st.session_state.rag_engine with @st.cache_resource.
   • The 200 MB embedding model loads once when the first user connects;
     every subsequent browser tab reuses the same instance.

2. Server-side rate limiter  (RateLimiter class)
   • The old approach was time.sleep(4) inside the browser session — two tabs
     could fire simultaneously and both hit Gemini's 15-req/min cap.
   • A threading.Lock-backed token bucket runs at the Streamlit-process level.
     acquire() blocks any caller that would exceed the limit, regardless of
     which browser tab the request came from.

3. Delete-confirmation bug fixed
   • confirm_delete is now reset to None every time the user changes the
     document in the selectbox, so every document always gets the two-click
     warning before deletion.

4. CSS consolidated
   • Three separate st.markdown(<style>…</style>) blocks merged into one
     injection at the top of the page.
"""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import time
import threading

from src.rag_engine import get_rag_engine
from src.db_utils import get_total_chunks, get_document_stats, delete_document
from src.ingest import ingest_pdf
from src.config import PROJECT_ROOT


# ============================================================================
# SERVER-SIDE RATE LIMITER
# ============================================================================
class RateLimiter:
    """
    Simple token-bucket rate limiter that is safe across Streamlit threads.

    • max_requests / per_seconds  →  e.g. 15 requests per 60 seconds
    • acquire() blocks the calling thread until a slot is available.
    • Used once, right before every Gemini API call, so no tab can sneak
      past the limit.
    """

    def __init__(self, max_requests: int = 15, per_seconds: float = 60.0):
        self.max_requests = max_requests
        self.per_seconds  = per_seconds
        self.timestamps   = []          # wall-clock times of recent requests
        self.lock         = threading.Lock()

    def acquire(self):
        """Block until a request slot is available, then claim it."""
        while True:
            with self.lock:
                now = time.time()
                # discard timestamps older than the window
                self.timestamps = [t for t in self.timestamps if now - t < self.per_seconds]

                if len(self.timestamps) < self.max_requests:
                    self.timestamps.append(now)
                    return                          # slot acquired — go ahead

                # window is full; figure out how long until the oldest slot expires
                wait = self.per_seconds - (now - self.timestamps[0])

            # sleep OUTSIDE the lock so other threads aren't blocked while waiting
            time.sleep(wait)

# Module-level singleton — shared by every Streamlit tab in this process
_rate_limiter = RateLimiter(max_requests=15, per_seconds=60.0)


# ============================================================================
# PAGE CONFIG  +  SINGLE CSS BLOCK
# ============================================================================
st.set_page_config(
    page_title="Document QA Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# All styles in ONE injection — easier to maintain and debug
st.markdown("""
<style>
    /* ── overall dark theme ──────────────────────────────── */
    .main {
        background-color: #0E1117;
    }
    .stMarkdown, .stText, p, span, div {
        color: #FAFAFA !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }

    /* ── custom components ───────────────────────────────── */
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

    /* ── chat messages ───────────────────────────────────── */
    .stChatMessage {
        background-color: #1F2937;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stMarkdown {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# RAG ENGINE  —  cached at process level (not per browser session)
# ============================================================================
@st.cache_resource
def load_rag_engine():
    """Load once per Streamlit process; every tab shares this instance."""
    return get_rag_engine()


# ============================================================================
# SESSION STATE  —  only per-user chat state lives here now
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None

# Track which document was selected last time so we can detect a change
if "prev_doc_selection" not in st.session_state:
    st.session_state.prev_doc_selection = None


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
                        <strong style="color: #60A5FA;">Source {i}:</strong>
                        <span style="color: #F9FAFB;">{source['document']} (Page {source['page']})</span><br>
                        <strong style="color: #60A5FA;">Relevance:</strong>
                        <span style="color: #F9FAFB;">{source['similarity']:.3f}</span><br>
                        <strong style="color: #60A5FA;">Preview:</strong>
                        <span style="color: #D1D5DB;">{source['content'][:200]}...</span>
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
        doc_stats = []              # so the rest of the sidebar doesn't crash

    st.divider()

    # ── PDF upload ────────────────────────────────────────────────────
    st.header("📤 Upload New PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
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

        # ── BUG FIX: reset confirm_delete whenever the selection changes ──
        # Before: switching from doc A to doc B kept confirm_delete = doc A,
        #         so the very next click on "Delete" would confirm doc B
        #         instantly — no warning shown.
        if doc_to_delete != st.session_state.prev_doc_selection:
            st.session_state.confirm_delete  = None
            st.session_state.prev_doc_selection = doc_to_delete

        if st.button("🗑️ Delete Document", type="secondary"):
            if st.session_state.confirm_delete == doc_to_delete:
                # second click — confirmed
                with st.spinner(f"Deleting {doc_to_delete}..."):
                    deleted = delete_document(doc_to_delete)
                st.success(f"✅ Deleted {deleted} chunks from {doc_to_delete}")
                st.session_state.confirm_delete     = None
                st.session_state.prev_doc_selection = None
                st.rerun()
            else:
                # first click — ask for confirmation
                st.session_state.confirm_delete = doc_to_delete
                st.warning("⚠️ Click again to confirm deletion")

    st.divider()

    # ── settings ──────────────────────────────────────────────────────
    st.header("⚙️ Settings")
    top_k = st.slider("Number of sources to retrieve", min_value=3, max_value=15, value=8)

    st.info("💡 **Tips for Better Answers:**\n- More sources = more detailed answers\n- Technical docs work best with 8-10 sources")
    st.info("💡 **Rate Limit Info:**\nFree tier: 15 requests/min\nServer-side limiter is active")

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

# replay saved messages
for message in st.session_state.messages:
    display_message(
        message["role"],
        message["content"],
        message.get("sources")
    )

# input box
if prompt := st.chat_input("Ask a question about your documents..."):

    # add user message immediately so it renders before the spinner
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_message("user", prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()

            try:
                # ── SERVER-SIDE RATE LIMIT ─────────────────────────
                # acquire() blocks THIS thread (not a sleep in the browser)
                # so concurrent tabs are serialised at the process level.
                _rate_limiter.acquire()

                # get the process-level cached engine
                rag = load_rag_engine()

                response = rag.query(prompt, top_k=top_k)

                elapsed_time = time.time() - start_time

                # display answer
                answer = response["answer"]
                st.markdown(answer)

                st.caption(f"⏱️ Response time: {elapsed_time:.2f}s | 📚 Sources: {response['num_sources']}")

                # source cards
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

                # save to chat history
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


# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #9CA3AF; font-size: 0.9rem; padding: 1rem;">
    Document QA Chatbot v1.1 | Powered by PostgreSQL + pgvector + Gemini AI
</div>
""", unsafe_allow_html=True)