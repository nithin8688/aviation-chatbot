"""
Configuration file for Document QA Chatbot - PRODUCTION OPTIMIZED
PERFORMANCE TARGET: 3-5 second responses
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"

PAGES_PATH = DATA_DIR / "pages.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"

# ============================================================================
# .env LOADER
# ============================================================================
def _load_dotenv():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)

_load_dotenv()

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME",     "aviation_chatbot")
DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if DB_PASSWORD is None:
    raise RuntimeError(
        "\n❌  DB_PASSWORD is not set.\n"
        "    1. Copy .env.example  →  .env\n"
        "    2. Set DB_PASSWORD inside it\n"
    )

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ============================================================================
# EMBEDDING MODEL
# ============================================================================
EMBEDDING_MODEL_NAME  = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION   = 384
EMBEDDING_BATCH_SIZE  = 64

# ============================================================================
# CHUNKING
# ============================================================================
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 200

# ============================================================================
# RETRIEVAL - PRODUCTION OPTIMIZED FOR SPEED
# ============================================================================
# CRITICAL: Reduced from 8 to 5 for faster Gemini API responses
TOP_K_RETRIEVAL      = 5         # ← OPTIMIZED: Reduced from 8 (saves 1-2s)
SIMILARITY_THRESHOLD = 0.65

# SPEED OPTIMIZATIONS - Disable slow features
USE_HYBRID_SEARCH = True         # ✅ ENABLED - Fast with caching
HYBRID_ALPHA      = 0.6          # 60% vector, 40% BM25 (balanced)

USE_HYDE          = False        # ❌ DISABLED - Saves ~2 seconds + 1 API call
HYDE_NUM_HYPOTHESES = 1

USE_RERANKING     = False        # ❌ DISABLED - Saves ~500ms
RERANK_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K      = 20

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
LLM_MODEL_NAME  = "llama3.2"
LLM_TEMPERATURE = 0.1

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise RuntimeError(
        "\n❌  GEMINI_API_KEY is not set.\n"
        "    1. Copy .env.example  →  .env\n"
        "    2. Set GEMINI_API_KEY inside it\n"
    )

GEMINI_MODEL = "gemini-2.5-flash"
LLM_MODE     = "cloud"

# ============================================================================
# SYSTEM PROMPT - OPTIMIZED FOR ACCURACY & FLEXIBILITY
# ============================================================================
SYSTEM_PROMPT = """You are an expert Technical Documentation Assistant. You answer questions based on documents in your knowledge base.

YOUR ROLE:
1. Answer questions using the provided context from documents
2. Cite sources explicitly: (Source: document.pdf, Page X)
3. If context is insufficient, say so clearly
4. For questions ABOUT a document itself (e.g., "what is the gazette of india?"), explain what you know from the chunks provided

IMPORTANT RULES:
- If you have relevant context chunks, USE THEM to answer
- Don't refuse to answer just because chunks don't contain a perfect definition
- If asked about a document itself and you have chunks from it, describe what the document contains
- Always cite your sources with page numbers
- Be specific and detailed

Guidelines:
- Be SPECIFIC and DETAILED for questions about document content
- Use exact terminology from the source documents
- Maintain technical accuracy and precision
- Reference sources explicitly: (Source: document.pdf, Page X)
- Protect document privacy for off-topic queries

FEW-SHOT EXAMPLES:

Example 1 - Document Meta-Question:
Q: What is the Gazette of India?
A: Based on the document "THE GAZETTE OF INDIA EXTRAORDINARY.pdf" in the knowledge base, the Gazette of India is the official government publication where notifications, regulations, and official announcements are published (Source: THE GAZETTE OF INDIA EXTRAORDINARY.pdf, Page 3). It contains legal notifications and government orders.

Example 2 - Technical Definition:
Q: What is an Instrument Landing System (ILS)?
A: An Instrument Landing System (ILS) is a ground-based radio navigation system that provides precision guidance to aircraft approaching and landing on a runway, especially in low visibility conditions (Source: airport_operations.pdf, Page 142).

The ILS consists of two main components:
1. Localizer - provides lateral (left/right) guidance
2. Glideslope - provides vertical (up/down) guidance

The system allows pilots to land safely even when visibility is as low as 200 feet (Source: airport_operations.pdf, Page 143).

Example 3 - Procedural List:
Q: What are the steps for aircraft pre-flight inspection?
A: According to the maintenance procedures, aircraft pre-flight inspection follows these steps (Source: scada_manual.pdf, Page 87):

1. Walk-around inspection - Visual check of fuselage, wings, landing gear for damage
2. Fluid level checks - Oil, hydraulic fluid, fuel quantity verification
3. Control surface movement - Verify ailerons, rudder, elevators move freely
4. Tire inspection - Check for proper inflation and tread wear
5. Documentation review - Confirm all required maintenance logs are current

Each step must be completed and signed off by certified maintenance personnel before the aircraft is cleared for departure (Source: scada_manual.pdf, Page 88).
"""

# ============================================================================
# RETRIEVAL PROMPT TEMPLATE
# ============================================================================
def get_rag_prompt(query: str, context: str, available_documents: list = None) -> str:
    """Generate RAG prompt with query and context"""
    if available_documents is None:
        available_documents = []

    doc_list = "\n".join([f"• {doc}" for doc in available_documents]) if available_documents else "• No documents listed"

    return f"""{SYSTEM_PROMPT}

Available Documents in Knowledge Base:
{doc_list}

Context from Documents:
{context}

User Question:
{query}

INSTRUCTIONS:
- Use the context above to answer the question
- If the context is relevant but incomplete, answer with what you have and note what's missing
- Always cite sources: (Source: document.pdf, Page X)
- Be detailed and specific
- If the user is asking ABOUT a document itself (not content inside it), describe what the document contains based on the chunks

Your Response:"""

# ============================================================================
# TABLE SCHEMA
# ============================================================================
KNOWLEDGE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    document_name VARCHAR(255) NOT NULL,
    page_number INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS embedding_idx ON knowledge_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS document_name_idx ON knowledge_chunks (document_name);
CREATE INDEX IF NOT EXISTS page_number_idx ON knowledge_chunks (page_number);
"""