"""
Configuration file for Aviation Chatbot
Supports both Local (FAISS) and Cloud (PostgreSQL + Gemini) modes
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"

# Legacy file-based storage (for backward compatibility)
PAGES_PATH = DATA_DIR / "pages.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"

# ============================================================================
# DATABASE CONFIGURATION (PostgreSQL + pgvector)
# ============================================================================
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "aviation_chatbot"
DB_USER = "postgres"
DB_PASSWORD = "aviation123"

# Connection string for SQLAlchemy
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE = 32

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
TOP_K_RETRIEVAL = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Local LLM (Ollama)
LLM_MODEL_NAME = "llama3.2"  # Your existing Ollama model
LLM_TEMPERATURE = 0.1

# Cloud LLM (Gemini)
GEMINI_API_KEY = "AIzaSyDrk-9RnExxmQ2ba2BWupNSrRBskCfXm40"  # Replace with your actual key
GEMINI_MODEL = "gemini-2.5-flash"  # Options: gemini-1.5-pro, gemini-1.5-flash

# LLM Mode: "local" or "cloud"
LLM_MODE = "cloud"  # Switch between local (Ollama) and cloud (Gemini)

# ============================================================================
# SYSTEM PROMPT
# ============================================================================
SYSTEM_PROMPT = """You are an expert Aviation Assistant with deep knowledge of airport operations, SCADA systems, and aviation safety regulations.

Your role is to:
1. Answer questions accurately based on the provided context from official aviation documents
2. Cite specific sources (document name and page number) when providing information
3. If the context doesn't contain enough information, clearly state this limitation
4. Use technical aviation terminology appropriately
5. Prioritize safety and regulatory compliance in your responses

Guidelines:
- Be precise and factual
- Reference the source documents explicitly
- If uncertain, acknowledge the uncertainty rather than speculating
- Break down complex technical concepts when helpful
- Maintain a professional but accessible tone
"""

# ============================================================================
# RETRIEVAL PROMPT TEMPLATE
# ============================================================================
def get_rag_prompt(query: str, context: str) -> str:
    """
    Generate a RAG prompt with query and retrieved context
    
    Args:
        query: User's question
        context: Retrieved context from documents
    
    Returns:
        Formatted prompt string
    """
    return f"""{SYSTEM_PROMPT}

Context from Aviation Documents:
{context}

User Question:
{query}

Instructions:
- Base your answer strictly on the provided context
- Cite sources using format: (Source: document_name.pdf, Page X)
- If the context doesn't fully answer the question, acknowledge this
- Provide a clear, structured response

Answer:"""

# ============================================================================
# TABLE SCHEMA (for PostgreSQL)
# ============================================================================
KNOWLEDGE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),  -- Matches EMBEDDING_DIMENSION
    document_name VARCHAR(255) NOT NULL,
    page_number INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB  -- For future extensibility (tags, categories, etc.)
);

-- Create index for faster vector similarity search
CREATE INDEX IF NOT EXISTS embedding_idx ON knowledge_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for text search (optional, for hybrid search)
CREATE INDEX IF NOT EXISTS document_name_idx ON knowledge_chunks (document_name);
CREATE INDEX IF NOT EXISTS page_number_idx ON knowledge_chunks (page_number);
"""