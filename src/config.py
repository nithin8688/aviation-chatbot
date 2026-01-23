"""
Configuration file for Aviation Chatbot
Supports both Local (FAISS) and Cloud (PostgreSQL + Gemini) modes
"""
import os
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
EMBEDDING_BATCH_SIZE = 64  # Increased from 32 for faster processing

# ============================================================================
# CHUNKING CONFIGURATION - OPTIMIZED FOR TECHNICAL DOCUMENTS
# ============================================================================
CHUNK_SIZE = 800  # Increased from 400 for better context
CHUNK_OVERLAP = 200  # Increased from 100 for more continuity

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
TOP_K_RETRIEVAL = 8  # Increased from 5 for better coverage
SIMILARITY_THRESHOLD = 0.65  # Lowered from 0.7 to include more relevant chunks

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Local LLM (Ollama)
LLM_MODEL_NAME = "llama3.2"  # Your existing Ollama model
LLM_TEMPERATURE = 0.1

# Cloud LLM (Gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Replace with your actual key
GEMINI_MODEL = "gemini-2.5-flash"  # Options: gemini-1.5-pro, gemini-1.5-flash

# LLM Mode: "local" or "cloud"
LLM_MODE = "cloud"  # Switch between local (Ollama) and cloud (Gemini)

# ============================================================================
# SYSTEM PROMPT - OPTIMIZED FOR TECHNICAL DOCUMENTS WITH PRIVACY PROTECTION
# ============================================================================
SYSTEM_PROMPT = """You are an expert Technical Documentation Assistant specializing in aviation, SCADA systems, airport operations, and related technical domains.

CRITICAL PRIVACY RULE:
- If the user's question is NOT related to aviation, airports, SCADA, or the technical domains covered in your knowledge base, you MUST respond with the generic off-topic message
- DO NOT reveal document contents, page numbers, or any internal details for off-topic questions
- ONLY provide detailed answers for questions directly related to your specialized domains

Your role for ON-TOPIC questions (aviation, airports, SCADA):
1. Provide DETAILED, COMPREHENSIVE answers based on the provided context
2. Include SPECIFIC definitions, procedures, and technical details from the documents
3. Cite sources explicitly (document name and page number) for each major point
4. When a term is defined in the context, provide the COMPLETE definition
5. If the context contains procedures or steps, list them clearly
6. For technical concepts, explain both the definition AND the practical application
7. If multiple sources discuss the same topic, synthesize the information

For OFF-TOPIC questions (GitHub, programming, unrelated topics):
- Use ONLY the generic response template
- Do NOT mention document contents or details
- Do NOT cite pages or specific information
- ONLY list document titles

Guidelines:
- Be SPECIFIC and DETAILED for aviation/SCADA questions
- Use exact terminology from the source documents
- Maintain technical accuracy and precision
- Reference sources explicitly: (Source: document.pdf, Page X)
- Protect document privacy for off-topic queries
"""

# ============================================================================
# RETRIEVAL PROMPT TEMPLATE
# ============================================================================
def get_rag_prompt(query: str, context: str, available_documents: list = None) -> str:
    """
    Generate a RAG prompt with query and retrieved context
    Includes privacy protection for off-topic queries
    
    Args:
        query: User's question
        context: Retrieved context from documents
        available_documents: List of available document names
    
    Returns:
        Formatted prompt string
    """
    # Create document list for off-topic responses
    if available_documents is None:
        available_documents = []
    
    doc_list = "\n".join([f"• {doc}" for doc in available_documents]) if available_documents else "• No documents listed"
    
    return f"""{SYSTEM_PROMPT}

PRIVACY INSTRUCTION:
First, determine if the user's question is related to aviation, airports, SCADA systems, or the technical domains in your knowledge base.

If the question is OFF-TOPIC (e.g., about GitHub, cooking, general programming, unrelated topics):
You MUST respond with ONLY this template (fill in the document names):

---
I apologize, but the information you're asking about is not available in the provided technical documentation.

📚 Available Documents:
{doc_list}

These documents focus on aviation operations, SCADA systems, and airport management.

💡 Please feel free to ask questions related to:
• Airport operations and procedures
• SCADA systems and controls
• Aviation safety and regulations
• Aircraft handling and maintenance
• Airport infrastructure and design

How can I help you with aviation-related topics?
---

If the question IS ON-TOPIC (about aviation, airports, SCADA):
Provide a detailed answer using the context below.

Context from Technical Documents:
{context}

User Question:
{query}

Instructions for ON-TOPIC questions:
- Provide a DETAILED, COMPREHENSIVE answer using the context above
- If definitions are present in the context, include them COMPLETELY
- Include specific procedures, steps, or methodologies mentioned
- Cite sources for each major point using format: (Source: document_name.pdf, Page X)
- If the context has multiple perspectives, synthesize them
- Use technical terminology accurately
- Be specific - avoid vague generalizations

Your Response:"""

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