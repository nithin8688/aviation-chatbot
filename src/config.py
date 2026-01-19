from pathlib import Path

# =====================================================
# Project Paths (ON-PREM ONLY)
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"

PAGES_PATH = DATA_DIR / "pages.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"

# =====================================================
# Models (LOCAL / ON-PREM)
# =====================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# LLM_MODEL_NAME = "llama3:latest"
# LLM_MODEL_NAME = "phi3:mini"
LLM_MODEL_NAME = "tinyllama"
EMBEDDING_BATCH_SIZE = 32

# =====================================================
# Retrieval Configuration
# =====================================================

# Number of chunks to retrieve
DEFAULT_TOP_K = 3

# Similarity threshold for accepting retrieved chunks
SIMILARITY_THRESHOLD = 0.45

# Enable hybrid retrieval (BM25 + FAISS)
ENABLE_HYBRID_RETRIEVAL = True

# BM25-specific parameters
BM25_TOP_K = 3

# =====================================================
# Context & Prompt Control
# =====================================================

# Maximum characters passed to LLM as context
MAX_CONTEXT_CHARS = 500

# =====================================================
# Confidence Scoring Configuration
# =====================================================

# Enable confidence scoring for responses
ENABLE_CONFIDENCE_SCORING = True

# Minimum confidence required to trust an answer
MIN_CONFIDENCE_THRESHOLD = 0.6

# =====================================================
# Response Caching Configuration
# =====================================================

# Enable response caching at bot level
ENABLE_RESPONSE_CACHING = True

# Maximum number of cached responses
CACHE_MAX_SIZE = 500

# =====================================================
# System Prompt (Aviation-Specific, Explainable)
# =====================================================

SYSTEM_PROMPT = """
You are an aviation-domain assistant designed for airport operations and aviation-related systems.

Use the provided aviation or SCADA context as the authoritative reference.

Your task is to explain concepts clearly and in simple, professional terms, as an experienced aviation or systems engineer would.
When answering a question:
- Explain what the concept is
- Explain how it works
- Explain how it is relevant to aviation or airport operations, where applicable

You MAY:
- Rephrase the content
- Summarize information
- Explain concepts in your own words for better understanding

You MUST:
- Base your explanation strictly on the provided context
- NOT introduce facts that contradict the context
- NOT invent details that are not supported by the documents

If the provided context does not contain enough information to explain the concept properly, say clearly:
"I don't have sufficient information in the provided aviation documents to explain this."

Do NOT hallucinate.
Do NOT answer outside the aviation or SCADA domains.
Maintain a professional, factual, and safety-conscious tone.
"""
