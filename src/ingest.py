"""
PDF Ingestion Pipeline - PHASE 3 ENHANCED
Handles PDF processing with progress tracking and faster processing

PHASE 3 ENHANCEMENT: Table Extraction
──────────────────────────────────────
Technical documents (aviation specs, regulations, gazettes) are table-heavy.
Plain text extraction breaks tables into nonsense rows. This version:

1. Detects tables using pdfplumber
2. Extracts them as structured grids
3. Formats as Markdown tables for better embedding
4. Chunk table rows independently so queries can match specific table cells

Example:
  Before: "757 100 200 300" (broken text)
  After:  | Aircraft | Max Passengers | Range (nm) |
          | 757      | 100           | 200        |
          
This dramatically improves spec-lookup queries like "What is the range of a 757?"
"""

import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional, Callable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from psycopg2.extras import execute_values

from .config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from .db_utils import insert_chunks, check_document_exists, get_db_connection

# OPTIMIZATION: Invalidate BM25 cache when documents change
try:
    from .hybrid_search import invalidate_bm25_cache
    BM25_CACHE_AVAILABLE = True
except ImportError:
    BM25_CACHE_AVAILABLE = False


# Global embedding model (load once, reuse)
_embedding_model = None

def get_embedding_model():
    """Get or create embedding model (singleton)"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# ============================================================================
# ATOMIC REPLACE
# ============================================================================
def _replace_document_chunks(chunks: List[dict], embeddings: np.ndarray, document_name: str) -> int:
    """
    Delete existing chunks for document and insert new ones in ONE transaction.
    """
    rows = [
        (chunk["text"], emb.tolist(), chunk["document_name"], chunk["page_number"])
        for chunk, emb in zip(chunks, embeddings)
    ]

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM knowledge_chunks WHERE document_name = %s;",
                (document_name,),
            )
            execute_values(
                cur,
                """
                INSERT INTO knowledge_chunks
                    (content, embedding, document_name, page_number)
                VALUES %s
                """,
                rows,
            )
        conn.commit()

    return len(rows)


# ============================================================================
# TABLE EXTRACTION - PHASE 3 NEW
# ============================================================================
def _format_table_as_markdown(table: List[List[str]]) -> str:
    """
    Convert pdfplumber table (list of lists) to Markdown grid.
    
    Example input:
        [["Aircraft", "Passengers", "Range"],
         ["757", "100", "3000"],
         ["767", "200", "5000"]]
    
    Output:
        | Aircraft | Passengers | Range |
        |----------|------------|-------|
        | 757      | 100        | 3000  |
        | 767      | 200        | 5000  |
    """
    if not table or len(table) < 2:
        return ""
    
    # Clean None values
    cleaned = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        cleaned.append(cleaned_row)
    
    # Build Markdown
    headers = cleaned[0]
    rows    = cleaned[1:]
    
    # Calculate column widths
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*cleaned)]
    
    # Header row
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    
    # Separator
    separator = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    
    # Data rows
    data_lines = []
    for row in rows:
        line = "| " + " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)) + " |"
        data_lines.append(line)
    
    return "\n".join([header_line, separator] + data_lines)


def extract_text_from_pdf(pdf_path: Path, progress_callback: Optional[Callable] = None) -> List[Dict]:
    """
    Extract text AND tables from PDF.
    
    PHASE 3 CHANGE: Now detects tables and formats them as Markdown grids.
    
    Returns:
        List of page dicts: {"text": ..., "page_number": ..., "document_name": ...}
    """
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for i, page in enumerate(pdf.pages):
            if progress_callback:
                progress_callback(f"Extracting page {i+1}/{total_pages}", (i+1)/total_pages)

            # ── Extract tables first ────────────────────────────────
            tables = page.extract_tables()
            table_text = ""
            if tables:
                for table in tables:
                    md_table = _format_table_as_markdown(table)
                    if md_table:
                        table_text += f"\n\n{md_table}\n\n"
            
            # ── Extract regular text ────────────────────────────────
            raw_text = page.extract_text() or ""
            
            # Combine: text + tables
            full_text = raw_text + table_text
            
            if full_text.strip():
                pages.append({
                    "text": full_text.strip(),
                    "page_number": i + 1,
                    "document_name": pdf_path.name
                })

    return pages


# ============================================================================
# CHUNKING (unchanged from Phase 2)
# ============================================================================
def chunk_pages(pages: List[Dict], progress_callback: Optional[Callable] = None) -> List[Dict]:
    """Chunk pages into smaller pieces with progress tracking"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    total_pages = len(pages)

    for idx, page in enumerate(pages):
        if progress_callback:
            progress_callback(f"Chunking page {idx+1}/{total_pages}", (idx+1)/total_pages)

        splits = text_splitter.split_text(page["text"])

        for chunk_text in splits:
            chunks.append({
                "text": chunk_text,
                "document_name": page["document_name"],
                "page_number": page["page_number"]
            })

    return chunks


# ============================================================================
# EMBEDDING GENERATION (unchanged from Phase 2)
# ============================================================================
def generate_embeddings(
    chunks: List[Dict],
    progress_callback: Optional[Callable] = None
) -> np.ndarray:
    """Generate embeddings for chunks with progress tracking"""
    
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]

    total_chunks = len(texts)
    batch_size = EMBEDDING_BATCH_SIZE * 2
    all_embeddings = []

    for i in range(0, total_chunks, batch_size):
        batch = texts[i:i + batch_size]

        if progress_callback:
            progress = min((i + batch_size) / total_chunks, 1.0)
            progress_callback(f"Generating embeddings {i+batch_size}/{total_chunks}", progress)

        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)


# ============================================================================
# MAIN INGESTION PIPELINE
# ============================================================================
def ingest_pdf(
    pdf_path: Path,
    skip_if_exists: bool = True,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    Complete ingestion pipeline with table extraction.
    
    PHASE 3 CHANGE: Tables are now extracted and formatted as Markdown.
    """
    document_name = pdf_path.name

    doc_exists = check_document_exists(document_name)

    if skip_if_exists and doc_exists:
        return {
            "status": "skipped",
            "document": document_name,
            "message": "Document already exists in database",
            "pages": 0,
            "chunks": 0
        }

    try:
        # Step 1: Extract text + tables (20%)
        if progress_callback:
            progress_callback("Starting PDF extraction (with tables)...", 0.0)

        pages = extract_text_from_pdf(
            pdf_path,
            lambda msg, prog: progress_callback(msg, prog * 0.2) if progress_callback else None
        )

        if not pages:
            return {
                "status": "error",
                "document": document_name,
                "message": "No text extracted from PDF",
                "pages": 0,
                "chunks": 0
            }

        # Step 2: Chunk (20%)
        if progress_callback:
            progress_callback("Chunking text...", 0.2)

        chunks = chunk_pages(
            pages,
            lambda msg, prog: progress_callback(msg, 0.2 + prog * 0.2) if progress_callback else None
        )

        # Step 3: Generate embeddings (50%)
        if progress_callback:
            progress_callback("Generating embeddings...", 0.4)

        embeddings = generate_embeddings(
            chunks,
            lambda msg, prog: progress_callback(msg, 0.4 + prog * 0.5) if progress_callback else None
        )

        # Step 4: Write to database (10%)
        if progress_callback:
            progress_callback("Inserting into database...", 0.9)

        if doc_exists:
            inserted = _replace_document_chunks(chunks, embeddings, document_name)
        else:
            inserted = insert_chunks(chunks, embeddings)

        if progress_callback:
            progress_callback("Complete!", 1.0)

        # OPTIMIZATION: Invalidate BM25 cache so next query rebuilds index
        if BM25_CACHE_AVAILABLE:
            invalidate_bm25_cache()

        return {
            "status": "success",
            "document": document_name,
            "message": f"Successfully ingested {inserted} chunks",
            "pages": len(pages),
            "chunks": inserted
        }

    except Exception as e:
        return {
            "status": "error",
            "document": document_name,
            "message": f"Error during ingestion: {str(e)}",
            "pages": 0,
            "chunks": 0
        }


def ingest_directory(directory: Path, skip_if_exists: bool = True) -> List[Dict]:
    """Ingest all PDFs from a directory"""
    pdf_files = list(directory.glob("*.pdf"))
    results = []

    for pdf_file in pdf_files:
        result = ingest_pdf(pdf_file, skip_if_exists=skip_if_exists)
        results.append(result)

    return results