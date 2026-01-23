"""
PDF Ingestion Pipeline for Aviation Chatbot - OPTIMIZED VERSION
Handles PDF processing with progress tracking and faster processing
"""

import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional, Callable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

from .config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from .db_utils import insert_chunks, check_document_exists


# Global embedding model (load once, reuse)
_embedding_model = None

def get_embedding_model():
    """Get or create embedding model (singleton)"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def extract_text_from_pdf(pdf_path: Path, progress_callback: Optional[Callable] = None) -> List[Dict]:
    """
    Extract text from PDF page by page with progress tracking
    
    Args:
        pdf_path: Path to PDF file
        progress_callback: Optional function to call with progress updates
    
    Returns:
        List of page dictionaries with text, page_number, document_name
    """
    pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        for i, page in enumerate(pdf.pages):
            if progress_callback:
                progress_callback(f"Extracting page {i+1}/{total_pages}", (i+1)/total_pages)
            
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "text": text.strip(),
                    "page_number": i + 1,
                    "document_name": pdf_path.name
                })
    
    return pages


def chunk_pages(pages: List[Dict], progress_callback: Optional[Callable] = None) -> List[Dict]:
    """
    Chunk pages into smaller pieces with progress tracking
    
    Args:
        pages: List of page dictionaries
        progress_callback: Optional function to call with progress updates
    
    Returns:
        List of chunk dictionaries
    """
    # Optimized chunking parameters for better context
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


def generate_embeddings(
    chunks: List[Dict], 
    progress_callback: Optional[Callable] = None
) -> np.ndarray:
    """
    Generate embeddings for chunks with progress tracking
    
    Args:
        chunks: List of chunk dictionaries
        progress_callback: Optional function to call with progress updates
    
    Returns:
        Numpy array of embeddings
    """
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    
    # Process in batches with progress tracking
    total_chunks = len(texts)
    batch_size = EMBEDDING_BATCH_SIZE * 2  # Doubled for faster processing
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


def ingest_pdf(
    pdf_path: Path, 
    skip_if_exists: bool = True,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    Complete ingestion pipeline for a single PDF with progress tracking
    
    Args:
        pdf_path: Path to PDF file
        skip_if_exists: If True, skip if document already exists in DB
        progress_callback: Function to call with (message, progress) updates
    
    Returns:
        Dictionary with ingestion statistics
    """
    document_name = pdf_path.name
    
    # Check if document already exists
    if skip_if_exists and check_document_exists(document_name):
        return {
            "status": "skipped",
            "document": document_name,
            "message": "Document already exists in database",
            "pages": 0,
            "chunks": 0
        }
    
    try:
        # Step 1: Extract text (20% of progress)
        if progress_callback:
            progress_callback("Starting PDF extraction...", 0.0)
        
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
        
        # Step 2: Chunk (20% of progress)
        if progress_callback:
            progress_callback("Chunking text...", 0.2)
        
        chunks = chunk_pages(
            pages,
            lambda msg, prog: progress_callback(msg, 0.2 + prog * 0.2) if progress_callback else None
        )
        
        # Step 3: Generate embeddings (50% of progress - this is the slowest)
        if progress_callback:
            progress_callback("Generating embeddings...", 0.4)
        
        embeddings = generate_embeddings(
            chunks,
            lambda msg, prog: progress_callback(msg, 0.4 + prog * 0.5) if progress_callback else None
        )
        
        # Step 4: Insert into database (10% of progress)
        if progress_callback:
            progress_callback("Inserting into database...", 0.9)
        
        inserted = insert_chunks(chunks, embeddings)
        
        if progress_callback:
            progress_callback("Complete!", 1.0)
        
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
    """
    Ingest all PDFs from a directory
    
    Args:
        directory: Path to directory containing PDFs
        skip_if_exists: If True, skip documents that already exist
    
    Returns:
        List of ingestion result dictionaries
    """
    pdf_files = list(directory.glob("*.pdf"))
    results = []
    
    for pdf_file in pdf_files:
        result = ingest_pdf(pdf_file, skip_if_exists=skip_if_exists)
        results.append(result)
    
    return results