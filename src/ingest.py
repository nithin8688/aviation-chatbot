"""
PDF Ingestion Pipeline for Aviation Chatbot
Handles PDF processing, chunking, embedding, and database insertion
"""

import pdfplumber
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from .db_utils import insert_chunks, check_document_exists


def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
    """
    Extract text from PDF page by page
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        List of page dictionaries with text, page_number, document_name
    """
    pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "text": text.strip(),
                    "page_number": i + 1,
                    "document_name": pdf_path.name
                })
    
    return pages


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Chunk pages into smaller pieces
    
    Args:
        pages: List of page dictionaries
    
    Returns:
        List of chunk dictionaries
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = []
    
    for page in pages:
        splits = text_splitter.split_text(page["text"])
        
        for chunk_text in splits:
            chunks.append({
                "text": chunk_text,
                "document_name": page["document_name"],
                "page_number": page["page_number"]
            })
    
    return chunks


def generate_embeddings(chunks: List[Dict], model_name: str = EMBEDDING_MODEL_NAME):
    """
    Generate embeddings for chunks
    
    Args:
        chunks: List of chunk dictionaries
        model_name: Name of embedding model to use
    
    Returns:
        Numpy array of embeddings
    """
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings


def ingest_pdf(pdf_path: Path, skip_if_exists: bool = True) -> Dict:
    """
    Complete ingestion pipeline for a single PDF
    
    Args:
        pdf_path: Path to PDF file
        skip_if_exists: If True, skip if document already exists in DB
    
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
        # Step 1: Extract text
        pages = extract_text_from_pdf(pdf_path)
        
        if not pages:
            return {
                "status": "error",
                "document": document_name,
                "message": "No text extracted from PDF",
                "pages": 0,
                "chunks": 0
            }
        
        # Step 2: Chunk
        chunks = chunk_pages(pages)
        
        # Step 3: Generate embeddings
        embeddings = generate_embeddings(chunks)
        
        # Step 4: Insert into database
        inserted = insert_chunks(chunks, embeddings)
        
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