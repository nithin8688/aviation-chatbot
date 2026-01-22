"""
Database utility functions for Aviation Chatbot
Handles PostgreSQL operations with pgvector
"""

import psycopg2
from typing import List, Tuple, Optional
import numpy as np

from .config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def get_total_chunks() -> int:
    """Get total number of chunks in database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM knowledge_chunks;")
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return count


def get_document_stats() -> List[Tuple[str, int]]:
    """
    Get statistics about documents in database
    
    Returns:
        List of (document_name, chunk_count) tuples
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT document_name, COUNT(*) as chunk_count
        FROM knowledge_chunks
        GROUP BY document_name
        ORDER BY chunk_count DESC;
    """)
    stats = cursor.fetchall()
    cursor.close()
    conn.close()
    return stats


def search_similar_chunks(query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple]:
    """
    Search for similar chunks using vector similarity
    
    Args:
        query_embedding: Query vector embedding
        top_k: Number of results to return
    
    Returns:
        List of (content, document_name, page_number, similarity_score) tuples
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            content,
            document_name,
            page_number,
            1 - (embedding <=> %s::vector) as similarity
        FROM knowledge_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding.tolist(), query_embedding.tolist(), top_k))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return results


def insert_chunks(chunks: List[dict], embeddings: np.ndarray) -> int:
    """
    Insert chunks and embeddings into database
    
    Args:
        chunks: List of chunk dictionaries with 'text', 'document_name', 'page_number'
        embeddings: Numpy array of embeddings
    
    Returns:
        Number of chunks inserted
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Prepare batch data
    batch_data = [
        (
            chunk["text"],
            embedding.tolist(),
            chunk["document_name"],
            chunk["page_number"]
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    
    # Execute batch insert
    cursor.executemany("""
        INSERT INTO knowledge_chunks (content, embedding, document_name, page_number)
        VALUES (%s, %s, %s, %s)
    """, batch_data)
    
    conn.commit()
    inserted = len(batch_data)
    
    cursor.close()
    conn.close()
    
    return inserted


def delete_document(document_name: str) -> int:
    """
    Delete all chunks from a specific document
    
    Args:
        document_name: Name of document to delete
    
    Returns:
        Number of chunks deleted
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM knowledge_chunks
        WHERE document_name = %s;
    """, (document_name,))
    
    deleted = cursor.rowcount
    conn.commit()
    
    cursor.close()
    conn.close()
    
    return deleted


def check_document_exists(document_name: str) -> bool:
    """Check if a document already exists in the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT EXISTS(
            SELECT 1 FROM knowledge_chunks 
            WHERE document_name = %s
            LIMIT 1
        );
    """, (document_name,))
    
    exists = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    return exists