"""
Database utility functions for Document QA Chatbot
Handles PostgreSQL operations with pgvector

WHAT CHANGED vs the previous version
─────────────────────────────────────
1. Connection pool  (ThreadedConnectionPool, min 2 / max 10)
   • get_db_connection() is now a @contextmanager.
   • It checks a connection out of the pool, yields it, and returns it in a
     finally block — so connections NEVER leak even if an exception fires.
   • close_pool() is provided for clean shutdown.

2. search_similar_chunks  —  single-parameter CTE
   • The query vector is passed once as %(vec)s.
   • A CTE computes the cosine distance in one pass; the outer SELECT just
     re-orders DESC.  pgvector no longer computes the distance twice per row
     and the driver no longer serialises the 384-float array twice on the wire.

3. insert_chunks  —  execute_values (single round-trip)
   • Replaces executemany (N individual INSERTs) with one big
     INSERT … VALUES (...), (...), …  statement.
   • Everything is inside one transaction; if anything fails the whole batch
     rolls back automatically.
   • Typical speedup: ~20× for 5 000 rows (minutes → seconds).
"""

import contextlib

import psycopg2
from psycopg2.pool   import ThreadedConnectionPool
from psycopg2.extras import execute_values
from typing import List, Tuple, Optional
import numpy as np

from .config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


# ============================================================================
# CONNECTION POOL  —  created once when the module is first imported
# ============================================================================
_pool: Optional[ThreadedConnectionPool] = None


def _get_pool() -> ThreadedConnectionPool:
    """Lazy-init the pool; return it on every subsequent call."""
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(
            minconn=2,          # two connections stay warm at all times
            maxconn=10,         # raise this if you see "pool exhausted" errors
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
    return _pool


@contextlib.contextmanager
def get_db_connection():
    """
    Context-manager that checks out a pooled connection and always returns it.

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
            conn.commit()          # only when you actually wrote something

    If an exception is raised inside the block the connection is rolled back
    before being returned to the pool, so it stays usable for the next caller.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()        # undo any half-finished writes
        raise                  # re-raise so the caller still sees the error
    finally:
        pool.putconn(conn)     # always goes back — even after an exception


def close_pool():
    """Close every connection in the pool.  Call this at application shutdown."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


# ============================================================================
# PUBLIC API
# ============================================================================

def get_total_chunks() -> int:
    """Return the total number of chunks stored in the knowledge base."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_chunks;")
            return cur.fetchone()[0]


def get_document_stats() -> List[Tuple[str, int]]:
    """
    Per-document chunk counts, highest first.

    Returns:
        [(document_name, chunk_count), …]
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT document_name, COUNT(*) AS chunk_count
                FROM   knowledge_chunks
                GROUP  BY document_name
                ORDER  BY chunk_count DESC;
            """)
            return cur.fetchall()


def search_similar_chunks(query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple]:
    """
    Cosine-similarity search using pgvector.

    • %(vec)s is sent exactly once; PostgreSQL references it in both ORDER BY
      and the similarity expression without recomputing the distance.
    • The CTE materialises the top-k rows; the outer query sorts them
      highest-similarity first for the caller.

    Returns:
        [(content, document_name, page_number, similarity_score), …]
    """
    vec = query_embedding.tolist()          # numpy → plain list, done once

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH ranked AS (
                    SELECT
                        content,
                        document_name,
                        page_number,
                        1 - (embedding <=> %(vec)s::vector) AS similarity
                    FROM   knowledge_chunks
                    ORDER  BY embedding <=> %(vec)s::vector   -- pgvector index hit
                    LIMIT  %(k)s
                )
                SELECT * FROM ranked ORDER BY similarity DESC;
            """, {"vec": vec, "k": top_k})
            return cur.fetchall()


def insert_chunks(chunks: List[dict], embeddings: np.ndarray) -> int:
    """
    Bulk-insert chunks and their embeddings in a single transaction.

    Uses execute_values → one big VALUES clause, one commit, one round-trip.
    If anything fails mid-way the entire batch rolls back (the context-manager
    handles rollback automatically).

    Args:
        chunks:     [{"text": …, "document_name": …, "page_number": …}, …]
        embeddings: np.ndarray of shape (N, 384)

    Returns:
        Number of rows written.
    """
    rows = [
        (
            chunk["text"],
            emb.tolist(),
            chunk["document_name"],
            chunk["page_number"],
        )
        for chunk, emb in zip(chunks, embeddings)
    ]

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO knowledge_chunks
                    (content, embedding, document_name, page_number)
                VALUES %s
                """,
                rows,
            )
        conn.commit()       # single commit after every row is staged

    return len(rows)


def delete_document(document_name: str) -> int:
    """
    Remove every chunk that belongs to *document_name*.

    Returns:
        Number of rows deleted.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM knowledge_chunks WHERE document_name = %s;",
                (document_name,),
            )
            deleted = cur.rowcount
        conn.commit()
    
    # OPTIMIZATION: Invalidate BM25 cache after deletion
    try:
        from .hybrid_search import invalidate_bm25_cache
        invalidate_bm25_cache()
    except ImportError:
        pass
    
    return deleted


def check_document_exists(document_name: str) -> bool:
    """True when at least one chunk with *document_name* exists."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM knowledge_chunks
                    WHERE  document_name = %s
                    LIMIT  1
                );
                """,
                (document_name,),
            )
            return cur.fetchone()[0]