"""
RAG Engine - OPTIMIZED FOR SPEED & ACCURACY
Handles retrieval and answer generation

OPTIMIZATIONS vs Phase 3
─────────────────────────
1. HyDE disabled by default (saves 200ms + 1 API call)
2. Cross-encoder disabled by default (saves 500ms)
3. Hybrid search uses CACHED BM25 index (10× faster)
4. Smarter query handling for meta-questions

TARGET PERFORMANCE
──────────────────
Total response time: 3-5 seconds
  • Embedding: ~50ms
  • Hybrid search: ~50ms (cached BM25)
  • Database query: ~100ms
  • Gemini API: 2-4 seconds
  • Total: ~3-5 seconds (vs 20 seconds before)
"""

from typing import List, Tuple
import time

from google import genai

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ sentence-transformers not available")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np

from .config import (
    EMBEDDING_MODEL_NAME,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TOP_K_RETRIEVAL,
    get_rag_prompt,
    USE_HYBRID_SEARCH,
    HYBRID_ALPHA,
    USE_HYDE,
    USE_RERANKING,
    RERANK_MODEL,
    RERANK_TOP_K,
)
from .db_utils import search_similar_chunks, get_document_stats

if USE_HYBRID_SEARCH:
    try:
        from .hybrid_search import hybrid_search
    except ImportError:
        print("⚠️ hybrid_search not available, using vector-only")
        USE_HYBRID_SEARCH = False


# ============================================================================
# FALLBACK EMBEDDING MODEL
# ============================================================================
class SimpleSentenceTransformer:
    def __init__(self, model_name):
        print(f"📦 Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=512, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**encoded)
        return outputs.last_hidden_state.mean(dim=1).numpy()


# ============================================================================
# RAG ENGINE - OPTIMIZED
# ============================================================================
class RAGEngine:
    """
    Optimized RAG Engine
    
    Targets:
    - 3-5 second response time
    - High accuracy
    - No false off-topic rejections
    """

    def __init__(self):
        print("🔄 Initializing RAG Engine (OPTIMIZED)...")

        # ── Embedding model ──────────────────────────────────────────
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                print(f"✅ Embedding model loaded: {EMBEDDING_MODEL_NAME}")
            else:
                self.embedding_model = SimpleSentenceTransformer(EMBEDDING_MODEL_NAME)
                print(f"✅ Embedding model loaded (fallback): {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise

        # ── Cross-encoder (optional) ─────────────────────────────────
        self.cross_encoder = None
        if USE_RERANKING and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(RERANK_MODEL)
                print(f"✅ Cross-encoder loaded: {RERANK_MODEL}")
            except Exception as e:
                print(f"⚠️ Cross-encoder not loaded: {e}")

        # ── Gemini client ────────────────────────────────────────────
        try:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"✅ Gemini client ready: {GEMINI_MODEL}")
        except Exception as e:
            print(f"❌ Error configuring Gemini: {e}")
            raise

    # ── HyDE (optional - disabled by default for speed) ─────────────
    def _generate_hypothesis(self, query: str) -> str:
        """HyDE: generate hypothetical answer"""
        try:
            prompt = f"Write a brief answer to: {query}"
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"⚠️ HyDE failed: {e}")
        return query

    # ── OPTIMIZED retrieval ──────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple]:
        """
        Retrieve relevant chunks - OPTIMIZED for speed
        
        Performance target: 100-150ms total
        """
        try:
            start = time.time()
            
            # ── Step 1: HyDE (optional - disabled by default) ───────
            search_text = query
            if USE_HYDE:
                hypothesis  = self._generate_hypothesis(query)
                search_text = hypothesis

            # ── Step 2: Embed query ──────────────────────────────────
            query_embedding = self.embedding_model.encode([search_text])[0]
            embed_time = time.time() - start
            print(f"⏱️  Embedding: {embed_time*1000:.0f}ms")

            # ── Step 3: Hybrid or vector search ──────────────────────
            search_start = time.time()
            if USE_HYBRID_SEARCH:
                results = hybrid_search(
                    query_text=query,
                    query_embedding=query_embedding,
                    top_k=RERANK_TOP_K if USE_RERANKING and self.cross_encoder else top_k,
                    alpha=HYBRID_ALPHA,
                )
                search_time = time.time() - search_start
                print(f"⏱️  Hybrid search: {search_time*1000:.0f}ms")
            else:
                results = search_similar_chunks(
                    query_embedding,
                    top_k=RERANK_TOP_K if USE_RERANKING and self.cross_encoder else top_k
                )
                search_time = time.time() - search_start
                print(f"⏱️  Vector search: {search_time*1000:.0f}ms")

            # ── Step 4: Re-ranking (optional - disabled by default) ─
            if USE_RERANKING and self.cross_encoder and len(results) > top_k:
                rerank_start = time.time()
                pairs = [(query, content) for content, _, _, _ in results]
                rerank_scores = self.cross_encoder.predict(pairs)
                scored = [
                    (content, doc, page, float(score))
                    for (content, doc, page, _), score in zip(results, rerank_scores)
                ]
                scored.sort(key=lambda x: x[3], reverse=True)
                results = scored[:top_k]
                rerank_time = time.time() - rerank_start
                print(f"⏱️  Re-ranking: {rerank_time*1000:.0f}ms")

            total_time = time.time() - start
            print(f"⏱️  Total retrieval: {total_time*1000:.0f}ms")
            
            return results

        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []

    # ── Generation ───────────────────────────────────────────────────
    def generate_answer(self, query: str, retrieved_chunks: List[Tuple]) -> str:
        """Generate answer using LLM"""
        
        if not retrieved_chunks:
            return "❌ No relevant information found. Please try rephrasing your question."

        # Format context
        context_parts = []
        for content, doc_name, page_num, similarity in retrieved_chunks:
            context_parts.append(
                f"[Source: {doc_name}, Page {page_num}, Relevance: {similarity:.2f}]\n{content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Get document list
        try:
            doc_stats = get_document_stats()
            available_docs = [doc_name for doc_name, _ in doc_stats]
        except Exception as e:
            print(f"⚠️ Could not fetch doc stats: {e}")
            available_docs = []

        prompt = get_rag_prompt(query, context, available_documents=available_docs)

        # Call Gemini with exponential back-off
        max_retries = 3
        base_delay  = 2

        for attempt in range(max_retries):
            try:
                api_start = time.time()
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                api_time = time.time() - api_start
                print(f"⏱️  Gemini API: {api_time:.2f}s")

                if response and response.text:
                    return response.text
                return "❌ Empty response from LLM."

            except Exception as e:
                error_str = str(e)

                if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        print(f"⚠️ Rate limit. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return """❌ Rate limit exceeded.

- 15 requests per minute
- 1,500 requests per day

Wait 1 minute and try again."""

                elif "blocked" in error_str.lower():
                    return "⚠️ Content blocked. Please rephrase your question."

                elif "api" in error_str.lower() and ("key" in error_str.lower() or "auth" in error_str.lower()):
                    return "❌ API authentication error. Check your GEMINI_API_KEY in .env."

                else:
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        print(f"⚠️ Error: {error_str}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return f"❌ Error: {error_str}"

        return "❌ Maximum retries exceeded."

    # ── Full pipeline ────────────────────────────────────────────────
    def query(self, question: str, top_k: int = TOP_K_RETRIEVAL) -> dict:
        """
        Complete RAG pipeline
        
        Target: 3-5 seconds total
        """
        try:
            pipeline_start = time.time()
            
            retrieved_chunks = self.retrieve(question, top_k=top_k)
            answer = self.generate_answer(question, retrieved_chunks)

            total_time = time.time() - pipeline_start
            print(f"⏱️  TOTAL PIPELINE: {total_time:.2f}s")

            return {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "content": content,
                        "document": doc_name,
                        "page": page_num,
                        "similarity": float(similarity)
                    }
                    for content, doc_name, page_num, similarity in retrieved_chunks
                ],
                "num_sources": len(retrieved_chunks),
                "status": "success"
            }

        except Exception as e:
            error_message = f"❌ Unexpected error: {str(e)}"
            print(error_message)
            return {
                "question": question,
                "answer": error_message,
                "sources": [],
                "num_sources": 0,
                "status": "error"
            }


# ============================================================================
# SINGLETON
# ============================================================================
_rag_engine = None

def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine

def reset_rag_engine():
    global _rag_engine
    _rag_engine = None