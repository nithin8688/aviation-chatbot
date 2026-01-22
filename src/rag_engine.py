"""
RAG Engine for Aviation Chatbot
Handles retrieval and answer generation
"""

from typing import List, Tuple
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_MODEL_NAME,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TOP_K_RETRIEVAL,
    get_rag_prompt
)
from .db_utils import search_similar_chunks


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine
    Combines vector search with LLM generation
    """
    
    def __init__(self):
        """Initialize embedding model and Gemini LLM"""
        print("🔄 Initializing RAG Engine...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"✅ Embedding model loaded: {EMBEDDING_MODEL_NAME}")
        
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.llm = genai.GenerativeModel(GEMINI_MODEL)
        print(f"✅ Gemini LLM configured: {GEMINI_MODEL}")
    
    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
        
        Returns:
            List of (content, document_name, page_number, similarity_score)
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search database
        results = search_similar_chunks(query_embedding, top_k=top_k)
        
        return results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Tuple]) -> str:
        """
        Generate answer using LLM with retrieved context
        
        Args:
            query: User's question
            retrieved_chunks: List of (content, doc_name, page_num, similarity)
        
        Returns:
            Generated answer string
        """
        # Format context from retrieved chunks
        context_parts = []
        for content, doc_name, page_num, similarity in retrieved_chunks:
            context_parts.append(
                f"[Source: {doc_name}, Page {page_num}, Relevance: {similarity:.2f}]\n{content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt
        prompt = get_rag_prompt(query, context)
        
        # Generate answer with retry logic
        import time
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(prompt)
                return response.text
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limit errors (429)
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"⚠️ Rate limit hit. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return f"❌ Rate limit exceeded. Please wait a minute and try again.\n\n💡 Tip: The free tier allows 15 requests per minute. Try asking fewer questions or wait between queries."
                
                # Handle other errors
                return f"❌ Error generating answer: {error_str}"
        
        return "❌ Max retries exceeded. Please try again later."
    
    def query(self, question: str, top_k: int = TOP_K_RETRIEVAL) -> dict:
        """
        Complete RAG pipeline: retrieve + generate
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
        
        Returns:
            Dictionary with answer and sources
        """
        # Step 1: Retrieve
        retrieved_chunks = self.retrieve(question, top_k=top_k)
        
        # Step 2: Generate
        answer = self.generate_answer(question, retrieved_chunks)
        
        # Format response
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
            "num_sources": len(retrieved_chunks)
        }


# Singleton instance (lazy loading)
_rag_engine = None

def get_rag_engine() -> RAGEngine:
    """Get or create RAG engine instance (singleton pattern)"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine