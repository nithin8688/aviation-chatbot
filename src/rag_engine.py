"""
RAG Engine for Aviation Chatbot
Handles retrieval and answer generation with privacy protection
"""

from typing import List, Tuple
import time
import os
import google.generativeai as genai

# Import with error handling
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ sentence-transformers not available, using alternative")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np

from .config import (
    EMBEDDING_MODEL_NAME,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TOP_K_RETRIEVAL,
    get_rag_prompt
)
from .db_utils import search_similar_chunks, get_document_stats


# Lightweight embedding model as fallback
class SimpleSentenceTransformer:
    """Fallback embedding model when sentence-transformers fails"""
    
    def __init__(self, model_name):
        print(f"📦 Loading model with transformers directly: {model_name}")
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
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine
    Combines vector search with LLM generation
    Includes privacy protection and rate limit handling
    """
    
    def __init__(self):
        """Initialize embedding model and Gemini LLM"""
        print("🔄 Initializing RAG Engine...")
        
        # Load embedding model with fallback
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                print(f"✅ Embedding model loaded: {EMBEDDING_MODEL_NAME}")
            else:
                self.embedding_model = SimpleSentenceTransformer(EMBEDDING_MODEL_NAME)
                print(f"✅ Embedding model loaded (fallback mode): {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
        
        # Configure Gemini
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.llm = genai.GenerativeModel(GEMINI_MODEL)
            print(f"✅ Gemini LLM configured: {GEMINI_MODEL}")
        except Exception as e:
            print(f"❌ Error configuring Gemini: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
        
        Returns:
            List of (content, document_name, page_number, similarity_score)
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search database
            results = search_similar_chunks(query_embedding, top_k=top_k)
            
            return results
        
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")
            return []
    
    def generate_answer(self, query: str, retrieved_chunks: List[Tuple]) -> str:
        """
        Generate answer using LLM with retrieved context
        Includes privacy protection and rate limit handling
        
        Args:
            query: User's question
            retrieved_chunks: List of (content, doc_name, page_num, similarity)
        
        Returns:
            Generated answer string
        """
        # Handle empty retrieval results
        if not retrieved_chunks:
            return "❌ No relevant information found in the knowledge base. Please try rephrasing your question."
        
        # Format context from retrieved chunks
        context_parts = []
        for content, doc_name, page_num, similarity in retrieved_chunks:
            context_parts.append(
                f"[Source: {doc_name}, Page {page_num}, Relevance: {similarity:.2f}]\n{content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Get available document names (for privacy protection in off-topic responses)
        try:
            doc_stats = get_document_stats()
            available_docs = [doc_name for doc_name, _ in doc_stats]
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch document stats: {e}")
            available_docs = []
        
        # Build prompt with document list for privacy-safe off-topic responses
        prompt = get_rag_prompt(query, context, available_documents=available_docs)
        
        # Generate answer with retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Generate content with Gemini
                response = self.llm.generate_content(prompt)
                
                # Check if response is valid
                if response and response.text:
                    return response.text
                else:
                    return "❌ Received empty response from LLM. Please try again."
            
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limit errors (429)
                if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"⚠️ Rate limit hit. Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return """❌ Rate limit exceeded. The Gemini API free tier has limits:
                        
- 15 requests per minute
- 1,500 requests per day

💡 **What you can do:**
1. Wait 1 minute and try again
2. Reduce the number of sources in settings
3. Upgrade to a paid API plan for higher limits

Please wait a moment before asking another question."""
                
                # Handle blocked content errors
                elif "blocked" in error_str.lower():
                    return f"⚠️ Content blocked by safety filters. Please rephrase your question in a different way."
                
                # Handle invalid API key
                elif "api" in error_str.lower() and ("key" in error_str.lower() or "auth" in error_str.lower()):
                    return "❌ API authentication error. Please check your Gemini API key in the configuration."
                
                # Handle other errors
                else:
                    if attempt < max_retries - 1:
                        print(f"⚠️ Error: {error_str}. Retrying (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return f"❌ Error generating answer: {error_str}\n\nPlease try again or contact support if the issue persists."
        
        return "❌ Maximum retry attempts exceeded. Please try again later."
    
    def query(self, question: str, top_k: int = TOP_K_RETRIEVAL) -> dict:
        """
        Complete RAG pipeline: retrieve + generate
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.retrieve(question, top_k=top_k)
            
            # Step 2: Generate answer
            answer = self.generate_answer(question, retrieved_chunks)
            
            # Step 3: Format response
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
            # Handle any unexpected errors
            error_message = f"❌ An unexpected error occurred: {str(e)}"
            print(error_message)
            
            return {
                "question": question,
                "answer": error_message,
                "sources": [],
                "num_sources": 0,
                "status": "error"
            }


# Singleton instance (lazy loading)
_rag_engine = None

def get_rag_engine() -> RAGEngine:
    """
    Get or create RAG engine instance (singleton pattern)
    
    Returns:
        RAGEngine instance
    """
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


def reset_rag_engine():
    """
    Reset the RAG engine singleton (useful for testing or config changes)
    """
    global _rag_engine
    _rag_engine = None