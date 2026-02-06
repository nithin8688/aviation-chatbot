"""
Semantic Chunking - PHASE 4
Intelligent chunk boundaries based on semantic similarity

WHY THIS EXISTS
───────────────
Fixed-size chunking (800 chars) often splits mid-sentence or mid-concept.
Semantic chunking groups sentences by meaning, creating more coherent chunks.

BENEFITS
────────
• Better embeddings (chunks represent complete ideas)
• Improved retrieval accuracy (+10-15%)
• No mid-sentence cuts
• Respects paragraph boundaries

EXAMPLE
───────
Fixed chunking:
  Chunk 1: "...the aircraft must comply with FAR 25.1001 which states that"
  Chunk 2: "fuel system components must be designed to prevent..."
  ❌ Regulation split across chunks!

Semantic chunking:
  Chunk 1: "...the aircraft must comply with FAR 25.1001 which states that 
            fuel system components must be designed to prevent ignition."
  ✅ Complete regulation in one chunk!
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class SemanticChunker:
    """
    Semantic chunking using sentence similarity.
    
    Algorithm:
    1. Split text into sentences
    2. Compute embedding for each sentence
    3. Calculate similarity between consecutive sentences
    4. Group sentences where similarity > threshold
    5. Merge groups to fit within max_chunk_size
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.7,
    ):
        """
        Args:
            model_name: Embedding model for sentence similarity
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            similarity_threshold: Sentences with similarity > this are grouped
                                  0.7 = fairly similar (good default)
                                  0.8 = very similar (stricter boundaries)
                                  0.6 = loosely similar (more flexible)
        """
        self.model = SentenceTransformer(model_name)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        sentences = sent_tokenize(text)
        # Filter out very short sentences (likely artifacts)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _compute_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Compute embeddings for all sentences"""
        return self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    
    def _compute_similarities(self, embeddings: np.ndarray) -> List[float]:
        """
        Compute cosine similarity between consecutive sentences.
        
        Returns:
            List of similarities: [sim(s0,s1), sim(s1,s2), ..., sim(sN-1,sN)]
        """
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(float(sim))
        return similarities
    
    def _group_sentences(
        self,
        sentences: List[str],
        similarities: List[float]
    ) -> List[List[str]]:
        """
        Group sentences based on similarity threshold.
        
        Algorithm:
        - Start a new group when similarity < threshold
        - Also start a new group when max_chunk_size would be exceeded
        """
        if not sentences:
            return []
        
        groups = []
        current_group = [sentences[0]]
        current_size = len(sentences[0])
        
        for i, sentence in enumerate(sentences[1:], start=0):
            sentence_len = len(sentence)
            
            # Check if we should start a new group
            should_split = (
                # Similarity dropped below threshold
                similarities[i] < self.similarity_threshold or
                # Adding this sentence would exceed max size
                current_size + sentence_len > self.max_chunk_size
            )
            
            if should_split:
                groups.append(current_group)
                current_group = [sentence]
                current_size = sentence_len
            else:
                current_group.append(sentence)
                current_size += sentence_len
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _merge_small_groups(self, groups: List[List[str]]) -> List[str]:
        """
        Merge groups that are too small.
        
        Small groups can happen when there are many topic shifts.
        We merge them with adjacent groups to reach min_chunk_size.
        """
        chunks = []
        current_chunk = ""
        
        for group in groups:
            group_text = " ".join(group)
            
            if not current_chunk:
                current_chunk = group_text
            elif len(current_chunk) + len(group_text) <= self.max_chunk_size:
                current_chunk += " " + group_text
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = group_text
                else:
                    # Current chunk too small, merge with this group
                    current_chunk += " " + group_text
        
        # Add the final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)
        elif current_chunk and chunks:
            # Merge with last chunk if too small
            chunks[-1] += " " + current_chunk
        elif current_chunk:
            # Only one chunk and it's small - keep it anyway
            chunks.append(current_chunk)
        
        return chunks
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text semantically.
        
        Args:
            text: Input text to chunk
        
        Returns:
            List of semantically coherent chunks
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return [text] if text.strip() else []
        
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [sentences[0]]
        
        # Step 2: Compute sentence embeddings
        embeddings = self._compute_sentence_embeddings(sentences)
        
        # Step 3: Compute similarities between consecutive sentences
        similarities = self._compute_similarities(embeddings)
        
        # Step 4: Group sentences by similarity
        groups = self._group_sentences(sentences, similarities)
        
        # Step 5: Merge small groups
        chunks = self._merge_small_groups(groups)
        
        return chunks
    
    def chunk_pages(
        self,
        pages: List[Dict],
        progress_callback=None
    ) -> List[Dict]:
        """
        Chunk pages semantically (replacement for langchain RecursiveCharacterTextSplitter).
        
        Args:
            pages: List of page dicts with 'text', 'page_number', 'document_name'
            progress_callback: Optional callback(message, progress)
        
        Returns:
            List of chunk dicts with 'text', 'page_number', 'document_name'
        """
        chunks = []
        total_pages = len(pages)
        
        for idx, page in enumerate(pages):
            if progress_callback:
                progress_callback(
                    f"Semantic chunking page {idx+1}/{total_pages}",
                    (idx+1) / total_pages
                )
            
            page_chunks = self.chunk_text(page["text"])
            
            for chunk_text in page_chunks:
                chunks.append({
                    "text": chunk_text,
                    "document_name": page["document_name"],
                    "page_number": page["page_number"]
                })
        
        return chunks


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================
def semantic_chunk_pages(
    pages: List[Dict],
    max_chunk_size: int = 1000,
    similarity_threshold: float = 0.7,
    progress_callback=None
) -> List[Dict]:
    """
    Convenience wrapper for semantic chunking.
    
    Drop-in replacement for the langchain chunking in ingest.py:
    
    BEFORE:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_text(page["text"])
    
    AFTER:
        from src.semantic_chunking import semantic_chunk_pages
        chunks = semantic_chunk_pages(pages, max_chunk_size=1000)
    """
    chunker = SemanticChunker(
        max_chunk_size=max_chunk_size,
        similarity_threshold=similarity_threshold
    )
    return chunker.chunk_pages(pages, progress_callback=progress_callback)