import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from rank_bm25 import BM25Okapi


from src.config import (
    CHUNKS_PATH,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
    MAX_CONTEXT_CHARS,
    SYSTEM_PROMPT,
)


class RAGEngine:
    """
    Core on-prem Retrieval-Augmented Generation engine
    for the Aviation Chatbot.
    """

    def __init__(self):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # BM25
        self.tokenized_corpus = [
            chunk["text"].lower().split()
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # FAISS
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        # Embeddings
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # LLM
        self.llm = Ollama(
            model=LLM_MODEL_NAME,
            temperature=0.1,
        )
        

    def detect_domain(self, query: str) -> str:
        query = query.lower()

        airport_keywords = [
            "apron", "airside", "ground", "runway",
            "terminal", "gate", "aircraft parking"
        ]

        scada_keywords = [
            "scada", "rtu", "plc", "alarm", "protocol"
        ]

        if any(k in query for k in airport_keywords):
            return "airport_ops"

        if any(k in query for k in scada_keywords):
            return "scada"

        return "general"


    # -------------------------
    # BM25 Retrieval (helper)
    # -------------------------
    def retrieve_bm25(self, query: str, top_k: int):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]

    # -------------------------
    # Hybrid Retrieval (FAISS + BM25)
    # -------------------------
    def retrieve(self, query: str, top_k: int | None = None):
        if top_k is None:
            top_k = DEFAULT_TOP_K

        # -------------------------
        # DOMAIN ROUTING
        # -------------------------
        domain = self.detect_domain(query)

        filtered_chunks = [
            c for c in self.chunks
            if (
                domain == "general"
                or (domain == "airport_ops" and "airport" in c["document_name"].lower())
                or (domain == "scada" and "scada" in c["document_name"].lower())
            )
        ]

        # -------------------------
        # FAISS (semantic search)
        # -------------------------
        query_embedding = self.embedding_model.encode([query])

        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"),
            top_k,
        )

        retrieved_chunks = []
        confidence_scores = []

        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 / (1 + dist)

            if similarity >= SIMILARITY_THRESHOLD:
                chunk = self.chunks[idx]

                # Optional domain guard (lightweight)
                if chunk in filtered_chunks or domain == "general":
                    retrieved_chunks.append(chunk)
                    confidence_scores.append(similarity)

        # -------------------------
        # BM25 (keyword search, domain-filtered)
        # -------------------------
        bm25_results = self.retrieve_bm25(query, top_k)

        bm25_results = [
            c for c in bm25_results
            if c in filtered_chunks
        ]

        if bm25_results:
            retrieved_chunks.extend(bm25_results)
            confidence_scores.append(0.6)  # keyword relevance boost

        # -------------------------
        # Deduplicate + confidence
        # -------------------------
        unique_chunks = {id(c): c for c in retrieved_chunks}

        confidence = (
            round(max(confidence_scores), 2)
            if confidence_scores else 0.0
        )

        return list(unique_chunks.values()), confidence




    # -------------------------
    # Generation
    # -------------------------
    def generate(self, query: str, retrieved_chunks: list[dict], confidence: float) -> str:
        if not retrieved_chunks:
            return (
                "This question is outside the aviation domain.\n"
                "Confidence: 0.00 (Very Low)"
            )

        context = "\n\n".join(
            f"(Source: {c['document_name']}, Page {c['page_number']})\n"
            f"{c['text'][:MAX_CONTEXT_CHARS]}"
            for c in retrieved_chunks
        )

        prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:
"""

        answer = self.llm.invoke(prompt)

        confidence_label = (
            "High" if confidence >= 0.75 else
            "Medium" if confidence >= 0.5 else
            "Low"
        )

        return f"{answer}\n\nConfidence: {confidence} ({confidence_label})"

