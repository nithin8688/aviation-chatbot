from functools import lru_cache
from src.rag_engine import RAGEngine
from src.config import ENABLE_RESPONSE_CACHING, CACHE_MAX_SIZE

# Create a single shared RAG engine instance
_rag_engine = RAGEngine()

# -------------------------
# Simple conversation memory (last turn only)
# -------------------------
_last_interaction = {
    "question": None,
    "answer": None
}

# -------------------------
# Follow-up detection
# -------------------------
def _is_follow_up(query: str) -> bool:
    follow_up_phrases = [
        "tell me more",
        "more information",
        "explain more",
        "elaborate",
        "continue",
        "yes",
        "go on"
    ]
    q = query.lower().strip()
    return any(q.startswith(p) or p in q for p in follow_up_phrases)

# -------------------------
# Cached internal function
# -------------------------
@lru_cache(maxsize=CACHE_MAX_SIZE)
def _cached_answer(combined_query: str) -> str:
    retrieved_chunks, confidence = _rag_engine.retrieve(combined_query)
    return _rag_engine.generate(combined_query, retrieved_chunks, confidence)

# -------------------------
# Public entry point
# -------------------------
def ask_aviation_bot(user_query: str) -> str:
    """
    Single public entry point for the aviation chatbot.
    Supports multi-turn follow-up questions.
    """

    if not user_query or not user_query.strip():
        return "Please enter a valid aviation-related question."

    user_query = user_query.strip()

    # -------------------------
    # Handle follow-up queries
    # -------------------------
    if (
        _is_follow_up(user_query)
        and _last_interaction["question"]
        and _last_interaction["answer"]
    ):
        combined_query = f"""
Previous question:
{_last_interaction['question']}

Previous answer:
{_last_interaction['answer']}

Follow-up request:
{user_query}
"""
    else:
        combined_query = user_query

    # -------------------------
    # Generate answer
    # -------------------------
    if ENABLE_RESPONSE_CACHING:
        answer = _cached_answer(combined_query)
    else:
        retrieved_chunks, confidence = _rag_engine.retrieve(combined_query)
        answer = _rag_engine.generate(combined_query, retrieved_chunks, confidence)

    # -------------------------
    # Update conversation memory
    # -------------------------
    _last_interaction["question"] = user_query
    _last_interaction["answer"] = answer

    return answer
