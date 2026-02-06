"""
Conversation Memory - PHASE 4
Multi-turn conversation with context awareness

WHY THIS EXISTS
───────────────
Current system treats every query independently. User asks:
  User: "What is ILS?"
  Bot: "ILS is..."
  User: "How does it work?"  ← Bot has no context that "it" = ILS
  Bot: "What are you referring to?"  ❌

With conversation memory:
  User: "What is ILS?"
  Bot: "ILS is..."
  User: "How does it work?"
  Bot: "ILS works by..." ✅ (remembers we're talking about ILS)

FEATURES
────────
• Short-term memory (last N messages in session)
• Conversation summarization (compress old messages)
• Context injection into prompts
• Conversation history persistence
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import hashlib


class ConversationMemory:
    """
    Manages conversation context across multiple turns.
    
    Memory structure:
    - messages: List of {role, content, timestamp} dicts
    - max_messages: Keep only last N messages in active memory
    - summary: Compressed summary of older messages
    """
    
    def __init__(
        self,
        session_id: str,
        max_messages: int = 10,
        storage_dir: Optional[Path] = None
    ):
        """
        Args:
            session_id: Unique ID for this conversation (e.g., user ID + timestamp)
            max_messages: Maximum messages to keep in active memory
            storage_dir: Directory to persist conversations
        """
        self.session_id = session_id
        self.max_messages = max_messages
        self.messages: List[Dict] = []
        self.summary: str = ""
        
        # Storage
        self.storage_dir = storage_dir
        if storage_dir:
            storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history.
        
        Args:
            role: "user" or "assistant"
            content: Message text
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        
        # Trim if exceeds max_messages
        if len(self.messages) > self.max_messages:
            self._compress_old_messages()
        
        # Persist to disk
        if self.storage_dir:
            self._save_to_disk()
    
    def get_recent_messages(self, n: int = 5) -> List[Dict]:
        """Get the last N messages"""
        return self.messages[-n:]
    
    def get_context_string(self, n: int = 5) -> str:
        """
        Get conversation context as a formatted string for prompt injection.
        
        Returns:
            Formatted context like:
            "Previous conversation:
             User: What is ILS?
             Assistant: ILS is an Instrument Landing System...
             User: How does it work?"
        """
        recent = self.get_recent_messages(n)
        
        if not recent:
            return ""
        
        context_lines = ["Previous conversation:"]
        for msg in recent:
            role = msg["role"].capitalize()
            content = msg["content"][:200]  # Truncate if too long
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)
    
    def clear(self):
        """Clear all messages and summary"""
        self.messages = []
        self.summary = ""
        
        if self.storage_dir:
            self._save_to_disk()
    
    def _compress_old_messages(self):
        """
        When messages exceed max_messages, summarize the oldest ones.
        
        This is a simple compression - just keep the text.
        A more sophisticated version would use an LLM to summarize.
        """
        # Take oldest messages beyond the limit
        excess = len(self.messages) - self.max_messages
        old_messages = self.messages[:excess]
        
        # Simple summary: just concatenate
        summary_parts = []
        for msg in old_messages:
            summary_parts.append(f"{msg['role']}: {msg['content'][:100]}")
        
        if self.summary:
            self.summary += "\n" + "\n".join(summary_parts)
        else:
            self.summary = "\n".join(summary_parts)
        
        # Keep only recent messages
        self.messages = self.messages[excess:]
    
    def _get_file_path(self) -> Path:
        """Get the file path for this conversation"""
        # Use hash of session_id as filename to avoid path issues
        file_hash = hashlib.md5(self.session_id.encode()).hexdigest()
        return self.storage_dir / f"conversation_{file_hash}.json"
    
    def _save_to_disk(self):
        """Persist conversation to disk"""
        if not self.storage_dir:
            return
        
        data = {
            "session_id": self.session_id,
            "messages": self.messages,
            "summary": self.summary,
            "last_updated": datetime.now().isoformat()
        }
        
        file_path = self._get_file_path()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_from_disk(self):
        """Load conversation from disk if it exists"""
        if not self.storage_dir:
            return
        
        file_path = self._get_file_path()
        if not file_path.exists():
            return
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.messages = data.get("messages", [])
            self.summary = data.get("summary", "")
        except Exception as e:
            print(f"⚠️ Could not load conversation: {e}")


class ConversationManager:
    """
    Manages multiple conversation sessions.
    
    Used by the Streamlit app to handle multiple users/sessions.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Args:
            storage_dir: Directory to store all conversations
        """
        self.storage_dir = storage_dir
        self.sessions: Dict[str, ConversationMemory] = {}
    
    def get_session(
        self,
        session_id: str,
        max_messages: int = 10
    ) -> ConversationMemory:
        """
        Get or create a conversation session.
        
        Args:
            session_id: Unique ID for the session
            max_messages: Max messages to keep in memory
        
        Returns:
            ConversationMemory instance
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(
                session_id=session_id,
                max_messages=max_messages,
                storage_dir=self.storage_dir
            )
        
        return self.sessions[session_id]
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            self.sessions[session_id].clear()
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs"""
        return list(self.sessions.keys())


# ============================================================================
# PROMPT INJECTION HELPERS
# ============================================================================
def inject_conversation_context(
    base_prompt: str,
    conversation: ConversationMemory,
    n_messages: int = 3
) -> str:
    """
    Inject conversation context into a prompt.
    
    Args:
        base_prompt: The original prompt
        conversation: ConversationMemory instance
        n_messages: Number of recent messages to include
    
    Returns:
        Modified prompt with conversation context
    
    Example:
        BEFORE:
            "Context: [chunks]
             User Question: How does it work?"
        
        AFTER:
            "Previous conversation:
             User: What is ILS?
             Assistant: ILS is...
             
             Context: [chunks]
             User Question: How does it work?"
    """
    context = conversation.get_context_string(n_messages)
    
    if not context:
        return base_prompt
    
    # Insert conversation context before the main query
    return f"{context}\n\n{base_prompt}"


def should_use_conversation_context(query: str) -> bool:
    """
    Determine if a query likely needs conversation context.
    
    Heuristics:
    - Contains pronouns: "it", "this", "that", "they"
    - Contains "also", "too", "additionally" (follow-up indicators)
    - Very short (< 5 words, likely a follow-up)
    
    Args:
        query: User's query
    
    Returns:
        True if conversation context should be used
    """
    query_lower = query.lower()
    
    # Check for pronouns
    pronouns = ["it", "this", "that", "they", "them", "its", "their"]
    has_pronoun = any(f" {p} " in f" {query_lower} " or f" {p}?" in f" {query_lower}" for p in pronouns)
    
    # Check for follow-up indicators
    follow_up_words = ["also", "too", "additionally", "moreover", "furthermore"]
    has_follow_up = any(word in query_lower for word in follow_up_words)
    
    # Check if very short
    word_count = len(query.split())
    is_short = word_count < 5
    
    return has_pronoun or has_follow_up or is_short