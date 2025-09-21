import sqlite3
from collections import deque
import logging
from utils.config import MAX_CHAT_HISTORY_LENGTH
from langchain_core.messages import HumanMessage, AIMessage
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChatMemoryManager:
    """Enhanced memory manager with persistent storage"""
    def __init__(self):
        # Use deque for efficient appending and popping from both ends (for max length)
        self.history = deque(maxlen=MAX_CHAT_HISTORY_LENGTH)
        self.db_path = 'chat_history.db'
        self._init_database()
        self.load_history_from_db()
        logging.info(f"ChatMemoryManager initialized with max history length: {MAX_CHAT_HISTORY_LENGTH}")

    def _init_database(self):
        """Initialize SQLite database for persistent chat history"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      role TEXT,
                      content TEXT,
                      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()

    def _get_connection(self):
        """Create a new database connection for each operation"""
        return sqlite3.connect(self.db_path)

    def add_message(self, role: str, content: str):
        """Adds a message to memory and database"""
        self.history.append({"role": role, "content": content})
        logging.debug(f"Message added: {role}: {content[:50]}... Current history length: {len(self.history)}")
        
        # Save to database
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content))
        conn.commit()
        conn.close()

    def get_history(self) -> list:
        """Returns the current chat history as a list of dictionaries."""
        return list(self.history)

    def clear_history(self):
        """Clears the entire chat history."""
        self.history.clear()
        logging.info("Chat history cleared.")
        
        # Clear database
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("DELETE FROM chat_history")
        conn.commit()
        conn.close()

    def get_last_n_messages(self, n: int) -> list:
        """Returns the last n messages from the history."""
        if n > 0:
            return list(self.history)[-n:]
        return []

    def get_langchain_messages(self) -> list:
        """
        Converts the internal history to a format compatible with LangChain's
        HumanMessage and AIMessage for conversational chains.
        """
        langchain_messages = []
        for msg in self.history:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        logging.debug(f"Converted {len(langchain_messages)} messages to LangChain format.")
        return langchain_messages

    def load_history_from_db(self):
        """Load chat history from database"""
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT role, content FROM chat_history ORDER BY timestamp ASC")
        for role, content in c.fetchall():
            self.history.append({"role": role, "content": content})
        conn.close()
        logging.info(f"Loaded {len(self.history)} messages from database")

    def __len__(self):
        return len(self.history)

    def __str__(self):
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.history])