# src/ui/chat_interface.py
import streamlit as st
from typing import Any, Callable, Dict, List

class ChatInterface:
    """Handle chat interface UI elements and interactions."""
    
    def __init__(
        self,
        handle_chat_input: Callable[[str], None],
        chat_history: List[Dict[str, str]],
        chain_initialized: bool
    ):
        self.handle_chat_input = handle_chat_input
        self.chat_history = chat_history
        self.chain_initialized = chain_initialized

    def render(self):
        """Render chat interface."""
        st.title("RAG Chatbot with Hybrid Search")
        
        if self.chain_initialized:
            self._render_chat_history()
            self._handle_new_input()
        else:
            st.info("Please process an input document to start chatting.")

    def _render_chat_history(self):
        """Render existing chat history."""
        for message in self.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _handle_new_input(self):
        """Handle new chat input."""
        if prompt := st.chat_input("You:"):
            with st.chat_message("user"):
                st.markdown(prompt)
            self.handle_chat_input(prompt)