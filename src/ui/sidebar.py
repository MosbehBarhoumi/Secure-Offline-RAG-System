# src/ui/sidebar.py
import streamlit as st
from typing import Any, Callable

class Sidebar:
    """Handle sidebar UI elements and interactions."""
    
    def __init__(
        self,
        process_input_callback: Callable[[Any], None],
        update_weights_callback: Callable[[float], None]
    ):
        self.process_input_callback = process_input_callback
        self.update_weights_callback = update_weights_callback

    def render(self):
        """Render sidebar with input processing and weight adjustment options."""
        with st.sidebar:
            st.header("Input Processing")
            
            st.session_state['show_debug_info'] = st.checkbox(
                "Show Debug Info", 
                value=False
            )
            
            input_type = st.selectbox(
                "Select input type", 
                ["File Upload", "URL", "Text Input"]
            )
            
            input_source = self._get_input_source(input_type)
            
            if st.button("Process Input"):
                self.process_input_callback(input_source)

            st.header("Retrieval Weights")
            faiss_weight = st.slider(
                "Semantic Search (FAISS) Weight", 
                0.0, 1.0, 0.5, 0.01
            )
            st.write(f"Keyword Search (BM25) Weight: {1 - faiss_weight:.2f}")
            
            if st.button("Update Weights"):
                self.update_weights_callback(faiss_weight)

    @staticmethod
    def _get_input_source(input_type: str):
        """Get input source based on type."""
        if input_type == "File Upload":
            return st.file_uploader("Choose a file", type=["txt", "pdf", "docx", "csv"])
        elif input_type == "URL":
            return st.text_input("Enter URL")
        else:
            return st.text_area("Enter text")