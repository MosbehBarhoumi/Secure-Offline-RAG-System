# src/app.py
import streamlit as st
import os
import logging
from typing import List, Optional, Tuple, Dict, Any
from models.session_state import SessionState
from models.embeddings import NomicEmbeddings
from services.text_processing import TextProcessor
from services.retriever_service import RetrieverService
from services.chain_service import ChainService
from ui.sidebar import Sidebar
from ui.chat_interface import ChatInterface
from utils.decorators import measure_time
from utils.logging_config import setup_logging
from document_processor import DocumentProcessor
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from rank_bm25 import BM25Okapi
from functools import lru_cache
from typing import Any, Dict
import time

logger = setup_logging()

class RAGChatbot:
    """Main class for RAG Chatbot implementation."""
    
    def __init__(self):
        self.initialize_session_state()
        self.document_processor = DocumentProcessor()
        self.setup_prompts()
        self.setup_ui_components()
        self.text_processor = TextProcessor()

    def setup_prompts(self):
        """Initialize prompt templates."""
        self.qa_prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always cite specific parts of the context that support your answer.

        Context: {context}

        Question: {question}
        Answer: Let me help you with that based on the provided context."""

    @staticmethod
    def initialize_session_state():
        """Initialize session state with default values."""
        if 'state' not in st.session_state:
            st.session_state.state = SessionState(
                chat_history=[],
                processing_stats={
                    'embedding_time': 0.0,
                    'retrieval_time': 0.0,
                    'total_processed_chunks': 0
                }
            )

    def setup_ui_components(self):
        """Initialize UI components."""
        self.sidebar = Sidebar(
            self._process_input,
            self._update_weights
        )
        self.chat_interface = ChatInterface(
            self.handle_chat_input,
            st.session_state.state.chat_history,
            st.session_state.state.chain is not None
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def initialize_llm() -> Ollama:
        """Initialize and cache the LLM instance."""
        return Ollama(
            model="qwen2.5:3b",
            temperature=0.7,
            top_p=0.9,
            callback_manager=None
        )

    @measure_time
    def _process_input(self, input_source: Any):
        """Process input source and initialize necessary components."""
        with st.spinner("Processing input..."):
            try:
                state = st.session_state.state
                
                # Process document
                temp_file_path = self.document_processor.process_input(input_source)
                
                # Read and split text
                with open(temp_file_path, 'r') as file:
                    text = file.read()
                state.texts = self.text_processor.split_text(text)
                
                # Initialize embeddings and vector store
                state.embeddings = NomicEmbeddings()
                state.vector_store = self._initialize_vector_store(state.texts)
                
                # Initialize BM25
                state.bm25 = self._initialize_bm25(state.texts)
                
                # Initialize LLM
                state.llm = self.initialize_llm()
                
                # Setup retriever and chain
                hybrid_retriever = RetrieverService.setup_hybrid_retriever(
                    state.vector_store,
                    state.texts,
                    state.embeddings,
                    0.5
                )
                state.chain = ChainService.setup_chain(
                    hybrid_retriever,
                    state.llm,
                    self.qa_prompt_template
                )
                
                st.success("Input processed successfully!")
                
            except Exception as e:
                st.error(f"Error processing input: {str(e)}")
                logger.error(f"Error details: {e}", exc_info=True)

    def _initialize_vector_store(self, texts: List[str]) -> FAISS:
        """Initialize vector store with embeddings."""
        batch_size = 32
        vector_store = None
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, st.session_state.state.embeddings)
            else:
                vector_store.add_texts(batch)
        
        return vector_store

    @staticmethod
    def _initialize_bm25(texts: List[str]) -> BM25Okapi:
        """Initialize BM25 retriever."""
        corpus = [text.split() for text in texts]
        return BM25Okapi(corpus)

    def _update_weights(self, faiss_weight: float):
        """Update retrieval weights."""
        state = st.session_state.state
        if state.vector_store is not None and state.texts is not None:
            try:
                hybrid_retriever = RetrieverService.setup_hybrid_retriever(
                    state.vector_store,
                    state.texts,
                    state.embeddings,
                    faiss_weight
                )
                state.chain = ChainService.setup_chain(
                    hybrid_retriever,
                    state.llm,
                    self.qa_prompt_template
                )
                st.success("Weights updated successfully!")
            except Exception as e:
                st.error(f"Error updating weights: {str(e)}")
                logger.error(f"Error details: {e}", exc_info=True)
        else:
            st.warning("Please process input before updating weights.")

    @measure_time
    def handle_chat_input(self, prompt: str):
        """Process chat input and generate response."""
        try:
            retrieval_start = time.time()
            result = st.session_state.state.chain({"question": prompt})
            retrieval_time = time.time() - retrieval_start
            
            st.session_state.state.processing_stats['retrieval_time'] = retrieval_time
            
            if os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true':
                for i, doc in enumerate(result['source_documents']):
                    logger.debug(f"Document {i + 1}:\n{doc.page_content}\n---")
            
            self._update_chat_history(prompt, result['answer'])
            self._display_response(result)
            
        except Exception as e:
            logger.error(f"Error in chat handling: {str(e)}", exc_info=True)
            self._handle_error(e)

    def _update_chat_history(self, prompt: str, response: str):
        """Update chat history with new messages."""
        st.session_state.state.chat_history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])

    def _display_response(self, result: Dict[str, Any]):
        """Display the chatbot response with optional debugging info."""
        with st.chat_message("assistant"):
            st.markdown(result['answer'])
            
            if st.session_state.get('show_debug_info', False):
                st.info(f"""
                Retrieval Time: {st.session_state.state.processing_stats['retrieval_time']:.2f}s
                Sources Used: {len(result['source_documents'])}
                """)

    def _handle_error(self, error: Exception):
        """Handle errors gracefully."""
        error_message = f"I apologize, but I encountered an error: {str(error)}"
        with st.chat_message("assistant"):
            st.error(error_message)
        logger.error(f"Error details: {error}", exc_info=True)

    def run(self):
        """Main method to run the Streamlit app."""
        self.sidebar.render()
        self.chat_interface.render()

def main():
    chatbot = RAGChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()