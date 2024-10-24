import streamlit as st
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from langchain.embeddings.base import Embeddings
from document_processor import DocumentProcessor
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from functools import lru_cache
import time
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NomicEmbeddings(BaseModel, Embeddings):
    """Custom embeddings class for Nomic's embedding model."""
    
    model_name: str = Field(default="nomic-ai/nomic-embed-text-v1.5")
    show_progress: bool = Field(default=True)
    _model: SentenceTransformer = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True  # Allow loading custom code
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # Add the required prefix for documents
        texts = [f"search_document: {text}" for text in texts]
        embeddings = self._model.encode(
            texts,
            show_progress_bar=self.show_progress,
            batch_size=32
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a query string."""
        # Add the required prefix for queries
        text = f"search_query: {text}"
        embedding = self._model.encode(
            [text],
            show_progress_bar=False
        )
        return embedding[0].tolist()


@dataclass
class SessionState:
    """Class to manage Streamlit session state variables."""
    chat_history: List[Dict[str, str]]
    vector_store: Optional[FAISS] = None
    chain: Optional[ConversationalRetrievalChain] = None
    embeddings: Optional[NomicEmbeddings] = None
    llm: Optional[Ollama] = None
    texts: Optional[List[str]] = None
    bm25: Optional[BM25Okapi] = None
    last_processed_input: Optional[str] = None
    processing_stats: Dict[str, float] = None

class RAGChatbot:
    """Main class for RAG Chatbot implementation."""
    
    def __init__(self):
        self.initialize_session_state()
        self.document_processor = DocumentProcessor()
        self.setup_prompts()

    def setup_prompts(self):
        """Initialize prompt templates."""
        self.qa_prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always cite specific parts of the context that support your answer.

        Context: {context}

        Question: {question}
        Answer: Let me help you with that based on the provided context."""

        self.system_prompt = """You are a helpful AI assistant powered by RAG technology.
        Always be clear about the source of your information and maintain a professional tone.
        If you're unsure about something, acknowledge it openly."""

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

    @staticmethod
    def measure_time(func):
        """Decorator to measure function execution time."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"{func.__name__} took {execution_time:.2f} seconds")
            return result
        return wrapper

    @measure_time
    def load_and_process_input(self, input_source) -> List[str]:
        """Process input and split into chunks."""
        if (st.session_state.state.last_processed_input == str(input_source) and 
            st.session_state.state.texts is not None):
            return st.session_state.state.texts

        temp_file_path = self.document_processor.process_input(input_source)
        
        with open(temp_file_path, 'r') as file:
            text = file.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
            is_separator_regex=False
        )
        
        chunks = text_splitter.split_text(text)
        st.session_state.state.last_processed_input = str(input_source)
        st.session_state.state.processing_stats['total_processed_chunks'] = len(chunks)
        return chunks

    @measure_time
    def initialize_vector_store(self, texts: List[str]) -> FAISS:
        """Initialize vector store with Nomic embeddings."""
        embeddings = NomicEmbeddings()
        st.session_state.state.embeddings = embeddings
        
        batch_size = 32
        vector_store = None
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embeddings)
            else:
                vector_store.add_texts(batch)
        
        return vector_store

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
    
    @staticmethod
    def initialize_bm25(texts: List[str]) -> BM25Okapi:
        """Initialize BM25 retriever."""
        corpus = [text.split() for text in texts]
        return BM25Okapi(corpus)

    def setup_hybrid_retriever(
        self, 
        vector_store: FAISS, 
        texts: List[str], 
        faiss_weight: float,
        k: int = 5
    ) -> ContextualCompressionRetriever:
        """Set up hybrid retriever with FAISS and BM25."""
        bm25_weight = 1 - faiss_weight
        
        faiss_retriever = vector_store.as_retriever(
            search_kwargs={"k": k, "fetch_k": k * 2}
        )
        bm25_retriever = BM25Retriever.from_texts(texts)
        bm25_retriever.k = k

        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[faiss_weight, bm25_weight]
        )
        
        embeddings_filter = EmbeddingsFilter(
            embeddings=st.session_state.state.embeddings,
            similarity_threshold=0.5
        )
        
        return ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=ensemble_retriever
        )

    def setup_chain(self, hybrid_retriever: ContextualCompressionRetriever, llm: Ollama) -> ConversationalRetrievalChain:
        """Set up the conversational chain."""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=hybrid_retriever,
            memory=memory,
            return_source_documents=True,
            return_generated_question=True,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=self.qa_prompt_template,
                    input_variables=["context", "question"]
                )
            }
        )

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

    def _get_input_source(self, input_type: str):
        """Get input source based on type."""
        if input_type == "File Upload":
            return st.file_uploader("Choose a file", type=["txt", "pdf", "docx", "csv"])
        elif input_type == "URL":
            return st.text_input("Enter URL")
        else:
            return st.text_area("Enter text")

    def _process_input(self, input_source):
        """Process input source."""
        with st.spinner("Processing input..."):
            try:
                state = st.session_state.state
                state.texts = self.load_and_process_input(input_source)
                state.vector_store = self.initialize_vector_store(state.texts)
                state.bm25 = self.initialize_bm25(state.texts)
                state.llm = self.initialize_llm()
                
                hybrid_retriever = self.setup_hybrid_retriever(
                    state.vector_store,
                    state.texts,
                    0.5
                )
                state.chain = self.setup_chain(hybrid_retriever, state.llm)
                
                st.success("Input processed successfully!")
            except Exception as e:
                st.error(f"Error processing input: {str(e)}")
                logger.error(f"Error details: {e}", exc_info=True)

    def _update_weights(self, faiss_weight: float):
        """Update retrieval weights."""
        state = st.session_state.state
        if state.vector_store is not None and state.texts is not None:
            try:
                hybrid_retriever = self.setup_hybrid_retriever(
                    state.vector_store,
                    state.texts,
                    faiss_weight
                )
                state.chain = self.setup_chain(hybrid_retriever, state.llm)
                st.success("Weights updated successfully!")
            except Exception as e:
                st.error(f"Error updating weights: {str(e)}")
                logger.error(f"Error details: {e}", exc_info=True)
        else:
            st.warning("Please process input before updating weights.")

    def render_sidebar(self):
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
                self._process_input(input_source)

            st.header("Retrieval Weights")
            faiss_weight = st.slider(
                "Semantic Search (FAISS) Weight", 
                0.0, 1.0, 0.5, 0.01
            )
            st.write(f"Keyword Search (BM25) Weight: {1 - faiss_weight:.2f}")
            
            if st.button("Update Weights"):
                self._update_weights(faiss_weight)

            if st.session_state['show_debug_info']:
                self._display_debug_info()

    def _display_debug_info(self):
        """Display debug information and performance metrics."""
        st.header("Debug Information")
        stats = st.session_state.state.processing_stats
        st.write(f"""
        - Embedding Time: {stats.get('embedding_time', 0):.2f}s
        - Retrieval Time: {stats.get('retrieval_time', 0):.2f}s
        - Processed Chunks: {stats.get('total_processed_chunks', 0)}
        """)

    def render_chat_interface(self):
        """Render chat interface."""
        st.title("RAG Chatbot with Hybrid Search")
        
        if st.session_state.state.chain is not None:
            for message in st.session_state.state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("You:"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                self.handle_chat_input(prompt)
        else:
            st.info("Please process an input document to start chatting.")

    def run(self):
        """Main method to run the Streamlit app."""
        self.render_sidebar()
        self.render_chat_interface()

def main():
    chatbot = RAGChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()