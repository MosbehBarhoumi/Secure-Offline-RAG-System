import streamlit as st
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from rank_bm25 import BM25Okapi
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from document_processor import DocumentProcessor
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
AVAILABLE_MODELS = {
    "Qwen 3B": "qwen2.5:3b",
    "Qwen 7B": "qwen2.5:7b",
    "Qwen 14B": "qwen2.5:14b"
}

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_RETRIEVER_K = 4

@dataclass
class ChatMessage:
    role: str
    content: str

class MultiBertEmbeddings(BaseModel, Embeddings):
    """Updated MultiBertEmbeddings class with proper hashing support"""
    model_name: str = Field(default=EMBEDDING_MODEL)
    show_progress: bool = Field(default=False)
    _model: Optional[SentenceTransformer] = None
    
    model_config = {
        'arbitrary_types_allowed': True,
        'protected_namespaces': ()
    }
    
    def __hash__(self):
        return hash(self.model_name)
    
    def __eq__(self, other):
        if isinstance(other, MultiBertEmbeddings):
            return self.model_name == other.model_name
        return False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            self._model = SentenceTransformer(self.model_name, device='cpu')
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            raise

    @lru_cache(maxsize=1024)
    def embed_query(self, text: str) -> List[float]:
        try:
            embedding = self._model.encode([text], show_progress_bar=False, 
                                         normalize_embeddings=True)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error in embed_query: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self._model.encode(texts, show_progress_bar=self.show_progress,
                                          batch_size=DEFAULT_BATCH_SIZE, 
                                          normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error in embed_documents: {e}")
            raise

class SessionState:
    def __init__(self):
        self.chat_history: List[ChatMessage] = []
        self.vector_store: Optional[FAISS] = None
        self.chain: Optional[Any] = None
        self.embeddings: Optional[MultiBertEmbeddings] = None
        self.llm: Optional[ChatOllama] = None
        self.texts: Optional[List[str]] = None
        self.bm25: Optional[BM25Okapi] = None
        self.selected_model: str = list(AVAILABLE_MODELS.values())[0]
        self.processing_error: Optional[str] = None
        self.document_processor: DocumentProcessor = DocumentProcessor()

    @staticmethod
    def initialize_llm(model_name: str) -> ChatOllama:
        try:
            return ChatOllama(
                model=model_name,
                temperature=0.7,
                top_p=0.9,
                context_window=2048,
                repeat_penalty=1.1,
                base_url="http://localhost:11434"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

class RAGChatbot:
    def __init__(self):
        self.initialize_session_state()
        self.setup_prompts()

    @staticmethod
    def initialize_session_state():
        if 'state' not in st.session_state:
            st.session_state.state = SessionState()

    def setup_prompts(self):
        self.qa_template = PromptTemplate.from_template("""
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know.

        IMPORTANT: If the user's question is in French, respond in French.
        If the user's question is in English, respond in English.
        
        Context: {context}
        
        Question: {question}
        Answer: Let me help you with that based on the provided context.
        """)

    def load_and_process_input(self, input_source: Any) -> List[str]:
        if not input_source:
            return []
        
        try:
            # Process the input using DocumentProcessor
            processed_file_path = st.session_state.state.document_processor.process_input(input_source)
            
            with open(processed_file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            chunks = text_splitter.split_text(text)
            if not chunks:
                raise ValueError("No text chunks were generated from the input")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            st.session_state.state.processing_error = str(e)
            raise

    def initialize_vector_store(self, texts: List[str]) -> FAISS:
        try:
            embeddings = MultiBertEmbeddings()
            st.session_state.state.embeddings = embeddings
            
            progress_bar = st.progress(0)
            vector_store = FAISS.from_texts(texts, embeddings)
            progress_bar.progress(1.0)
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def setup_hybrid_retriever(
        self,
        vector_store: FAISS,
        texts: List[str],
        semantic_weight: float
    ) -> ContextualCompressionRetriever:
        try:
            faiss_retriever = vector_store.as_retriever(
                search_kwargs={"k": DEFAULT_RETRIEVER_K}
            )
            bm25_retriever = BM25Retriever.from_texts(texts)
            bm25_retriever.k = DEFAULT_RETRIEVER_K
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever],
                weights=[semantic_weight, 1 - semantic_weight]
            )
            
            embeddings_filter = EmbeddingsFilter(
                embeddings=st.session_state.state.embeddings,
                similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD
            )
            
            return ContextualCompressionRetriever(
                base_compressor=embeddings_filter,
                base_retriever=ensemble_retriever
            )
            
        except Exception as e:
            logger.error(f"Error setting up hybrid retriever: {e}")
            raise

    def setup_chain(
        self,
        hybrid_retriever: ContextualCompressionRetriever,
        llm: ChatOllama
    ) -> Any:
        try:
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            def get_context(query):
                docs = hybrid_retriever.get_relevant_documents(query)
                return format_docs(docs)
            
            retrieval_chain = RunnablePassthrough() | {
                "context": get_context,
                "question": lambda x: x
            } | self.qa_template | llm | StrOutputParser()
            
            return retrieval_chain
            
        except Exception as e:
            logger.error(f"Error setting up chain: {e}")
            raise

    def handle_chat_input(self, prompt: str):
        try:
            with st.spinner("Generating response..."):
                result = st.session_state.state.chain.invoke(prompt)
                
                self._update_chat_history(prompt, result)
                
                with st.chat_message("assistant"):
                    st.markdown(result)
                    
        except Exception as e:
            logger.error(f"Error handling chat input: {e}")
            with st.chat_message("assistant"):
                error_msg = str(e)
                is_french = prompt[:2].lower() == "fr"
                st.error(
                    f"Une erreur s'est produite : {error_msg}" if is_french
                    else f"An error occurred: {error_msg}"
                )

    def _update_chat_history(self, prompt: str, response: str):
        st.session_state.state.chat_history.extend([
            ChatMessage(role="user", content=prompt),
            ChatMessage(role="assistant", content=response)
        ])

    def render_sidebar(self):
        with st.sidebar:
            st.header("Model Selection")
            
            selected_model_name = st.selectbox(
                "Select Language Model",
                options=list(AVAILABLE_MODELS.keys()),
                format_func=lambda x: f"{x}",
                index=list(AVAILABLE_MODELS.keys()).index(
                    next(k for k, v in AVAILABLE_MODELS.items() 
                         if v == st.session_state.state.selected_model)
                )
            )
            
            new_model = AVAILABLE_MODELS[selected_model_name]
            if new_model != st.session_state.state.selected_model:
                st.session_state.state.selected_model = new_model
                if st.session_state.state.texts is not None:
                    st.warning("Model changed. Please reprocess input.")
                    st.session_state.state.chain = None
            
            st.header("Input Processing")
            
            input_type = st.selectbox(
                "Select input type",
                ["File Upload", "URL", "Text Input"]
            )
            
            input_source = self._get_input_source(input_type)
            
            semantic_weight = st.slider(
                "Semantic Search Weight",
                0.0, 1.0, 0.7, 0.1,
                help="Adjust the balance between semantic and keyword search"
            )
            
            if st.button("Process Input"):
                self._process_input(input_source, semantic_weight)

    def _get_input_source(self, input_type: str) -> Any:
        if input_type == "File Upload":
            return st.file_uploader(
                "Choose a file",
                type=["txt", "pdf", "docx", "csv", "py", "js", "html", "css"]
            )
        elif input_type == "URL":
            return st.text_input("Enter URL")
        else:
            return st.text_area("Enter text")

    def _process_input(self, input_source: Any, semantic_weight: float):
        with st.spinner("Processing input..."):
            try:
                state = st.session_state.state
                state.processing_error = None
                
                with st.status("Processing document...") as status:
                    status.update(label="Loading document...")
                    state.texts = self.load_and_process_input(input_source)
                    
                    status.update(label="Initializing vector store...")
                    state.vector_store = self.initialize_vector_store(state.texts)
                    
                    status.update(label="Initializing language model...")
                    state.llm = SessionState.initialize_llm(state.selected_model)
                    
                    status.update(label="Setting up retriever...")
                    hybrid_retriever = self.setup_hybrid_retriever(
                        state.vector_store,
                        state.texts,
                        semantic_weight
                    )
                    
                    status.update(label="Finalizing setup...")
                    state.chain = self.setup_chain(hybrid_retriever, state.llm)
                    
                    status.update(label="Complete!", state="complete")
                
                st.success("Input processed successfully!")
                
            except Exception as e:
                logger.error(f"Error processing input: {str(e)}")
                st.session_state.state.processing_error = str(e)
                st.error(f"Error processing input: {str(e)}")
                raise

    def render_chat_interface(self):
        st.title("Multilingual RAG Chatbot")
        
        if st.session_state.state.chain is not None:
            st.success("System is ready for chat!")
            
            for message in st.session_state.state.chat_history:
                with st.chat_message(message.role):
                    st.markdown(message.content)
            
            prompt = st.chat_input("Ask a question in English or French:")
            if prompt:
                with st.chat_message("user"):
                    st.markdown(prompt)
                self.handle_chat_input(prompt)
        else:
            st.info("Please process an input document using the sidebar controls to start chatting.")

    def run(self):
        try:
            self.render_sidebar()
            self.render_chat_interface()
        except Exception as e:
            logger.error(f"Error running chatbot: {e}")
            st.error(
                "An unexpected error occurred. Please refresh the page and try again."
            )

def main():
    try:
        st.set_page_config(
            page_title="Multilingual RAG Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        chatbot = RAGChatbot()
        chatbot.run()
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        st.error(
            "A fatal error occurred while starting the application. "
            "Please check the logs and try again."
        )

if __name__ == "__main__":
    main()