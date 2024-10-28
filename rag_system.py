import streamlit as st
import os
from dataclasses import dataclass
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from rank_bm25 import BM25Okapi
from langchain.embeddings.base import Embeddings
from document_processor import DocumentProcessor
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
AVAILABLE_MODELS = {
    "Qwen 3B": "qwen2.5:3b",
    "Qwen 7B": "qwen2.5:7b",
    "Qwen 14B": "qwen2.5:14b"
}

class MultiBertEmbeddings(BaseModel, Embeddings):
    model_name: str = Field(default=EMBEDDING_MODEL)
    show_progress: bool = Field(default=False)
    _model: SentenceTransformer = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = SentenceTransformer(self.model_name, device='cpu')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=self.show_progress, 
                                      batch_size=16, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode([text], show_progress_bar=False, 
                                     normalize_embeddings=True)
        return embedding[0].tolist()

class SessionState:
    def __init__(self):
        self.chat_history: List[Dict[str, str]] = []
        self.vector_store: Optional[FAISS] = None
        self.chain: Optional[ConversationalRetrievalChain] = None
        self.embeddings: Optional[MultiBertEmbeddings] = None
        self.llm: Optional[Ollama] = None
        self.texts: Optional[List[str]] = None
        self.bm25: Optional[BM25Okapi] = None
        self.selected_model: str = "qwen2.5:3b"

    @staticmethod
    def initialize_llm(model_name: str) -> Ollama:
        return Ollama(model=model_name, temperature=0.7, top_p=0.9, 
                     num_ctx=2048, repeat_penalty=1.1)

class RAGChatbot:
    def __init__(self):
        self.initialize_session_state()
        self.document_processor = DocumentProcessor()
        self.setup_prompts()

    @staticmethod
    def initialize_session_state():
        if 'state' not in st.session_state:
            st.session_state.state = SessionState()

    def setup_prompts(self):
        self.qa_prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always cite specific parts of the context that support your answer.

        IMPORTANT: If the user's question is in French, you must respond in French.
        If the user's question is in English, you must respond in English.
        
        When responding in French:
        - Use proper French grammar and vocabulary
        - Include French punctuation (e.g., « guillemets » for quotes)
        - Maintain a formal tone using "vous"
        - Translate any technical terms appropriately
        
        When responding in English:
        - Use proper English grammar and vocabulary
        - Follow English punctuation rules
        - Maintain a professional tone
        - Use standard technical terminology

        For code-related questions in either language:
        - Include code examples with explanations in the same language as the question
        - Keep code comments and output messages in the question's language

        Context: {context}

        Question: {question}
        Answer: Let me help you with that based on the provided context."""

    def load_and_process_input(self, input_source) -> List[str]:
        if not input_source:
            return []
        temp_file_path = self.document_processor.process_input(input_source)
        with open(temp_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""])
        return text_splitter.split_text(text)

    def initialize_vector_store(self, texts: List[str]) -> FAISS:
        embeddings = MultiBertEmbeddings()
        st.session_state.state.embeddings = embeddings
        vector_store = None
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embeddings)
            else:
                vector_store.add_texts(batch)
        return vector_store

    def setup_hybrid_retriever(self, vector_store: FAISS, texts: List[str], 
                             semantic_weight: float) -> ContextualCompressionRetriever:
        faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        bm25_retriever = BM25Retriever.from_texts(texts)
        bm25_retriever.k = 4
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[semantic_weight, 1 - semantic_weight]
        )
        embeddings_filter = EmbeddingsFilter(
            embeddings=st.session_state.state.embeddings,
            similarity_threshold=0.6
        )
        return ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=ensemble_retriever
        )

    def setup_chain(self, hybrid_retriever: ContextualCompressionRetriever, 
                   llm: Ollama) -> ConversationalRetrievalChain:
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
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=self.qa_prompt_template,
                    input_variables=["context", "question"]
                )
            }
        )

    def handle_chat_input(self, prompt: str):
        try:
            with st.spinner("Generating response..."):
                result = st.session_state.state.chain({"question": prompt})
                self._update_chat_history(prompt, result['answer'])
                with st.chat_message("assistant"):
                    st.markdown(result['answer'])
        except Exception as e:
            with st.chat_message("assistant"):
                error_msg = str(e)
                if "fr" in prompt[:2].lower():  # Simple check for French
                    st.error(f"Une erreur s'est produite : {error_msg}")
                else:
                    st.error(f"An error occurred: {error_msg}")

    def _update_chat_history(self, prompt: str, response: str):
        st.session_state.state.chat_history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])

    def _get_input_source(self, input_type: str):
        if input_type == "File Upload":
            return st.file_uploader("Choose a file", 
                                  type=["txt", "pdf", "docx", "csv", "py", "js", "html", "css"])
        elif input_type == "URL":
            return st.text_input("Enter URL")
        else:
            return st.text_area("Enter text")

    def _process_input(self, input_source, semantic_weight: float):
        with st.spinner("Processing input..."):
            try:
                state = st.session_state.state
                state.texts = self.load_and_process_input(input_source)
                state.vector_store = self.initialize_vector_store(state.texts)
                state.bm25 = self.initialize_bm25(state.texts)
                state.llm = SessionState.initialize_llm(state.selected_model)
                hybrid_retriever = self.setup_hybrid_retriever(
                    state.vector_store,
                    state.texts,
                    semantic_weight
                )
                state.chain = self.setup_chain(hybrid_retriever, state.llm)
                st.success("Input processed successfully!")
            except Exception as e:
                st.error(f"Error processing input: {str(e)}")

    @staticmethod
    def initialize_bm25(texts: List[str]) -> BM25Okapi:
        corpus = [text.split() for text in texts]
        return BM25Okapi(corpus)

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
                0.0, 1.0, 0.7, 0.1
            )
            
            if st.button("Process Input"):
                self._process_input(input_source, semantic_weight)

    def render_chat_interface(self):
        st.title("Multilingual RAG Chatbot")
        if st.session_state.state.chain is not None:
            for message in st.session_state.state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if prompt := st.chat_input("Ask a question in English or French:"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                self.handle_chat_input(prompt)
        else:
            st.info("Please process an input document to start chatting.")

    def run(self):
        self.render_sidebar()
        self.render_chat_interface()

def main():
    chatbot = RAGChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()