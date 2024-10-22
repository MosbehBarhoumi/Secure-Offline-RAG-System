import streamlit as st
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
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
from document_processor import DocumentProcessor

@dataclass
class SessionState:
    """Class to manage Streamlit session state variables."""
    chat_history: List[dict]
    vector_store: Optional[FAISS]
    chain: Optional[ConversationalRetrievalChain]
    embeddings: Optional[HuggingFaceEmbeddings]
    llm: Optional[Ollama]
    texts: Optional[List[str]]
    bm25: Optional[BM25Okapi]

class RAGChatbot:
    """Main class for RAG Chatbot implementation."""
    
    def __init__(self):
        self.initialize_session_state()
        self.document_processor = DocumentProcessor()
        self.prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""

    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables."""
        if 'state' not in st.session_state:
            st.session_state.state = SessionState(
                chat_history=[],
                vector_store=None,
                chain=None,
                embeddings=None,
                llm=None,
                texts=None,
                bm25=None
            )

    def load_and_process_input(self, input_source) -> List[str]:
        """Process input and split into chunks."""
        temp_file_path = self.document_processor.process_input(input_source)
        
        with open(temp_file_path, 'r') as file:
            text = file.read()
        
        # Using RecursiveCharacterTextSplitter for better chunk handling
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_text(text)

    def initialize_vector_store(self, texts: List[str]) -> FAISS:
        """Initialize embeddings and vector store."""
        # Force CPU usage for embeddings
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        st.session_state.state.embeddings = embeddings
        return FAISS.from_texts(texts, embeddings)

    @staticmethod
    def initialize_bm25(texts: List[str]) -> BM25Okapi:
        """Initialize BM25 retriever."""
        corpus = [text.split() for text in texts]
        return BM25Okapi(corpus)

    @staticmethod
    @st.cache_resource
    def initialize_llm() -> Ollama:
        """Initialize the Ollama model."""
        return Ollama(model="qwen2.5:3b")

    def setup_hybrid_retriever(self, vector_store: FAISS, texts: List[str], faiss_weight: float) -> ContextualCompressionRetriever:
        """Set up hybrid retriever with FAISS and BM25."""
        bm25_weight = 1 - faiss_weight
        faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        bm25_retriever = BM25Retriever.from_texts(texts)
        bm25_retriever.k = 5

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
                    template=self.prompt_template, 
                    input_variables=["context", "question"]
                )
            }
        )

    def handle_chat_input(self, prompt: str):
        """Process chat input and generate response."""
        st.session_state.state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                result = st.session_state.state.chain({"question": prompt})
                
                # Log retrieved context for debugging
                for i, doc in enumerate(result['source_documents']):
                    print(f"Document {i + 1}:")
                    print(doc.page_content)
                    print("---")
                
                message_placeholder.markdown(result['answer'])
                
                st.session_state.state.chat_history.append(
                    {"role": "assistant", "content": result['answer']}
                )
            except Exception as e:
                error_message = f"Error processing request: {str(e)}"
                message_placeholder.error(error_message)
                print(f"Error details: {e}")

    def render_sidebar(self):
        """Render sidebar with input processing and weight adjustment options."""
        with st.sidebar:
            st.header("Input Processing")
            input_type = st.selectbox("Select input type", ["File Upload", "URL", "Text Input"])
            
            input_source = None
            if input_type == "File Upload":
                input_source = st.file_uploader("Choose a file", type=["txt", "pdf", "docx", "csv"])
            elif input_type == "URL":
                input_source = st.text_input("Enter URL")
            else:
                input_source = st.text_area("Enter text")
            
            if st.button("Process Input"):
                with st.spinner("Processing input..."):
                    try:
                        state = st.session_state.state
                        state.texts = self.load_and_process_input(input_source)
                        state.vector_store = self.initialize_vector_store(state.texts)
                        state.bm25 = self.initialize_bm25(state.texts)
                        state.llm = self.initialize_llm()
                        
                        # Initialize chain with default weights
                        hybrid_retriever = self.setup_hybrid_retriever(
                            state.vector_store,
                            state.texts,
                            0.5  # Default FAISS weight
                        )
                        state.chain = self.setup_chain(hybrid_retriever, state.llm)
                        
                        st.success("Input processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing input: {str(e)}")
                        print(f"Error details: {e}")

            st.header("Retrieval Weights")
            faiss_weight = st.slider("Semantic Search (FAISS) Weight", 0.0, 1.0, 0.5, 0.01)
            st.write(f"Keyword Search (BM25) Weight: {1 - faiss_weight:.2f}")
            
            if st.button("Update Weights"):
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
                        print(f"Error details: {e}")
                else:
                    st.warning("Please process input before updating weights.")

    def render_chat_interface(self):
        """Render main chat interface."""
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
            st.info("Please process an input to start chatting.")

    def run(self):
        """Main method to run the Streamlit app."""
        self.render_sidebar()
        self.render_chat_interface()

def main():
    chatbot = RAGChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()