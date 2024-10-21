import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from document_processor import DocumentProcessor
from rank_bm25 import BM25Okapi
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate


# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

# Load and process the input
def load_and_process_input(input_source):
    processor = DocumentProcessor()
    temp_file_path = processor.process_input(input_source)
    
    with open(temp_file_path, 'r') as file:
        text = file.read()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    return texts

# Initialize embeddings and vector store
def initialize_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts, embeddings)
    st.session_state.embeddings = embeddings
    return vector_store

def initialize_bm25(texts):
    corpus = [text.split() for text in texts]
    return BM25Okapi(corpus)

# Initialize the Ollama model
@st.cache_resource
def initialize_llm():
    return Ollama(model="qwen2.5:3b")



# Function to check if retrieved documents are relevant
def are_documents_relevant(docs):
    return len(docs) > 0  # You might want to implement a more sophisticated check here



def setup_hybrid_retriever(vector_store, bm25, texts):
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.3, 0.7]  # Give more weight to BM25 for keyword matching
    )
    
    embeddings_filter = EmbeddingsFilter(embeddings=st.session_state.embeddings, similarity_threshold=0.5)  # Lower threshold
    hybrid_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=ensemble_retriever)
    
    return hybrid_retriever

def setup_chain(hybrid_retriever, llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=hybrid_retriever,
        memory=memory,
        return_source_documents=True,
        return_generated_question=True,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
    )
    return chain

# Main Streamlit app
def main():
    st.title("RAG Chatbot with Hybrid Search")

    # Sidebar for input processing
    with st.sidebar:
        st.header("Input Processing")
        input_type = st.selectbox("Select input type", ["File Upload", "URL", "Text Input"])
        
        if input_type == "File Upload":
            uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx", "csv"])
            input_source = uploaded_file
        elif input_type == "URL":
            input_source = st.text_input("Enter URL")
        else:
            input_source = st.text_area("Enter text")
        
        if st.button("Process Input"):
            with st.spinner("Processing input..."):
                texts = load_and_process_input(input_source)
                st.session_state.vector_store = initialize_vector_store(texts)
                st.session_state.bm25 = initialize_bm25(texts)
                st.session_state.llm = initialize_llm()
                hybrid_retriever = setup_hybrid_retriever(st.session_state.vector_store, st.session_state.bm25, texts)
                st.session_state.chain = setup_chain(hybrid_retriever, st.session_state.llm)
            st.success("Input processed successfully!")

    # Main chat interface
    if st.session_state.chain is not None:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("You:"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                # Get the response from the chain
                result = st.session_state.chain({"question": prompt})
                
                # Print retrieved context
                print("Retrieved context:")
                for i, doc in enumerate(result['source_documents']):
                    print(f"Document {i + 1}:")
                    print(doc.page_content)
                    print("---")
                
                full_response = result['answer']
                message_placeholder.markdown(full_response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    else:
        st.info("Please process an input to start chatting.")

if __name__ == "__main__":
    main()