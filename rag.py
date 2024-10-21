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

# Load and process the text file
def load_and_process_text(file):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(file.getvalue().decode())
    return texts

# Initialize embeddings and vector store
def initialize_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts, embeddings)
    st.session_state.embeddings = embeddings
    return vector_store

# Initialize the Ollama model
@st.cache_resource
def initialize_llm():
    return Ollama(model="qwen2.5:3b")

# Set up the conversational chain with relevance checking
def setup_chain(vector_store, llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'  # Specify the output key
    )
    
    # Create a retriever with relevance filtering
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    embeddings_filter = EmbeddingsFilter(embeddings=st.session_state.embeddings, similarity_threshold=0.7)
    retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=base_retriever)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        return_generated_question=True
    )
    return chain

# Function to check if retrieved documents are relevant
def are_documents_relevant(docs):
    return len(docs) > 0  # You might want to implement a more sophisticated check here

# Main Streamlit app
def main():
    st.title("RAG Chatbot with Relevance Checking")

    # Sidebar for file upload and processing
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Choose a text file", type="txt")
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    texts = load_and_process_text(uploaded_file)
                    st.session_state.vector_store = initialize_vector_store(texts)
                    st.session_state.llm = initialize_llm()
                    st.session_state.chain = setup_chain(st.session_state.vector_store, st.session_state.llm)
                st.success("Document processed successfully!")

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
                
                # Check if retrieved documents are relevant
                if are_documents_relevant(result['source_documents']):
                    full_response = result['answer']
                else:
                    # If not relevant, just use the user's query without context
                    full_response = st.session_state.llm(prompt)

                message_placeholder.markdown(full_response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    else:
        st.info("Please upload and process a document to start chatting.")

if __name__ == "__main__":
    main()