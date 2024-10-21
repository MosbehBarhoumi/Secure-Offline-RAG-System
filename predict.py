import pandas as pd
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.prompts import PromptTemplate
from tqdm import tqdm


CONTEXT = """
Matter is a communication standard for smart home devices by the Connectivity Standards Alliance (CSA). 
"""

ANSWER_GUIDELINES = """
When answering, follow these rules:

1. Keep the response concise and within **1-3 sentences**. 
2. Avoid redundant details; answer the question with the most relevant information only.
3. Use vocabulary similar to the retrieved examples; aim for simple and direct language.
4. Only provide **specific details** if they are explicitly mentioned in the question.
5. Maintain the tone and style of the retrieved answers, mirroring brevity and clarity.
6. If the question requires a detailed process, outline it **in a few short steps**.
7. Refrain from adding new or unrelated information; focus only on answering the core question.

"""

# Load and process the CSV file
def load_and_process_csv(file_path, is_test_data=False):
    df = pd.read_csv(file_path)
    if is_test_data:
        df = df.dropna(subset=['Query'])
    else:
        df = df.dropna(subset=['Query', 'Response'])
    df = df.reset_index(drop=True)
    return df

# Initialize embeddings and vector store
def initialize_vector_store(df):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    texts = df['Query'].tolist()
    metadatas = df.to_dict('records')
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vector_store, embeddings

# Initialize the Ollama model
def initialize_llm():
    return Ollama(model="qwen2.5:3b")

def setup_chain(vector_store, llm, embeddings):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
    retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=base_retriever)
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""Context: {{context}}

Question: {{question}}

{ANSWER_GUIDELINES}

Please answer the question based on the given context and your knowledge of the Matter protocol. If the context doesn't provide sufficient information, use your own knowledge to provide the best possible answer."""
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        return_generated_question=True,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )
    return chain



# Function to check if retrieved documents are relevant
def are_documents_relevant(docs):
    return len(docs) > 0

def generate_response(chain, llm, query):
    full_query = f"""Context: {CONTEXT}

Question: {query}

{ANSWER_GUIDELINES}

Please answer the question based on the given context and your knowledge of the Matter protocol. If the context doesn't provide sufficient information, use your own knowledge to provide the best possible answer."""
    
    result = chain({"question": full_query})
    
    if are_documents_relevant(result['source_documents']):
        return result['answer']
    else:
        return llm(full_query)

# Main function to run the prediction process
def main():
    # Load and process training data
    print("Loading and processing training data...")
    train_df = load_and_process_csv('train.csv', is_test_data=False)
    print(f"Training data loaded. Shape: {train_df.shape}")

    # Initialize vector store and LLM
    print("Initializing vector store and language model...")
    vector_store, embeddings = initialize_vector_store(train_df)
    llm = initialize_llm()
    chain = setup_chain(vector_store, llm, embeddings)
    print("Initialization complete.")

    # Load test data
    print("Loading test data...")
    test_df = load_and_process_csv('test.csv', is_test_data=True)
    print(f"Test data loaded. Shape: {test_df.shape}")

    # Generate predictions
    print("Generating predictions...")
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating responses"):
        response = generate_response(chain, llm, row['Query'])
        results.append({
            'trustii_id': row['trustii_id'],
            'Query': row['Query'],
            'Response': response
        })

    # Create submission CSV
    submission_df = pd.DataFrame(results)
    submission_df.to_csv('submission.csv', index=False)
    print("Submission CSV created successfully.")

if __name__ == "__main__":
    main()