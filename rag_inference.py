import pandas as pd
from typing import List, Dict
import logging
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
import csv
from langchain_core.prompts import PromptTemplate
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
from pydantic import BaseModel, Field
from langchain.embeddings.base import Embeddings

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
            trust_remote_code=True
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [f"search_document: {text}" for text in texts]
        embeddings = self._model.encode(
            texts,
            show_progress_bar=self.show_progress,
            batch_size=32
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        text = f"search_query: {text}"
        embedding = self._model.encode(
            [text],
            show_progress_bar=False
        )
        return embedding[0].tolist()

@dataclass
class InferenceConfig:
    """Configuration for inference"""
    train_file: str
    test_file: str
    output_file: str
    faiss_weight: float = 0.5
    batch_size: int = 32

class RAGInference:
    """Class to handle batch inference using the RAG system"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.chain = None
        self.qa_prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always cite specific parts of the context that support your answer.

        Context: {context}

        Question: {question}
        Answer: Let me help you with that based on the provided context."""

    def process_text(self, text: str) -> List[str]:
        """Process input text into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return text_splitter.split_text(text)

    def initialize_system(self):
        """Initialize the RAG system with training data"""
        logger.info("Loading training data...")
        train_df = pd.read_csv(self.config.train_file)
        
        # Combine all training text into a single document
        training_text = ""
        for _, row in train_df.iterrows():
            training_text += f"Question: {row['Query']}\nAnswer: {row['Response']}\n\n"

        # Process text into chunks
        logger.info("Processing training data...")
        texts = self.process_text(training_text)
        
        # Initialize embeddings and vector store
        logger.info("Initializing embeddings and vector store...")
        embeddings = NomicEmbeddings()
        vector_store = FAISS.from_texts(texts, embeddings)
        
        # Initialize BM25
        corpus = [text.split() for text in texts]
        bm25 = BM25Okapi(corpus)
        
        # Setup retrievers
        faiss_retriever = vector_store.as_retriever(
            search_kwargs={"k": 5, "fetch_k": 10}
        )
        bm25_retriever = BM25Retriever.from_texts(texts)
        bm25_retriever.k = 5

        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[self.config.faiss_weight, 1 - self.config.faiss_weight]
        )
        
        # Setup embeddings filter
        embeddings_filter = EmbeddingsFilter(
            embeddings=embeddings,
            similarity_threshold=0.5
        )
        
        # Create compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=ensemble_retriever
        )

        # Initialize LLM
        logger.info("Initializing LLM...")
        llm = Ollama(
            model="qwen2.5:3b",
            temperature=0.7
        )

        # Setup conversation chain
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=compression_retriever,
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

    def generate_responses(self) -> List[Dict]:
        """Generate responses for test queries"""
        if self.chain is None:
            self.initialize_system()

        logger.info("Loading test data...")
        test_df = pd.read_csv(self.config.test_file)
        results = []

        logger.info("Generating responses...")
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            try:
                result = self.chain({"question": row['Query']})
                results.append({
                    'trustii_id': row['trustii_id'],
                    'Query': row['Query'],
                    'Response': result['answer']
                })
            except Exception as e:
                logger.error(f"Error processing query {row['trustii_id']}: {str(e)}")
                results.append({
                    'trustii_id': row['trustii_id'],
                    'Query': row['Query'],
                    'Response': "An error occurred while generating the response."
                })

        return results

    def save_results(self, results: List[Dict]):
        """Save results to CSV file"""
        logger.info(f"Saving results to {self.config.output_file}")
        
        with open(self.config.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['trustii_id', 'Query', 'Response'])
            writer.writeheader()
            writer.writerows(results)

def main():
    config = InferenceConfig(
        train_file="train.csv",
        test_file="test.csv",
        output_file="submission.csv",
        faiss_weight=0.5
    )

    inference = RAGInference(config)
    results = inference.generate_responses()
    inference.save_results(results)
    logger.info("Inference completed successfully!")

if __name__ == "__main__":
    main()