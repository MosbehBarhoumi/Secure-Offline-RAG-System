# src/services/retriever_service.py
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.embeddings.base import Embeddings

class RetrieverService:
    """Handle setup and configuration of retrievers."""
    
    @staticmethod
    def setup_hybrid_retriever(
        vector_store: FAISS,
        texts: List[str],
        embeddings: Embeddings,
        faiss_weight: float = 0.5,
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
            embeddings=embeddings,
            similarity_threshold=0.5
        )
        
        return ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=ensemble_retriever
        )