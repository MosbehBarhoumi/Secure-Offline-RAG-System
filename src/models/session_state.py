# src/models/session_state.py
from dataclasses import dataclass
from typing import List, Optional, Dict
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from rank_bm25 import BM25Okapi
from .embeddings import NomicEmbeddings
from langchain_community.llms import Ollama

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