# src/models/embeddings.py
from typing import List
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from langchain.embeddings.base import Embeddings

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