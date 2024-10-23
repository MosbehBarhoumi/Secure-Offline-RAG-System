# src/services/text_processing.py
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    """Handle text processing and chunking operations."""
    
    @staticmethod
    def split_text(text: str) -> List[str]:
        """Split text into chunks using RecursiveCharacterTextSplitter."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
            is_separator_regex=False
        )
        return text_splitter.split_text(text)