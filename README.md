# Secure-Offline-RAG-System
This restructured version of the code is organized into multiple files with clear separation of concerns. Here's a breakdown of the structure:

src/models/:

embeddings.py: Contains the NomicEmbeddings class
session_state.py: Defines the SessionState dataclass


src/services/:

text_processing.py: Handles text splitting and processing
retriever_service.py: Manages retriever setup and configuration
chain_service.py: Handles conversation chain setup


src/ui/:

sidebar.py: Manages sidebar UI components
chat_interface.py: Handles chat interface rendering


src/utils/:

logging_config.py: Logging configuration
decorators.py: Contains utility decorators


src/app.py: Main application file