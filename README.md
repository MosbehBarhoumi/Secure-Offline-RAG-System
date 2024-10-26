# RAG Chatbot with Hybrid Search

A sophisticated Retrieval-Augmented Generation (RAG) chatbot implementation using Streamlit, combining semantic search (FAISS) and keyword-based search (BM25) for enhanced document retrieval and question answering.

## 🌟 Features

- **Hybrid Search System**
  - Semantic search using FAISS vector store
  - Keyword-based search using BM25
  - Adjustable weights between search methods
  - Contextual compression for better retrieval accuracy

- **Multiple Input Options**
  - File upload (supports txt, pdf, docx, csv)
  - URL input
  - Direct text input

- **Advanced Document Processing**
  - Intelligent text chunking
  - Document embedding using Nomic AI's embedding model
  - Efficient batch processing
  - Caching mechanisms for performance

- **Interactive UI**
  - Real-time chat interface
  - Debug information display
  - Performance metrics monitoring
  - Adjustable search weights

## 🔧 Technical Stack

- **Framework**: Streamlit
- **Language Models**: 
  - Qwen 3B (via Ollama)
  - Nomic AI embeddings (nomic-embed-text-v1.5)
- **Vector Store**: FAISS
- **Text Processing**: LangChain
- **Search Algorithms**: 
  - FAISS for semantic search
  - BM25 for keyword search
- **Memory**: ConversationBufferMemory for chat history

## 📋 Prerequisites

```bash
pip install streamlit langchain sentence-transformers faiss-cpu rank-bm25 ollama
```

Additionally, ensure you have Ollama installed and running with the Qwen 2.5 3B model.

## 🚀 Getting Started

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

## 💡 Usage

1. **Input Processing**
   - Select input type (File/URL/Text)
   - Upload or input your document
   - Click "Process Input" to initialize the system

2. **Adjust Search Weights**
   - Use the slider to balance between semantic and keyword search
   - Higher FAISS weight (>0.5) favors semantic search
   - Lower FAISS weight (<0.5) favors keyword search

3. **Chat Interface**
   - Start asking questions about your document
   - View source documents and retrieval metrics in debug mode

## ⚙️ Configuration

- **Chunk Size**: 500 characters (adjustable in code)
- **Chunk Overlap**: 50 characters
- **LLM Temperature**: 0.7
- **Top P**: 0.9
- **Similarity Threshold**: 0.5 for embeddings filter

## 🔍 Debug Mode

Enable "Show Debug Info" to view:
- Embedding time
- Retrieval time
- Number of processed chunks
- Source documents used for responses

## 🛠️ Architecture

The system uses a multi-stage pipeline:
1. Document processing and chunking
2. Embedding generation using Nomic AI
3. Dual-retrieval system (FAISS + BM25)
4. Contextual compression for relevance filtering
5. LLM-based response generation

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## ⚠️ Important Notes

- The system requires significant RAM for processing large documents
- Performance depends on the quality of input chunking and embedding
- Adjust chunk sizes and overlap based on your specific use case
- Debug mode may impact performance when enabled

## 📝 License

[Your License Here]

## 👥 Authors

[Your Name/Organization]