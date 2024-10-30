# Multilingual RAG Chatbot

A powerful multilingual Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, supporting both English and French conversations. The system combines semantic search and keyword-based retrieval to provide accurate responses based on your documents.

## Features

- 🌐 **Multilingual Support**
  - Seamlessly handles both English and French queries
  - Automatic language detection and response matching

- 📑 **Flexible Document Input**
  - File Upload (supports txt, pdf, docx, csv, py, js, html, css)
  - URL import
  - Direct text input

- 🔍 **Advanced Retrieval System**
  - Hybrid retrieval combining semantic search and BM25
  - Adjustable semantic search weight
  - Contextual compression for better result relevance

- 🤖 **Multiple LLM Options**
  - Qwen 3B
  - Qwen 7B
  - Qwen 14B

- 💼 **Professional Features**
  - Progress tracking for document processing
  - Error handling and logging
  - Clean and intuitive user interface
  - Real-time chat history
  - Customizable retrieval parameters

## Prerequisites

- Python 3.8+
- Ollama (for running local LLMs)
- 8GB+ RAM recommended
- GPU optional but recommended for better performance

## Installation

1. **Install Ollama**

   For Ubuntu:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
   
   For other Linux distributions, check the installation instructions at:
   https://ollama.com/download/linux

2. **Pull the Qwen Model**
   ```bash
   ollama pull qwen2.5
   ```

3. **Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd <repository-name>
   ```

4. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start Ollama Server**
   ```bash
   ollama serve
   ```

2. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the Interface**
   - Open your browser and navigate to `http://localhost:8501`
   - The sidebar will appear with configuration options

4. **Process Documents**
   - Choose your input type (File/URL/Text)
   - Upload or input your document
   - Adjust the semantic search weight if desired
   - Click "Process Input"

5. **Start Chatting**
   - Once processing is complete, you can start asking questions
   - Type in English or French - the system will respond in the same language

## Configuration

Key parameters can be adjusted in the code:

```python
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_RETRIEVER_K = 4
```

## Models

The system uses the following components:
- **Embedding Model**: intfloat/multilingual-e5-small
- **LLM Options**: 
  - Qwen 3B (Fastest)
  - Qwen 7B (Balanced)
  - Qwen 14B (Most capable)

## Error Handling

The system includes comprehensive error handling:
- Input processing errors
- Model initialization failures
- Runtime errors
- All errors are logged and displayed user-friendly messages

## Performance Tips

1. **Memory Usage**
   - Start with smaller documents when testing
   - Monitor RAM usage with larger documents

2. **Response Speed**
   - Adjust chunk size based on your needs
   - Use Qwen 3B for faster responses
   - Consider reducing retriever_k for quicker retrieval

3. **Accuracy**
   - Increase semantic_weight for conceptual questions
   - Decrease semantic_weight for factual queries
   - Adjust similarity_threshold based on your needs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your License Here]

## Acknowledgments

- Qwen by Alibaba
- Streamlit
- LangChain
- FAISS
- Sentence-Transformers

## Support

For issues and questions:
1. Check the logs (logging is enabled by default)
2. Create an issue in the repository
3. Contact the maintainer

---
**Note**: This project is for educational and research purposes. Please ensure you comply with all relevant licenses and terms of service.