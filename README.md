# Document Analysis Chatbot

A simple RAG (Retrieval-Augmented Generation) based chatbot that analyzes uploaded documents and answers questions about their content in real-time.

## Features

- **Document Upload**: Supports PDF, Word (.docx), and text files
- **Real-time Processing**: Instant document analysis and indexing
- **Context-aware Responses**: Uses RAG to provide accurate answers based on document content
- **Clean UI**: Simple and intuitive Streamlit interface
- **Local Processing**: Everything runs locally, no external API calls needed

## Technologies Used

- **Streamlit**: Web interface
- **LangChain**: Text processing and chunking
- **Sentence Transformers**: Document embeddings
- **FAISS**: Vector similarity search
- **Transformers**: Question-answering model
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing

## Installation & Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd document-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and go to `http://localhost:8501`

## How to Use

1. **Upload Document**: Use the sidebar to upload a PDF, Word, or text file
2. **Process**: Click "Process Document" to analyze the content
3. **Ask Questions**: Use the chat interface to ask questions about your document
4. **Get Answers**: The chatbot will provide context-aware responses

## Project Structure

```
document-chatbot/
├── app.py                 # Main Streamlit application
├── chatbot.py            # ChatBot class with RAG implementation
├── document_processor.py # Document processing utilities
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Model Information

- **Embedding Model**: all-MiniLM-L6-v2 (lightweight and fast)
- **QA Model**: distilbert-base-cased-distilled-squad (optimized for question-answering)

Both models are chosen for being lightweight while maintaining good performance for local execution.

## Limitations

- Works best with text-heavy documents
- Performance depends on document quality and question clarity
- Requires decent RAM for larger documents (>10MB)

## Troubleshooting

1. **Out of Memory**: Try smaller documents or reduce chunk_size in document_processor.py
2. **Slow Performance**: The first run downloads models, subsequent runs are faster
3. **Poor Answers**: Try rephrasing questions or ensure the information exists in the document
### Common Issues:
4. **ImportError with huggingface_hub or sentence_transformers**:
   ```bash
   pip install --upgrade huggingface_hub sentence-transformers transformers

## Recommended Python Version:

-Python 3.9 - 3.11 (Python 3.10 or 3.11 recommended)
-Avoid Python 3.12+ as some dependencies may not be fully compatible yet
