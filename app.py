import streamlit as st
import os
import tempfile
from document_processor import DocumentProcessor
from chatbot import ChatBot
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="",
        layout="wide"
    )
    
    st.title("RAG Chatbot")
    st.markdown("Upload a document and ask questions about it")
    
    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, Word document, or text file"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Process document
                        processor = DocumentProcessor()
                        chunks = processor.process_document(tmp_file_path)
                        # Add this after: chunks = processor.process_document(tmp_file_path)
                        st.write(f"Debug: Found {len(chunks)} chunks")
                        st.write(f"First chunk preview: {chunks[0][:200]}...")
                        
                        # Initialize chatbot with processed chunks
                        st.session_state.chatbot = ChatBot(chunks)
                        st.session_state.document_processed = True
                        st.session_state.messages = []  # Clear previous messages
                        
                        # Clean up temp file
                        os.unlink(tmp_file_path)
                        
                        st.success(f"Document processed! Found {len(chunks)} text chunks.")
                        
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        logger.error(f"Document processing error: {e}")
    
    # Main chat interface
    if st.session_state.document_processed and st.session_state.chatbot:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Wait for response pleasee"):
                    response = st.session_state.chatbot.get_response(prompt)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    else:
        st.info("Please upload a document in the sidebar to start chatting!")

if __name__ == "__main__":
    main()
