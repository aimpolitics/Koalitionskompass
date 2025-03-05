from pinecone_processor import PineconePDFProcessor
import logging
import os
from pinecone import Pinecone
import streamlit as st
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Create vector store from PDF document."""
    logger.info("Starting vector store creation with Pinecone...")
    
    # Debug Pinecone credentials
    api_key = PINECONE_API_KEY
    environment = PINECONE_ENVIRONMENT
    index_name = PINECONE_INDEX_NAME
    
    logger.info(f"API key present: {'Yes' if api_key else 'No'}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Index name: {index_name}")
    
    # Explicitly initialize Pinecone first
    try:
        logger.info("Setting up Pinecone client directly")
        pc = Pinecone(api_key=api_key)
        index_list = pc.list_indexes().names()
        logger.info(f"Available Pinecone indexes: {index_list}")
        
        if index_name not in index_list:
            logger.error(f"Index {index_name} not found in Pinecone")
            raise ValueError(f"Index {index_name} does not exist")
    except Exception as e:
        logger.error(f"Error initializing Pinecone directly: {str(e)}", exc_info=True)
        raise
    
    # Initialize Pinecone PDF processor
    pdf_processor = PineconePDFProcessor()
    
    try:
        # Process PDF and create vector store
        logger.info("Processing PDF and creating embeddings...")
        
        # Log PDF path
        logger.info(f"PDF path: {pdf_processor.pdf_path if hasattr(pdf_processor, 'pdf_path') else 'Not set directly'}")
        
        # Load and process PDF
        logger.info("Loading and processing PDF...")
        documents = pdf_processor.load_and_process_pdf()
        logger.info(f"PDF processed. Number of document chunks: {len(documents)}")
        
        # Create vector store from documents
        logger.info("Creating vector store from documents...")
        vector_store = pdf_processor.create_vector_store(documents)
        logger.info("Vector store created successfully in Pinecone")
        
        # Check document count
        from pinecone_processor import count_documents
        doc_count = count_documents()
        logger.info(f"Document count in Pinecone: {doc_count}")
        
        # Test the vector store
        logger.info("Testing vector store...")
        results = vector_store.similarity_search("test", k=1)
        logger.info(f"Vector store test results: {len(results)} results found")
        logger.info("Vector store is working correctly!")
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 