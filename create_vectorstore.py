from pinecone_processor import PineconePDFProcessor, get_pinecone_instance, count_documents
import logging
import os
from pinecone import Pinecone
import streamlit as st
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Create vector store from PDF document using Pinecone's integrated embedding."""
    logger.info("Starting vector store creation with Pinecone integrated embedding...")
    
    # Debug Pinecone credentials
    api_key = PINECONE_API_KEY
    environment = PINECONE_ENVIRONMENT
    index_name = PINECONE_INDEX_NAME
    
    logger.info(f"API key present: {'Yes' if api_key else 'No'}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Index name: {index_name}")
    
    # Use the singleton pattern to initialize Pinecone
    try:
        logger.info("Setting up Pinecone client")
        pc = get_pinecone_instance()
        index_list = pc.list_indexes().names()
        logger.info(f"Available Pinecone indexes: {index_list}")
        
        if index_name not in index_list:
            logger.error(f"Index {index_name} not found in Pinecone")
            print(f"""
            The index '{index_name}' does not exist. You need to create an index configured for integrated embedding.
            
            Example code to create a suitable index:
            
            from pinecone import Pinecone
            
            pc = Pinecone(api_key="{api_key}")
            
            # Create an index specifically configured for a hosted embedding model
            pc.create_index_for_model(
                name="{index_name}",
                cloud="aws",
                region="us-east-1",
                embed={{
                    "model": "multilingual-e5-large",  # Choose appropriate model
                    "field_map": {{"text": "page_content"}}
                }}
            )
            """)
            raise ValueError(f"Index {index_name} does not exist")
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}", exc_info=True)
        raise
    
    # Initialize Pinecone processor
    pdf_processor = PineconePDFProcessor()
    
    try:
        logger.info("Processing PDF and creating vector store")
        vector_store = pdf_processor.process_pdf()
        logger.info("Vector store created successfully!")
        
        # Count documents
        count = count_documents()
        logger.info(f"Vector store contains {count} vectors")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
        raise
    
if __name__ == "__main__":
    main() 