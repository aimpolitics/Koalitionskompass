from pinecone_processor import PineconePDFProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Create vector store from PDF document."""
    logger.info("Starting vector store creation with Pinecone...")
    
    # Initialize Pinecone PDF processor
    pdf_processor = PineconePDFProcessor()
    
    try:
        # Process PDF and create vector store
        logger.info("Processing PDF and creating embeddings...")
        vector_store = pdf_processor.process_pdf()
        logger.info("Vector store created successfully in Pinecone")
        
        # Test the vector store
        logger.info("Testing vector store...")
        results = vector_store.similarity_search("test", k=1)
        logger.info("Vector store is working correctly!")
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

if __name__ == "__main__":
    main() 