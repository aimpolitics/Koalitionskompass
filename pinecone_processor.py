from typing import List, Optional, Dict, Any
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from config import PDF_PATH, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, PINECONE_NAMESPACE
from text_processor import TextProcessor
from pinecone import Pinecone, PodSpec
import os
import logging
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use configuration from config.py, which now properly prioritizes Streamlit secrets
pinecone_api_key = PINECONE_API_KEY
pinecone_environment = PINECONE_ENVIRONMENT

# Log configuration status (without exposing API keys)
logger.info(f"Pinecone configuration from config.py: API Key: {'Found' if pinecone_api_key else 'Not found'}, Environment: {pinecone_environment}")

# Global singleton instances
_pinecone_instance = None
_vector_store_instance = None

# Create a passthrough embedding class for use with integrated embedding
class PassthroughEmbeddings(Embeddings):
    """A dummy embeddings class for use with Pinecone integrated embedding.
    This class implements the Embeddings interface but doesn't actually compute embeddings,
    as Pinecone will handle the embedding process internally."""
    
    def __init__(self, dimension: int = 1024):
        """Initialize with the dimension matching your Pinecone index.
        
        Args:
            dimension: The dimension of your Pinecone index. Default is 1024 for
                       multilingual-e5-large model used in integrated embedding.
        """
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """This method is not used with Pinecone integrated embedding."""
        # Return placeholder values - these aren't used since Pinecone does the embedding
        return [[0.0] * self.dimension] * len(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """This method is not used with Pinecone integrated embedding."""
        # Return placeholder values - these aren't used since Pinecone does the embedding
        return [0.0] * self.dimension

def get_pinecone_instance():
    """Get or create the Pinecone singleton instance."""
    global _pinecone_instance
    
    if _pinecone_instance is None:
        logger.info("Initializing Pinecone instance")
        pinecone_api_key = PINECONE_API_KEY
        pinecone_environment = PINECONE_ENVIRONMENT
        
        if not pinecone_api_key:
            error_msg = "Pinecone API key is missing. Please check your Streamlit secrets or environment variables."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Initializing Pinecone with environment {pinecone_environment}")
        
        try:
            # Create Pinecone instance
            _pinecone_instance = Pinecone(api_key=pinecone_api_key)
            
            # Check if index exists
            index_names = _pinecone_instance.list_indexes().names()
            if PINECONE_INDEX_NAME not in index_names:
                logger.warning(f"Pinecone index {PINECONE_INDEX_NAME} does not exist")
                raise ValueError(f"Pinecone index {PINECONE_INDEX_NAME} does not exist. Please create it first.")
            else:
                logger.info(f"Pinecone index {PINECONE_INDEX_NAME} exists. Available indexes: {index_names}")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    return _pinecone_instance

def get_vector_store_instance():
    """Get or create the vector store singleton instance.
    This now uses Pinecone's integrated embedding API."""
    global _vector_store_instance
    
    if _vector_store_instance is None:
        try:
            logger.info("Creating vector store instance with Pinecone's integrated embedding")
            pc = get_pinecone_instance()
            
            # Access the index using Pinecone's client
            index = pc.Index(PINECONE_INDEX_NAME)
            
            # Create a PineconeVectorStore instance configured for integrated embedding
            _vector_store_instance = PineconeVectorStore(
                index=index,
                embedding=PassthroughEmbeddings(),  # Use our dummy embeddings class
                namespace=PINECONE_NAMESPACE
            )
            
            logger.info("Vector store instance created successfully")
        except Exception as e:
            logger.error(f"Error creating vector store instance: {str(e)}")
            raise
    
    return _vector_store_instance

class PineconePDFProcessor:
    """Class to process PDF documents and create/load a Pinecone vector store.
    Modified to use Pinecone's integrated embedding API."""
    
    def __init__(self):
        """Initialize the processor."""
        logger.info("Initializing PineconePDFProcessor")
        try:
            # Initialize Pinecone (using singleton pattern)
            self.pc = get_pinecone_instance()
            logger.info("PineconePDFProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing PineconePDFProcessor: {str(e)}")
            raise

    def load_and_process_pdf(self) -> List:
        """Load and process a PDF document."""
        logger.info(f"Loading PDF from {PDF_PATH}")
        try:
            # Load the PDF
            loader = PyPDFLoader(PDF_PATH)
            documents = loader.load()
            
            # Split the documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            return text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def initialize_pinecone(self):
        """Initialize Pinecone with API key and environment.
        Now using singleton pattern."""
        try:
            # Just use the get_pinecone_instance function
            self.pc = get_pinecone_instance()
        except Exception as e:
            error_msg = f"Failed to initialize Pinecone: {str(e)}"
            logger.error(error_msg)
            raise PineconeConnectionError(error_msg)

    def create_vector_store(self, documents: List) -> PineconeVectorStore:
        """Create a vector store from documents using Pinecone's integrated embedding."""
        logger.info("Creating vector store with Pinecone's integrated embedding")
        try:
            # Get the Pinecone index
            index = self.pc.Index(PINECONE_INDEX_NAME)
            
            # Prepare records for integrated embedding
            # Maximum batch size for upsert_records is 96 as per Pinecone limitation
            batch_size = 96
            total_records = len(documents)
            total_batches = (total_records + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_records)
                batch_docs = documents[start_idx:end_idx]
                
                # Create records for this batch
                records = []
                for i, doc in enumerate(batch_docs):
                    record_id = f"doc_{start_idx + i}"
                    
                    # Create a base record with _id and text field
                    record = {
                        "_id": record_id,
                        "text": doc.page_content,  # Field that contains the text to be embedded
                    }
                    
                    # Add flattened metadata fields as top-level fields
                    # Only add fields that are of supported types
                    for key, value in doc.metadata.items():
                        if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                            # Use key as is for simple types
                            record[key] = value
                        else:
                            # Convert complex types to strings
                            record[key] = str(value)
                    
                    records.append(record)
                
                logger.info(f"Upserting batch {batch_idx + 1}/{total_batches} ({len(records)} records)")
                # Use upsert_records for integrated embedding
                index.upsert_records(
                    PINECONE_NAMESPACE,
                    records
                )
            
            # Create and return the vector store interface
            # Use the correct initialization parameters for PineconeVectorStore
            # Remove the text_field parameter as it's not supported
            vector_store = PineconeVectorStore(
                index=index,
                embedding=PassthroughEmbeddings(),
                namespace=PINECONE_NAMESPACE
            )
            
            return vector_store
            
        except Exception as e:
            error_msg = f"Error creating vector store: {str(e)}"
            logger.error(error_msg)
            raise e

    def load_vector_store(self) -> PineconeVectorStore:
        """Load an existing vector store."""
        logger.info("Loading existing vector store")
        return get_vector_store_instance()

    def process_pdf(self) -> PineconeVectorStore:
        """Process a PDF document and create or load a vector store."""
        try:
            documents = self.load_and_process_pdf()
            return self.create_vector_store(documents)
        except Exception as e:
            logger.error(f"Error in process_pdf: {str(e)}")
            raise

class PineconeConnectionError(Exception):
    """Exception raised for Pinecone connection errors."""
    pass

def initialize_pinecone():
    """Initialize Pinecone. Now uses the singleton pattern."""
    return get_pinecone_instance()

def count_documents():
    """Count documents in the vector store."""
    try:
        pc = get_pinecone_instance()
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        total_count = stats.get("total_vector_count", 0)
        logger.info(f"Total vector count: {total_count}")
        return total_count
    except Exception as e:
        logger.error(f"Error counting documents: {str(e)}")
        return 0 
