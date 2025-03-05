from typing import List
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from config import PDF_PATH, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, PINECONE_NAMESPACE
from text_processor import TextProcessor
from pinecone import Pinecone, ServerlessSpec
import os
import logging
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use configuration from config.py, which now properly prioritizes Streamlit secrets
pinecone_api_key = PINECONE_API_KEY
pinecone_environment = PINECONE_ENVIRONMENT

# Log configuration status (without exposing API keys)
logger.info(f"Pinecone configuration from config.py: API Key: {'Found' if pinecone_api_key else 'Not found'}, Environment: {pinecone_environment}")

class PineconePDFProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.text_processor = TextProcessor()
        self.api_key = pinecone_api_key
        self.environment = pinecone_environment
        self.index_name = PINECONE_INDEX_NAME
        self.namespace = PINECONE_NAMESPACE
        self.pc = None
        
        # Debug
        logger.info(f"PineconePDFProcessor initialized with API key: {'Present' if self.api_key else 'Missing'}")
        logger.info(f"PineconePDFProcessor initialized with environment: {'Present' if self.environment else 'Missing'}")
        logger.info(f"PineconePDFProcessor initialized with index name: {self.index_name}")
        logger.info(f"PineconePDFProcessor initialized with namespace: {self.namespace}")
        
    def load_and_process_pdf(self) -> List:
        """Load PDF and process its content."""
        # Load PDF
        logger.info(f"Loading PDF from {PDF_PATH}")
        loader = PyPDFLoader(PDF_PATH)
        raw_documents = loader.load()
        
        # Process documents (clean and split)
        logger.info("Processing documents")
        return self.text_processor.process_documents(raw_documents)
    
    def initialize_pinecone(self):
        """Initialize Pinecone client."""
        
        # Check if we have the required credentials
        if not self.api_key:
            error_msg = """
Pinecone API key is missing. Please check your Streamlit secrets or environment variables.

For Streamlit Cloud deployment, ensure your secrets.toml has the correct format:
[pinecone]
api_key = "your-pinecone-api-key"
environment = "your-pinecone-environment"
index_name = "koalitionskompass"
namespace = "default"
            """
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not self.environment and not hasattr(self.pc, 'list_indexes'):
            # Only check environment if we don't already have a working Pinecone instance
            error_msg = """
Pinecone environment is missing. Please check your Streamlit secrets or environment variables.

For Streamlit Cloud deployment, ensure your secrets.toml has the correct format:
[pinecone]
api_key = "your-pinecone-api-key"
environment = "your-pinecone-environment"  # This should be something like "us-east-1" or "gcp-starter"
index_name = "koalitionskompass"
namespace = "default"
            """
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Initializing Pinecone with environment {self.environment}")
        
        try:
            # Create Pinecone instance
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists
            index_names = self.pc.list_indexes().names()
            if self.index_name not in index_names:
                logger.warning(f"Pinecone index {self.index_name} does not exist")
                raise ValueError(f"Pinecone index {self.index_name} does not exist. Please create it first.")
            else:
                logger.info(f"Pinecone index {self.index_name} exists. Available indexes: {index_names}")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def create_vector_store(self, documents: List) -> PineconeVectorStore:
        """Create vector store from documents."""
        # Initialize Pinecone
        self.initialize_pinecone()
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        # Create vector store
        return PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name,
            namespace=self.namespace,
            pinecone_api_key=self.api_key
        )
    
    def load_vector_store(self) -> PineconeVectorStore:
        """Load existing vector store."""
        # Initialize Pinecone
        self.initialize_pinecone()
        
        logger.info("Loading existing vector store")
        # Load vector store
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=self.namespace,
            pinecone_api_key=self.api_key
        )
    
    def process_pdf(self) -> PineconeVectorStore:
        """Process PDF and return vector store."""
        documents = self.load_and_process_pdf()
        vector_store = self.create_vector_store(documents)
        return vector_store 
