from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from config import PDF_PATH, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, PINECONE_NAMESPACE
from text_processor import TextProcessor
from pinecone import Pinecone, ServerlessSpec
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable for langchain_pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENVIRONMENT

class PineconePDFProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.text_processor = TextProcessor()
        self.api_key = PINECONE_API_KEY
        self.environment = PINECONE_ENVIRONMENT
        self.index_name = PINECONE_INDEX_NAME
        self.namespace = PINECONE_NAMESPACE
        self.pc = None
        
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
        logger.info(f"Initializing Pinecone with environment {self.environment}")
        # Create Pinecone instance
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check if index exists
        if self.index_name not in self.pc.list_indexes().names():
            logger.warning(f"Pinecone index {self.index_name} does not exist")
            raise ValueError(f"Pinecone index {self.index_name} does not exist. Please create it first.")
        else:
            logger.info(f"Pinecone index {self.index_name} already exists")
    
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