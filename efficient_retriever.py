import logging
from typing import List, Optional, Dict, Any
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import PINECONE_NAMESPACE, PINECONE_INDEX_NAME
from pinecone_processor import get_pinecone_instance

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientPineconeRetriever(BaseRetriever):
    """
    Custom retriever that uses Pinecone's integrated embedding API for efficient retrieval.
    This bypasses the need for local embeddings and directly uses Pinecone's search_records API.
    """
    
    def __init__(self, index_name=PINECONE_INDEX_NAME, namespace=PINECONE_NAMESPACE, top_k=3):
        """
        Initialize the efficient retriever.
        
        Args:
            index_name: Name of the Pinecone index to query
            namespace: Namespace within the index (optional)
            top_k: Number of results to return
        """
        super().__init__()
        self._index_name = index_name
        self._namespace = namespace
        self._top_k = top_k
        self._pinecone_client = None
        self._index = None
        
        # Initialize Pinecone client
        self._init_pinecone()
    
    def _init_pinecone(self):
        """Initialize the Pinecone client and index."""
        try:
            self._pinecone_client = get_pinecone_instance()
            self._index = self._pinecone_client.Index(self._index_name)
            logger.info(f"Initialized EfficientPineconeRetriever with index: {self._index_name}")
        except Exception as e:
            logger.error(f"Error initializing Pinecone in EfficientPineconeRetriever: {str(e)}")
            raise
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Get documents relevant to a query using Pinecone's integrated embedding.
        
        Args:
            query: Query text
            run_manager: Callback manager
            
        Returns:
            List of relevant Document objects
        """
        logger.info(f"Retrieving documents for query: '{query}' using efficient method")
        
        try:
            # Use search_records with text input for integrated embedding
            search_response = self._index.search_records(
                namespace=self._namespace,
                query={
                    "inputs": {"text": query},
                    "top_k": self._top_k
                },
                # Include all fields in the response
                fields=["text", "page", "source"]
            )
            
            # Extract matches from response
            matches = search_response.get("result", {}).get("hits", [])
            logger.info(f"Found {len(matches)} matches with efficient query")
            
            # Convert Pinecone matches to LangChain Document objects
            documents = []
            for match in matches:
                fields = match.get("fields", {})
                score = match.get("_score")
                
                # Create metadata dictionary
                metadata = {
                    "score": score,
                    "source": fields.get("source", "Unknown"),
                    "page": fields.get("page", "N/A")
                }
                
                # Create Document object
                doc = Document(
                    page_content=fields.get("text", ""),
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in efficient retrieval: {str(e)}")
            # Return empty list on error
            return [] 