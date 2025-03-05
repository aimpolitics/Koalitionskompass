import logging
from typing import List, Optional, Dict, Any
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import PINECONE_NAMESPACE, PINECONE_INDEX_NAME
from pinecone_processor import get_pinecone_instance, PassthroughEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientPineconeRetriever(BaseRetriever):
    """
    Custom retriever that uses Pinecone's integrated embedding API for efficient retrieval.
    This bypasses the need for local embeddings and directly uses Pinecone's text API.
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
        self._embeddings = PassthroughEmbeddings(dimension=1024)
        
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
            # Use search_records for integrated embedding as per Pinecone documentation
            search_response = self._index.search_records(
                namespace=self._namespace,
                query={
                    "inputs": {"text": query},  # The text query for integrated embedding
                    "top_k": self._top_k
                },
                fields=["text", "source", "page"]  # Specify fields to return
            )
            
            # Process the response based on Pinecone v6.x response format
            documents = []
            
            # Check if we have results
            if hasattr(search_response, 'result') and hasattr(search_response.result, 'hits') and search_response.result.hits:
                hits = search_response.result.hits
                logger.info(f"Found {len(hits)} hits with efficient query")
                
                for hit in hits:
                    # Extract metadata and score safely
                    record_id = hit._id if hasattr(hit, '_id') else "Unknown"
                    score = hit._score if hasattr(hit, '_score') else 0
                    fields = hit.fields if hasattr(hit, 'fields') else {}
                    
                    # Create metadata dictionary for the document
                    doc_metadata = {
                        "score": score,
                        "id": record_id,
                        "source": fields.get("source", "Unknown"),
                        "page": fields.get("page", "N/A")
                    }
                    
                    # The text content should be in the fields
                    page_content = fields.get("text", "")
                    
                    # Create Document object
                    doc = Document(
                        page_content=page_content,
                        metadata=doc_metadata
                    )
                    documents.append(doc)
            else:
                logger.warning("No hits found in the search response")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in efficient retrieval: {str(e)}")
            # Return empty list on error
            return [] 