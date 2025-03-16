import argparse
import logging
import time
from typing import List, Dict, Any

# Import the necessary components
from pinecone_processor import get_vector_store_instance
from config import PINECONE_INDEX_NAME, PINECONE_NAMESPACE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def query_pinecone_langchain(query: str, top_k: int = 3):
    """
    Query Pinecone through the LangChain vector store interface.
    
    Args:
        query: The query text
        top_k: Number of results to return
        
    Returns:
        List of document objects from LangChain
    """
    logger.info(f"Querying Pinecone via LangChain with: '{query}', top_k={top_k}")
    
    try:
        # Get the vector store instance
        vector_store = get_vector_store_instance()
        
        # Time the query operation
        start_time = time.time()
        
        # Perform the similarity search
        results = vector_store.similarity_search(query, k=top_k)
        
        query_time = time.time() - start_time
        logger.info(f"Query completed in {query_time:.2f} seconds")
        
        logger.info(f"Found {len(results)} results")
        return results
    
    except Exception as e:
        logger.error(f"Error querying Pinecone via LangChain: {str(e)}")
        raise

def print_langchain_results(results):
    """
    Print the LangChain document results in a readable format.
    
    Args:
        results: List of Document objects from LangChain
    """
    if not results:
        print("No results found.")
        return
    
    print("\n===== LANGCHAIN QUERY RESULTS =====\n")
    
    for i, doc in enumerate(results, 1):
        print(f"=== Result {i} ===")
        
        # Get metadata
        metadata = getattr(doc, 'metadata', {})
        
        # Print page number if available
        page = metadata.get("page", "N/A")
        source = metadata.get("source", "Unknown")
        if source and '/' in source:
            source = source.split('/')[-1]
        
        print(f"Source: {source}, Page: {page}")
        
        # Print content
        content = getattr(doc, 'page_content', "No content available")
        # Truncate if too long
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"Content: {content}\n")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test Pinecone query with LangChain")
    parser.add_argument("query", nargs="?", default="", help="Query text")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()
    
    # If no query provided, use the default test queries
    if not args.query:
        test_queries = [
            "Wie fördert das Regierungsprogramm die Wettbewerbsfähigkeit und wirtschaftliche Entwicklung durch steuerliche Reformen und Investitionsanreize?",
            "Welche Maßnahmen gibt es für pflegende Angehörige?"
        ]
        
        for query in test_queries:
            print(f"\n\n==== TESTING QUERY: {query} ====\n")
            results = query_pinecone_langchain(query, args.top_k)
            print_langchain_results(results)
    else:
        # Use the provided query
        results = query_pinecone_langchain(args.query, args.top_k)
        print_langchain_results(results)

if __name__ == "__main__":
    main() 