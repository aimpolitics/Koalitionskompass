import argparse
import logging
import os
import time
from typing import List, Dict, Any

# Import the necessary components
from pinecone import Pinecone
from pinecone_processor import PassthroughEmbeddings, get_pinecone_instance
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, PINECONE_NAMESPACE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def query_pinecone_direct(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Query Pinecone directly using the integrated embedding API.
    
    Args:
        query: The query text
        top_k: Number of results to return
        
    Returns:
        List of match dictionaries with metadata and score
    """
    logger.info(f"Querying Pinecone directly with: '{query}', top_k={top_k}")
    
    try:
        # Get Pinecone instance
        pc = get_pinecone_instance()
        
        # Get the index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Time the query operation
        start_time = time.time()
        
        # Query using the index's query method with the sparseVector parameter set to None
        query_response = index.query(
            namespace=PINECONE_NAMESPACE,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
            vector=None,  # No need to provide a vector since we'll use text
            text=query,  # Use the integrated embedding API
        )
        
        query_time = time.time() - start_time
        logger.info(f"Query completed in {query_time:.2f} seconds")
        
        # Process and return results
        matches = query_response.get("matches", [])
        logger.info(f"Found {len(matches)} matches")
        
        return matches
    
    except Exception as e:
        logger.error(f"Error querying Pinecone: {str(e)}")
        raise

def print_results(matches: List[Dict[str, Any]]) -> None:
    """
    Print the query results in a readable format.
    
    Args:
        matches: List of match dictionaries from Pinecone
    """
    if not matches:
        print("No matches found.")
        return
    
    print("\n===== QUERY RESULTS =====\n")
    
    for i, match in enumerate(matches, 1):
        print(f"=== Result {i} (Score: {match.get('score', 'N/A'):.4f}) ===")
        
        # Get metadata
        metadata = match.get("metadata", {})
        
        # Print page number if available
        page = metadata.get("page", "N/A")
        source = metadata.get("source", "Unknown")
        if source and '/' in source:
            source = source.split('/')[-1]
        
        print(f"Source: {source}, Page: {page}")
        
        # Print content
        text = metadata.get("text", "No content available")
        # Truncate if too long
        if len(text) > 300:
            text = text[:300] + "..."
        print(f"Content: {text}\n")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test Pinecone query with integrated embedding")
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
            matches = query_pinecone_direct(query, args.top_k)
            print_results(matches)
    else:
        # Use the provided query
        matches = query_pinecone_direct(args.query, args.top_k)
        print_results(matches)

if __name__ == "__main__":
    main() 