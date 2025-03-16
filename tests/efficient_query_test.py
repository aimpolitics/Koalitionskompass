import argparse
import logging
import time
from typing import List, Dict, Any

# Import the necessary components
from pinecone import Pinecone
from pinecone_processor import get_pinecone_instance
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, PINECONE_NAMESPACE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def query_pinecone_efficient(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Query Pinecone efficiently using the search_records method with text input.
    
    Args:
        query: The query text
        top_k: Number of results to return
        
    Returns:
        List of match dictionaries with metadata and score
    """
    logger.info(f"Querying Pinecone efficiently with: '{query}', top_k={top_k}")
    
    try:
        # Get Pinecone instance
        pc = get_pinecone_instance()
        
        # Get the index with the correct name from config
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Time the query operation
        start_time = time.time()
        
        # Use search_records with text input for integrated embedding
        search_response = index.search_records(
            namespace=PINECONE_NAMESPACE,
            query={
                "inputs": {"text": query},
                "top_k": top_k
            },
            # Include all fields in the response
            fields=["text", "page", "source"]
        )
        
        query_time = time.time() - start_time
        logger.info(f"Efficient query completed in {query_time:.2f} seconds")
        
        # Process results - the format is different from the regular query method
        matches = search_response.get("result", {}).get("hits", [])
        logger.info(f"Found {len(matches)} matches")
        
        return matches, query_time
    
    except Exception as e:
        logger.error(f"Error querying Pinecone efficiently: {str(e)}")
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
        print(f"=== Result {i} (Score: {match.get('_score', 'N/A'):.4f}) ===")
        
        # Get fields
        fields = match.get("fields", {})
        
        # Print page number if available
        page = fields.get("page", "N/A")
        source = fields.get("source", "Unknown")
        if source and '/' in source:
            source = source.split('/')[-1]
        
        print(f"Source: {source}, Page: {page}")
        
        # Print content
        text = fields.get("text", "No content available")
        # Truncate if too long
        if len(text) > 300:
            text = text[:300] + "..."
        print(f"Content: {text}\n")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test efficient Pinecone querying")
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
            try:
                matches, query_time = query_pinecone_efficient(query, args.top_k)
                print(f"Query completed in {query_time:.4f} seconds")
                print_results(matches)
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Use the provided query
        try:
            matches, query_time = query_pinecone_efficient(args.query, args.top_k)
            print(f"Query completed in {query_time:.4f} seconds")
            print_results(matches)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 