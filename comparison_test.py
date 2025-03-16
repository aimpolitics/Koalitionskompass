import argparse
import logging
import time
from typing import List, Dict, Any

# Import the necessary components
from pinecone import Pinecone
from pinecone_processor import PassthroughEmbeddings, get_pinecone_instance, get_vector_store_instance
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
        logger.info(f"Direct query completed in {query_time:.2f} seconds")
        
        # Process and return results
        matches = query_response.get("matches", [])
        logger.info(f"Direct query found {len(matches)} matches")
        
        return matches, query_time
    
    except Exception as e:
        logger.error(f"Error querying Pinecone directly: {str(e)}")
        raise

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
        logger.info(f"LangChain query completed in {query_time:.2f} seconds")
        
        logger.info(f"LangChain query found {len(results)} results")
        return results, query_time
    
    except Exception as e:
        logger.error(f"Error querying Pinecone via LangChain: {str(e)}")
        raise

def print_direct_results(matches: List[Dict[str, Any]]) -> None:
    """
    Print the direct query results in a readable format.
    
    Args:
        matches: List of match dictionaries from Pinecone
    """
    if not matches:
        print("No matches found.")
        return
    
    print("\n===== DIRECT QUERY RESULTS =====\n")
    
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
    parser = argparse.ArgumentParser(description="Compare Pinecone query methods")
    parser.add_argument("query", nargs="?", default="", help="Query text")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--method", choices=["direct", "langchain", "both"], default="both", 
                        help="Query method to use (direct, langchain, or both)")
    args = parser.parse_args()
    
    # If no query provided, use the default test queries
    queries = []
    if not args.query:
        queries = [
            "Wie fördert das Regierungsprogramm die Wettbewerbsfähigkeit und wirtschaftliche Entwicklung durch steuerliche Reformen und Investitionsanreize?",
            "Welche Maßnahmen gibt es für pflegende Angehörige?"
        ]
    else:
        queries = [args.query]
    
    # Track timing statistics
    direct_times = []
    langchain_times = []
    
    for query in queries:
        print(f"\n\n==== TESTING QUERY: {query} ====\n")
        
        if args.method in ["direct", "both"]:
            try:
                direct_matches, direct_time = query_pinecone_direct(query, args.top_k)
                direct_times.append(direct_time)
                print_direct_results(direct_matches)
            except Exception as e:
                print(f"Error with direct query: {e}")
        
        if args.method in ["langchain", "both"]:
            try:
                langchain_results, langchain_time = query_pinecone_langchain(query, args.top_k)
                langchain_times.append(langchain_time)
                print_langchain_results(langchain_results)
            except Exception as e:
                print(f"Error with LangChain query: {e}")
    
    # Print timing comparison if both methods were used
    if args.method == "both" and direct_times and langchain_times:
        avg_direct = sum(direct_times) / len(direct_times)
        avg_langchain = sum(langchain_times) / len(langchain_times)
        
        print("\n===== PERFORMANCE COMPARISON =====")
        print(f"Direct Pinecone query average time: {avg_direct:.4f} seconds")
        print(f"LangChain query average time: {avg_langchain:.4f} seconds")
        print(f"Difference: {avg_langchain - avg_direct:.4f} seconds")
        if avg_direct < avg_langchain:
            print(f"Direct method is {(avg_langchain / avg_direct):.2f}x faster")
        else:
            print(f"LangChain method is {(avg_direct / avg_langchain):.2f}x faster")

if __name__ == "__main__":
    main() 