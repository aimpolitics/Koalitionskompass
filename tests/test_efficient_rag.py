import logging
import time
from pinecone_processor import get_vector_store_instance, get_efficient_retriever_instance
from chatbot import ChatBot
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_standard_rag(query, top_k=3):
    """Test the standard LangChain RAG setup with vector store retriever."""
    logger.info(f"Testing standard LangChain RAG with query: '{query}'")
    
    # Get vector store instance (using integrated embedding API but with LangChain interface)
    vector_store = get_vector_store_instance()
    
    # Initialize chatbot with standard retriever
    start_time = time.time()
    chatbot = ChatBot(vector_store, use_efficient_retriever=False)
    setup_time = time.time() - start_time
    logger.info(f"Standard ChatBot initialized in {setup_time:.4f} seconds")
    
    # Get response using standard method
    start_time = time.time()
    response = chatbot.get_response(query)
    query_time = time.time() - start_time
    
    logger.info(f"Standard RAG query completed in {query_time:.4f} seconds")
    
    # Return response and timing
    return response, query_time

def test_efficient_rag(query, top_k=3):
    """Test the LangChain RAG with efficient retriever."""
    logger.info(f"Testing efficient LangChain RAG with query: '{query}'")
    
    # Get vector store instance
    vector_store = get_vector_store_instance()
    
    # Initialize chatbot with efficient retriever
    start_time = time.time()
    chatbot = ChatBot(vector_store, use_efficient_retriever=True)
    setup_time = time.time() - start_time
    logger.info(f"Efficient ChatBot initialized in {setup_time:.4f} seconds")
    
    # Get response using efficient method
    start_time = time.time()
    response = chatbot.get_response(query)
    query_time = time.time() - start_time
    
    logger.info(f"Efficient RAG query completed in {query_time:.4f} seconds")
    
    # Return response and timing
    return response, query_time

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test LangChain RAG with efficient retriever")
    parser.add_argument("query", nargs="?", default="", help="Query text")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--mode", choices=["standard", "efficient", "both"], default="both", 
                        help="Which mode to test: standard, efficient, or both")
    args = parser.parse_args()
    
    # If no query provided, use a default test query
    if not args.query:
        args.query = "Wie fördert das Regierungsprogramm die Wettbewerbsfähigkeit und wirtschaftliche Entwicklung durch steuerliche Reformen und Investitionsanreize?"
    
    # Run tests based on selected mode
    if args.mode in ["standard", "both"]:
        print(f"\n=== TESTING STANDARD LANGCHAIN RAG ===\n")
        response, query_time = test_standard_rag(args.query, args.top_k)
        print(f"\nStandard RAG query completed in {query_time:.4f} seconds")
        print("\n=== RESPONSE ===\n")
        print(response)
    
    if args.mode in ["efficient", "both"]:
        print(f"\n=== TESTING EFFICIENT LANGCHAIN RAG ===\n")
        response, query_time = test_efficient_rag(args.query, args.top_k)
        print(f"\nEfficient RAG query completed in {query_time:.4f} seconds")
        print("\n=== RESPONSE ===\n")
        print(response)

if __name__ == "__main__":
    main() 