import logging
from pinecone_processor import get_vector_store_instance, get_efficient_retriever_instance
from config import PINECONE_INDEX_NAME, PINECONE_NAMESPACE
from chatbot import ChatBot

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vector_store():
    """Test the vector store initialization."""
    try:
        vector_store = get_vector_store_instance()
        logger.info("Vector store initialized successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        return None

def test_efficient_retriever():
    """Test the efficient retriever initialization."""
    try:
        retriever = get_efficient_retriever_instance()
        logger.info(f"Efficient retriever initialized successfully with index {PINECONE_INDEX_NAME}")
        return retriever
    except Exception as e:
        logger.error(f"Error initializing efficient retriever: {str(e)}")
        return None

def test_chatbot_with_efficient_retriever():
    """Test the ChatBot with efficient retriever."""
    vector_store = test_vector_store()
    if vector_store:
        try:
            chatbot = ChatBot(vector_store, use_efficient_retriever=True)
            logger.info("ChatBot with efficient retriever initialized successfully")
            return chatbot
        except Exception as e:
            logger.error(f"Error initializing ChatBot with efficient retriever: {str(e)}")
            return None
    else:
        logger.error("Cannot test ChatBot: vector store initialization failed")
        return None

def test_query(chatbot):
    """Test a query with the ChatBot."""
    if chatbot:
        try:
            query = "Wie fördert das Regierungsprogramm die Wettbewerbsfähigkeit und wirtschaftliche Entwicklung?"
            logger.info(f"Testing query: {query}")
            response = chatbot.get_response(query)
            logger.info("Query successful")
            logger.info(f"Response length: {len(response if response else '')}")
            return response
        except Exception as e:
            logger.error(f"Error testing query: {str(e)}")
            return None
    else:
        logger.error("Cannot test query: chatbot is None")
        return None

def main():
    """Run all tests."""
    logger.info("=== TESTING VECTOR STORE ===")
    vector_store = test_vector_store()
    
    logger.info("\n=== TESTING EFFICIENT RETRIEVER ===")
    retriever = test_efficient_retriever()
    
    logger.info("\n=== TESTING CHATBOT WITH EFFICIENT RETRIEVER ===")
    chatbot = test_chatbot_with_efficient_retriever()
    
    if chatbot:
        logger.info("\n=== TESTING QUERY ===")
        response = test_query(chatbot)
        if response:
            print("\n=== RESPONSE ===\n")
            print(response)
    
    logger.info("\n=== TESTING COMPLETE ===")

if __name__ == "__main__":
    main() 