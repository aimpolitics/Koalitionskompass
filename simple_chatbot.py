import os
import requests
import json
import logging
from langchain_pinecone import PineconeVectorStore
from config import OPENAI_API_KEY, SYSTEM_PROMPT
from pinecone_processor import get_vector_store_instance
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleChatbot:
    def __init__(self):
        # Use the singleton vector store instance
        self.vector_store = get_vector_store_instance()
        
        # Check if OpenAI API key is available
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            logger.error("OpenAI API key is missing! Please check your Streamlit secrets or environment variables.")
            raise ValueError("""
OpenAI API key is missing! 

For Streamlit Cloud deployment, ensure your secrets.toml has the correct format:
[openai]
api_key = "your-openai-api-key"

For local development, either use .streamlit/secrets.toml or set the OPENAI_API_KEY environment variable.
            """)
            
        self.base_url = "https://oai.hconeai.com/v1"
        self.history = []
        logger.info("SimpleChatbot initialized successfully with OpenAI API key")
        
    def add_to_history(self, role, content):
        """Add a message to chat history."""
        self.history.append({"role": role, "content": content})
        
    def get_context_from_query(self, query, max_results=3):
        """Get relevant context from vector store using Pinecone's integrated embedding."""
        logger.info(f"Getting context for query: {query}")
        try:
            # The vector_store's similarity_search will use Pinecone's integrated embedding
            # to convert the query to a vector on the server side
            results = self.vector_store.similarity_search(query, k=max_results)
            context = "\n\n".join([doc.page_content for doc in results])
            logger.info(f"Retrieved {len(results)} documents from vector store")
            return context, results
        except Exception as e:
            logger.error(f"Error getting context from query: {str(e)}")
            return "", []
    
    def get_response(self, query):
        """Get response for user query."""
        try:
            logger.info(f"Getting response for: {query}")
            
            # Get relevant context from the vector store
            context, source_docs = self.get_context_from_query(query)
            
            if not context:
                logger.warning("No context found for query")
                return {
                    "answer": "Ich konnte leider keine relevanten Informationen zu Ihrer Anfrage finden.",
                    "sources": []
                }
            
            # Add the user's message to history
            self.add_to_history("user", query)
            
            # Create the system message with context
            system_message = {
                "role": "system", 
                "content": f"""Du bist ein hilfreicher Assistent, der Fragen zum Koalitionsvertrag der Bundesregierung beantwortet.
Nutze die folgenden Informationen, um die Frage des Nutzers zu beantworten:

{context}

Falls du die Antwort nicht in den bereitgestellten Informationen findest, sage ehrlich, dass du es nicht weißt.
Gib immer eine sachliche und objektive Antwort. Beziehe dich nur auf Fakten aus dem Text.
"""
            }
            
            # Create messages array
            messages = [system_message] + self.history
            
            # Create a client for the OpenAI API
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Generate response
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract the assistant's message
            answer = response.choices[0].message.content
            
            # Add the assistant's response to history
            self.add_to_history("assistant", answer)
            
            # Extract source information
            sources = []
            for doc in source_docs:
                if hasattr(doc, 'metadata') and doc.metadata:
                    source = {
                        'page': doc.metadata.get('page', None),
                        'source': doc.metadata.get('source', None),
                        'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                    }
                    sources.append(source)
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return {
                "answer": f"Ein Fehler ist aufgetreten: {str(e)}",
                "sources": []
            }
    
    def clear_history(self):
        """Clear chat history."""
        self.history = []
        logger.info("Chat history cleared") 