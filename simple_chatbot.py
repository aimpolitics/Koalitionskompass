import os
import requests
import json
import logging
from langchain_pinecone import PineconeVectorStore
from config import OPENAI_API_KEY, SYSTEM_PROMPT
from pinecone_processor import PineconePDFProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleChatbot:
    def __init__(self):
        # Lade die Pinecone-Vektordatenbank
        pdf_processor = PineconePDFProcessor()
        self.vector_store = pdf_processor.load_vector_store()
        
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
        """Get relevant context from vector store."""
        results = self.vector_store.similarity_search(query, k=max_results)
        context = "\n\n".join([doc.page_content for doc in results])
        return context, results
    
    def get_response(self, query):
        """Get response for user query."""
        try:
            # Get relevant context
            context, sources = self.get_context_from_query(query)
            
            # Add user query to history
            self.add_to_history("user", query)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT + f"\nKontext: {context}"}
            ] + self.history
            
            # Check if API key is valid before making API call
            if not self.api_key:
                return {
                    "answer": "Fehler: OpenAI API-Schlüssel fehlt. Bitte konfigurieren Sie den API-Schlüssel in den Streamlit Secrets.",
                    "sources": []
                }
            
            # Make direct API call
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            # Check if request was successful
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data["choices"][0]["message"]["content"]
                
                # Add response to history
                self.add_to_history("assistant", answer)
                
                return {
                    "answer": answer,
                    "sources": [doc.page_content for doc in sources]
                }
            elif response.status_code == 401:
                logger.error(f"Authentication error with OpenAI API: {response.text}")
                return {
                    "answer": "Fehler: Die Authentifizierung mit der OpenAI API ist fehlgeschlagen. Bitte überprüfen Sie Ihren API-Schlüssel.",
                    "sources": []
                }
            else:
                logger.error(f"API Error: {response.status_code}, {response.text}")
                return {
                    "answer": f"API-Fehler: {response.status_code}. Bitte versuchen Sie es später erneut.",
                    "sources": []
                }
                
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return {
                "answer": f"Es ist ein Fehler aufgetreten: {str(e)}. Bitte versuchen Sie es später erneut.",
                "sources": []
            }
    
    def clear_history(self):
        """Clear chat history."""
        self.history = [] 