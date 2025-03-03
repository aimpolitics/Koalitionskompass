import os
import requests
import json
from langchain_pinecone import PineconeVectorStore
from config import OPENAI_API_KEY, SYSTEM_PROMPT
from pinecone_processor import PineconePDFProcessor

class SimpleChatbot:
    def __init__(self):
        # Lade die Pinecone-Vektordatenbank
        pdf_processor = PineconePDFProcessor()
        self.vector_store = pdf_processor.load_vector_store()
        self.api_key = OPENAI_API_KEY
        self.base_url = "https://oai.hconeai.com/v1"
        self.history = []
        
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
            else:
                print(f"API Error: {response.status_code}, {response.text}")
                return {
                    "answer": f"API-Fehler: {response.status_code}. Bitte versuchen Sie es später erneut.",
                    "sources": []
                }
                
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return {
                "answer": f"Es ist ein Fehler aufgetreten: {str(e)}. Bitte versuchen Sie es später erneut.",
                "sources": []
            }
    
    def clear_history(self):
        """Clear chat history."""
        self.history = [] 