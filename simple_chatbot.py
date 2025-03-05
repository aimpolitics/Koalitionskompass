import os
import requests
import json
import logging
import re
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
    
    def format_response(self, answer, sources):
        """Format the response with answer and sources in proper markdown."""
        # Format the main answer
        formatted_response = f"{answer}\n\n"
        
        # Add sources section if there are any sources
        if sources:
            formatted_response += "---\n\n"
            formatted_response += "### 📚 Quellen\n\n"
            
            for i, source in enumerate(sources, 1):
                page = source.get('page', 'N/A')
                content = source.get('content', '')
                doc_source = source.get('source', 'Unbekannt')
                
                # Extract just the filename from the document source path if it exists
                if doc_source and '/' in doc_source:
                    doc_source = doc_source.split('/')[-1]
                
                # Format source with markdown
                formatted_response += f"**[{i}] Seite {page}** - *{doc_source}*\n"
                formatted_response += f"> {content}\n\n"
                
        return formatted_response
    
    def get_response(self, query, simple_language=False):
        """Get response for user query."""
        try:
            logger.info(f"Getting response for: {query}")
            
            # Get relevant context from the vector store
            context, source_docs = self.get_context_from_query(query)
            
            if not context:
                logger.warning("No context found for query")
                return "Ich konnte leider keine relevanten Informationen zu Ihrer Anfrage finden."
            
            # Add the user's message to history
            self.add_to_history("user", query)
            
            # Create the system message with context
            system_message = {
                "role": "system", 
                "content": f"""{SYSTEM_PROMPT}

Nutze die folgenden Informationen, um die Frage des Nutzers zu beantworten:

{context}

{("Verwende einfache Sprache ohne Fremdwörter oder Fachbegriffe. Erkläre komplexe Konzepte in einfachen Worten und verwende kurze Sätze." if simple_language else "")}
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
                model="gpt-4o-mini-2024-07-18",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract the assistant's message
            answer = response.choices[0].message.content
            
            # Add the assistant's response to history
            self.add_to_history("assistant", answer)
            
            # Clean and extract source information
            sources = []
            for doc in source_docs:
                if hasattr(doc, 'metadata') and doc.metadata:
                    # Clean up the content by replacing Unicode bullet points and formatting
                    content = doc.page_content
                    # Replace Unicode bullet points with dashes
                    content = content.replace('\uf0b7', '-')
                    # Replace Unicode bullets with normal dash-space
                    content = content.replace('\no', '- ')
                    # Also handle standalone 'o ' bullets
                    content = content.replace('o ', '- ')
                    # Clean up excessive newlines
                    content = ' '.join([line.strip() for line in content.split('\n') if line.strip()])
                    # Remove any page numbers at the beginning of the content
                    content = re.sub(r'^\d+\s+', '', content)
                    
                    # Truncate at a word boundary if too long
                    if len(content) > 200:
                        content = content[:200].rsplit(' ', 1)[0] + '...'
                    
                    # Format page number as integer if possible
                    page = doc.metadata.get('page', None)
                    if page is not None:
                        try:
                            page = int(page)
                        except (ValueError, TypeError):
                            pass  # Keep as is if not convertible
                    
                    source = {
                        'page': page,
                        'source': doc.metadata.get('source', None),
                        'content': content
                    }
                    sources.append(source)
            
            # Instead of returning a dictionary, return a formatted markdown string
            return self.format_response(answer, sources)
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return f"Ein Fehler ist aufgetreten: {str(e)}"
    
    def clear_history(self):
        """Clear chat history."""
        self.history = []
        logger.info("Chat history cleared") 