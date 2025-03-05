from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS, SYSTEM_PROMPT
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, vector_store):
        """Initialize ChatBot with the vector store instance.
        
        The vector_store now uses Pinecone's integrated embedding API, so no local
        embedding model is required.
        """
        self.vector_store = vector_store
        
        # Check if OpenAI API key is available
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key is missing")
            raise ValueError(
                "OpenAI API key is missing. Please make sure you have configured a valid "
                "API key in .streamlit/secrets.toml or environment variables."
            )
        
        # Configure OpenAI settings
        os.environ["OPENAI_API_BASE"] = "https://oai.hconeai.com/v1"
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        try:
            # Initialize language model
            llm = ChatOpenAI(
                model_name=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            # Create conversational retrieval chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(),
                memory=self.memory,
                return_source_documents=True,
                verbose=True,
                chain_type="stuff",
                output_key="answer"
            )
            logger.info("ChatBot initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ChatBot: {str(e)}")
            raise ValueError(f"Error initializing ChatBot: {str(e)}")
        
    def get_response(self, query, simple_language=False):
        """Get response for a user query."""
        try:
            # Send query to the chain and get response
            logger.info(f"Getting response for query: {query}")
            result = self.chain({"question": query})
            
            # Extract answer and source documents
            answer = result.get("answer", "")
            source_documents = result.get("source_documents", [])
            
            # Extract source information
            sources = []
            for doc in source_documents:
                # Extract metadata
                metadata = doc.metadata
                page = metadata.get("page", None)
                
                # Create source object
                source = {
                    "page": page,
                    "content": doc.page_content,
                    "source": metadata.get("source", None)
                }
                sources.append(source)
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise
    
    def clear_history(self):
        """Clear conversation history."""
        self.memory.clear()
        logger.info("Conversation history cleared") 