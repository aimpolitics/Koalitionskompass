from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS, SYSTEM_PROMPT, PINECONE_INDEX_NAME, PINECONE_NAMESPACE
import os
import logging
import re
from langchain_core.prompts import ChatPromptTemplate
from pinecone_processor import get_efficient_retriever_instance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, vector_store, use_efficient_retriever=True):
        """Initialize ChatBot with the vector store instance.
        
        Args:
            vector_store: Vector store instance
            use_efficient_retriever: Whether to use the efficient Pinecone retriever
                                     that uses integrated embedding API
        """
        self.vector_store = vector_store
        self.use_efficient_retriever = use_efficient_retriever
        
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
            
            # Create prompt template with system prompt from config
            # We'll create two different prompt templates - one for regular and one for simple language
            # We'll choose the appropriate one in the get_response method
            self.regular_prompt = ChatPromptTemplate.from_messages([
                ("system", f"{SYSTEM_PROMPT}\n\nContext:\n{{context}}"),
                ("human", "{question}")
            ])
            
            self.simple_prompt = ChatPromptTemplate.from_messages([
                ("system", f"{SYSTEM_PROMPT}\n\nVerwende einfache Sprache ohne Fremdwörter oder Fachbegriffe. Erkläre komplexe Konzepte in einfachen Worten und verwende kurze Sätze.\n\nContext:\n{{context}}"),
                ("human", "{question}")
            ])
            
            # Select retriever based on configuration
            if use_efficient_retriever:
                logger.info("Using efficient retriever with integrated embedding")
                self.retriever = get_efficient_retriever_instance()
            else:
                logger.info("Using standard LangChain retriever")
                self.retriever = vector_store.as_retriever()
            
            # Create conversational retrieval chain with default prompt (will be changed as needed)
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=True,
                chain_type="stuff",
                output_key="answer",
                combine_docs_chain_kwargs={"prompt": self.regular_prompt}
            )
            logger.info("ChatBot initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ChatBot: {str(e)}")
            raise ValueError(f"Error initializing ChatBot: {str(e)}")
        
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
                content = source.get('content', '')  # Get full content without truncation
                doc_source = source.get('source', 'Unbekannt')
                
                # Extract just the filename from the document source path if it exists
                if doc_source and '/' in doc_source:
                    doc_source = doc_source.split('/')[-1]
                
                # Format page number as integer if possible
                if page is not None:
                    try:
                        page = int(page)
                    except (ValueError, TypeError):
                        pass  # Keep as is if not convertible
                
                # Clean up the content
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
                
                # Format source with markdown
                formatted_response += f"**[{i}] Seite {page}** - *{doc_source}*\n"
                formatted_response += f"> {content}\n\n"
                
        return formatted_response
        
    def get_response(self, query, simple_language=False):
        """Get response for a user query."""
        try:
            # Select the appropriate retriever based on configuration
            if self.use_efficient_retriever and not hasattr(self, 'retriever'):
                logger.info("Setting up efficient retriever")
                self.retriever = get_efficient_retriever_instance()
            elif not self.use_efficient_retriever and not hasattr(self, 'retriever'):
                logger.info("Setting up standard LangChain retriever")
                self.retriever = self.vector_store.as_retriever()
            
            # Recreate the chain with the appropriate prompt based on the simple_language parameter
            if simple_language:
                logger.info("Using simple language prompt")
                # Recreate the chain with the simple language prompt
                self.chain = ConversationalRetrievalChain.from_llm(
                    llm=ChatOpenAI(
                        model_name=MODEL_NAME,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS
                    ),
                    retriever=self.retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True,
                    chain_type="stuff",
                    output_key="answer",
                    combine_docs_chain_kwargs={"prompt": self.simple_prompt}
                )
            else:
                logger.info("Using regular prompt")
                # Recreate the chain with the regular prompt
                self.chain = ConversationalRetrievalChain.from_llm(
                    llm=ChatOpenAI(
                        model_name=MODEL_NAME,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS
                    ),
                    retriever=self.retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True,
                    chain_type="stuff",
                    output_key="answer",
                    combine_docs_chain_kwargs={"prompt": self.regular_prompt}
                )
                
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
            
            # Instead of returning a dictionary, return a formatted markdown string
            return self.format_response(answer, sources)
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise
    
    def clear_history(self):
        """Clear conversation history."""
        self.memory.clear()
        logger.info("Conversation history cleared") 