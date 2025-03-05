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
        self.vector_store = vector_store
        
        # Überprüfen, ob der OpenAI API-Schlüssel vorhanden ist
        if not OPENAI_API_KEY:
            logger.error("OpenAI API-Schlüssel fehlt")
            raise ValueError(
                "OpenAI API-Schlüssel fehlt. Bitte stellen Sie sicher, dass Sie einen gültigen "
                "API-Schlüssel in der .streamlit/secrets.toml oder in Umgebungsvariablen konfiguriert haben."
            )
        
        # Configure OpenAI settings
        os.environ["OPENAI_API_BASE"] = "https://oai.hconeai.com/v1"
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        try:
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000,
                model_kwargs={"stop": None}
            )
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(),
                memory=self.memory,
                return_source_documents=True,
                verbose=True,
                chain_type="stuff",
                rephrase_question=False,
                output_key="answer"
            )
            logger.info("ChatBot erfolgreich initialisiert")
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des ChatBot: {str(e)}")
            raise ValueError(f"Fehler bei der Initialisierung des ChatBot: {str(e)}")
        
    def get_response(self, query: str, simple_language=False) -> str:
        """Get response for user query with optional simple language formatting."""
        try:
            # Modifiziere den Prompt für einfache Sprache wenn nötig
            prompt_to_use = f"Bitte erkläre in einfacher Sprache: {query}" if simple_language else query
            
            response = self.chain.invoke({"question": prompt_to_use})
            
            # Extrahiere Antworttext und Quellen
            answer_text = response.get("answer", "Keine Antwort gefunden.")
            sources = [doc.page_content for doc in response.get("source_documents", [])]
            
            # Antwort formatieren
            formatted_response = answer_text
            
            # Quellen hinzufügen, wenn vorhanden
            if sources:
                formatted_response += "\n\n**Quellen:**\n"
                for source in sources:
                    formatted_response += f"- {source}\n"
                    
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return f"Entschuldigung, es gab einen Fehler bei der Verarbeitung Ihrer Anfrage: {str(e)}"
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear() 