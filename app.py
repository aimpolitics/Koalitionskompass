import streamlit as st
import os
import logging
from PIL import Image
from pinecone_processor import get_vector_store_instance, PineconeConnectionError
from simple_chatbot import SimpleChatbot
from chatbot import ChatBot
from supabase_service import SupabaseService
import traceback
import hashlib
from time import time

# Konfiguration der Streamlit-App
st.set_page_config(
    page_title="Koalitionskompass",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def initialize_session_state():
#     """Initialisiert die Session-Variablen, wenn sie noch nicht existieren."""
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
    
#     if "simple_chat_history" not in st.session_state:
#         st.session_state.simple_chat_history = []
    
#     if "vector_store" not in st.session_state:
#         # Vector store should already be initialized by ensure_vectorstore_exists
#         pass
    
#     if "session_id" not in st.session_state:
#         st.session_state.session_id = hashlib.md5(str(os.urandom(32)).encode()).hexdigest()
    
#     if "chatbot" not in st.session_state:
#         try:
#             # Use efficient retriever by default (integrated embedding)
#             logger.info("Initializing ChatBot with efficient retriever")
            
#             # Ensure the vector store exists
#             if "vector_store" not in st.session_state or st.session_state.vector_store is None:
#                 logger.warning("Vector store not initialized, initializing now")
#                 ensure_vectorstore_exists()
            
#             if "vector_store" in st.session_state and st.session_state.vector_store is not None:
#                 st.session_state.chatbot = ChatBot(st.session_state.vector_store, use_efficient_retriever=True)
#                 logger.info("ChatBot with efficient retriever initialized successfully")
#             else:
#                 raise ValueError("Vector store initialization failed")
                
#         except ValueError as e:
#             st.error(f"Fehler bei der Initialisierung des Standard-Chatbots: {str(e)}")
#             logger.error(f"Error initializing ChatBot: {str(e)}")
#             st.warning("""
#             ## OpenAI API-Schlüssel fehlt oder ist ungültig
            
#             Bitte stellen Sie sicher, dass Sie einen gültigen OpenAI API-Schlüssel konfiguriert haben:
            
#             ### Für Streamlit Cloud:
#             Gehen Sie zu den Streamlit Cloud-Einstellungen > Secrets und fügen Sie die folgende Konfiguration hinzu:
            
# toml
#             [openai]
#             api_key = "sk-Ihr-OpenAI-API-Schlüssel"

            
#             ### Für lokale Entwicklung:
#             Erstellen Sie eine .env-Datei oder .streamlit/secrets.toml mit der gleichen Konfiguration.
#             """)
            
#             # Fall back to standard retriever if efficient retriever fails
#             try:
#                 logger.info("Trying to initialize ChatBot with standard retriever")
#                 st.session_state.chatbot = ChatBot(st.session_state.vector_store, use_efficient_retriever=False)
#                 logger.info("ChatBot with standard retriever initialized successfully")
#             except Exception as fallback_error:
#                 logger.error(f"Error initializing ChatBot with standard retriever: {str(fallback_error)}")
#                 st.session_state.chatbot = None
                
#         except Exception as e:
#             st.error(f"Unerwarteter Fehler bei der Initialisierung des Standard-Chatbots: {str(e)}")
#             logger.error(f"Unexpected error initializing ChatBot: {str(e)}")
#             st.session_state.chatbot = None
    
#     if "simple_chatbot" not in st.session_state:
#         try:
#             st.session_state.simple_chatbot = SimpleChatbot()
#         except ValueError as e:
#             st.error(f"Fehler bei der Initialisierung des einfachen Chatbots: {str(e)}")
#             st.warning("""
#             ## OpenAI API-Schlüssel fehlt oder ist ungültig
            
#             Bitte stellen Sie sicher, dass Sie einen gültigen OpenAI API-Schlüssel konfiguriert haben:
            
#             ### Für Streamlit Cloud:
#             Gehen Sie zu den Streamlit Cloud-Einstellungen > Secrets und fügen Sie die folgende Konfiguration hinzu:
            
# toml
#             [openai]
#             api_key = "sk-Ihr-OpenAI-API-Schlüssel"

            
#             ### Für lokale Entwicklung:
#             Erstellen Sie eine .env-Datei oder .streamlit/secrets.toml mit der gleichen Konfiguration.
#             """)
#             st.session_state.simple_chatbot = None
#         except Exception as e:
#             st.error(f"Unerwarteter Fehler bei der Initialisierung des einfachen Chatbots: {str(e)}")
#             st.session_state.simple_chatbot = None
    
#     if "active_tab" not in st.session_state:
#         st.session_state.active_tab = "standard"

def generate_message_hash(content):
    """Generate a unique hash based on message content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def render_chat_interface(simple_language=False):
    """Rendert das Chat-Interface je nach ausgewähltem Modus"""
    
    # Input für Benutzer
    # Input für Benutzer (wird immer ganz unten angezeigt)
    user_input = st.chat_input("Stellen Sie eine Frage zum Regierungsprogramm...")
    
    # Auswahl der richtigen Chat-Historie basierend auf dem aktiven Modus
    chat_history = st.session_state.simple_chat_history if simple_language else st.session_state.chat_history

    # Erstelle Container für die Chat-Elemente (in umgekehrter Reihenfolge, damit das Input ganz unten ist)
    chat_container = st.container()
    
    # Verarbeite Benutzer-Eingabe
    if user_input:
        # Lade den richtigen Chatbot basierend auf dem ausgewählten Modus
        if simple_language:
            if st.session_state.simple_chatbot is None:
                with st.chat_message("assistant"):
                    st.markdown("Der Chatbot konnte aufgrund eines Konfigurationsproblems nicht initialisiert werden. Bitte prüfen Sie die Fehlermeldungen oben.")
                return
            chatbot = st.session_state.simple_chatbot
        else:
            if st.session_state.chatbot is None:
                with st.chat_message("assistant"):
                    st.markdown("Der Chatbot konnte aufgrund eines Konfigurationsproblems nicht initialisiert werden. Bitte prüfen Sie die Fehlermeldungen oben.")
                return
            chatbot = st.session_state.chatbot
            
        # Antwort-Platzhalter - Der key hier ist wichtig, damit die temporäre Anzeige nicht später auch nochmal gerendert wird
        message_placeholder = st.empty()
                
        try:
            # Antwort vom Chatbot
            response = chatbot.get_response(user_input, simple_language=simple_language)
            
            # Generiere Hash-Werte für Nachrichten
            user_hash = generate_message_hash(user_input)
            response_hash = generate_message_hash(response)
            
            # Speichere Nachricht und Antwort in der Chat-Historie mit Hash
            if simple_language:
                st.session_state.simple_chat_history.insert(0, {"role": "assistant", "content": response, "hash": response_hash})
                st.session_state.simple_chat_history.insert(0, {"role": "user", "content": user_input, "hash": user_hash})
            else:
                st.session_state.chat_history.insert(0, {"role": "assistant", "content": response, "hash": response_hash})
                st.session_state.chat_history.insert(0, {"role": "user", "content": user_input, "hash": user_hash})
            
            # Save to Supabase
            st.session_state.supabase_service.save_prompt(
                
                # use a timestamp + random string as session_id             
                session_id=str(time()) + str(os.urandom(32)),
                prompt_text=user_input,
                response_text=response,
                language_mode="simple" if simple_language else "standard",
                is_einfache_sprache=simple_language
            )
                    
        except Exception as e:
            error_message = f"Entschuldigung, ich konnte keine Antwort generieren: {str(e)}"
            
            # Generiere Hash für Fehlermeldung
            error_hash = generate_message_hash(error_message)
            user_hash = generate_message_hash(user_input)
            
            # Fehler in der Chat-Historie speichern
            if simple_language:
                st.session_state.simple_chat_history.insert(0, {"role": "assistant", "content": error_message, "hash": error_hash})
                st.session_state.simple_chat_history.insert(0, {"role": "user", "content": user_input, "hash": user_hash})
            else:
                st.session_state.chat_history.insert(0, {"role": "assistant", "content": error_message, "hash": error_hash})
                st.session_state.chat_history.insert(0, {"role": "user", "content": user_input, "hash": user_hash})
        
        # Trigger rerun to immediately show the updated chat history
        st.rerun()
    
    # Dedupliziere die Chat-Historie basierend auf Hash-Werten
    deduplicated_history = []
    seen_hashes = set()
    
    for message in chat_history:
        # Wenn es ältere Nachrichten ohne Hash gibt, füge einen hinzu
        if "hash" not in message:
            message["hash"] = generate_message_hash(message["content"])
            
        if message["hash"] not in seen_hashes:
            seen_hashes.add(message["hash"])
            deduplicated_history.append(message)
    
    # Zeige die deduplizierte Chat-Historie im Container
    with chat_container:
        # Gruppiere Nachrichten in Paare (Benutzer + Antwort)
        for i in range(0, len(deduplicated_history), 2):
            if i + 1 < len(deduplicated_history):  # Stelle sicher, dass sowohl Benutzer als auch Antwort vorhanden sind
                # Nachrichten in der richtigen Reihenfolge anzeigen (Benutzer zuerst, dann Antwort)
                with st.chat_message("user"):
                    st.markdown(deduplicated_history[i]["content"])
                with st.chat_message("assistant"):
                    st.markdown(deduplicated_history[i+1]["content"])
            elif i < len(deduplicated_history):  # Falls nur eine Benutzer-Nachricht ohne Antwort existiert
                with st.chat_message("user"):
                    st.markdown(deduplicated_history[i]["content"])

def reset_current_chat():
    """Setzt den aktuellen Chat-Verlauf zurück"""
    if st.session_state.active_tab == "standard":
        st.session_state.chat_history = []
    else:
        st.session_state.simple_chat_history = []
    
    st.success("Der aktuelle Chat wurde zurückgesetzt!")

def ensure_vectorstore_exists():
    """Ensure vector store exists and is accessible."""
    try:
        # Use our singleton pattern to get the vector store instance
        # This will now use the integrated embedding approach
        logger.info("Initializing vector store with integrated embedding")
        vector_store = get_vector_store_instance()
        
        # Store in session state
        st.session_state.vector_store = vector_store
        logger.info("Vector store initialized successfully")
        
        return vector_store
    except ValueError as e:
        if "API key is missing" in str(e) or "environment is missing" in str(e):
            # Special handling for API key issues
            st.error("Fehler: Pinecone API-Schlüssel oder Umgebungsvariablen fehlen")
            logger.error(f"Pinecone configuration error: {str(e)}")
            st.warning("""
            ## Pinecone API-Konfiguration fehlt
            
            Bitte stellen Sie sicher, dass Sie Ihre Pinecone API-Konfiguration korrekt eingerichtet haben:
            
            ### Für Streamlit Cloud:
            Gehen Sie zu den Streamlit Cloud-Einstellungen > Secrets und fügen Sie die folgende Konfiguration hinzu:
            
toml
            [pinecone]
            api_key = "Ihr-Pinecone-API-Schlüssel"
            environment = "Ihre-Pinecone-Umgebung"
            index_name = "Ihr-Index-Name"
            namespace = "Ihr-Namespace"

            
            ### Für lokale Entwicklung:
            Erstellen Sie eine .env-Datei oder .streamlit/secrets.toml mit der gleichen Konfiguration.
            """)
        elif "does not exist" in str(e):
            # Special handling for missing index
            st.error(f"Fehler: Pinecone-Index existiert nicht: {str(e)}")
            logger.error(f"Pinecone index error: {str(e)}")
            st.warning("""
            ## Pinecone-Index existiert nicht
            
            Der angegebene Pinecone-Index existiert nicht. Bitte überprüfen Sie, ob Sie den richtigen Index-Namen angegeben haben.
            """)
        else:
            # Generic error handling
            st.error(f"Fehler bei der Verbindung zur Pinecone-Vektordatenbank: {str(e)}")
            logger.error(f"Pinecone vector store error: {str(e)}")
            
        st.session_state.vector_store = None
        return None
    except Exception as e:
        # Generic error handling
        st.error(f"Unerwarteter Fehler bei der Verbindung zur Pinecone-Vektordatenbank: {str(e)}")
        logger.error(f"Unexpected error initializing vector store: {str(e)}")
        st.session_state.vector_store = None
        return None

def main():
    
    # Initialize Supabase service
    if 'supabase_service' not in st.session_state:
        supabase_service = SupabaseService()
        st.session_state.supabase_service = supabase_service
        logger.info("Supabase service initialized successfully")

    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(os.urandom(32)).encode()).hexdigest()
        logger.info("Session ID initialized successfully")
    
    # Titel ohne Logo
    st.title("Koalitionskompass")
    st.markdown("Dein interaktiver Programm-Guide")
    
    # Info-Box mit Beschreibung des Chatbots
    st.info("""
        **📖 Über diesen Chatbot:**
        
        Dieser Chatbot beantwortet Ihre Fragen zum Regierungsprogramm 2025-2029. Die Antworten basieren auf den Inhalten des offiziellen Dokuments, mit Quellenangaben zu den entsprechenden Seiten.
        
        *Bitte beachten Sie: Der Chatbot kann unvollständige oder falsche Antworten geben und in manchen Fällen halluzinieren. Überprüfen Sie bitte immer die angezeigten Quellenangaben.*
    """)
    
    # Initialize session variables that don't depend on the vector store
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "simple_chat_history" not in st.session_state:
        st.session_state.simple_chat_history = []
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "standard"
    
    # Ensure the vector store exists - must happen before chatbot initialization
    vector_store = ensure_vectorstore_exists()
    
    # Initialize chatbot only if vector store is available
    if "chatbot" not in st.session_state:
        if vector_store is not None:
            try:
                logger.info("Initializing ChatBot with efficient retriever")
                st.session_state.chatbot = ChatBot(vector_store, use_efficient_retriever=True)
                logger.info("ChatBot with efficient retriever initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing ChatBot with efficient retriever: {str(e)}")
                try:
                    logger.info("Falling back to standard retriever")
                    st.session_state.chatbot = ChatBot(vector_store, use_efficient_retriever=False)
                    logger.info("ChatBot with standard retriever initialized successfully")
                except Exception as fallback_error:
                    logger.error(f"Error initializing ChatBot with standard retriever: {str(fallback_error)}")
                    st.session_state.chatbot = None
        else:
            logger.error("Cannot initialize chatbot: vector store is None")
            st.session_state.chatbot = None
    
    # Initialize simple chatbot if needed
    if "simple_chatbot" not in st.session_state:
        try:
            st.session_state.simple_chatbot = SimpleChatbot()
            logger.info("SimpleChatbot initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SimpleChatbot: {str(e)}")
            st.session_state.simple_chatbot = None
    
    # Einfache Modusauswahl mit Buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Standard", key="standard_mode", 
                    type="primary" if st.session_state.active_tab == "standard" else "secondary",
                    use_container_width=True):
            st.session_state.active_tab = "standard"    
            st.rerun()
            
    with col2:
        if st.button("Einfache Sprache", key="simple_mode", 
                    type="primary" if st.session_state.active_tab == "simple" else "secondary",
                    use_container_width=True):
            st.session_state.active_tab = "simple"
            st.rerun()
    
    # Hinweis zum aktiven Modus
    if st.session_state.active_tab == "standard":
        st.caption("Sie sind im Standard-Modus")
    else:
        st.caption("Sie sind im Modus 'Einfache Sprache'")
    
    # Chat Container erstellen - wir packen das Chat Interface in ein Container,
    # damit es in der Rangfolge vor dem Footer erscheint
    chat_container = st.container()
    
    # Reset-Button für den aktuellen Chat
    st.button("Aktuellen Chat zurücksetzen", on_click=reset_current_chat, use_container_width=True)
    
    # Footer - jetzt am Ende, nach dem Chat-Container
    footer_container = st.container()
    with footer_container:
        st.markdown("---")
        st.markdown("© 2025 AI Empowered Politics and Dejan Đukić | Koalitionskompass")
    
    # Chat Interface innerhalb des Containers rendern
    with chat_container:
        if st.session_state.active_tab == "standard":
            render_chat_interface(simple_language=False)
        else:
            render_chat_interface(simple_language=True)

if __name__ == "__main__":
    main()