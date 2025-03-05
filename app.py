import streamlit as st
import os
import logging
from PIL import Image
from pinecone_processor import get_vector_store_instance, PineconeConnectionError
from simple_chatbot import SimpleChatbot
from chatbot import ChatBot
import traceback

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

# Constants
PAGE_TITLE = "Koalitionskompass"
PAGE_ICON = "📄"

def initialize_session_state():
    """Initialize session state variables."""
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "sources" not in st.session_state:
        st.session_state.sources = []
        
    if "use_simple_language" not in st.session_state:
        st.session_state.use_simple_language = False
        
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = False
        
    if "is_initial_load" not in st.session_state:
        st.session_state.is_initial_load = True

def format_source(source):
    """Format a source document for display."""
    source_html = ""
    if isinstance(source, dict):
        # Handle the new source format (dictionary with metadata)
        page = source.get('page', 'Unbekannt')
        if page is None:
            page = 'Unbekannt'
        
        source_text = source.get('content', '')
        
        source_html = f"""
        <div style="margin-bottom: 15px; padding: 10px; border-radius: 5px; background-color: #f0f2f6;">
            <div style="font-size: 0.8em; color: #606060; margin-bottom: 5px;">
                Seite {page}
            </div>
            <div style="font-size: 0.9em;">
                {source_text}
            </div>
        </div>
        """
    else:
        # Handle the old format (plain text)
        preview = source[:200] + "..." if len(source) > 200 else source
        source_html = f"""
        <div style="margin-bottom: 15px; padding: 10px; border-radius: 5px; background-color: #f0f2f6;">
            <div style="font-size: 0.9em;">
                {preview}
            </div>
        </div>
        """
    
    return source_html

def render_chat_interface(simple_language=False):
    """Render the chat interface."""
    # Initialize the chatbot if not already initialized
    if st.session_state.chatbot is None:
        try:
            if simple_language:
                st.session_state.chatbot = SimpleChatbot()
                logger.info("SimpleChatbot initialized")
            else:
                vector_store = get_vector_store_instance()
                st.session_state.chatbot = ChatBot(vector_store)
                logger.info("Regular ChatBot initialized")
        except Exception as e:
            logger.error(f"Error initializing chatbot: {str(e)}")
            if "API key is missing" in str(e):
                st.error("⚠️ OpenAI API-Schlüssel fehlt. Bitte konfigurieren Sie den API-Schlüssel in den Streamlit Secrets oder Umgebungsvariablen.")
            else:
                traceback_str = traceback.format_exc()
                st.error(f"⚠️ Fehler beim Initialisieren des Chatbots: {str(e)}\n\n{traceback_str}")
            return

    # Display chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.write(content)
    
    # Chat input
    user_input = st.chat_input("Stellen Sie eine Frage zum Koalitionsvertrag...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get response from chatbot
        with st.chat_message("assistant"):
            with st.spinner("Denke nach..."):
                try:
                    response = st.session_state.chatbot.get_response(user_input)
                    answer = response.get("answer", "Keine Antwort erhalten.")
                    sources = response.get("sources", [])
                    
                    # Display assistant response
                    st.write(answer)
                    
                    # Update sources
                    st.session_state.sources = sources
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_message = f"Fehler beim Generieren der Antwort: {str(e)}"
                    st.error(error_message)
                    logger.error(error_message)
        
        # Display sources if enabled
        if st.session_state.show_sources and st.session_state.sources:
            with st.expander("Quellen", expanded=True):
                sources_html = ""
                for source in st.session_state.sources:
                    sources_html += format_source(source)
                
                st.markdown(sources_html, unsafe_allow_html=True)

def reset_current_chat():
    """Reset the current chat."""
    if st.session_state.chatbot:
        st.session_state.chatbot.clear_history()
    
    st.session_state.chat_history = []
    st.session_state.sources = []
    st.session_state.is_initial_load = False
    
    # Display a success message
    st.success("Chat zurückgesetzt!")
    
    # Rerun the app to clear the interface
    st.rerun()

def ensure_vectorstore_exists():
    """Ensure vector store exists and is accessible."""
    try:
        # Use our singleton pattern to get the vector store instance
        # This will now use the integrated embedding approach
        vector_store = get_vector_store_instance()
        return vector_store
    except PineconeConnectionError as e:
        logger.error(f"Pinecone connection error: {str(e)}")
        st.error(f"""
        ⚠️ Fehler bei der Verbindung zur Pinecone-Vektordatenbank: {str(e)}
        
        Bitte stellen Sie sicher, dass Sie einen gültigen Pinecone API-Schlüssel und eine gültige Umgebung konfiguriert haben.
        
        Für Streamlit Cloud Deployment, stellen Sie sicher, dass Ihre secrets.toml das richtige Format hat:
        [pinecone]
        api_key = "ihr-pinecone-api-key"
        environment = "ihre-pinecone-umgebung"
        index_name = "koalitionskompass"
        namespace = "default"
        """)
        raise
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        # More specific error for missing API key or environment
        if "API key is missing" in str(e):
            st.error(f"""
            ⚠️ Pinecone API-Schlüssel fehlt: {str(e)}
            
            Für Streamlit Cloud Deployment, stellen Sie sicher, dass Ihre secrets.toml das richtige Format hat:
            [pinecone]
            api_key = "ihr-pinecone-api-key"
            environment = "ihre-pinecone-umgebung"
            index_name = "koalitionskompass"
            namespace = "default"
            """)
        else:
            st.error(f"⚠️ Fehler: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        st.error(f"""
        ⚠️ Unerwarteter Fehler beim Zugriff auf die Vektordatenbank: {str(e)}
        
        Bitte überprüfen Sie die Logs für weitere Informationen oder erstellen Sie die Datenbank, falls sie nicht existiert.
        """)
        raise

def main():
    # Titel ohne Logo
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    
    # Initialize session state
    initialize_session_state()
    
    # First message on initial load
    if st.session_state.is_initial_load:
        with st.chat_message("assistant"):
            st.write("Hallo! Ich bin der Koalitionskompass-Assistent. Ich kann Ihnen Fragen zum Koalitionsvertrag der Bundesregierung beantworten. Wie kann ich Ihnen helfen?")
        st.session_state.is_initial_load = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Einstellungen")
        
        # Language toggle
        simple_language = st.toggle("Einfache Sprache verwenden", value=st.session_state.use_simple_language)
        
        # Update session state and reset chatbot if language preference changed
        if simple_language != st.session_state.use_simple_language:
            st.session_state.use_simple_language = simple_language
            st.session_state.chatbot = None  # Force re-initialization with new language setting
            reset_current_chat()
        
        # Sources toggle
        st.session_state.show_sources = st.toggle("Quellen anzeigen", value=st.session_state.show_sources)
        
        # Reset chat button
        st.button("Chat zurücksetzen", on_click=reset_current_chat, type="primary")
        
        # About section
        st.markdown("---")
        st.markdown("## Über")
        st.markdown("""
        Der Koalitionskompass nutzt KI, um den Koalitionsvertrag der Bundesregierung zugänglich zu machen.
        
        **Hinweis:** Die bereitgestellten Informationen dienen nur zur Orientierung und ersetzen nicht das Lesen des vollständigen Vertrags.
        """)
    
    # Try to ensure vector store connection
    try:
        ensure_vectorstore_exists()
        # Render chat interface with language preference
        render_chat_interface(simple_language=st.session_state.use_simple_language)
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        # Error messages are displayed by the ensure_vectorstore_exists function

if __name__ == "__main__":
    main()