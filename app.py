import streamlit as st
import os
from simple_chatbot import SimpleChatbot
from chatbot import ChatBot
import logging
from pinecone_processor import PineconePDFProcessor, initialize_pinecone, create_embeddings, count_documents, PineconeConnectionError

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

def initialize_session_state():
    """Initialisiert die Session-Variablen, wenn sie noch nicht existieren."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "simple_chat_history" not in st.session_state:
        st.session_state.simple_chat_history = []
    
    if "vector_store" not in st.session_state:
        # Vector store should already be initialized by ensure_vectorstore_exists
        pass
    
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = ChatBot(st.session_state.vector_store)
        except ValueError as e:
            st.error(f"Fehler bei der Initialisierung des Standard-Chatbots: {str(e)}")
            st.warning("""
            ## OpenAI API-Schlüssel fehlt oder ist ungültig
            
            Bitte stellen Sie sicher, dass Sie einen gültigen OpenAI API-Schlüssel konfiguriert haben:
            
            ### Für Streamlit Cloud:
            Gehen Sie zu den Streamlit Cloud-Einstellungen > Secrets und fügen Sie die folgende Konfiguration hinzu:
            ```toml
            [openai]
            api_key = "sk-Ihr-OpenAI-API-Schlüssel"
            ```
            
            ### Für lokale Entwicklung:
            Erstellen Sie eine `.env`-Datei oder `.streamlit/secrets.toml` mit der gleichen Konfiguration.
            """)
            st.session_state.chatbot = None
        except Exception as e:
            st.error(f"Unerwarteter Fehler bei der Initialisierung des Standard-Chatbots: {str(e)}")
            st.session_state.chatbot = None
    
    if "simple_chatbot" not in st.session_state:
        try:
            st.session_state.simple_chatbot = SimpleChatbot()
        except ValueError as e:
            st.error(f"Fehler bei der Initialisierung des einfachen Chatbots: {str(e)}")
            st.warning("""
            ## OpenAI API-Schlüssel fehlt oder ist ungültig
            
            Bitte stellen Sie sicher, dass Sie einen gültigen OpenAI API-Schlüssel konfiguriert haben:
            
            ### Für Streamlit Cloud:
            Gehen Sie zu den Streamlit Cloud-Einstellungen > Secrets und fügen Sie die folgende Konfiguration hinzu:
            ```toml
            [openai]
            api_key = "sk-Ihr-OpenAI-API-Schlüssel"
            ```
            
            ### Für lokale Entwicklung:
            Erstellen Sie eine `.env`-Datei oder `.streamlit/secrets.toml` mit der gleichen Konfiguration.
            """)
            st.session_state.simple_chatbot = None
        except Exception as e:
            st.error(f"Unerwarteter Fehler bei der Initialisierung des einfachen Chatbots: {str(e)}")
            st.session_state.simple_chatbot = None
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "standard"

def format_source(source):
    """Formatiert eine Quelle für die Anzeige."""
    try:
        if isinstance(source, dict):
            # Wenn die Quelle ein Dictionary ist
            page = source.get("page", "Unbekannte Seite")
            text = source.get("text", "")
            return f"<strong>Seite {page}:</strong> {text[:150]}..."
        elif isinstance(source, str):
            # Wenn die Quelle ein String ist
            return source
        else:
            # Fallback für unerwartete Datentypen
            return str(source)
    except Exception as e:
        # Bei Fehlern geben wir einen Standardtext zurück
        return "Quelle konnte nicht formatiert werden"

def render_chat_interface(simple_language=False):
    """Rendert das Chat-Interface je nach ausgewähltem Modus"""
    
    # Auswahl der richtigen Chat-Historie basierend auf dem aktiven Modus
    chat_history = st.session_state.simple_chat_history if simple_language else st.session_state.chat_history
    
    # Chat-Verlauf anzeigen
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input für Benutzer
    user_input = st.chat_input("Stellen Sie eine Frage zum Regierungsprogramm...")
    
    if user_input:
        # Benutzer-Nachricht speichern und anzeigen
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Speichere Nachricht in der richtigen Chat-Historie
        if simple_language:
            st.session_state.simple_chat_history.append({"role": "user", "content": user_input})
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
        
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
        
        # Antwort-Platzhalter
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Denke...")
            
            try:
                # Antwort vom Chatbot
                response = chatbot.get_response(user_input, simple_language=simple_language)
                
                # Antwort anzeigen
                message_placeholder.markdown(response)
                
                # Speichere Antwort in der richtigen Chat-Historie
                if simple_language:
                    st.session_state.simple_chat_history.append({"role": "assistant", "content": response})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
            except Exception as e:
                error_message = f"Entschuldigung, ich konnte keine Antwort generieren: {str(e)}"
                message_placeholder.markdown(error_message)
                
                # Fehler in der Chat-Historie speichern
                if simple_language:
                    st.session_state.simple_chat_history.append({"role": "assistant", "content": error_message})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})

def reset_current_chat():
    """Setzt den aktuellen Chat-Verlauf zurück"""
    if st.session_state.active_tab == "standard":
        st.session_state.chat_history = []
    else:
        st.session_state.simple_chat_history = []
    
    st.success("Der aktuelle Chat wurde zurückgesetzt!")

# Überprüfen und erstellen der Pinecone-Vektordatenbank, falls sie nicht existiert
def ensure_vectorstore_exists():
    try:
        # Initialisiere Pinecone
        vector_store = initialize_pinecone()
        st.success("Verbindung zur Pinecone-Vektordatenbank hergestellt!")
        return vector_store
    except ValueError as e:
        if "API key is missing" in str(e) or "environment is missing" in str(e):
            # Special handling for API key issues
            st.error("Fehler: Pinecone API-Schlüssel oder Umgebungsvariablen fehlen")
            st.warning("""
            ## Pinecone API-Konfiguration fehlt
            
            ### Für Streamlit Cloud:
            Gehen Sie zu den Streamlit Cloud-Einstellungen > Secrets und fügen Sie die folgende Konfiguration hinzu:
            ```toml
            [pinecone]
            api_key = "Ihr-Pinecone-API-Schlüssel"
            environment = "Ihre-Pinecone-Region"
            index_name = "koalitionskompass"
            namespace = "default"
            ```
            
            ### Für lokale Entwicklung:
            Erstellen Sie eine `.env`-Datei oder `.streamlit/secrets.toml` mit der gleichen Konfiguration.
            """)
            st.stop()
        else:
            # General error handling
            st.error(f"Fehler beim Verbinden mit der Pinecone-Vektordatenbank: {str(e)}")
            st.info("Bitte stellen Sie sicher, dass die Pinecone-Datenbank bereits erstellt wurde.")
            st.info("Führen Sie lokal 'python create_vectorstore.py' aus, um die Datenbank zu erstellen.")
            st.stop()
    except Exception as e:
        st.error(f"Unerwarteter Fehler: {str(e)}")
        st.info("Bitte überprüfen Sie die Logs für weitere Details.")
        st.stop()

def main():
    # Titel ohne Logo
    st.title("Koalitionskompass")
    st.markdown("Dein interaktiver Programm-Guide")
    
    # Info-Box mit Beschreibung des Chatbots
    st.info("""
        **📖 Über diesen Chatbot:**
        
        Dieser Chatbot beantwortet Ihre Fragen zum Regierungsprogramm 2025-2029. Die Antworten basieren auf den Inhalten des offiziellen Dokuments, mit Quellenangaben zu den entsprechenden Seiten.
        
        *Bitte beachten Sie: Der Chatbot kann unvollständige oder falsche Antworten geben und in manchen Fällen halluzinieren. Überprüfen Sie bitte immer die angezeigten Quellenangaben.*
    """)
    
    # Stellen Sie sicher, dass die Vektordatenbank existiert
    # Dies muss vor der Initialisierung des ChatBots geschehen
    vector_store = ensure_vectorstore_exists()
    if vector_store:
        st.session_state.vector_store = vector_store
    
    # Session State initialisieren (inklusive Chatbot)
    initialize_session_state()
    
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
        st.markdown("© 2023 AimPolitics | Koalitionskompass")
    
    # Chat Interface innerhalb des Containers rendern
    with chat_container:
        if st.session_state.active_tab == "standard":
            render_chat_interface(simple_language=False)
        else:
            render_chat_interface(simple_language=True)

if __name__ == "__main__":
    main()