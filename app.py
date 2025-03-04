import streamlit as st
import os
from simple_chatbot import SimpleChatbot
from pinecone_processor import PineconePDFProcessor
import logging

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
    # Separate Chat-Historien für die beiden Modi
    if "standard_messages" not in st.session_state:
        st.session_state.standard_messages = []
        
    if "simple_messages" not in st.session_state:
        st.session_state.simple_messages = []
        
    if "chatbot" not in st.session_state:
        # Chatbot initialisieren
        st.session_state.chatbot = SimpleChatbot()
        
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "standard"

    if "simple_chatbot" not in st.session_state:
        # Verwende Pinecone für die Vektordatenbank
        st.session_state.simple_chatbot = SimpleChatbot()

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
    """Rendert die Chat-Oberfläche und verarbeitet Nutzereingaben."""
    # Wähle die richtige Chat-Historie basierend auf dem Modus
    messages_key = "simple_messages" if simple_language else "standard_messages"
    
    # Chat-Container
    chat_container = st.container()
    
    with chat_container:
        # Zeige alle bisherigen Nachrichten aus der entsprechenden Historie an
        for message in st.session_state[messages_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input für neue Nachrichten
        placeholder_text = "Stellen Sie eine Frage zum Regierungsprogramm (Einfache Sprache)..." if simple_language else "Stellen Sie eine Frage zum Regierungsprogramm..."
        if prompt := st.chat_input(placeholder_text):
            # Füge Nutzer-Nachricht zur entsprechenden Chat-Historie hinzu
            st.session_state[messages_key].append({"role": "user", "content": prompt})
            
            # Zeige Nutzer-Nachricht an
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Verarbeite die Anfrage
            with st.chat_message("assistant"):
                with st.spinner("Suche im Regierungsprogramm..."):
                    # Modifiziere den Prompt für einfache Sprache wenn nötig
                    if simple_language:
                        prompt_to_use = f"Bitte erkläre in einfacher Sprache: {prompt}"
                    else:
                        prompt_to_use = prompt
                    
                    # Erhalte Antwort vom Chatbot mit der korrekten Methode get_response
                    response = st.session_state.chatbot.get_response(prompt_to_use)
                    
                    # Extrahiere Antworttext und Quellen
                    answer_text = response.get("content", response.get("answer", "Keine Antwort gefunden."))
                    sources = response.get("sources", [])
                    
                    # Zeige die Antwort an
                    st.markdown(answer_text)
                    
                    # Quellen anzeigen, wenn vorhanden
                    if sources:
                        sources_html = "<div class='sources'><h4>Quellen:</h4><ul>"
                        for source in sources:
                            sources_html += f"<li>{format_source(source)}</li>"
                        sources_html += "</ul></div>"
                        st.markdown(sources_html, unsafe_allow_html=True)
                    
                    # Speichere die Antwort in der entsprechenden Chat-Historie
                    full_response = answer_text
                    if sources:
                        full_response += "\n\n**Quellen:**\n" + "\n".join([format_source(s) for s in sources])
                    
                    st.session_state[messages_key].append({"role": "assistant", "content": full_response})

def reset_current_chat():
    """Setzt nur den aktuell aktiven Chat zurück"""
    if st.session_state.active_tab == "standard":
        st.session_state.standard_messages = []
    else:
        st.session_state.simple_messages = []

# Überprüfen und erstellen der Pinecone-Vektordatenbank, falls sie nicht existiert
def ensure_vectorstore_exists():
    try:
        # Versuche, die Vektordatenbank zu laden
        processor = PineconePDFProcessor()
        processor.load_vector_store()
        st.success("Verbindung zur Pinecone-Vektordatenbank hergestellt!")
    except Exception as e:
        st.error(f"Fehler beim Verbinden mit der Pinecone-Vektordatenbank: {str(e)}")
        st.info("Bitte stellen Sie sicher, dass die Pinecone-Datenbank bereits erstellt wurde.")
        st.info("Führen Sie lokal 'python create_vectorstore.py' aus, um die Datenbank zu erstellen.")
        st.stop()

def main():
    # Stellen Sie sicher, dass die Vektordatenbank existiert
    ensure_vectorstore_exists()
    
    # Session State initialisieren
    initialize_session_state()
    
    # Titel ohne Logo
    st.title("Koalitionskompass")
    st.markdown("Dein interaktiver Programm-Guide")
    
    # Info-Box mit Beschreibung des Chatbots
    st.info("""
        **📖 Über diesen Chatbot:**
        
        Dieser Chatbot beantwortet Ihre Fragen zum Regierungsprogramm 2025-2029. Die Antworten basieren auf den Inhalten des offiziellen Dokuments, mit Quellenangaben zu den entsprechenden Seiten.
        
        *Bitte beachten Sie: Der Chatbot kann unvollständige oder falsche Antworten geben und in manchen Fällen halluzinieren. Überprüfen Sie bitte immer die angezeigten Quellenangaben.*
    """)
    
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
    
    # Chat Interface basierend auf aktivem Tab rendern
    if st.session_state.active_tab == "standard":
        render_chat_interface(simple_language=False)
    else:
        render_chat_interface(simple_language=True)
    
    # Reset-Button für den aktuellen Chat
    st.button("Aktuellen Chat zurücksetzen", on_click=reset_current_chat, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2023 AimPolitics | Koalitionskompass")

if __name__ == "__main__":
    main()