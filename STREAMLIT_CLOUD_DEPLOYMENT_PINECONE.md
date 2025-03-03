# Deployment auf Streamlit Cloud mit Pinecone

Diese Anleitung beschreibt, wie Sie den Koalitionskompass-Chatbot auf Streamlit Cloud deployen, wobei eine bereits existierende Pinecone-Vektordatenbank verwendet wird.

## Voraussetzungen

1. **GitHub-Repository**: Ihr Code muss in einem GitHub-Repository sein.
2. **Pinecone-Konto**: Sie benötigen ein Pinecone-Konto mit einem bereits erstellten Index.
3. **OpenAI-API-Schlüssel**: Sie benötigen einen gültigen OpenAI-API-Schlüssel.

## Vorbereitung der Vektordatenbank

Bevor Sie die Anwendung auf Streamlit Cloud deployen, müssen Sie die Vektordatenbank lokal erstellen:

1. Klonen Sie das Repository auf Ihren lokalen Computer.
2. Erstellen Sie eine `.env` Datei mit Ihren API-Schlüsseln:
   ```
   PINECONE_API_KEY=Ihr_Pinecone_API_Schlüssel
   PINECONE_ENVIRONMENT=Ihre_Pinecone_Region
   PINECONE_INDEX_NAME=koalitionskompass
   PINECONE_NAMESPACE=default
   OPENAI_API_KEY=Ihr_OpenAI_API_Schlüssel
   ```
3. Erstellen Sie eine virtuelle Umgebung und installieren Sie die Abhängigkeiten:
   ```
   python -m venv venv
   source venv/bin/activate  # Unter Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Führen Sie das Skript zur Erstellung der Vektordatenbank aus:
   ```
   python create_vectorstore.py
   ```
5. Überprüfen Sie in der Pinecone-Konsole, ob der Index erstellt wurde und Daten enthält.

## Deployment auf Streamlit Cloud

1. **Streamlit Cloud Account erstellen**:
   - Besuchen Sie [share.streamlit.io](https://share.streamlit.io/)
   - Melden Sie sich mit Ihrem GitHub-Account an.

2. **Neue App erstellen**:
   - Klicken Sie auf "New app"
   - Wählen Sie Ihr GitHub-Repository aus
   - Wählen Sie den Branch (normalerweise "main")
   - Geben Sie den Pfad zur Hauptdatei an (in Ihrem Fall "app.py")

3. **Secrets konfigurieren**:
   - Klicken Sie auf "Advanced settings" > "Secrets"
   - Fügen Sie Ihre Konfigurationswerte als TOML-Format hinzu:
   ```toml
   [openai]
   api_key = "Ihr_OpenAI_API_Schlüssel"

   [pinecone]
   api_key = "Ihr_Pinecone_API_Schlüssel"
   environment = "Ihre_Pinecone_Region"
   index_name = "koalitionskompass"
   namespace = "default"
   ```

4. **Deployment starten**:
   - Klicken Sie auf "Deploy"
   - Warten Sie, bis die Anwendung gebaut und gestartet ist

5. **Anwendung testen**:
   - Nach erfolgreichem Deployment erhalten Sie eine URL, unter der Ihre Anwendung erreichbar ist
   - Testen Sie die Anwendung, um sicherzustellen, dass sie korrekt auf die Pinecone-Datenbank zugreift

## Wichtige Hinweise

- Die Anwendung ist so konfiguriert, dass sie **keine** neue Vektordatenbank erstellt, sondern nur auf eine bereits existierende zugreift.
- Wenn die Verbindung zur Pinecone-Datenbank fehlschlägt, wird eine Fehlermeldung angezeigt.
- Stellen Sie sicher, dass die Pinecone-Datenbank immer verfügbar ist, da die Anwendung ohne sie nicht funktioniert.
- Wenn Sie Änderungen am PDF-Dokument vornehmen, müssen Sie die Vektordatenbank lokal neu erstellen und dann die Anwendung auf Streamlit Cloud neu deployen.

## Fehlerbehebung

- **Fehler bei der Verbindung zur Pinecone-Datenbank**: Überprüfen Sie, ob die API-Schlüssel und die Umgebungsvariablen korrekt konfiguriert sind.
- **Keine Ergebnisse bei der Suche**: Stellen Sie sicher, dass die Vektordatenbank korrekt erstellt wurde und Daten enthält.
- **Timeout bei der Anfrage**: Streamlit Cloud hat Zeitbeschränkungen für Anfragen. Wenn die Antwort zu lange dauert, kann es zu einem Timeout kommen. 