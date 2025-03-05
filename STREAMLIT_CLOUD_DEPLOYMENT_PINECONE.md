# Deployment auf Streamlit Cloud mit Pinecone

Diese Anleitung beschreibt, wie Sie den Koalitionskompass-Chatbot auf Streamlit Cloud deployen, wobei eine bereits existierende Pinecone-Vektordatenbank verwendet wird.

## Voraussetzungen

1. **GitHub-Repository**: Ihr Code muss in einem GitHub-Repository sein.
2. **Pinecone-Konto**: Sie benötigen ein Pinecone-Konto mit einem bereits erstellten Index.
3. **OpenAI-API-Schlüssel**: Sie benötigen einen gültigen OpenAI-API-Schlüssel.

## Vorbereitung der Vektordatenbank

Bevor Sie die Anwendung auf Streamlit Cloud deployen, müssen Sie die Vektordatenbank lokal erstellen:

1. Klonen Sie das Repository auf Ihren lokalen Computer.
2. Konfigurieren Sie Ihre API-Schlüssel entweder über:
   
   **Option A: Streamlit Secrets** (empfohlen)
   ```
   mkdir -p .streamlit
   ```

   Erstellen Sie eine Datei `.streamlit/secrets.toml`:
   ```toml
   [openai]
   api_key = "Ihr_OpenAI_API_Schlüssel"

   [pinecone]
   api_key = "Ihr_Pinecone_API_Schlüssel"
   environment = "Ihre_Pinecone_Region"
   index_name = "koalitionskompass"
   namespace = "default"
   ```

   **Option B: .env Datei**
   ```
   OPENAI_API_KEY=Ihr_OpenAI_API_Schlüssel
   PINECONE_API_KEY=Ihr_Pinecone_API_Schlüssel
   PINECONE_ENVIRONMENT=Ihre_Pinecone_Region
   PINECONE_INDEX_NAME=koalitionskompass
   PINECONE_NAMESPACE=default
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

3. **Secrets konfigurieren** (wichtig!):
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

   **Wichtig:** Die Struktur der Secrets in Streamlit Cloud muss exakt dem Format oben entsprechen. Die Anwendung erwartet diese hierarchische Struktur mit Abschnitten ([openai], [pinecone]) und den dazugehörigen Schlüssel-Wert-Paaren.

4. **Deployment starten**:
   - Klicken Sie auf "Deploy"
   - Warten Sie, bis die Anwendung gebaut und gestartet ist

5. **Anwendung testen**:
   - Nach erfolgreichem Deployment erhalten Sie eine URL, unter der Ihre Anwendung erreichbar ist
   - Testen Sie die Anwendung, um sicherzustellen, dass sie korrekt auf die Pinecone-Datenbank zugreift

## Fehlersuche bei Konfigurationsproblemen

Wenn Ihre Anwendung auf Streamlit Cloud nicht funktioniert, prüfen Sie folgende häufige Probleme:

1. **Secrets-Format überprüfen**: Stellen Sie sicher, dass die Secrets genau im oben beschriebenen Format konfiguriert sind.

2. **Logs anzeigen**: In der Streamlit Cloud können Sie die Logs Ihrer Anwendung einsehen, um Fehler zu identifizieren.

3. **Häufige Fehler und Lösungen**:
   - `Pinecone API key is missing`: Überprüfen Sie, ob der API-Schlüssel korrekt in den Secrets unter `[pinecone].api_key` konfiguriert ist.
   - `Pinecone environment is missing`: Überprüfen Sie, ob die Umgebung korrekt in den Secrets unter `[pinecone].environment` konfiguriert ist.
   - `OpenAI API key is missing`: Überprüfen Sie, ob der API-Schlüssel korrekt in den Secrets unter `[openai].api_key` konfiguriert ist.

## Wichtige Hinweise

- Die Anwendung ist so konfiguriert, dass sie **keine** neue Vektordatenbank erstellt, sondern nur auf eine bereits existierende zugreift.
- Wenn die Verbindung zur Pinecone-Datenbank fehlschlägt, wird eine Fehlermeldung angezeigt.
- Stellen Sie sicher, dass die Pinecone-Datenbank immer verfügbar ist, da die Anwendung ohne sie nicht funktioniert.
- Wenn Sie Änderungen am PDF-Dokument vornehmen, müssen Sie die Vektordatenbank lokal neu erstellen und dann die Anwendung auf Streamlit Cloud neu deployen. 