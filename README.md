# Koalitionskompass

Ein interaktiver Chatbot, der Fragen zum Regierungsprogramm 2025-2029 beantwortet, basierend auf dem offiziellen PDF-Dokument.

## Schritt-für-Schritt Anleitung zum Testen des Projekts

### 1. Voraussetzungen

Stellen Sie sicher, dass folgende Komponenten auf Ihrem System installiert sind:

- Python 3.9 oder höher (empfohlen: Python 3.12)
- Git
- pip (Python Paketmanager)
- Ein Terminal oder eine Kommandozeile
- Das Regierungsprogramm 2025 PDF (kann von hier heruntergeladen werden: https://www.dievolkspartei.at/Download/Regierungsprogramm_2025.pdf)

### 2. Repository klonen

```bash
git clone [repository-url]
cd Koalitionskompass
```

### 3. Python-Umgebung erstellen und aktivieren

#### Für macOS/Linux:
```bash
python -m venv ven
source venv/bin/activate
```

#### Für Windows:
```bash
python -m venv venv312
venv312\Scripts\activate
```

Wichtig: Im Terminal sollte nun `(venv312)` am Anfang der Zeile erscheinen, was bedeutet, dass die virtuelle Umgebung aktiv ist.

### 4. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

Dieser Vorgang kann einige Minuten dauern, da die Pakete und ihre Abhängigkeiten heruntergeladen und installiert werden.

### 5. Umgebungsvariablen konfigurieren

Sie benötigen API-Schlüssel für OpenAI und Pinecone. Erstellen Sie eine `.env` Datei im Projektverzeichnis mit folgenden Inhalten:

```
OPENAI_API_KEY=ihr-openai-api-key
PINECONE_API_KEY=ihr-pinecone-api-key
PINECONE_ENVIRONMENT=ihre-pinecone-environment (z.B. us-east-1)
PINECONE_INDEX_NAME=ihr-pinecone-index-name
```

Alternativ können Sie diese Konfiguration auch über Streamlit Secrets vornehmen (siehe Schritt 7).

### 6. Pinecone Einrichtung (falls nicht vorhanden)

Falls Sie noch keinen Pinecone-Index haben, befolgen Sie diese Schritte:

1. Registrieren Sie sich für ein kostenloses Konto bei [Pinecone](https://www.pinecone.io/)
2. Erstellen Sie einen neuen Index:
   - Wählen Sie einen Namen (z.B. "koalitionskompass")
   - Dimension: 1536 (für OpenAI Embeddings)
   - Metric: cosine
   - Server-Region: (empfohlen: us-east-1)
3. Notieren Sie sich den API-Schlüssel, die Umgebung und den Index-Namen für Schritt 5

Für detailliertere Anweisungen siehe die Datei `PINECONE_SETUP.md` im Repository.

### 7. Streamlit Secrets konfigurieren (alternative zu .env)

Erstellen Sie eine Datei `.streamlit/secrets.toml` mit folgendem Inhalt:

```toml
[pinecone]
api_key = "ihr-pinecone-api-key"
environment = "ihre-pinecone-environment"
index_name = "ihr-pinecone-index-name"
namespace = "koalitionskompass"

[openai]
api_key = "ihr-openai-api-key"
```

### 8. Vektordatenbank erstellen (wenn noch nicht vorhanden)

Wenn Sie das Projekt zum ersten Mal ausführen, müssen Sie die PDF-Datei in die Vektordatenbank laden:

1. Laden Sie das Regierungsprogramm 2025 PDF herunter: https://www.dievolkspartei.at/Download/Regierungsprogramm_2025.pdf
2. Erstellen Sie einen `data` Ordner im Projektverzeichnis, falls dieser noch nicht existiert
3. Speichern Sie die PDF-Datei als `Regierungsprogramm_2025.pdf` im `data` Ordner
4. Führen Sie den folgenden Befehl aus:

```bash
python create_vectorstore.py
```

Dieser Prozess extrahiert Text aus dem PDF, teilt ihn in Chunks auf, erstellt Embeddings und speichert alles in Pinecone.

### 9. Anwendung starten

```bash
streamlit run app.py
```

Öffnen Sie einen Webbrowser und navigieren Sie zu:
```
http://localhost:8501
```

Die App sollte nun laufen und Sie können Fragen zum Regierungsprogramm stellen.

### 10. Fehlerbehebung

#### ModuleNotFoundError
Wenn Fehler wie `ModuleNotFoundError: No module named 'xyz'` auftreten:
```bash
pip install xyz
```

#### Pinecone Verbindungsprobleme
Überprüfen Sie Ihre API-Schlüssel und Umgebungsvariablen. Stellen Sie sicher, dass Ihr Index korrekt eingerichtet ist.

#### Streamlit läuft, aber der Chat funktioniert nicht
Überprüfen Sie die Console-Ausgabe auf Fehlermeldungen. Häufige Probleme sind fehlerhafte API-Schlüssel oder Probleme mit der Vektordatenbank.

## Git-Verwaltung

Wichtiger Hinweis: Die virtuelle Umgebung (`venv312/`) sollte nicht in Git aufgenommen werden, da sie sehr große Dateien enthält. Die `.gitignore`-Datei ist bereits so konfiguriert, dass der `venv312/`-Ordner ignoriert wird.

Falls dieser Ordner bereits verfolgt wird, können Sie ihn mit folgendem Befehl aus dem Git-Index entfernen:

```bash
git rm --cached -r venv312/
```

## Features

- PDF-Verarbeitung und semantische Suche mit Pinecone Vektordatenbank
- Benutzerfreundliche Chat-Oberfläche mit Streamlit
- Quellenangaben für Antworten
- Standard- und "Einfache Sprache"-Modus
- Mehrere Deployment-Optionen (Streamlit Cloud, Docker, API)

## Installation

1. Repository klonen:
```bash
git clone [repository-url]
cd Koalitionskompass
```

2. Python-Umgebung erstellen und aktivieren:
```bash
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate
```

3. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

4. Umgebungsvariablen konfigurieren:
Erstellen Sie eine `.env` Datei im Projektverzeichnis:
```
OPENAI_API_KEY=ihr-openai-api-key
```

## Lokale Verwendung

Streamlit-App starten:
```bash
streamlit run app.py
```

Öffnen Sie einen Webbrowser und navigieren Sie zu:
```
http://localhost:8501
```

## Deployment-Optionen

### Option 1: Streamlit Cloud (empfohlen)

1. Pushen Sie Ihr Repository zu GitHub/GitLab/Bitbucket
2. Melden Sie sich bei [Streamlit Cloud](https://streamlit.io/cloud) an
3. Verknüpfen Sie Ihr Repository und deployen Sie die Anwendung
4. Konfigurieren Sie Umgebungsvariablen (OPENAI_API_KEY) in den Streamlit Cloud Secrets

### Option 2: Docker

Mit dem beiliegenden Dockerfile können Sie die Anwendung containerisieren:

```bash
docker build -t koalitionskompass .
docker run -p 8501:8501 koalitionskompass
```

### Option 3: FastAPI Backend

Für fortgeschrittene Integrationen können Sie die beiliegende API verwenden:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Website-Integration

Der Chatbot kann auf zwei Arten in Ihre Website integriert werden:

1. **iframe-Integration (Streamlit UI):**
```html
<iframe src="https://ihre-streamlit-url.streamlit.app" width="100%" height="800px" frameborder="0"></iframe>
```

2. **API-Integration (für angepasste UIs):**
Verwenden Sie unsere REST-API-Endpunkte für eine nahtlose Integration mit Ihrem Frontend.

```javascript
// Beispiel für einen API-Aufruf
async function askQuestion(question, simpleLanguage = false) {
  const response = await fetch('https://ihre-api-url/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: question,
      simple_language: simpleLanguage
    }),
  });
  return await response.json();
}
```

## Datenschutz und DSGVO

- Alle Daten werden lokal verarbeitet
- Keine Speicherung von Benutzerinformationen
- Transparente Quellenangaben
- Konform mit DSGVO-Anforderungen

## Projektstruktur und Dateiabhängigkeiten

Die Anwendung besteht aus folgenden Hauptkomponenten:

- `app.py` - Hauptanwendungsdatei mit der Streamlit-Benutzeroberfläche. Enthält Funktionen für die Initialisierung des Session-States, das Formatieren von Quellen, die Chat-Oberfläche und das Zurücksetzen des Chats.

- `simple_chatbot.py` - Implementiert die Kern-Chatbot-Funktionalität, verarbeitet Benutzeranfragen und generiert Antworten basierend auf dem Regierungsprogramm.

- `pdf_processor.py` - Verantwortlich für das Einlesen und Verarbeiten des PDF-Dokuments, das Aufteilen in Chunks und das Speichern in der Vektordatenbank (ChromaDB).

- `api.py` - Stellt eine REST-API-Schnittstelle für die Integration in andere Anwendungen bereit.

- `config.py` - Zentrale Konfigurationsdatei mit Einstellungen für das Modell, Vektordatenbank und andere Parameter.

- `data/` - Verzeichnis für das PDF-Dokument und die generierte Vektordatenbank.

- `Dockerfile` - Definition für die Containerisierung der Anwendung.

- `requirements.txt` - Liste aller Python-Abhängigkeiten.

**Datenfluss:**
1. Der Benutzer stellt eine Frage über die Streamlit-UI in `app.py`
2. Die Anfrage wird an den `simple_chatbot.py` weitergeleitet
3. Der Chatbot sucht relevante Abschnitte aus der durch `pdf_processor.py` erstellten Vektordatenbank
4. Die gefundenen Informationen werden verwendet, um eine Antwort zu generieren
5. Die Antwort mit Quellenangaben wird an die UI zurückgegeben

## Lizenz

MIT License 