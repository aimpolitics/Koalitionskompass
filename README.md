# Koalitionskompass

Ein interaktiver Chatbot, der Fragen zum Regierungsprogramm 2025-2029 beantwortet, basierend auf dem offiziellen PDF-Dokument.

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