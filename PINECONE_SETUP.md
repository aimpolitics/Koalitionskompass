# Pinecone Einrichtung

Diese Anleitung beschreibt, wie Sie Pinecone für Ihren Koalitionskompass einrichten können.

## 1. Konto bei Pinecone erstellen

1. Besuchen Sie [Pinecone](https://www.pinecone.io/) und klicken Sie auf "Sign Up"
2. Erstellen Sie ein Konto mit Ihrer E-Mail-Adresse oder über Google/GitHub
3. Bestätigen Sie Ihre E-Mail-Adresse und melden Sie sich an

## 2. Neuen Index erstellen

Hinweis: Sie müssen keinen Index manuell erstellen, da die Anwendung dies automatisch tut. Diese Schritte sind nur für Ihre Information.

1. Klicken Sie auf "Create Index"
2. Wählen Sie einen Namen für Ihren Index (z.B. "koalitionskompass")
3. Wählen Sie die Dimension 384 (für das verwendete Embedding-Modell)
4. Wählen Sie "cosine" als Metrik
5. Wählen Sie "Starter" als Pod-Typ für den kostenlosen Plan

## 3. API-Key generieren

1. Klicken Sie auf "API Keys" in der linken Seitenleiste
2. Kopieren Sie den API-Key (er wird nur einmal angezeigt)

## 4. Environment-Name notieren

1. Auf der Index-Übersichtsseite sehen Sie den "Environment"-Namen
2. Dieser sieht typischerweise aus wie "gcp-starter" oder "us-west1-gcp"
3. Notieren Sie sich diesen Namen

## 5. Konfiguration in Streamlit Cloud

Fügen Sie die folgenden Werte in Ihre Streamlit Cloud Secrets ein:

```toml
[pinecone]
api_key = "Ihr-Pinecone-API-Key"
environment = "Ihre-Pinecone-Environment" # z.B. "gcp-starter"
index_name = "koalitionskompass"
namespace = "default"
```

## 6. Lokale Entwicklung

Für die lokale Entwicklung kopieren Sie die gleichen Werte in Ihre `.streamlit/secrets.toml` Datei oder setzen Sie die entsprechenden Umgebungsvariablen:

```bash
export PINECONE_API_KEY="Ihr-Pinecone-API-Key"
export PINECONE_ENVIRONMENT="Ihre-Pinecone-Environment"
export PINECONE_INDEX_NAME="koalitionskompass"
export PINECONE_NAMESPACE="default"
```

## 7. Erste Verwendung

Bei der ersten Verwendung wird die Anwendung automatisch:

1. Einen neuen Index in Pinecone erstellen, falls er noch nicht existiert
2. Das PDF-Dokument verarbeiten und die Vektoren in Pinecone speichern

Dies kann einige Minuten dauern, abhängig von der Größe des Dokuments.

## Vorteile von Pinecone

- Kostenloser Starter-Plan mit 1 Million Vektoren
- Hohe Skalierbarkeit für größere Projekte
- Schnelle Abfragen auch bei großen Datenmengen
- Einfache API und gute Dokumentation
- Keine Größenbeschränkungen durch GitHub

## Kostenloser Plan Limits

Der kostenlose "Starter"-Plan von Pinecone bietet:

- 1 Index
- 1 Million Vektoren
- 10.000 Abfragen pro Tag
- 1 Replikat
- 1 Pod

Dies ist mehr als ausreichend für den Koalitionskompass mit einem einzelnen PDF-Dokument. 