from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from simple_chatbot import SimpleChatbot
from pdf_processor import PDFProcessor
import uvicorn

app = FastAPI(title="Regierungsprogramm Chatbot API")

# CORS für die API aktivieren (wichtig für Web-Integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion einschränken: z.B. ["https://ihre-website.de"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chatbot initialisieren
pdf_processor = PDFProcessor(mode="local")
vector_store = pdf_processor.load_vector_store()
chatbot = SimpleChatbot(vector_store)

class QueryRequest(BaseModel):
    query: str
    simple_language: bool = False

@app.post("/api/chat")
async def get_chatbot_response(request: QueryRequest):
    try:
        # Query anpassen für einfache Sprache, falls gewünscht
        prompt_to_use = f"Bitte erkläre in einfacher Sprache: {request.query}" if request.simple_language else request.query
        
        # Antwort vom Chatbot erhalten
        response = chatbot.get_response(prompt_to_use)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 