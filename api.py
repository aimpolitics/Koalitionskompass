from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from simple_chatbot import SimpleChatbot
from chatbot import ChatBot
from pinecone_processor import get_vector_store_instance
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Regierungsprogramm Chatbot API")

# CORS für die API aktivieren (wichtig für Web-Integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion einschränken: z.B. ["https://ihre-website.de"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get vector store instance
vector_store = get_vector_store_instance()

# Initialize both chatbots
simple_chatbot = SimpleChatbot()
efficient_chatbot = ChatBot(vector_store, use_efficient_retriever=True)

logger.info("API initialized with efficient ChatBot")

class QueryRequest(BaseModel):
    query: str
    simple_language: bool = False
    use_efficient_retriever: bool = True  # Default to using efficient retriever

@app.post("/api/chat")
async def get_chatbot_response(request: QueryRequest):
    try:
        logger.info(f"Received query: '{request.query}', simple_language: {request.simple_language}, use_efficient_retriever: {request.use_efficient_retriever}")
        
        if request.use_efficient_retriever:
            # Use the efficient retriever-based ChatBot
            response = efficient_chatbot.get_response(request.query, simple_language=request.simple_language)
            logger.info(f"Generated response using efficient retriever")
        else:
            # Use SimpleChatbot with standard retrieval
            prompt_to_use = f"Bitte erkläre in einfacher Sprache: {request.query}" if request.simple_language else request.query
            response = simple_chatbot.get_response(prompt_to_use)
            logger.info(f"Generated response using standard method")
        
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 