from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS, SYSTEM_PROMPT
import os

class Chatbot:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
        # Configure OpenAI settings
        os.environ["OPENAI_API_BASE"] = "https://oai.hconeai.com/v1"
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            model_kwargs={"stop": None}
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=self.memory,
            return_source_documents=True,
            verbose=True,
            chain_type="stuff",
            rephrase_question=False,
            output_key="answer"
        )
        
    def get_response(self, query: str) -> Dict:
        """Get response for user query."""
        try:
            response = self.chain.invoke({"question": query})
            return {
                "answer": response["answer"],
                "sources": [doc.page_content for doc in response["source_documents"]]
            }
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return {
                "answer": "Entschuldigung, es gab einen Fehler bei der Verarbeitung Ihrer Anfrage. Bitte versuchen Sie es erneut.",
                "sources": []
            }
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear() 