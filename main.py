from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# We import these but don't initialize them yet
from app.rag import RAGSystem

app = FastAPI(title="AI Policy Assistant API")

# Enable CORS for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold our RAG engine
rag_engine = None

@app.on_event("startup")
async def startup_event():
    # This allows the server to start FIRST, then load the AI
    logging.info("Server is waking up...")

def get_rag():
    global rag_engine
    if rag_engine is None:
        logging.info("Loading AI Models (Lazy Loading)...")
        rag_engine = RAGSystem()
    return rag_engine

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"status": "online", "message": "Policy AI Backend is Live"}

@app.post("/api/query")
async def handle_query(request: QueryRequest):
    try:
        # Load the engine only when the first question is asked
        engine = get_rag()
        result = await engine.query(request.query)
        return result
    except Exception as e:
        logging.error(f"Query Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
