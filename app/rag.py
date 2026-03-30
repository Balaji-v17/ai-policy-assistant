from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crucial: Keep this as None. DO NOT call RAGSystem() here.
rag_engine = None

@app.get("/")
async def root():
    # This route MUST be lightning fast so Render sees it immediately
    return {"status": "online", "message": "Policy AI Backend is Live"}

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def handle_query(request: QueryRequest):
    global rag_engine
    if rag_engine is None:
        from app.rag import RAGSystem
        rag_engine = RAGSystem()
    
    try:
        result = await rag_engine.query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
