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

# Global variable for the engine
rag_engine = None

@app.get("/")
async def root():
    return {"status": "online", "message": "Policy AI Backend is Live"}

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def handle_query(request: QueryRequest):
    global rag_engine
    
    # Lazy Loading: We only import and load the AI when someone asks a question
    if rag_engine is None:
        try:
            from app.rag import RAGSystem
            rag_engine = RAGSystem()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load AI: {str(e)}")
    
    try:
        result = await rag_engine.query(request.query)
        return {"context": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
