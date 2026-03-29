# main.py — Full Professional Feature Set
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import PolicyDatabase
from generator import AIGenerator
from models import QueryRequest, ComparisonRequest, VoiceRequest, HealthResponse
from speech import SpeechProcessor

# Initialize all components
policy_db = PolicyDatabase()
ai_generator = AIGenerator()
speech_processor = SpeechProcessor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load policies and check AI
    policy_db.load_policies_from_directory(settings.policy_data_path)
    await ai_generator.is_available()
    yield

app = FastAPI(title="Multilingual AI Policy Assistant", lifespan=lifespan)
from fastapi.middleware.cors import CORSMiddleware

# This is the "Permission Slip" (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your HTML file to connect
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET and POST requests
    allow_headers=["*"],  # Allows all data headers
)
# --- 1. Main Search Tool ---
@app.post("/api/query", tags=["Search"])
async def process_query(request: QueryRequest):
    query = request.question.strip()
    language = request.language or policy_db.detect_language(query)
    search_results = policy_db.search_policies(query, language=language, top_k=settings.top_k_results)
    
    if not search_results:
        return {"answer": "No matching policies found.", "detected_language": language}

    context_docs = [f"Title: {r.title}\nContent: {r.content}" for r in search_results]
    query_en = policy_db.translate_to_english(query, source_lang=language)
    ai_result = await ai_generator.generate_response(query=query_en, context_docs=context_docs)
    
    answer = ai_result.get("answer") or ""
    if language != "en" and answer:
        answer = policy_db.translate_from_english(answer, target_lang=language)
    
    return {
        "query": query, "detected_language": language, "answer": answer,
        "sources": [{"title": r.title, "relevance": r.relevance} for r in search_results]
    }

# --- 2. Comparison Tool ---
@app.post("/api/compare", tags=["Analytics"])
async def compare_policies(request: ComparisonRequest):
    result = policy_db.compare_policies(request.policy_ids, request.criteria)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

# --- 3. Policy Management ---
@app.get("/api/policies", tags=["Policies"])
async def list_policies():
    return {
        "count": len(policy_db.policies),
        "policies": [{"id": p.id, "title": p.title, "category": p.category} for p in policy_db.policies]
    }

# --- 4. System Health ---
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="healthy", policies_loaded=len(policy_db.policies),
        index_ready=policy_db.index is not None,
        ai_model=settings.ai_model_name, supported_languages=settings.supported_languages
    )
