# models.py
from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import BaseModel, Field

# ─── Internal data structures ────────────────────────────────────────────────

@dataclass
class PolicyDocument:
    """Represents a single policy document stored in the vector index."""
    id: str
    title: str
    content: str
    section: str
    category: str
    language: str
    keywords: List[str]
    date: str
    source: str

@dataclass
class QueryResult:
    """A single hit returned by the vector search."""
    policy_id: str
    title: str
    content: str          # truncated excerpt
    score: float          # cosine similarity [0, 1]
    section: str
    relevance: str        # human-readable percentage

# ─── API request / response schemas ──────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    language: Optional[str] = Field(None, pattern=r"^[a-z]{2}$")
    use_voice: bool = False
    voice_timeout: int = Field(5, ge=1, le=30)

class ComparisonRequest(BaseModel):
    policy_ids: List[str] = Field(..., min_length=2)
    criteria: Optional[List[str]] = None

class VoiceRequest(BaseModel):
    timeout: int = Field(5, ge=1, le=30)

class SourceInfo(BaseModel):
    id: int
    title: str
    section: str
    excerpt: str
    relevance: str
    full_content_available: bool = True

class QueryResponse(BaseModel):
    query: str
    detected_language: str
    answer: str
    sources: List[SourceInfo]
    confidence: str
    search_results: int

class HealthResponse(BaseModel):
    status: str
    policies_loaded: int
    index_ready: bool
    ai_model: Optional[str]
    supported_languages: List[str]
