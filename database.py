# database.py
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import faiss
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException, DetectorFactory
from sentence_transformers import SentenceTransformer

# Make language detection deterministic across runs
DetectorFactory.seed = 0

from config import settings
from models import PolicyDocument, QueryResult

logger = logging.getLogger(__name__)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows so that inner-product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid div-by-zero
    return (vectors / norms).astype("float32")


class PolicyDatabase:
    """Manages policy documents and vector similarity search."""

    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        self.policies: List[PolicyDocument] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self._dimension: int = 384

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model '%s' …", settings.embedding_model)
            self._model = SentenceTransformer(settings.embedding_model)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def load_policies_from_directory(self, directory_path: str) -> None:
        """Load every *.json file in *directory_path* as a PolicyDocument."""
        policy_dir = Path(directory_path)
        if not policy_dir.exists():
            logger.warning("Policy directory '%s' does not exist — skipping.", directory_path)
            return

        loaded: List[PolicyDocument] = []
        for json_file in sorted(policy_dir.glob("*.json")):
            doc = self._parse_policy_file(json_file)
            if doc:
                loaded.append(doc)

        self.policies = loaded
        self._build_index()
        logger.info("Loaded %d policy document(s).", len(self.policies))

    def detect_language(self, text: str) -> str:
        """Detect language using langdetect (offline)."""
        if len(text.strip()) < 15:
            return settings.default_language
        try:
            lang = detect(text)
            return lang if lang in settings.supported_languages else settings.default_language
        except LangDetectException:
            return settings.default_language

    def translate_to_english(self, text: str, source_lang: str = "auto") -> str:
        if source_lang == "en":
            return text
        try:
            return GoogleTranslator(source=source_lang, target="en").translate(text)
        except Exception as exc:
            logger.warning("Translation to English failed: %s", exc)
            return text

    def translate_from_english(self, text: str, target_lang: str) -> str:
        if target_lang == "en":
            return text
        try:
            chunk_size = 4500
            if len(text) <= chunk_size:
                return GoogleTranslator(source="en", target=target_lang).translate(text)
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            translated_chunks = [
                GoogleTranslator(source="en", target=target_lang).translate(chunk)
                for chunk in chunks
            ]
            return " ".join(translated_chunks)
        except Exception as exc:
            logger.warning("Translation from English to '%s' failed: %s", target_lang, exc)
            return text

    def search_policies(self, query: str, language: str = "en", top_k: int = 5) -> List[QueryResult]:
        if self.index is None or not self.policies:
            return []

        query_en = self.translate_to_english(query, source_lang=language)
        raw = self.embedding_model.encode([query_en], convert_to_numpy=True)
        query_vec = _normalise(raw)

        k = min(top_k, len(self.policies))
        scores, indices = self.index.search(query_vec, k)

        results: List[QueryResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.policies): continue
            if float(score) < settings.similarity_threshold: continue

            policy = self.policies[idx]
            results.append(QueryResult(
                policy_id=policy.id, title=policy.title, 
                content=policy.content[:500] + "…", score=float(score),
                section=policy.section, relevance=f"{min(100, int(float(score) * 100))}%"
            ))
        return results

    def get_policy_by_id(self, policy_id: str) -> Optional[PolicyDocument]:
        return next((p for p in self.policies if p.id == policy_id), None)

    def compare_policies(self, policy_ids: List[str], criteria: Optional[List[str]] = None) -> Dict:
        if criteria is None: criteria = ["benefits", "eligibility"]
        matched = [p for p in self.policies if p.id in policy_ids]
        if len(matched) < 2: return {"error": "Not enough policies matched."}
        
        policy_details = []
        for policy in matched:
            details = {c: [s.strip() for s in policy.content.split(".") if c.lower() in s.lower()][:2] for c in criteria}
            policy_details.append({"id": policy.id, "title": policy.title, "details": details})
        return {"policies": policy_details}

    def _parse_policy_file(self, path: Path) -> Optional[PolicyDocument]:
        try:
            with open(path, encoding="utf-8") as fh: data = json.load(fh)
            return PolicyDocument(
                id=str(uuid.uuid4()), title=data.get("title", ""),
                content=data.get("content", ""), section=data.get("section", ""),
                category=data.get("category", ""), language=data.get("language", "en"),
                keywords=data.get("keywords", []), date=data.get("date", ""), source=data.get("source", "")
            )
        except Exception: return None

    def _build_index(self) -> None:
        if not self.policies: return
        texts = [f"{p.title} {p.content}" for p in self.policies]
        raw = self.embedding_model.encode(texts, convert_to_numpy=True)
        self.index = faiss.IndexFlatIP(self._dimension)
        self.index.add(_normalise(raw))
