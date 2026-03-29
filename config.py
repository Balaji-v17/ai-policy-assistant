# config.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Paths
    policy_data_path: str = "./policy_data"
    models_cache_path: str = "./models"

    # AI Model settings
    ai_model_name: str = "mistral"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Search settings
    top_k_results: int = 5
    similarity_threshold: float = 0.30

    # Language settings
    supported_languages: List[str] = ["en", "kn", "hi", "ta"]
    default_language: str = "en"

    # Speech settings
    speech_timeout: int = 5
    speech_phrase_limit: int = 10
    speech_language: str = "en-IN"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
