from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os

class Settings(BaseSettings):
    """
    Central configuration for the Multimodal RAG system.
    Uses Pydantic Settings for environment variable management.
    """
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    log_level: str = "INFO"
    
    # Storage Paths
    vector_db_path: str = "./data/faiss_index"
    metadata_db_path: str = "./data/metadata.json"
    
    # Model Config
    embedding_model_name: str = "all-MiniLM-L6-v2"
    llm_model_name: str = "gemini-1.5-flash"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Singleton instance
settings = Settings()
