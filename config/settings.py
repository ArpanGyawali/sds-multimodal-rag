"""Configuration settings for the RAG system."""

import os
from typing import Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    
    # Weaviate Configuration
    weaviate_url: str = Field(..., env="WEAVIATE_URL")
    weaviate_api_key: str = Field(..., env="WEAVIATE_API_KEY")
    weaviate_class_name: str = Field(default="SDS_MultimodalDocument", env="WEAVIATE_CLASS_NAME")
    
    # LangSmith Configuration
    langchain_tracing_v2: str = Field(default="true", env="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(..., env="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="pdf-rag-system", env="LANGCHAIN_PROJECT")
    
    # Processing Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_tokens: int = Field(default=4000, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def get_weaviate_schema() -> Dict[str, Any]:
    """Define Weaviate schema for document chunks."""
    return {
        "class": "DocumentChunk",
        "description": "A chunk of document content with metadata",
        "vectorizer": "none",  # We'll provide vectors manually
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "The text content of the chunk"
            },
            {
                "name": "content_type",
                "dataType": ["string"],
                "description": "Type of content: text, table, or figure"
            },
            {
                "name": "page_number",
                "dataType": ["int"],
                "description": "Page number in the document"
            },
            {
                "name": "chunk_index",
                "dataType": ["int"],
                "description": "Index of the chunk within the page"
            },
            {
                "name": "document_name",
                "dataType": ["string"],
                "description": "Name of the source document"
            },
            {
                "name": "bbox",
                "dataType": ["string"],
                "description": "Bounding box coordinates as JSON string"
            },
            {
                "name": "metadata",
                "dataType": ["string"],
                "description": "Additional metadata as JSON string"
            }
        ]
    }


# Global settings instance
settings = Settings()