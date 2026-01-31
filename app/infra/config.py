from dotenv import load_dotenv

load_dotenv()

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    open_ai_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENAI_BASE_URL")
    open_ai_api_key: str = Field(alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    qdrant_host: str = Field(default="qdrant", alias="QDRANT_HOST")
    qdrant_port: str = Field(default=6333, alias="QDRANT_PORT")
    top_k_limit: int = Field(default=3, alias="TOP_K_LIMIT")
    rag_attempts: int = Field(default=3, alias="RAG_ATTEMPTS")

    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")

    app_name: str = Field(default="Simple RAG", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")

    langsmith_tracing: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="rag-project", alias="LANGSMITH_PROJECT")


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
