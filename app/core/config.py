from functools import lru_cache
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:5500"]
    frontend_url: str = "http://localhost:3000"

    # Database
    database_url: str
    database_url_sync: str = ""

    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # Groq (Qwen3-32B)
    groq_api_key: str
    groq_model: str = "qwen/qwen3-32b"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Redis
    redis_url: str = "redis://localhost:6379"

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_async_database_url(cls, v: str) -> str:
        if isinstance(v, str) and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @model_validator(mode="after")
    def normalize_database_urls(self) -> "Settings":
        if self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        if not self.database_url_sync:
            self.database_url_sync = self.database_url

        if self.database_url_sync.startswith("postgresql+asyncpg://"):
            self.database_url_sync = self.database_url_sync.replace("postgresql+asyncpg://", "postgresql://", 1)

        return self

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
