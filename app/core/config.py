from functools import lru_cache
from urllib.parse import urlparse

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


def _origin_variants(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    # Common copy/paste issue in env vars
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()

    if raw == "*":
        return ["*"]

    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        return [f"{scheme}://{netloc}"]

    # If the scheme is missing, assume the value is a host[:port] (optionally with a path)
    candidate = raw.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].strip().rstrip("/")
    if not candidate:
        return []
    candidate = candidate.lower()
    return [f"http://{candidate}", f"https://{candidate}"]


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:5500"
    cors_origin_regex: str = ""
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

    @model_validator(mode="after")
    def normalize_database_urls(self) -> "Settings":
        if self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        if not self.database_url_sync:
            self.database_url_sync = self.database_url

        if self.database_url_sync.startswith("postgresql+asyncpg://"):
            self.database_url_sync = self.database_url_sync.replace("postgresql+asyncpg://", "postgresql://", 1)

        return self

    @property
    def cors_origin_list(self) -> list[str]:
        raw_values: list[str] = []
        if self.cors_origins:
            raw_values.extend(self.cors_origins.split(","))
        if self.frontend_url:
            raw_values.append(self.frontend_url)

        expanded: list[str] = []
        for raw in raw_values:
            expanded.extend(_origin_variants(raw))

        # Preserve order but remove duplicates
        seen: set[str] = set()
        origins: list[str] = []
        for origin in expanded:
            if origin not in seen:
                seen.add(origin)
                origins.append(origin)
        return origins

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
