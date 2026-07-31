from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/rag_vault"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    SECRET_KEY: str = "your-secret-key-change-this"
    DATABASE_ENCRYPTION_KEY: Optional[str] = None
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    APP_ENV: str = "development"
    RESEND_API_KEY: Optional[str] = None
    FRONTEND_URL: str = "http://localhost:5173"

    GOOGLE_CLIENT_ID: str = "your-google-client-id"
    GOOGLE_CLIENT_SECRET: str = "your-google-client-secret"
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"

    REDIS_URL: str = "redis://localhost:6379/0"

    ENABLE_CROSS_ENCODER_RERANKING: bool = True
    ENABLE_SPARSE_SEARCH: bool = True

    CHAT_BURST_LIMIT: int = 3
    CHAT_BURST_WINDOW_SECONDS: int = 10
    CHAT_SUSTAINED_LIMIT: int = 10
    CHAT_SUSTAINED_WINDOW_SECONDS: int = 60

    SQL_HISTORY_LIMIT: int = 5
    SQL_RESULT_SUMMARY_THRESHOLD: int = 20
    REPORTS_DIR: str = "./reports"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
