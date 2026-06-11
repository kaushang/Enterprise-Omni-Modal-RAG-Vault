from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    APP_ENV: str = "development"
    RESEND_API_KEY: str = "re_YuPvPF1Q_D896vzzUHrrqzmcE1wFqPDju"
    FRONTEND_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
