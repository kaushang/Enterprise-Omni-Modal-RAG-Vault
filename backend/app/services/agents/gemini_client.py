from openai import AsyncOpenAI
from app.core.config import settings

GEMINI_MODEL = "gemini-2.5-flash"


def get_gemini_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=settings.GEMINI_API_KEY,
    )
