from openai import AsyncOpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.1"


def get_ollama_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
    )
