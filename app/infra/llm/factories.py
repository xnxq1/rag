from openai import AsyncOpenAI

from app.infra.config import settings
from app.infra.llm.client import LLMClient


def llm_client_factory() -> LLMClient:
    return LLMClient(
        client=AsyncOpenAI(
            base_url=settings.open_ai_base_url,
            api_key=settings.open_ai_api_key,
        )
    )
