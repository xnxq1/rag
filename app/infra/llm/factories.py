from langsmith import wrappers
from openai import AsyncOpenAI

from app.infra.config import settings
from app.infra.llm.client import LLMClient


def create_openai_client() -> AsyncOpenAI:
    client = AsyncOpenAI(
        api_key=settings.open_ai_api_key,
        base_url=settings.open_ai_base_url,
    )

    if settings.langsmith_tracing:
        client = wrappers.wrap_openai(client)

    return client


def llm_client_factory() -> LLMClient:
    return LLMClient(client=create_openai_client())
