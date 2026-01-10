from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from app.infra.config import settings


class LLMClient:
    def __init__(
        self,
        client: AsyncOpenAI,
    ):
        self.client = client

    async def completions_create(self, system_prompt: str, user_query: str) -> str:
        # TODO: добавить лог по метаданным запроса
        result = await self.client.chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=0,
            messages=[
                ChatCompletionSystemMessageParam(content=system_prompt, role="system"),
                ChatCompletionUserMessageParam(content=user_query, role="user"),
            ],
        )
        return result.choices[0].message.content
