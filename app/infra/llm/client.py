from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from app.infra.config import settings
from app.infra.logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(
        self,
        client: AsyncOpenAI,
    ):
        self.client = client

    async def completions_create(self, system_prompt: str, user_query: str) -> str:
        result = await self.client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.2,
            messages=[
                ChatCompletionSystemMessageParam(content=system_prompt, role="system"),
                ChatCompletionUserMessageParam(content=user_query, role="user"),
            ],
        )
        total_tokens = result.usage.total_tokens
        input_tokens = result.usage.prompt_tokens
        output_tokens = result.usage.completion_tokens
        logger.info(
            f"Query to {settings.llm_model} "
            f"input_tokens: {input_tokens}, output_tokens: {output_tokens}, total_tokens: {total_tokens} "
        )

        return result.choices[0].message.content
