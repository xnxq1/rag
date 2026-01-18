from app.infra.llm.client import LLMClient
from app.logic.use_cases.base import UseCaseInterface


class QueryRewritingUseCase(UseCaseInterface):
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def handle(self, user_query: str):
        system_prompt = """Перепиши вопрос так, чтобы он был полным, явным и грамматически правильным.

        Правила:
        - Исправляй грамматику и опечатки
        - НЕ добавляй информацию, которой нет в оригинальном вопросе
        - НЕ придумывай контекст (сервис, продукт, компанию)
        - Сохраняй оригинальный смысл
        - Верни ТОЛЬКО переписанный вопрос, без пояснений

        Примеры:
        - "скока стоит подписка" → "Сколько стоит подписка?"
        - "как получ токен" → "Как получить токен?"
        - "отпуск сколько дней" → "Сколько дней длится отпуск?"
        """

        user_prompt = """
        Вопрос: {question}
        """

        result = await self.llm_client.completions_create(
            system_prompt=system_prompt,
            user_query=user_prompt.format(question=user_query),
        )
        return result
