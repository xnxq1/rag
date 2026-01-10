from app.infra.llm.client import LLMClient
from app.infra.qdrant.repos.repos import QdrantRepo
from app.logic.use_cases.embedding import EmbeddingInterface

system_prompt = """Ты — ассистент для ответа на FAQ компании.
Отвечай строго на основе предоставленного контекста.

Правила:
- используй только факты из контекста
- не придумывай информацию и не галлюцинируй
- если информации недостаточно — скажи "В контексте нет ответа"
- отвечай кратко (1–3 предложения)
- если есть несколько пунктов — делай списком
"""

prompt = """
        Контекст:
        {context}

        Вопрос пользователя: "{query}"

        Ответ:
        """


class RAGPipeline:
    def __init__(
        self,
        embedding_use_case: EmbeddingInterface,
        qdrant_repo: QdrantRepo,
        llm_client: LLMClient,
    ):
        self.embedding_use_case = embedding_use_case
        self.qdrant_repo = qdrant_repo
        self.llm_client = llm_client

    async def execute(self, query: str, collection_name: str) -> str:
        # TODO: добавить tiktoken, проверку на контекстное окно
        embedding = await self.embedding_use_case.handle(sentences=[query], mode="query")
        search_result = await self.qdrant_repo.search(
            vector=embedding[0].tolist(), collection_name=collection_name
        )
        context = "\n".join([point.payload["text"] for point in search_result.points])
        user_prompt = prompt.format(context=context, query=query)
        return await self.llm_client.completions_create(
            system_prompt=system_prompt, user_query=user_prompt
        )
