from app.infra.config import settings
from app.infra.llm.client import LLMClient
from app.infra.logging import get_logger
from app.infra.qdrant.repos.repos import QdrantRepo
from app.logic.use_cases.embedding import EmbeddingInterface
from app.logic.use_cases.reranking import CrossEncoderRerankingUseCase

logger = get_logger(__name__)
system_prompt = """Ты — ассистент по внутреннему FAQ компании.
Отвечай строго на основе предоставленного контекста.

Правила интерпретации:
1) Ключевой смысл вопроса — это то, что пользователь хочет узнать в терминах FAQ, даже если формулировка непрямая.
2) Если в контексте есть информация, которая по смыслу отвечает на вопрос — это считается ответом (даже если детали не указаны).
3) Если ответ дан частично — ответь частично и укажи, чего не хватает.
4) Если информации действительно нет — скажи: “Информации недостаточно”.
5) Не додумывай внешние факты и законы вне контекста.
6) Если вопрос не относится к тематике FAQ (шутка, личное, бытовое) — отвечай: “Вопрос вне тематики FAQ”.

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
        cross_encode_rerank_use_case: CrossEncoderRerankingUseCase,
    ):
        self.embedding_use_case = embedding_use_case
        self.qdrant_repo = qdrant_repo
        self.llm_client = llm_client
        self.cross_encode_rerank_use_case = cross_encode_rerank_use_case

    async def execute(self, query: str, collection_name: str) -> str:
        # TODO: добавить tiktoken, проверку на контекстное окно
        # TODO: Добавить qeury rewriting, cross-encoder and llm reranking, переписатьь все на llamaindex
        embedding = await self.embedding_use_case.handle(sentences=[query], mode="query")
        search_result = await self.qdrant_repo.search(
            vector=embedding[0].tolist(), collection_name=collection_name, limit=settings.top_k_limit,
        )
        context = [point.payload["text"] for point in search_result.points]
        rerank_context = await self.cross_encode_rerank_use_case.handle(query=query, docs=context, limit=3)
        context = "\n-------\n".join([text for text in rerank_context])
        logger.info(f'Final context: {context}')
        user_prompt = prompt.format(context=context, query=query)
        return await self.llm_client.completions_create(
            system_prompt=system_prompt, user_query=user_prompt
        )
