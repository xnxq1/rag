from app.infra.config import settings
from app.infra.llm.client import LLMClient
from app.infra.logging import get_logger
from app.infra.qdrant.repos.repos import QdrantRepo
from app.logic.use_cases.embedding import EmbeddingInterface
from app.logic.use_cases.query_rewriting import QueryRewritingUseCase
from app.logic.use_cases.reranking import CrossEncoderRerankingUseCase

logger = get_logger(__name__)
system_prompt = """Ты — ассистент по базе знаний.
Отвечай строго на основе предоставленного контекста.

Правила интерпретации:
1) Если в контексте есть информация, которая по смыслу отвечает на вопрос — это считается ответом (даже если детали не указаны).
2) Если ответ дан частично — ответь частично и укажи, чего не хватает.
3) Если информации действительно нет — скажи: “Информации недостаточно”.
4) Не додумывай внешние факты и законы вне контекста.

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
        query_rewriting_use_case: QueryRewritingUseCase,
    ):
        self.embedding_use_case = embedding_use_case
        self.qdrant_repo = qdrant_repo
        self.llm_client = llm_client
        self.cross_encode_rerank_use_case = cross_encode_rerank_use_case
        self.query_rewriting_use_case = query_rewriting_use_case

    async def execute(self, query: str, collection_name: str) -> str:
        # TODO: добавить tiktoken, проверку на контекстное окно
        # TODO: Добавить qeury rewriting, cross-encoder and llm reranking, переписатьь все на llamaindex
        query = await self.query_rewriting_use_case.handle(query)
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
