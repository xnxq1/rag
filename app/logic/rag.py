from app.infra.llm.client import LLMClient
from app.infra.llm.tracing import tracer
from app.infra.logging import get_logger
from app.logic.retrieval import RetrievalContextSubPipeline
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
        llm_client: LLMClient,
        cross_encode_rerank_use_case: CrossEncoderRerankingUseCase,
        retrieval_context_sub_pipeline: RetrievalContextSubPipeline,
    ):
        self.llm_client = llm_client
        self.cross_encode_rerank_use_case = cross_encode_rerank_use_case
        self.retrieval_context_sub_pipeline = retrieval_context_sub_pipeline

    @tracer.trace(name="RAG Pipeline", run_type="chain")
    async def execute(self, query: str, collection_name: str) -> dict:
        # TODO: добавить tiktoken, проверку на контекстное окно
        # TODO: Добавить qeury rewriting, cross-encoder and llm reranking, переписатьь все на llamaindex
        context = await self.retrieval_context_sub_pipeline.execute(
            query=query, collection_name=collection_name
        )
        rerank_context = await self.cross_encode_rerank_use_case.handle(
            query=query, docs=context, limit=3
        )
        context = "\n-------\n".join([text for text in rerank_context])
        logger.info(f"Final context: {context}")
        user_prompt = prompt.format(context=context, query=query)
        answer = await self.llm_client.completions_create(
            system_prompt=system_prompt, user_query=user_prompt
        )
        return {
            "answer": answer,
            "context": context,
        }
