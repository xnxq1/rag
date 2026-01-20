import asyncio

from app.infra.config import settings
from app.infra.llm.tracing import tracer
from app.infra.logging import get_logger
from app.infra.qdrant.repos.repos import QdrantRepo
from app.logic.use_cases.bm25 import BM25UseCase
from app.logic.use_cases.embedding import EmbeddingInterface
from app.logic.use_cases.query_rewriting import QueryRewritingUseCase

logger = get_logger(__name__)


class RetrievalContextSubPipeline:
    def __init__(
        self,
        embedding_use_case: EmbeddingInterface,
        qdrant_repo: QdrantRepo,
        query_rewriting_use_case: QueryRewritingUseCase,
        bm25_use_case: BM25UseCase,
    ):
        self.embedding_use_case = embedding_use_case
        self.qdrant_repo = qdrant_repo
        self.query_rewriting_use_case = query_rewriting_use_case
        self.bm25_use_case = bm25_use_case

    @tracer.trace(name="Retrieval sub pipeline", run_type="retriever")
    async def execute(self, query: str, collection_name: str) -> list:
        query = await self.query_rewriting_use_case.handle(query)
        logger.info(f"Rewriting query {query}")
        embedding, sparse_embedding = await asyncio.gather(
            self.embedding_use_case.handle(sentences=[query], mode="query"),
            self.bm25_use_case.handle(docs=[query]),
        )
        search_result = await self.qdrant_repo.hybrid_search(
            vector=embedding[0].tolist(),
            sparse_vector=sparse_embedding[0],
            collection_name=collection_name,
            limit=settings.top_k_limit,
        )
        context = [point.payload["text"] for point in search_result.points]
        return context
