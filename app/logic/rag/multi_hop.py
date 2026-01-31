import asyncio
from itertools import chain

from app.infra.llm.tracing import tracer
from app.infra.logging import get_logger
from app.logic.retrieval import RetrievalContextSubPipeline
from app.logic.use_cases.reranking import CrossEncoderRerankingUseCase

logger = get_logger(__name__)

class MultiHopContextSubPipeline:
    def __init__(
        self,
        cross_encode_rerank_use_case: CrossEncoderRerankingUseCase,
        retrieval_context_sub_pipeline: RetrievalContextSubPipeline,
    ):
        self.cross_encode_rerank_use_case = cross_encode_rerank_use_case
        self.retrieval_context_sub_pipeline = retrieval_context_sub_pipeline

    @tracer.trace(name="Multi Hop SubPipeline", run_type="chain")
    async def execute(self, queries: list, collection_name: str) -> list:
        contexts = await asyncio.gather(*(self.retrieval_context_sub_pipeline.execute(
            query=query, collection_name=collection_name
        ) for query in queries))
        rerank_contexts = await asyncio.gather(*(self.cross_encode_rerank_use_case.handle(
            query=query, docs=context, limit=3
        ) for context, query in zip(contexts, queries)))
        seen = set()
        unique_docs = []
        for doc in chain.from_iterable(rerank_contexts):
            if doc not in seen:
                seen.add(doc)
                unique_docs.append(doc)

        print(seen)
        return unique_docs
