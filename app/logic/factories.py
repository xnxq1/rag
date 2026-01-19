from app.infra.llm.factories import llm_client_factory
from app.infra.qdrant.repos.factories import qdrant_repo_factory
from app.logic.collections import CollectionService
from app.logic.ingest import IngestPipeline
from app.logic.rag import RAGPipeline
from app.logic.retrieval import RetrievalContextSubPipeline
from app.logic.use_cases.factories import (
    bm25_use_case_factory,
    cross_encode_rerank_use_case_factory,
    embedding_use_case_factory,
    load_url_content_reader_use_case_factory,
    query_rewriting_use_case_factory,
    recurcive_text_splitter_use_case_factory,
)


def ingest_pipeline_factory() -> IngestPipeline:
    return IngestPipeline(
        embedding_use_case=embedding_use_case_factory(),
        chunking_use_case=recurcive_text_splitter_use_case_factory(),
        qdrant_repo=qdrant_repo_factory(),
        load_url_content_use_case=load_url_content_reader_use_case_factory(),
        bm25_use_case=bm25_use_case_factory(),
    )


def rag_pipeline_factory() -> RAGPipeline:
    return RAGPipeline(
        llm_client=llm_client_factory(),
        cross_encode_rerank_use_case=cross_encode_rerank_use_case_factory(),
        retrieval_context_sub_pipeline=retrieval_context_sub_pipeline(),
    )


def collection_service_factory() -> CollectionService:
    return CollectionService(
        qdrant_repo=qdrant_repo_factory(),
    )


def retrieval_context_sub_pipeline() -> RetrievalContextSubPipeline:
    return RetrievalContextSubPipeline(
        embedding_use_case=embedding_use_case_factory(),
        query_rewriting_use_case=query_rewriting_use_case_factory(),
        qdrant_repo=qdrant_repo_factory(),
        bm25_use_case=bm25_use_case_factory(),
    )
