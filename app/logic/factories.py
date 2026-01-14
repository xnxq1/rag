from app.infra.llm.factories import llm_client_factory
from app.infra.qdrant.repos.factories import qdrant_repo_factory
from app.logic.collections import CollectionService
from app.logic.ingest import IngestPipeline
from app.logic.rag import RAGPipeline
from app.logic.use_cases.factories import (
    embedding_use_case_factory,
    pdf_reader_use_case_factory,
    recurcive_text_splitter_use_case_factory, cross_encode_rerank_use_case_factory,
)


def ingest_pipeline_factory() -> IngestPipeline:
    return IngestPipeline(
        embedding_use_case=embedding_use_case_factory(),
        chunking_use_case=recurcive_text_splitter_use_case_factory(),
        pdf_reader_use_case=pdf_reader_use_case_factory(),
        qdrant_repo=qdrant_repo_factory(),
    )


def rag_pipeline_factory() -> RAGPipeline:
    return RAGPipeline(
        embedding_use_case=embedding_use_case_factory(),
        qdrant_repo=qdrant_repo_factory(),
        llm_client=llm_client_factory(),
        cross_encode_rerank_use_case=cross_encode_rerank_use_case_factory(),
    )


def collection_service_factory() -> CollectionService:
    return CollectionService(
        qdrant_repo=qdrant_repo_factory(),
    )
