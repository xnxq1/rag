import asyncio
import dataclasses
import uuid
from typing import BinaryIO

from numpy import ndarray
from qdrant_client.http.models import models
from yarl import URL

from app.infra.logging import get_logger
from app.infra.qdrant.repos.interfaces import QdrantPoint
from app.infra.qdrant.repos.repos import QdrantRepo
from app.logic.exceptions import NotSupportedFormatError
from app.logic.use_cases.bm25 import BM25UseCase, SparseEmbedding
from app.logic.use_cases.chunking import ChunkingUseCase
from app.logic.use_cases.embedding import CreateEmbeddingFromRussianWordsUseCase
from app.logic.use_cases.reader import LoadUrlContentUseCase, PageMetaData

logger = get_logger(__name__)


@dataclasses.dataclass
class File:
    data: BinaryIO
    filename: str


@dataclasses.dataclass
class ProcessedPage:
    chunks: list
    metadata: dict
    embeddings: ndarray
    sparse_embeddings: list[SparseEmbedding]


class IngestPipeline:
    def __init__(
        self,
        embedding_use_case: CreateEmbeddingFromRussianWordsUseCase,
        chunking_use_case: ChunkingUseCase,
        qdrant_repo: QdrantRepo,
        load_url_content_use_case: LoadUrlContentUseCase,
        bm25_use_case: BM25UseCase,
    ):
        self.embedding_use_case = embedding_use_case
        self.chunking_use_case = chunking_use_case
        self.qdrant_repo = qdrant_repo
        self.load_url_content_use_case = load_url_content_use_case
        self.bm25_use_case = bm25_use_case
        self.format_to_reader_map = {
            "url": self.load_url_content_use_case,
        }

    async def execute(self, content_type: str, collection_name: str, url: URL):
        reader = self.format_to_reader_map.get(content_type)
        if not reader:
            raise NotSupportedFormatError(
                f"Incorrect file format, we only support {list(self.format_to_reader_map.keys())}"
            )
        pages = await reader.handle(url=url)
        processed_pages = await self._process_pages_parallel(pages)
        points = []

        for page in processed_pages:
            # logger.info(f"Processing page {page.metadata}. Chunks: {page.chunks}")
            for embedding, sparse_embedding in zip(page.embeddings, page.sparse_embeddings):
                points.append(
                    QdrantPoint(
                        vector={
                            "dense": embedding,
                            "sparse": models.SparseVector(
                                indices=sparse_embedding['indices'].tolist(),
                                values=sparse_embedding['values'].tolist(),
                            ),
                        },
                        id=uuid.uuid4(),
                        payload=page.metadata,
                    )
                )

        await self.qdrant_repo.create_or_update_vector(
            collection_name=collection_name, points=points
        )
        logger.debug(f"Processed {len(processed_pages)} pages")

    async def _process_pages_parallel(self, pages: list[PageMetaData]) -> list[ProcessedPage]:
        semaphore = asyncio.Semaphore(5)

        async def process_single_page(page: PageMetaData) -> ProcessedPage:
            async with semaphore:
                chunks = await self.chunking_use_case.handle(page.text)
                embeddings = await self.embedding_use_case.handle(chunks, mode="passage")
                sparse_embeddings = await self.bm25_use_case.handle(chunks)
                return ProcessedPage(
                    chunks=chunks,
                    metadata=page.metadata | {"text": page.text},
                    embeddings=embeddings,
                    sparse_embeddings=sparse_embeddings,
                )

        return await asyncio.gather(*[process_single_page(page) for page in pages])
