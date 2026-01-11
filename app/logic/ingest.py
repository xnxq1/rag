import asyncio
import dataclasses
import uuid
from typing import BinaryIO

from numpy import ndarray

from app.infra.logging import get_logger
from app.infra.qdrant.repos.interfaces import QdrantPoint
from app.infra.qdrant.repos.repos import QdrantRepo
from app.logic.exceptions import NotSupportedFormatError
from app.logic.use_cases.chunking import ChunkingUseCase
from app.logic.use_cases.embedding import CreateEmbeddingFromRussianWordsUseCase
from app.logic.use_cases.reader import PageMetaData, PdfReaderUseCase

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


class IngestPipeline:
    def __init__(
        self,
        embedding_use_case: CreateEmbeddingFromRussianWordsUseCase,
        chunking_use_case: ChunkingUseCase,
        pdf_reader_use_case: PdfReaderUseCase,
        qdrant_repo: QdrantRepo,
    ):
        self.embedding_use_case = embedding_use_case
        self.chunking_use_case = chunking_use_case
        self.pdf_reader_use_case = pdf_reader_use_case
        self.qdrant_repo = qdrant_repo
        self.format_to_reader_map = {
            "pdf": self.pdf_reader_use_case,
        }

    async def execute(self, file: File, collection_name: str):
        reader = self.format_to_reader_map.get(file.filename.split(".")[-1])
        if not reader:
            raise NotSupportedFormatError(
                f"Incorrect file format, we only support {list(self.format_to_reader_map.keys())}"
            )
        pages = await reader.handle(document=file.data)
        processed_pages = await self._process_pages_parallel(pages, file.filename)
        points = []

        for page in processed_pages:
            logger.info(f"Processing page {page.metadata}. Chunks: {page.chunks}")
            for embedding in page.embeddings:
                points.append(QdrantPoint(vector=embedding, id=uuid.uuid4(), payload=page.metadata))

        await self.qdrant_repo.create_or_update_vector(
            collection_name=collection_name, points=points
        )
        logger.debug(f"Processed {len(processed_pages)} pages")

    async def _process_pages_parallel(
        self, pages: list[PageMetaData], filename: str
    ) -> list[ProcessedPage]:
        semaphore = asyncio.Semaphore(5)

        async def process_single_page(page: PageMetaData) -> ProcessedPage:
            async with semaphore:
                chunks = await self.chunking_use_case.handle(page.text)
                embeddings = await self.embedding_use_case.handle(chunks, mode="passage")

                return ProcessedPage(
                    chunks=chunks,
                    metadata=page.metadata | {"filename": filename, "text": page.text},
                    embeddings=embeddings,
                )

        return await asyncio.gather(*[process_single_page(page) for page in pages])
