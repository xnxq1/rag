from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.infra.config import settings
from app.logic.use_cases.chunking import ChunkingUseCase
from app.logic.use_cases.embedding import CreateEmbeddingFromRussianWordsUseCase
from app.logic.use_cases.reader import PdfReaderUseCase


def embedding_use_case_factory() -> CreateEmbeddingFromRussianWordsUseCase:
    return CreateEmbeddingFromRussianWordsUseCase(
        model=SentenceTransformer("intfloat/multilingual-e5-base")
    )


def recurcive_text_splitter_use_case_factory() -> ChunkingUseCase:
    return ChunkingUseCase(
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    )


def pdf_reader_use_case_factory() -> PdfReaderUseCase:
    return PdfReaderUseCase()
