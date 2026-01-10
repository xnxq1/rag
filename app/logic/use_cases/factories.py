from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.infra.config import settings
from app.logic.use_cases.chunking import ChunkingUseCase
from app.logic.use_cases.embedding import CreateEmbeddingFromRussianWordsUseCase
from app.logic.use_cases.reader import DocxReaderUseCase, PdfReaderUseCase


def embedding_use_case_factory() -> CreateEmbeddingFromRussianWordsUseCase:
    return CreateEmbeddingFromRussianWordsUseCase(
        model=SentenceTransformer("intfloat/multilingual-e5-base")
    )


def recurcive_text_splitter_use_case_factory() -> ChunkingUseCase:
    return ChunkingUseCase(
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    )


def pdf_reader_use_case_factory() -> PdfReaderUseCase:
    return PdfReaderUseCase()


def docx_reader_use_case_factory() -> DocxReaderUseCase:
    return DocxReaderUseCase()
