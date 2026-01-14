from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.infra.config import settings
from app.logic.use_cases.chunking import ChunkingUseCase
from app.logic.use_cases.embedding import CreateEmbeddingFromRussianWordsUseCase
from app.logic.use_cases.reader import PdfReaderUseCase
from app.logic.use_cases.reranking import CrossEncoderRerankingUseCase


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

def cross_encode_rerank_use_case_factory() -> CrossEncoderRerankingUseCase:
    return CrossEncoderRerankingUseCase(
        cross_encoder_model=CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'))