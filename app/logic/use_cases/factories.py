from sentence_transformers import SentenceTransformer

from app.logic.use_cases.embedding import CreateEmbeddingFromRussianWordsUseCase


def embedding_use_case_factory() -> CreateEmbeddingFromRussianWordsUseCase:
    return CreateEmbeddingFromRussianWordsUseCase(model=SentenceTransformer("intfloat/multilingual-e5-base"))
