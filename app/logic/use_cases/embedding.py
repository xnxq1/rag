import abc
import asyncio
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from app.logic.use_cases.base import UseCaseInterface


class EmbeddingInterface(UseCaseInterface, abc.ABC):
    def __init__(self, model: SentenceTransformer):
        self.model = model


class CreateEmbeddingFromRussianWordsUseCase(EmbeddingInterface):
    async def handle(self, sentences: list[str], mode: Literal["passage", "query"]) -> np.ndarray:
        prefixed = [
            f"{mode}: {text}" if not text.startswith(f"{mode}:") else text for text in sentences
        ]

        return await asyncio.to_thread(self.model.encode, prefixed, normalize_embeddings=True)
