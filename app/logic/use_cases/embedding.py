import abc
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingInterface(abc.ABC):
    def __init__(self, model: SentenceTransformer):
        self.model = model

    @abc.abstractmethod
    def handle(self, *args, **kwargs) -> np.ndarray: ...


class CreateEmbeddingFromRussianWordsUseCase(EmbeddingInterface):
    def handle(self, sentences: list[str], mode: Literal["passage", "query"]) -> np.ndarray:
        prefixed = [
            f"{mode}: {text}" if not text.startswith(f"{mode}:") else text for text in sentences
        ]

        return self.model.encode(prefixed, normalize_embeddings=True)
