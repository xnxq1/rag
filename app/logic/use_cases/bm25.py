from typing import TypedDict

from fastembed import SparseTextEmbedding
from fastembed.common.types import NumpyArray
from numpy.typing import NDArray

from app.logic.use_cases.base import UseCaseInterface


class SparseEmbedding(TypedDict):
    values: NumpyArray
    indices: NDArray


class BM25UseCase(UseCaseInterface):
    def __init__(self, model: SparseTextEmbedding):
        self.model = model

    async def handle(self, docs: list) -> list[SparseEmbedding]:
        embeddings = list(self.model.embed(docs))
        return [emb.as_object() for emb in embeddings]
