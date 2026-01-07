from abc import ABC, abstractmethod

from qdrant_client.conversions.common_types import VectorStruct


class QdrantInterface(ABC):

    @abstractmethod
    async def create_collection(self, collection_name: str, size: int) -> None:
        ...

    @abstractmethod
    async def create_or_update_vector(self, collection_name: str, vector: VectorStruct, payload: dict):
        ...