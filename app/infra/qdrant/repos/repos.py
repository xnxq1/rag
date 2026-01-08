from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.infra.qdrant.repos.exceptions import CollectionNotExistError
from app.infra.qdrant.repos.interfaces import QdrantInterface, QdrantPoint


class QdrantRepo(QdrantInterface):
    def __init__(self, client: AsyncQdrantClient):
        self.client = client

    async def create_collection(self, collection_name: str, size: int) -> None:
        if not await self.client.collection_exists(collection_name=collection_name):
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )

    async def create_or_update_vector(
        self, collection_name: str, points: list[QdrantPoint]
    ) -> None:
        if not await self.client.collection_exists(collection_name=collection_name):
            raise CollectionNotExistError(f"Collection {collection_name} does not exist")
        await self.client.upsert(
            collection_name=collection_name,
            points=points,
        )
