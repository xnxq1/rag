from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, QueryResponse, VectorParams

from app.infra.config import settings
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

    async def search(
        self, collection_name, vector, limit: int = settings.top_k_limit
    ) -> QueryResponse:
        return await self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
        )
