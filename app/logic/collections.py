from qdrant_client.http.models import Distance, VectorParams

from app.infra.qdrant.repos.repos import QdrantRepo


class CollectionService:
    def __init__(self, qdrant_repo: QdrantRepo):
        self.qdrant_repo = qdrant_repo

    async def get_all(self):
        result = await self.qdrant_repo.client.get_collections()
        return result.collections

    async def create(self, collection_name: str, size: int):
        result = await self.qdrant_repo.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(distance=Distance.COSINE, size=size),
        )
        return result
