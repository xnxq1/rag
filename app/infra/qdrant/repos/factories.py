from qdrant_client import AsyncQdrantClient

from app.infra.config import settings
from app.infra.qdrant.repos.repos import QdrantRepo


def qdrant_repo_factory() -> QdrantRepo:
    return QdrantRepo(
        client=AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    )
