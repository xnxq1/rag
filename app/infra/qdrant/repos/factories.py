from qdrant_client import AsyncQdrantClient

from app.infra.qdrant.repos.repos import QdrantRepo


def qdrant_repo_factory():
    # TODO: вынести параметры в env
    return QdrantRepo(
        client=AsyncQdrantClient(
            host="qdrant",
            port=6333,
        )
    )
